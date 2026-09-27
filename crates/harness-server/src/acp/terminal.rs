//! Client-side `terminal/*` implementation.

use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use agent_client_protocol_schema::v1::{
    CreateTerminalRequest, CreateTerminalResponse, KillTerminalResponse, ReleaseTerminalResponse,
    TerminalExitStatus, TerminalId, TerminalOutputResponse, WaitForTerminalExitResponse,
};
use uuid::Uuid;

use super::{AcpError, Result};

/// Final captured output and exit status for later `commandExecution` mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalSnapshot {
    pub output: String,
    pub truncated: bool,
    pub exit_code: Option<u32>,
    pub signal: Option<String>,
}

struct TerminalInner {
    output: Vec<u8>,
    truncated: bool,
    output_byte_limit: Option<u64>,
    exit: Option<TerminalExitStatus>,
    pgid: Option<i32>,
    child: Option<Child>,
}

struct LiveTerminal {
    inner: Mutex<TerminalInner>,
    exit_cv: Condvar,
    remaining_readers: AtomicUsize,
}

/// Runs `sh -c` commands in their own process group and serves ACP terminal RPCs.
pub struct TerminalManager {
    session_cwd: Mutex<PathBuf>,
    live: Mutex<HashMap<String, Arc<LiveTerminal>>>,
    snapshots: Mutex<HashMap<String, TerminalSnapshot>>,
}

impl TerminalManager {
    pub fn new(session_cwd: PathBuf) -> Self {
        Self {
            session_cwd: Mutex::new(session_cwd),
            live: Mutex::new(HashMap::new()),
            snapshots: Mutex::new(HashMap::new()),
        }
    }

    pub fn set_session_cwd(&self, cwd: PathBuf) {
        *self.session_cwd.lock().expect("session cwd") = cwd;
    }

    pub fn create(&self, params: &CreateTerminalRequest) -> Result<CreateTerminalResponse> {
        let session_cwd = self.session_cwd.lock().expect("session cwd").clone();
        let cwd = params.cwd.clone().unwrap_or(session_cwd);
        let script = shell_script(&params.command, &params.args);
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg(script)
            .current_dir(&cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for variable in &params.env {
            command.env(&variable.name, &variable.value);
        }
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }

        let mut child = command.spawn()?;
        let pgid = child.id() as i32;
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let reader_count = usize::from(stdout.is_some()) + usize::from(stderr.is_some());
        let terminal_id = Uuid::new_v4().to_string();
        let live = Arc::new(LiveTerminal {
            inner: Mutex::new(TerminalInner {
                output: Vec::new(),
                truncated: false,
                output_byte_limit: params.output_byte_limit,
                exit: None,
                pgid: Some(pgid),
                child: Some(child),
            }),
            exit_cv: Condvar::new(),
            remaining_readers: AtomicUsize::new(reader_count),
        });

        self.live
            .lock()
            .expect("terminals")
            .insert(terminal_id.clone(), Arc::clone(&live));

        if let Some(stdout) = stdout {
            spawn_pipe_reader(Arc::clone(&live), stdout);
        }
        if let Some(stderr) = stderr {
            spawn_pipe_reader(Arc::clone(&live), stderr);
        }
        spawn_waiter(Arc::clone(&live));

        Ok(CreateTerminalResponse::new(TerminalId::new(terminal_id)))
    }

    pub fn output(&self, terminal_id: &TerminalId) -> Result<TerminalOutputResponse> {
        let live = self.live_or_err(terminal_id)?;
        let inner = live.inner.lock().expect("terminal");
        let mut response =
            TerminalOutputResponse::new(bytes_to_string(&inner.output), inner.truncated);
        if let Some(exit) = inner.exit.clone() {
            response = response.exit_status(exit);
        }
        Ok(response)
    }

    pub fn wait_for_exit(&self, terminal_id: &TerminalId) -> Result<WaitForTerminalExitResponse> {
        let live = self.live_or_err(terminal_id)?;
        let mut inner = live.inner.lock().expect("terminal");
        while inner.exit.is_none() {
            inner = live.exit_cv.wait(inner).expect("terminal wait");
        }
        let exit = inner.exit.clone().unwrap_or_default();
        Ok(WaitForTerminalExitResponse::new(exit))
    }

    pub fn kill(&self, terminal_id: &TerminalId) -> Result<KillTerminalResponse> {
        let live = self.live_or_err(terminal_id)?;
        kill_live(&live);
        Ok(KillTerminalResponse::new())
    }

    pub fn release(&self, terminal_id: &TerminalId) -> Result<ReleaseTerminalResponse> {
        let live = {
            let mut live_map = self.live.lock().expect("terminals");
            live_map.remove(terminal_id.0.as_ref())
        };
        let Some(live) = live else {
            return Err(AcpError::UnknownTerminal(terminal_id.0.to_string()));
        };
        kill_live(&live);
        // Wait briefly so the snapshot has an exit status when possible.
        let mut inner = live.inner.lock().expect("terminal");
        if inner.exit.is_none() {
            let (guard, _) = live
                .exit_cv
                .wait_timeout(inner, Duration::from_millis(200))
                .expect("release wait");
            inner = guard;
        }
        let snapshot = snapshot_from(&inner);
        drop(inner);
        self.snapshots
            .lock()
            .expect("snapshots")
            .insert(terminal_id.0.to_string(), snapshot);
        Ok(ReleaseTerminalResponse::new())
    }

    pub fn kill_all(&self) {
        let live = {
            let mut live_map = self.live.lock().expect("terminals");
            std::mem::take(&mut *live_map)
        };
        for (id, terminal) in live {
            kill_live(&terminal);
            let inner = terminal.inner.lock().expect("terminal");
            let snapshot = snapshot_from(&inner);
            self.snapshots
                .lock()
                .expect("snapshots")
                .insert(id, snapshot);
        }
    }

    pub fn snapshot(&self, terminal_id: &TerminalId) -> Option<TerminalSnapshot> {
        if let Some(snapshot) = self
            .snapshots
            .lock()
            .expect("snapshots")
            .get(terminal_id.0.as_ref())
            .cloned()
        {
            return Some(snapshot);
        }
        let live = self.live.lock().expect("terminals");
        live.get(terminal_id.0.as_ref()).map(|terminal| {
            let inner = terminal.inner.lock().expect("terminal");
            snapshot_from(&inner)
        })
    }

    fn live_or_err(&self, terminal_id: &TerminalId) -> Result<Arc<LiveTerminal>> {
        self.live
            .lock()
            .expect("terminals")
            .get(terminal_id.0.as_ref())
            .cloned()
            .ok_or_else(|| AcpError::UnknownTerminal(terminal_id.0.to_string()))
    }
}

impl Drop for TerminalManager {
    fn drop(&mut self) {
        self.kill_all();
    }
}

fn shell_script(command: &str, args: &[String]) -> String {
    if args.is_empty() {
        return command.to_string();
    }
    let mut script = command.to_string();
    for arg in args {
        script.push(' ');
        script.push_str(&shell_words::quote(arg));
    }
    script
}

fn spawn_pipe_reader(live: Arc<LiveTerminal>, mut pipe: impl Read + Send + 'static) {
    thread::spawn(move || {
        let mut buf = [0_u8; 4096];
        loop {
            match pipe.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    let mut inner = live.inner.lock().expect("terminal");
                    append_output(&mut inner, &buf[..n]);
                }
                Err(_) => break,
            }
        }
        let _guard = live.inner.lock().expect("terminal");
        live.remaining_readers.fetch_sub(1, Ordering::SeqCst);
        live.exit_cv.notify_all();
    });
}

fn spawn_waiter(live: Arc<LiveTerminal>) {
    thread::spawn(move || {
        let child = {
            let mut inner = live.inner.lock().expect("terminal");
            inner.child.take()
        };
        let status = child.and_then(|mut child| child.wait().ok());
        let mut inner = live.inner.lock().expect("terminal");
        while live.remaining_readers.load(Ordering::SeqCst) > 0 {
            inner = live.exit_cv.wait(inner).expect("pipe drain");
        }
        let mut exit = TerminalExitStatus::new();
        if let Some(status) = status {
            exit = exit.exit_code(status.code().map(|code| code as u32));
            #[cfg(unix)]
            {
                use std::os::unix::process::ExitStatusExt;
                if let Some(signal) = status.signal() {
                    exit = exit.signal(signal_name(signal));
                }
            }
        }
        inner.exit = Some(exit);
        inner.pgid = None;
        live.exit_cv.notify_all();
    });
}

fn append_output(inner: &mut TerminalInner, data: &[u8]) {
    inner.output.extend_from_slice(data);
    if let Some(limit) = inner.output_byte_limit
        && truncate_from_start(&mut inner.output, limit as usize)
    {
        inner.truncated = true;
    }
}

/// Drop bytes from the start until `buf.len() <= limit`, landing on a UTF-8
/// character boundary (which may drop slightly more than `len - limit`).
pub(crate) fn truncate_from_start(buf: &mut Vec<u8>, limit: usize) -> bool {
    if buf.len() <= limit {
        return false;
    }
    let mut start = buf.len() - limit;
    while start < buf.len() && (buf[start] & 0b1100_0000) == 0b1000_0000 {
        start += 1;
    }
    buf.drain(..start);
    true
}

fn bytes_to_string(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn snapshot_from(inner: &TerminalInner) -> TerminalSnapshot {
    TerminalSnapshot {
        output: bytes_to_string(&inner.output),
        truncated: inner.truncated,
        exit_code: inner.exit.as_ref().and_then(|exit| exit.exit_code),
        signal: inner.exit.as_ref().and_then(|exit| exit.signal.clone()),
    }
}

fn kill_live(live: &LiveTerminal) {
    let pgid = live.inner.lock().expect("terminal").pgid;
    if let Some(pgid) = pgid {
        kill_process_group(pgid);
    }
}

fn kill_process_group(pgid: i32) {
    #[cfg(unix)]
    unsafe {
        libc::kill(-pgid, libc::SIGKILL);
    }
    #[cfg(not(unix))]
    let _ = pgid;
}

fn signal_name(signal: i32) -> String {
    #[cfg(unix)]
    {
        match signal {
            libc::SIGKILL => "SIGKILL".to_string(),
            libc::SIGTERM => "SIGTERM".to_string(),
            libc::SIGINT => "SIGINT".to_string(),
            other => other.to_string(),
        }
    }
    #[cfg(not(unix))]
    {
        signal.to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use agent_client_protocol_schema::v1::{
        CreateTerminalRequest, EnvVariable, SessionId, TerminalId,
    };
    use uuid::Uuid;

    use super::{TerminalManager, truncate_from_start};
    use crate::acp::AcpError;

    fn manager() -> TerminalManager {
        TerminalManager::new(std::env::current_dir().expect("cwd"))
    }

    fn create(command: &str) -> CreateTerminalRequest {
        CreateTerminalRequest::new(SessionId::new("s"), command)
    }

    #[test]
    fn exit_code_zero() {
        let manager = manager();
        let created = manager.create(&create("exit 0")).expect("create");
        let wait = manager.wait_for_exit(&created.terminal_id).expect("wait");
        assert_eq!(wait.exit_status.exit_code, Some(0));
        assert_eq!(wait.exit_status.signal, None);
        let snapshot = manager.snapshot(&created.terminal_id).expect("snapshot");
        assert_eq!(snapshot.exit_code, Some(0));
    }

    #[test]
    fn exit_code_nonzero() {
        let manager = manager();
        let created = manager.create(&create("exit 7")).expect("create");
        let wait = manager.wait_for_exit(&created.terminal_id).expect("wait");
        assert_eq!(wait.exit_status.exit_code, Some(7));
    }

    #[test]
    fn cwd_and_env_are_applied() {
        let dir = std::env::temp_dir().join(format!("acp-term-cwd-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("mkdir");
        let cwd = fs::canonicalize(&dir).expect("canonicalize");
        let manager = manager();
        let mut params = CreateTerminalRequest::new(
            SessionId::new("s"),
            r#"printf '%s\n%s' "$(pwd)" "$ACP_HARNESS_TEST_VAR""#,
        );
        params.cwd = Some(cwd.clone());
        params.env = vec![EnvVariable::new("ACP_HARNESS_TEST_VAR", "merged-value")];
        let created = manager.create(&params).expect("create");
        manager.wait_for_exit(&created.terminal_id).expect("wait");
        let output = manager.output(&created.terminal_id).expect("output");
        let lines: Vec<&str> = output.output.lines().collect();
        assert_eq!(lines.len(), 2, "output={:?}", output.output);
        assert_eq!(PathBuf::from(lines[0]), cwd);
        assert_eq!(lines[1], "merged-value");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn truncation_from_start_on_multibyte_boundary() {
        // é is two bytes (c3 a9). Three of them is 6 bytes; limit 5 must drop
        // the first character entirely rather than split it.
        let mut buf = "ééé".as_bytes().to_vec();
        assert!(truncate_from_start(&mut buf, 5));
        assert_eq!(std::str::from_utf8(&buf).expect("utf8"), "éé");

        let manager = manager();
        let mut params = CreateTerminalRequest::new(SessionId::new("s"), "printf 'ééé'");
        params.output_byte_limit = Some(5);
        let created = manager.create(&params).expect("create");
        manager.wait_for_exit(&created.terminal_id).expect("wait");
        let output = manager.output(&created.terminal_id).expect("output");
        assert!(output.truncated);
        assert_eq!(output.output, "éé");
        let snapshot = manager.snapshot(&created.terminal_id).expect("snapshot");
        assert!(snapshot.truncated);
        assert_eq!(snapshot.output, "éé");
    }

    #[test]
    fn kill_and_release() {
        let manager = manager();
        let created = manager.create(&create("sleep 30")).expect("create");
        let started = Instant::now();
        manager.kill(&created.terminal_id).expect("kill");
        let wait = manager.wait_for_exit(&created.terminal_id).expect("wait");
        assert!(started.elapsed() < Duration::from_secs(5), "kill was slow");
        assert!(
            wait.exit_status.signal.is_some() || wait.exit_status.exit_code != Some(0),
            "killed process should not look like a clean exit 0: {wait:?}"
        );
        manager.release(&created.terminal_id).expect("release");
        let err = manager
            .output(&created.terminal_id)
            .expect_err("released id is unknown");
        assert!(matches!(err, AcpError::UnknownTerminal(_)));
        assert!(manager.snapshot(&created.terminal_id).is_some());
    }

    #[test]
    fn unknown_terminal_id_errors() {
        let manager = manager();
        let id = TerminalId::new("does-not-exist");
        assert!(matches!(
            manager.output(&id),
            Err(AcpError::UnknownTerminal(_))
        ));
        assert!(matches!(
            manager.wait_for_exit(&id),
            Err(AcpError::UnknownTerminal(_))
        ));
        assert!(matches!(
            manager.kill(&id),
            Err(AcpError::UnknownTerminal(_))
        ));
        assert!(matches!(
            manager.release(&id),
            Err(AcpError::UnknownTerminal(_))
        ));
    }

    #[test]
    fn stdout_and_stderr_are_merged_in_arrival_order() {
        let manager = manager();
        let created = manager
            .create(&create("printf 'out-'; printf 'err' >&2; printf 'more'"))
            .expect("create");
        manager.wait_for_exit(&created.terminal_id).expect("wait");
        let output = manager.output(&created.terminal_id).expect("output");
        assert!(
            output.output.contains("out-") && output.output.contains("err"),
            "merged output: {:?}",
            output.output
        );
    }
}
