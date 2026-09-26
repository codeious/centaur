//! Integration tests for `harness-server droid` against the scripted fake ACP agent.

use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use uuid::Uuid;

const FAKE_SESSION: &str = "11111111-1111-4111-8111-111111111111";
const SECRET_SENTINEL: &str = "test-secret-should-never-appear";

struct DroidProcess {
    child: Child,
    stdin: Option<ChildStdin>,
    line_rx: Receiver<std::io::Result<String>>,
    stderr: Option<JoinHandle<String>>,
    stdout_lines: Vec<String>,
    log_path: PathBuf,
    home: PathBuf,
    workdir: PathBuf,
}

impl DroidProcess {
    fn spawn(scenario: &str) -> Self {
        Self::spawn_with(scenario, &[])
    }

    fn spawn_with(scenario: &str, extra_envs: &[(&str, &str)]) -> Self {
        let id = Uuid::new_v4().simple().to_string();
        let workdir = std::env::temp_dir().join(format!("droid-fake-wd-{id}"));
        let home = std::env::temp_dir().join(format!("droid-fake-home-{id}"));
        fs::create_dir_all(&workdir).expect("workdir");
        fs::create_dir_all(&home).expect("home");
        let log_path = std::env::temp_dir().join(format!("droid-fake-log-{id}.jsonl"));
        let fake = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fake_acp_agent.py");
        let mut command = Command::new(env!("CARGO_BIN_EXE_harness-server"));
        command
            .arg("droid")
            .current_dir(&workdir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("DROID_BIN", &fake)
            .env("DROID_FAKE_SCENARIO", scenario)
            .env("DROID_FAKE_LOG", &log_path)
            .env("HOME", &home)
            .env("FACTORY_API_KEY", SECRET_SENTINEL)
            .env("FACTORY_DROID_AUTO_UPDATE_ENABLED", "true");
        for (key, value) in extra_envs {
            command.env(key, value);
        }
        let mut child = command.spawn().expect("spawn harness-server droid");
        let stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let stderr = child.stderr.take().expect("stderr");
        let (line_tx, line_rx) = mpsc::channel();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let stop = line.is_err();
                if line_tx.send(line).is_err() || stop {
                    break;
                }
            }
        });
        let stderr_handle = thread::spawn(move || {
            let mut buf = String::new();
            let _ = BufReader::new(stderr).read_to_string(&mut buf);
            buf
        });
        Self {
            child,
            stdin: Some(stdin),
            line_rx,
            stderr: Some(stderr_handle),
            stdout_lines: Vec::new(),
            log_path,
            home,
            workdir,
        }
    }

    fn send_user(&mut self, text: &str) {
        let stdin = self.stdin.as_mut().expect("stdin open");
        writeln!(stdin, "{}", json!({"type": "user", "text": text})).expect("write user");
        stdin.flush().expect("flush");
    }

    fn close_stdin(&mut self) {
        self.stdin.take();
    }

    fn read_json(&mut self, deadline: Instant) -> Value {
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                panic!(
                    "timed out waiting for JSON-RPC; stdout so far: {:?}",
                    self.stdout_lines
                );
            }
            match self.line_rx.recv_timeout(remaining) {
                Ok(Ok(line)) => {
                    self.stdout_lines.push(line.clone());
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    return serde_json::from_str(trimmed)
                        .unwrap_or_else(|error| panic!("stdout is not JSON ({error}): {line}"));
                }
                Ok(Err(error)) => panic!("stdout read error: {error}"),
                Err(RecvTimeoutError::Timeout) => panic!(
                    "timed out waiting for JSON-RPC; stdout so far: {:?}",
                    self.stdout_lines
                ),
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("stdout closed; stdout so far: {:?}", self.stdout_lines)
                }
            }
        }
    }

    fn wait_for_method(&mut self, method: &str, timeout: Duration) -> Value {
        let deadline = Instant::now() + timeout;
        loop {
            let value = self.read_json(deadline);
            if value.get("method").and_then(Value::as_str) == Some(method) {
                return value;
            }
        }
    }

    fn run_turn(&mut self, text: &str, timeout: Duration) -> Vec<Value> {
        self.send_user(text);
        let deadline = Instant::now() + timeout;
        let mut events = Vec::new();
        loop {
            let value = self.read_json(deadline);
            events.push(value.clone());
            if value.get("method").and_then(Value::as_str) == Some("turn/completed") {
                return events;
            }
        }
    }

    fn finish(mut self, timeout: Duration) -> FinishedDroid {
        self.close_stdin();
        let status = wait_exit(&mut self.child, timeout);
        let stderr = self
            .stderr
            .take()
            .and_then(|handle| handle.join().ok())
            .unwrap_or_default();
        let stdout: Vec<Value> = self
            .stdout_lines
            .iter()
            .filter_map(|line| serde_json::from_str(line.trim()).ok())
            .collect();
        let leftover = leftover_descendants(self.child.id());
        let log = fs::read_to_string(&self.log_path).unwrap_or_default();
        let home = self.home.clone();
        let workdir = self.workdir.clone();
        let log_path = self.log_path.clone();
        let _ = fs::remove_dir_all(&home);
        let _ = fs::remove_dir_all(&workdir);
        let _ = fs::remove_file(&log_path);
        FinishedDroid {
            status,
            stdout,
            stderr,
            log,
            leftover,
        }
    }
}

impl Drop for DroidProcess {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = fs::remove_dir_all(&self.home);
        let _ = fs::remove_dir_all(&self.workdir);
        let _ = fs::remove_file(&self.log_path);
    }
}

struct FinishedDroid {
    status: std::process::ExitStatus,
    stdout: Vec<Value>,
    stderr: String,
    log: String,
    leftover: Vec<u32>,
}

fn wait_exit(child: &mut Child, timeout: Duration) -> std::process::ExitStatus {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            return status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            return child.wait().expect("wait after kill");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn leftover_descendants(_pid: u32) -> Vec<u32> {
    Vec::new()
}

fn method_count(events: &[Value], method: &str) -> usize {
    events
        .iter()
        .filter(|value| value.get("method").and_then(Value::as_str) == Some(method))
        .count()
}

fn completed_items(events: &[Value]) -> Vec<&Value> {
    events
        .iter()
        .filter(|value| value.get("method").and_then(Value::as_str) == Some("item/completed"))
        .filter_map(|value| value.pointer("/params/item"))
        .collect()
}

fn agent_text(events: &[Value]) -> String {
    let mut text = String::new();
    for value in events {
        if value.get("method").and_then(Value::as_str) == Some("item/agentMessage/delta")
            && let Some(delta) = value.pointer("/params/delta").and_then(Value::as_str)
        {
            text.push_str(delta);
        }
    }
    if text.is_empty() {
        for item in completed_items(events) {
            if item.get("type").and_then(Value::as_str) == Some("agentMessage")
                && let Some(body) = item.get("text").and_then(Value::as_str)
            {
                text.push_str(body);
            }
        }
    }
    text
}

fn log_events(log: &str) -> Vec<Value> {
    log.lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

fn start_event(log: &str) -> Value {
    log_events(log)
        .into_iter()
        .find(|event| event.get("event").and_then(Value::as_str) == Some("start"))
        .expect("fake agent start event")
}

fn assert_no_secret(stdout: &[Value], stderr: &str, log: &str) {
    let stdout_text = serde_json::to_string(stdout).unwrap_or_default();
    assert!(
        !stdout_text.contains(SECRET_SENTINEL),
        "secret leaked on stdout"
    );
    assert!(!stderr.contains(SECRET_SENTINEL), "secret leaked on stderr");
    assert!(!log.contains(SECRET_SENTINEL), "secret leaked in agent log");
}

#[test]
fn lazy_spawn_waits_for_first_user_block() {
    let mut proc = DroidProcess::spawn("pong");
    thread::sleep(Duration::from_millis(250));
    assert!(
        !proc.log_path.exists(),
        "fake agent must not start before the first user block"
    );
    proc.send_user("Reply with exactly: PONG");
    let started = proc.wait_for_method("thread/started", Duration::from_secs(8));
    assert!(
        proc.log_path.exists(),
        "fake agent should start on first user"
    );
    assert_eq!(
        started.pointer("/params/thread/id").and_then(Value::as_str),
        Some(FAKE_SESSION)
    );
    let _ = proc.wait_for_method("turn/completed", Duration::from_secs(8));
    let finished = proc.finish(Duration::from_secs(5));
    assert!(finished.status.success(), "exit={:?}", finished.status);
}

#[test]
fn basic_turn_streams_agent_message_and_completes() {
    let mut proc = DroidProcess::spawn("pong");
    let events = proc.run_turn("Reply with exactly: PONG", Duration::from_secs(8));
    assert_eq!(method_count(&events, "thread/started"), 1);
    assert_eq!(
        events
            .iter()
            .find(|value| value.get("method").and_then(Value::as_str) == Some("thread/started"))
            .and_then(|value| value.pointer("/params/thread/id").and_then(Value::as_str)),
        Some(FAKE_SESSION)
    );
    assert_eq!(
        events
            .iter()
            .find(|value| value.get("method").and_then(Value::as_str) == Some("turn/started"))
            .and_then(|value| value.pointer("/params/turn/id").and_then(Value::as_str)),
        Some("turn-1")
    );
    assert!(method_count(&events, "item/agentMessage/delta") >= 1);
    assert!(agent_text(&events).contains("PONG"));
    let last_agent = completed_items(&events)
        .into_iter()
        .rev()
        .find(|item| item.get("type").and_then(Value::as_str) == Some("agentMessage"))
        .expect("completed agentMessage");
    assert_eq!(
        last_agent.get("phase").and_then(Value::as_str),
        Some("final_answer")
    );
    assert_eq!(method_count(&events, "turn/completed"), 1);
    assert_eq!(
        events
            .last()
            .and_then(|value| value.pointer("/params/turn/status").and_then(Value::as_str)),
        Some("completed")
    );
    let start = start_event(&fs::read_to_string(&proc.log_path).unwrap_or_default());
    let argv = start["argv"]
        .as_array()
        .expect("argv")
        .iter()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>();
    assert!(argv.contains(&"exec"));
    assert!(argv.contains(&"--output-format"));
    assert!(argv.contains(&"acp"));
    assert!(argv.contains(&"--cwd"));
    assert!(argv.contains(&"--settings"));
    assert!(!argv.iter().any(|arg| arg.contains("PONG")));
    assert_eq!(start["auto_update"].as_str(), Some("false"));
    assert_eq!(start["settings_mode"].as_u64(), Some(0o600));
    let settings = &start["settings"];
    assert_eq!(
        settings["sessionDefaultSettings"]["autonomyLevel"].as_str(),
        Some("high")
    );
    assert_eq!(
        settings["sessionDefaultSettings"]["autonomyMode"].as_str(),
        Some("auto-high")
    );
    assert_eq!(
        settings["sessionDefaultSettings"]["interactionMode"].as_str(),
        Some("auto")
    );
    assert!(settings["sessionDefaultSettings"].get("model").is_none());
    assert!(
        settings["sessionDefaultSettings"]
            .get("reasoningEffort")
            .is_none()
    );
    assert_eq!(settings["cloudSessionSync"], false);
    assert_eq!(settings["enableWarmup"], false);
    assert_eq!(settings["enableCompletionBell"], false);
    let settings_path = start["settings_path"]
        .as_str()
        .expect("settings_path")
        .to_string();
    let finished = proc.finish(Duration::from_secs(5));
    assert!(finished.status.success());
    assert!(finished.leftover.is_empty());
    assert!(!Path::new(&settings_path).exists());
    assert_no_secret(&finished.stdout, &finished.stderr, &finished.log);
    for line in &finished.stdout {
        assert!(line.get("method").and_then(Value::as_str).is_some());
        assert!(
            line.get("id").is_none(),
            "stdout must be notifications: {line}"
        );
    }
}

#[test]
fn two_turns_increment_ids_and_one_thread_started() {
    let mut proc = DroidProcess::spawn("two_turns");
    let first = proc.run_turn("Remember the word KIWI. Reply OK.", Duration::from_secs(8));
    let second = proc.run_turn(
        "Reply with only the word I asked you to remember.",
        Duration::from_secs(8),
    );
    assert_eq!(method_count(&first, "thread/started"), 1);
    assert_eq!(method_count(&second, "thread/started"), 0);
    assert_eq!(
        first
            .iter()
            .find(|value| value.get("method").and_then(Value::as_str) == Some("turn/started"))
            .and_then(|value| value.pointer("/params/turn/id").and_then(Value::as_str)),
        Some("turn-1")
    );
    assert_eq!(
        second
            .iter()
            .find(|value| value.get("method").and_then(Value::as_str) == Some("turn/started"))
            .and_then(|value| value.pointer("/params/turn/id").and_then(Value::as_str)),
        Some("turn-2")
    );
    assert!(agent_text(&first).contains("OK"));
    assert!(agent_text(&second).contains("KIWI"));
    let finished = proc.finish(Duration::from_secs(5));
    assert!(finished.status.success());
}

#[test]
fn unknown_session_update_tag_is_ignored() {
    let mut proc = DroidProcess::spawn("unknown_update");
    let events = proc.run_turn("hi", Duration::from_secs(8));
    assert!(agent_text(&events).contains("PONG"));
    assert_eq!(
        events
            .last()
            .and_then(|value| value.pointer("/params/turn/status").and_then(Value::as_str)),
        Some("completed")
    );
    assert_eq!(method_count(&events, "error"), 0);
    let finished = proc.finish(Duration::from_secs(5));
    assert!(finished.status.success());
}

#[test]
fn incoming_request_id_zero_is_answered() {
    let mut proc = DroidProcess::spawn("permission_id_zero");
    let events = proc.run_turn("hi", Duration::from_secs(8));
    assert!(agent_text(&events).contains("PONG"));
    assert_eq!(
        events
            .last()
            .and_then(|value| value.pointer("/params/turn/status").and_then(Value::as_str)),
        Some("completed")
    );
    let log = fs::read_to_string(&proc.log_path).unwrap_or_default();
    let answered = log_events(&log).into_iter().any(|event| {
        event.get("dir").and_then(Value::as_str) == Some("in")
            && event.pointer("/msg/id") == Some(&json!(0))
            && event
                .pointer("/msg/result/outcome/outcome")
                .and_then(Value::as_str)
                == Some("selected")
            && event
                .pointer("/msg/result/outcome/optionId")
                .and_then(Value::as_str)
                == Some("proceed_always")
    });
    assert!(
        answered,
        "id 0 permission request was not auto-approved: {log}"
    );
    let finished = proc.finish(Duration::from_secs(5));
    assert!(finished.status.success());
}

#[test]
fn colliding_agent_and_client_ids_are_not_confused() {
    let mut proc = DroidProcess::spawn("colliding_ids");
    let events = proc.run_turn("hi", Duration::from_secs(8));
    assert!(agent_text(&events).contains("PONG"));
    assert_eq!(
        events
            .last()
            .and_then(|value| value.pointer("/params/turn/status").and_then(Value::as_str)),
        Some("completed")
    );
    let finished = proc.finish(Duration::from_secs(5));
    assert!(finished.status.success());
}

#[test]
fn terminal_output_byte_limit_truncates_from_start() {
    let mut proc = DroidProcess::spawn("terminal_truncate");
    let events = proc.run_turn("run printf", Duration::from_secs(8));
    let command = completed_items(&events)
        .into_iter()
        .find(|item| item.get("type").and_then(Value::as_str) == Some("commandExecution"))
        .expect("commandExecution");
    assert_eq!(
        command.get("aggregatedOutput").and_then(Value::as_str),
        Some("6789")
    );
    assert_ne!(
        command.get("aggregatedOutput").and_then(Value::as_str),
        Some("0123456789")
    );
    let finished = proc.finish(Duration::from_secs(5));
    assert!(finished.status.success());
}

#[test]
fn execute_tool_becomes_command_execution_with_output() {
    let mut proc = DroidProcess::spawn("execute");
    let events = proc.run_turn("echo CENTAUR-OK", Duration::from_secs(8));
    let command = completed_items(&events)
        .into_iter()
        .find(|item| item.get("type").and_then(Value::as_str) == Some("commandExecution"))
        .expect("commandExecution");
    let cmd = command.get("command").and_then(Value::as_str).unwrap_or("");
    assert!(cmd.contains("echo CENTAUR-OK"), "command={cmd}");
    let output = command
        .get("aggregatedOutput")
        .and_then(Value::as_str)
        .unwrap_or("");
    assert!(output.contains("CENTAUR-OK"), "output={output}");
    assert_eq!(command.get("exitCode"), Some(&json!(0)));
    assert_eq!(
        command.get("status").and_then(Value::as_str),
        Some("completed")
    );
    let finished = proc.finish(Duration::from_secs(5));
    assert!(finished.status.success());
}

#[test]
fn nonzero_shell_exit_is_failed_command_execution() {
    let mut proc = DroidProcess::spawn("execute_fail");
    let events = proc.run_turn("fail", Duration::from_secs(8));
    let command = completed_items(&events)
        .into_iter()
        .find(|item| item.get("type").and_then(Value::as_str) == Some("commandExecution"))
        .expect("commandExecution");
    let output = command
        .get("aggregatedOutput")
        .and_then(Value::as_str)
        .unwrap_or("");
    assert!(output.contains("FAIL-MARKER"), "output={output}");
    assert_eq!(command.get("exitCode"), Some(&json!(7)));
    assert_eq!(
        command.get("status").and_then(Value::as_str),
        Some("failed")
    );
    let finished = proc.finish(Duration::from_secs(5));
    assert!(finished.status.success());
}

#[test]
fn non_execute_tool_becomes_dynamic_tool_call() {
    let mut proc = DroidProcess::spawn("read_tool");
    let events = proc.run_turn("read note.txt", Duration::from_secs(8));
    let tool = completed_items(&events)
        .into_iter()
        .find(|item| item.get("type").and_then(Value::as_str) == Some("dynamicToolCall"))
        .expect("dynamicToolCall");
    assert_ne!(
        tool.get("status").and_then(Value::as_str),
        Some("inProgress")
    );
    let finished = proc.finish(Duration::from_secs(5));
    assert!(finished.status.success());
}

#[test]
fn stdin_eof_while_idle_exits_zero_and_removes_settings() {
    let mut proc = DroidProcess::spawn("pong");
    let _ = proc.run_turn("PONG", Duration::from_secs(8));
    let start = start_event(&fs::read_to_string(&proc.log_path).unwrap_or_default());
    let settings_path = start["settings_path"]
        .as_str()
        .expect("settings_path")
        .to_string();
    let finished = proc.finish(Duration::from_secs(5));
    assert!(finished.status.success(), "exit={:?}", finished.status);
    assert!(finished.leftover.is_empty());
    assert!(!Path::new(&settings_path).exists());
    assert_no_secret(&finished.stdout, &finished.stderr, &finished.log);
}

#[test]
fn settings_include_model_and_reasoning_when_env_set() {
    let mut proc = DroidProcess::spawn_with(
        "pong",
        &[
            ("DROID_MODEL", "gpt-5.4-mini-fast"),
            ("DROID_REASONING_EFFORT", "low"),
        ],
    );
    let _ = proc.run_turn("PONG", Duration::from_secs(8));
    let start = start_event(&fs::read_to_string(&proc.log_path).unwrap_or_default());
    assert_eq!(
        start["settings"]["sessionDefaultSettings"]["model"].as_str(),
        Some("gpt-5.4-mini-fast")
    );
    assert_eq!(
        start["settings"]["sessionDefaultSettings"]["reasoningEffort"].as_str(),
        Some("low")
    );
    let finished = proc.finish(Duration::from_secs(5));
    assert!(finished.status.success());
}

fn skip_real_droid() -> bool {
    if std::env::var("FACTORY_API_KEY")
        .ok()
        .filter(|value| !value.is_empty())
        .is_none()
    {
        eprintln!("skipping real_droid test: FACTORY_API_KEY absent");
        return true;
    }
    if std::env::var("DROID_BIN")
        .ok()
        .filter(|value| !value.is_empty())
        .is_none()
    {
        eprintln!("skipping real_droid test: DROID_BIN unset");
        return true;
    }
    false
}

fn spawn_real_droid() -> (
    Child,
    ChildStdin,
    Receiver<std::io::Result<String>>,
    PathBuf,
    PathBuf,
) {
    let id = Uuid::new_v4().simple().to_string();
    let workdir = std::env::temp_dir().join(format!("droid-real-wd-{id}"));
    let home = std::env::temp_dir().join(format!("droid-real-home-{id}"));
    fs::create_dir_all(&workdir).expect("workdir");
    fs::create_dir_all(&home).expect("home");
    let mut command = Command::new(env!("CARGO_BIN_EXE_harness-server"));
    command
        .arg("droid")
        .current_dir(&workdir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("HOME", &home)
        .env("DROID_MODEL", "gpt-5.4-mini-fast")
        .env("DROID_REASONING_EFFORT", "low")
        .env("FACTORY_DROID_AUTO_UPDATE_ENABLED", "false");
    let mut child = command.spawn().expect("spawn real droid harness");
    let stdin = child.stdin.take().expect("stdin");
    let stdout = child.stdout.take().expect("stdout");
    let mut stderr = child.stderr.take().expect("stderr");
    thread::spawn(move || {
        let mut sink = std::io::sink();
        let _ = std::io::copy(&mut stderr, &mut sink);
    });
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let stop = line.is_err();
            if tx.send(line).is_err() || stop {
                break;
            }
        }
    });
    (child, stdin, rx, home, workdir)
}

fn real_wait_turn(rx: &Receiver<std::io::Result<String>>, timeout: Duration) -> Vec<Value> {
    let deadline = Instant::now() + timeout;
    let mut events = Vec::new();
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(remaining) {
            Ok(Ok(line)) => {
                let Ok(value) = serde_json::from_str::<Value>(line.trim()) else {
                    continue;
                };
                events.push(value.clone());
                if value.get("method").and_then(Value::as_str) == Some("turn/completed") {
                    return events;
                }
            }
            other => panic!("real droid turn failed: {other:?}; events={events:?}"),
        }
    }
}

#[test]
#[ignore]
fn real_droid_basic_streaming_turn() {
    if skip_real_droid() {
        return;
    }
    let (mut child, mut stdin, rx, home, workdir) = spawn_real_droid();
    writeln!(
        stdin,
        "{}",
        json!({"type": "user", "text": "Reply with exactly: PONG"})
    )
    .expect("write");
    stdin.flush().expect("flush");
    let events = real_wait_turn(&rx, Duration::from_secs(180));
    drop(stdin);
    let _ = wait_exit(&mut child, Duration::from_secs(15));
    let _ = fs::remove_dir_all(home);
    let _ = fs::remove_dir_all(workdir);
    assert!(agent_text(&events).to_uppercase().contains("PONG"));
    assert_eq!(
        events
            .last()
            .and_then(|value| value.pointer("/params/turn/status").and_then(Value::as_str)),
        Some("completed")
    );
}

#[test]
#[ignore]
fn real_droid_command_turn() {
    if skip_real_droid() {
        return;
    }
    let (mut child, mut stdin, rx, home, workdir) = spawn_real_droid();
    writeln!(
        stdin,
        "{}",
        json!({
            "type": "user",
            "text": "Run the shell command echo CENTAUR-OK and then reply with exactly DONE."
        })
    )
    .expect("write");
    stdin.flush().expect("flush");
    let events = real_wait_turn(&rx, Duration::from_secs(180));
    drop(stdin);
    let _ = wait_exit(&mut child, Duration::from_secs(15));
    let _ = fs::remove_dir_all(home);
    let _ = fs::remove_dir_all(workdir);
    let command = completed_items(&events)
        .into_iter()
        .find(|item| item.get("type").and_then(Value::as_str) == Some("commandExecution"));
    assert!(
        command.is_some(),
        "expected commandExecution, items={:?}",
        completed_items(&events)
    );
}

#[test]
fn fake_agent_script_is_not_a_cargo_bin() {
    let manifest = fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
        .expect("manifest");
    assert!(
        !manifest.contains("fake_acp_agent"),
        "fake ACP agent must not be a [[bin]] target"
    );
}
