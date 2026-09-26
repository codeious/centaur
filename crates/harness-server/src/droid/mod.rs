//! Factory Droid harness — drives `droid exec --output-format acp` as a
//! Centaur blocks-mode runtime.
//!
//! One long-lived Droid process per harness process. Lazy-spawned on the first
//! user block. Turns follow the Idle / Prompting / Steering / Cancelling state
//! machine in architecture.md §3.3.

mod config;
mod profile;
mod prompt;

use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

use agent_client_protocol_schema::v1::{ContentBlock, SessionId, StopReason};
use codex_app_server_protocol::{ServerNotification, UserInput};
use serde_json::Value;
use uuid::Uuid;

use crate::acp::{AcpClient, AcpError, AcpMapper, TerminalManager};
use crate::server::{BlocksCommand, BlocksState, parse_blocks_line_with_state, write_blocks_error};
use crate::traits::NormalizedEvent;
use crate::turn::{BridgeConfig, CodexTurnNormalizer};
use crate::util::write_value;
use crate::wire::notification_to_wire_value;
use crate::{HarnessServerError, Result};

use config::ConfigCatalog;
use profile::{env_opt, nonempty};
use prompt::prompt_blocks;

pub use profile::{
    droid_argv, droid_profile, droid_program, settings_document, write_settings_file,
};

type DroidMapper = AcpMapper<Arc<TerminalManager>>;

const HANDSHAKE_DRAIN: Duration = Duration::from_millis(20);
const PROMPT_POLL: Duration = Duration::from_millis(20);
const CHILD_EXIT_WAIT: Duration = Duration::from_secs(2);
const LAST_SESSION_FILE: &str = "centaur-last-session-id";

/// Entry point for `harness-server droid`.
pub fn run_droid_blocks_server() -> Result<()> {
    let mut stdout = io::stdout().lock();
    let mut runtime = Runtime::new();
    let (command_tx, command_rx) = mpsc::channel();
    let (interrupt_tx, interrupt_rx) = mpsc::channel();

    thread::spawn(move || {
        let stdin = io::stdin();
        let mut blocks_state = BlocksState::default();
        for raw in stdin.lock().lines() {
            let Ok(line) = raw else { break };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let sent = match parse_blocks_line_with_state(trimmed, &mut blocks_state) {
                Ok(BlocksCommand::Interrupt) => interrupt_tx.send(()).is_ok(),
                Ok(command) => command_tx.send(Ok(command)).is_ok(),
                Err(error) => command_tx.send(Err(error.to_string())).is_ok(),
            };
            if !sent {
                break;
            }
        }
    });

    loop {
        if runtime.is_active() {
            match runtime.pump(&mut stdout, &command_rx, &interrupt_rx)? {
                Pump::Continue => {}
                Pump::Idle => while interrupt_rx.try_recv().is_ok() {},
                Pump::Exit => break,
            }
        } else {
            match command_rx.recv_timeout(PROMPT_POLL) {
                Ok(Ok(BlocksCommand::User {
                    input,
                    client_user_message_id,
                    model,
                    provider: _,
                    reasoning,
                    trace_context: _,
                })) => {
                    while interrupt_rx.try_recv().is_ok() {}
                    runtime.begin_turn(
                        &mut stdout,
                        QueuedUser {
                            input,
                            client_user_message_id,
                            model,
                            reasoning,
                        },
                    )?;
                }
                Ok(Ok(BlocksCommand::Interrupt)) => {}
                Ok(Ok(BlocksCommand::AttachmentChunk)) => {}
                Ok(Err(error)) => {
                    let thread_id = runtime.thread_id();
                    eprintln!("invalid Droid blocks input: {error}");
                    write_blocks_error(&mut stdout, &thread_id, "input", error)?;
                }
                Err(RecvTimeoutError::Timeout) => while interrupt_rx.try_recv().is_ok() {},
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
    }
    Ok(())
}

/// Test-only override. Production default is 30s (architecture §3.3).
fn cancel_timeout() -> Duration {
    env::var("DROID_CANCEL_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_secs(30))
}

#[derive(Debug, Clone)]
struct QueuedUser {
    input: Vec<UserInput>,
    client_user_message_id: Option<String>,
    model: Option<String>,
    reasoning: Option<String>,
}

enum Phase {
    Prompting,
    Steering,
    Cancelling { kind: CancelKind },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CancelKind {
    Interrupt,
    StdinEof,
}

struct ActiveTurn {
    normalizer: CodexTurnNormalizer,
    phase: Phase,
    prompt_rx: Receiver<crate::acp::Result<agent_client_protocol_schema::v1::PromptResponse>>,
    drop_updates: bool,
    cancel_at: Option<Instant>,
    pending_steer: Option<QueuedUser>,
    queued_next: Option<QueuedUser>,
}

enum Pump {
    Continue,
    Idle,
    Exit,
}

struct Runtime {
    child: Option<DroidChild>,
    turn: u64,
    thread_started: bool,
    continue_session_id: Option<String>,
    last_session_id: Option<String>,
    active: Option<ActiveTurn>,
}

impl Runtime {
    fn new() -> Self {
        Self {
            child: None,
            turn: 0,
            thread_started: false,
            continue_session_id: env_opt("DROID_CONTINUE_SESSION_ID")
                .or_else(load_persisted_session_id),
            last_session_id: None,
            active: None,
        }
    }

    fn remember_session_id(&mut self, id: impl Into<String>) {
        let id = id.into();
        persist_last_session_id(&id);
        self.last_session_id = Some(id);
    }

    fn is_active(&self) -> bool {
        self.active.is_some()
    }

    fn thread_id(&self) -> String {
        self.child
            .as_ref()
            .map(DroidChild::thread_id)
            .map(str::to_owned)
            .or_else(|| self.last_session_id.clone())
            .or_else(|| self.continue_session_id.clone())
            .unwrap_or_else(|| "droid".to_string())
    }

    fn ensure_child(&mut self) -> Result<()> {
        if let Some(child) = self.child.as_mut() {
            if child.is_alive() {
                return Ok(());
            }
            child.kill_agent();
            self.child = None;
        }
        let resume = self
            .last_session_id
            .clone()
            .or_else(|| self.continue_session_id.clone());
        let child = DroidChild::start(resume)?;
        if child.session_ready {
            self.remember_session_id(child.thread_id());
        }
        self.child = Some(child);
        Ok(())
    }

    fn begin_turn<W: Write>(&mut self, stdout: &mut W, queued: QueuedUser) -> Result<()> {
        self.turn += 1;
        let turn_id = format!("turn-{}", self.turn);
        if let Err(error) = self.ensure_child() {
            let thread_id = self.thread_id();
            eprintln!("Droid blocks turn failed: {error:#}");
            write_blocks_error(stdout, &thread_id, &turn_id, error.to_string())?;
            return Ok(());
        }
        let child = self.child.as_mut().expect("droid started");
        if !child.session_ready {
            child.retry_load();
        }
        let thread_id = child.thread_id().to_string();
        let mut config = BridgeConfig::new(thread_id.clone(), turn_id.clone());
        config.cli_version = child.cli_version.clone();
        config.model_provider = child.model_provider.clone();
        let mut normalizer = CodexTurnNormalizer::new(config);
        let include_thread = !self.thread_started && child.session_ready;
        if include_thread {
            self.thread_started = true;
        }
        for notification in normalizer.start_notifications(include_thread)? {
            write_notification(stdout, &notification)?;
        }
        for notification in normalizer
            .emit_user_message(queued.client_user_message_id.clone(), queued.input.clone())?
        {
            write_notification(stdout, &notification)?;
        }

        if !child.session_ready {
            let message = child
                .resume_error
                .clone()
                .unwrap_or_else(|| format!("failed to resume session {thread_id}"));
            fail_turn(&mut normalizer, stdout, message)?;
            return Ok(());
        }
        persist_last_session_id(&thread_id);
        self.last_session_id = Some(thread_id);

        if let Err(message) =
            child.apply_desired_config(queued.model.clone(), queued.reasoning.clone())
        {
            fail_turn(&mut normalizer, stdout, message)?;
            return Ok(());
        }

        let blocks = prompt_blocks(&queued.input)?;
        let prompt_rx = child.spawn_prompt(blocks);
        self.active = Some(ActiveTurn {
            normalizer,
            phase: Phase::Prompting,
            prompt_rx,
            drop_updates: false,
            cancel_at: None,
            pending_steer: None,
            queued_next: None,
        });
        Ok(())
    }

    fn pump<W: Write>(
        &mut self,
        stdout: &mut W,
        command_rx: &Receiver<std::result::Result<BlocksCommand, String>>,
        interrupt_rx: &Receiver<()>,
    ) -> Result<Pump> {
        if self
            .active
            .as_ref()
            .and_then(|active| active.cancel_at)
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            return self.watchdog_fire(stdout);
        }

        if let Some(active) = self.active.as_ref() {
            match active.prompt_rx.try_recv() {
                Ok(result) => return self.on_prompt_result(stdout, result),
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => return self.on_child_dead(stdout),
            }
        }

        if interrupt_rx.try_recv().is_ok() {
            self.begin_cancel(stdout, CancelKind::Interrupt)?;
        }

        match command_rx.try_recv() {
            Ok(Ok(BlocksCommand::User {
                input,
                client_user_message_id,
                model,
                provider: _,
                reasoning,
                trace_context: _,
            })) => {
                self.on_user_during_turn(
                    stdout,
                    QueuedUser {
                        input,
                        client_user_message_id,
                        model,
                        reasoning,
                    },
                )?;
            }
            Ok(Ok(BlocksCommand::Interrupt)) => {}
            Ok(Ok(BlocksCommand::AttachmentChunk)) => {}
            Ok(Err(error)) => {
                let thread_id = self.thread_id();
                let turn_id = format!("turn-{}", self.turn);
                eprintln!("invalid Droid blocks input: {error}");
                write_blocks_error(stdout, &thread_id, &turn_id, error)?;
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => return self.handle_eof(stdout),
        }

        if self.child.as_mut().is_some_and(|child| !child.is_alive()) {
            return self.on_child_dead(stdout);
        }

        let Some(client) = self.child.as_ref().and_then(|child| child.client.clone()) else {
            return Ok(Pump::Continue);
        };
        match client.recv_notification_timeout(PROMPT_POLL) {
            Ok((method, params)) => self.on_notification(stdout, &method, params)?,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return self.on_child_dead(stdout),
        }
        Ok(Pump::Continue)
    }

    fn on_notification<W: Write>(
        &mut self,
        stdout: &mut W,
        method: &str,
        params: Value,
    ) -> Result<()> {
        let Some(child) = self.child.as_mut() else {
            return Ok(());
        };
        let Some(active) = self.active.as_mut() else {
            child.catalog.ingest_update(method, &params);
            return Ok(());
        };
        if active.drop_updates {
            child.catalog.ingest_update(method, &params);
            return Ok(());
        }
        apply_notification(
            &mut child.mapper,
            &mut child.catalog,
            &mut active.normalizer,
            stdout,
            method,
            params,
        )
    }

    fn on_user_during_turn<W: Write>(&mut self, stdout: &mut W, queued: QueuedUser) -> Result<()> {
        let Some(phase_is) = self.active.as_ref().map(|active| match active.phase {
            Phase::Prompting => 0,
            Phase::Steering => 1,
            Phase::Cancelling { .. } => 2,
        }) else {
            return Ok(());
        };
        match phase_is {
            0 => self.begin_steer(stdout, queued),
            1 => {
                let active = self.active.as_mut().expect("steering");
                for notification in active.normalizer.emit_user_message(
                    queued.client_user_message_id.clone(),
                    queued.input.clone(),
                )? {
                    write_notification(stdout, &notification)?;
                }
                if let Some(pending) = active.pending_steer.as_mut() {
                    pending.input.extend(queued.input);
                    if queued.model.is_some() {
                        pending.model = queued.model;
                    }
                    if queued.reasoning.is_some() {
                        pending.reasoning = queued.reasoning;
                    }
                    if queued.client_user_message_id.is_some() {
                        pending.client_user_message_id = queued.client_user_message_id;
                    }
                }
                Ok(())
            }
            _ => {
                if let Some(active) = self.active.as_mut() {
                    active.queued_next = Some(queued);
                }
                Ok(())
            }
        }
    }

    fn begin_steer<W: Write>(&mut self, stdout: &mut W, queued: QueuedUser) -> Result<()> {
        let Some(active) = self.active.as_mut() else {
            return Ok(());
        };
        let Some(child) = self.child.as_mut() else {
            return Err(HarnessServerError::Protocol(
                "droid client missing".to_string(),
            ));
        };
        for notification in active
            .normalizer
            .emit_user_message(queued.client_user_message_id.clone(), queued.input.clone())?
        {
            write_notification(stdout, &notification)?;
        }
        close_open(child, &mut active.normalizer, stdout)?;
        if let Err(error) = child.session_cancel() {
            eprintln!("Droid session/cancel failed: {error}");
        }
        active.drop_updates = true;
        active.phase = Phase::Steering;
        active.pending_steer = Some(queued);
        active.cancel_at = Some(Instant::now() + cancel_timeout());
        Ok(())
    }

    fn begin_cancel<W: Write>(&mut self, stdout: &mut W, kind: CancelKind) -> Result<()> {
        let Some(active) = self.active.as_mut() else {
            return Ok(());
        };
        match active.phase {
            Phase::Cancelling { .. } => {
                if kind == CancelKind::StdinEof {
                    active.queued_next = None;
                    active.phase = Phase::Cancelling { kind };
                }
                return Ok(());
            }
            Phase::Steering => {
                active.pending_steer = None;
                active.phase = Phase::Cancelling { kind };
                if kind == CancelKind::StdinEof {
                    active.queued_next = None;
                }
                return Ok(());
            }
            Phase::Prompting => {}
        }
        if let Some(child) = self.child.as_mut() {
            close_open(child, &mut active.normalizer, stdout)?;
            if let Err(error) = child.session_cancel() {
                eprintln!("Droid session/cancel failed: {error}");
            }
        }
        active.drop_updates = true;
        active.phase = Phase::Cancelling { kind };
        active.cancel_at = Some(Instant::now() + cancel_timeout());
        if kind == CancelKind::StdinEof {
            active.queued_next = None;
        }
        Ok(())
    }

    fn handle_eof<W: Write>(&mut self, stdout: &mut W) -> Result<Pump> {
        if self.active.is_none() {
            return Ok(Pump::Exit);
        }
        self.begin_cancel(stdout, CancelKind::StdinEof)?;
        Ok(Pump::Continue)
    }

    fn on_prompt_result<W: Write>(
        &mut self,
        stdout: &mut W,
        result: crate::acp::Result<agent_client_protocol_schema::v1::PromptResponse>,
    ) -> Result<Pump> {
        let phase = match self.active.as_ref().map(|active| match &active.phase {
            Phase::Prompting => 0,
            Phase::Steering => 1,
            Phase::Cancelling { kind } => {
                if *kind == CancelKind::StdinEof {
                    3
                } else {
                    2
                }
            }
        }) {
            Some(phase) => phase,
            None => return Ok(Pump::Idle),
        };
        if phase == 1 {
            let pending = self
                .active
                .as_mut()
                .and_then(|active| {
                    active.drop_updates = false;
                    active.cancel_at = None;
                    active.pending_steer.take()
                })
                .unwrap_or(QueuedUser {
                    input: Vec::new(),
                    client_user_message_id: None,
                    model: None,
                    reasoning: None,
                });
            if pending.input.is_empty() {
                return self.finish_active(stdout, FinishKind::Interrupted, None);
            }
            if self.child.is_none() {
                return self.finish_active(stdout, FinishKind::Failed, Some("droid exited".into()));
            }
            let config_result = {
                let child = self.child.as_mut().expect("child");
                child.apply_desired_config(pending.model.clone(), pending.reasoning.clone())
            };
            if let Err(message) = config_result {
                return self.finish_active(stdout, FinishKind::Failed, Some(message));
            }
            let blocks = prompt_blocks(&pending.input)?;
            let prompt_rx = self.child.as_ref().expect("child").spawn_prompt(blocks);
            if let Some(active) = self.active.as_mut() {
                active.prompt_rx = prompt_rx;
                active.phase = Phase::Prompting;
            }
            return Ok(Pump::Continue);
        }
        if phase == 2 || phase == 3 {
            let queued = self
                .active
                .as_mut()
                .and_then(|active| active.queued_next.take());
            self.finish_active(stdout, FinishKind::Interrupted, None)?;
            if phase == 3 {
                return Ok(Pump::Exit);
            }
            if let Some(queued) = queued {
                self.begin_turn(stdout, queued)?;
                return Ok(if self.is_active() {
                    Pump::Continue
                } else {
                    Pump::Idle
                });
            }
            return Ok(Pump::Idle);
        }
        match result {
            Ok(response) => {
                let failed = match response.stop_reason {
                    StopReason::Refusal => Some("agent refused".to_string()),
                    StopReason::Cancelled => {
                        return self.finish_active(stdout, FinishKind::Interrupted, None);
                    }
                    StopReason::EndTurn | StopReason::MaxTokens | StopReason::MaxTurnRequests => {
                        None
                    }
                    _ => None,
                };
                if let Some(message) = failed {
                    self.finish_active(stdout, FinishKind::Failed, Some(message))
                } else {
                    self.finish_active(stdout, FinishKind::Completed, None)
                }
            }
            Err(error) => {
                let message = error.to_string();
                let closed = matches!(error, AcpError::TransportClosed);
                let pump = self.finish_active(stdout, FinishKind::Failed, Some(message))?;
                if closed {
                    self.drop_child();
                }
                Ok(pump)
            }
        }
    }

    fn watchdog_fire<W: Write>(&mut self, stdout: &mut W) -> Result<Pump> {
        let (failed, eof, queued) = match self.active.as_mut() {
            Some(active) => {
                let failed = matches!(active.phase, Phase::Steering);
                let eof = matches!(
                    active.phase,
                    Phase::Cancelling {
                        kind: CancelKind::StdinEof
                    }
                );
                let queued = active.queued_next.take();
                (failed, eof, queued)
            }
            None => return Ok(Pump::Idle),
        };
        if let Some(child) = self.child.as_mut() {
            child.kill_agent();
        }
        let kind = if failed {
            FinishKind::Failed
        } else {
            FinishKind::Interrupted
        };
        self.finish_active(stdout, kind, Some("droid cancel timed out".to_string()))?;
        self.drop_child();
        if eof {
            return Ok(Pump::Exit);
        }
        if let Some(queued) = queued {
            self.begin_turn(stdout, queued)?;
            return Ok(if self.is_active() {
                Pump::Continue
            } else {
                Pump::Idle
            });
        }
        Ok(Pump::Idle)
    }

    fn on_child_dead<W: Write>(&mut self, stdout: &mut W) -> Result<Pump> {
        let eof = matches!(
            self.active.as_ref().map(|active| &active.phase),
            Some(Phase::Cancelling {
                kind: CancelKind::StdinEof
            })
        );
        let queued = self
            .active
            .as_mut()
            .and_then(|active| active.queued_next.take());
        let kind = if eof {
            FinishKind::Interrupted
        } else {
            FinishKind::Failed
        };
        self.finish_active(stdout, kind, Some("droid exited".to_string()))?;
        self.drop_child();
        if eof {
            return Ok(Pump::Exit);
        }
        if let Some(queued) = queued {
            self.begin_turn(stdout, queued)?;
            return Ok(if self.is_active() {
                Pump::Continue
            } else {
                Pump::Idle
            });
        }
        Ok(Pump::Idle)
    }

    fn finish_active<W: Write>(
        &mut self,
        stdout: &mut W,
        kind: FinishKind,
        message: Option<String>,
    ) -> Result<Pump> {
        let Some(mut active) = self.active.take() else {
            return Ok(Pump::Idle);
        };
        if let Some(child) = self.child.as_mut() {
            let mapped = match kind {
                FinishKind::Completed => child.mapper.finish("end_turn"),
                _ => child.mapper.abandon_open(),
            };
            write_mapped(&mut active.normalizer, stdout, mapped)?;
            for notification in active.normalizer.close_open_items()? {
                write_notification(stdout, &notification)?;
            }
        } else {
            for notification in active.normalizer.close_open_items()? {
                write_notification(stdout, &notification)?;
            }
        }
        if let Some(message) = message.as_ref() {
            for notification in active.normalizer.process_event(&NormalizedEvent::Error {
                message: message.clone(),
            })? {
                write_notification(stdout, &notification)?;
            }
        }
        let notification = match kind {
            FinishKind::Completed => active.normalizer.finish_turn(None)?,
            FinishKind::Failed => active.normalizer.finish_turn(message)?,
            FinishKind::Interrupted => active.normalizer.finish_turn_interrupted()?,
        };
        if let Some(notification) = notification {
            write_notification(stdout, &notification)?;
        }
        Ok(Pump::Idle)
    }

    fn drop_child(&mut self) {
        if let Some(mut child) = self.child.take() {
            child.kill_agent();
        }
    }
}

enum FinishKind {
    Completed,
    Failed,
    Interrupted,
}

struct DroidChild {
    child: Child,
    client: Option<Arc<AcpClient>>,
    mapper: DroidMapper,
    session_id: SessionId,
    settings_path: PathBuf,
    cli_version: String,
    model_provider: String,
    model_config_id: String,
    reasoning_config_id: String,
    catalog: ConfigCatalog,
    env_model: Option<String>,
    env_reasoning: Option<String>,
    env_applied: bool,
    cwd: PathBuf,
    session_ready: bool,
    resume_error: Option<String>,
}

impl Drop for DroidChild {
    fn drop(&mut self) {
        self.kill_agent();
        let _ = fs::remove_file(&self.settings_path);
    }
}

impl DroidChild {
    fn start(resume: Option<String>) -> Result<Self> {
        let cwd = env::current_dir()?;
        let cwd = cwd.canonicalize().unwrap_or(cwd);
        let env_model = env_opt("DROID_MODEL");
        let env_reasoning = env_opt("DROID_REASONING_EFFORT");
        let settings_path =
            env::temp_dir().join(format!("centaur-droid-settings-{}.json", Uuid::new_v4()));
        write_settings_file(
            &settings_path,
            &settings_document(env_model.as_deref(), env_reasoning.as_deref()),
        )?;
        let profile = droid_profile(&cwd, &settings_path);

        let mut command = ProcessCommand::new(&profile.program);
        command
            .args(&profile.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        for (key, value) in &profile.extra_env {
            command.env(key, value);
        }

        let mut child = command
            .spawn()
            .map_err(|source| HarnessServerError::SpawnHarness {
                cwd: cwd.clone(),
                source,
            })?;
        let stdin = child
            .stdin
            .take()
            .ok_or(HarnessServerError::HarnessStdinUnavailable)?;
        let stdout = child
            .stdout
            .take()
            .ok_or(HarnessServerError::HarnessStdoutUnavailable)?;
        let mut stderr = child
            .stderr
            .take()
            .ok_or(HarnessServerError::HarnessStderrUnavailable)?;
        thread::spawn(move || {
            let mut parent_stderr = io::stderr();
            let _ = io::copy(&mut stderr, &mut parent_stderr);
        });

        let client = Arc::new(AcpClient::from_rw(stdin, stdout, cwd.clone()));
        if let Err(error) = client.initialize() {
            return Err(map_acp_error(error, &mut child));
        }
        let mut catalog = ConfigCatalog::default();
        drain_notifications(&client, &mut catalog, HANDSHAKE_DRAIN);

        let (session_id, session_ready, resume_error) =
            handshake_session(&client, &cwd, resume.as_deref(), &mut catalog, &mut child)?;
        drain_notifications(&client, &mut catalog, HANDSHAKE_DRAIN);
        let mapper = AcpMapper::with_terminals(Arc::clone(client.terminals()));
        Ok(Self {
            child,
            client: Some(client),
            mapper,
            session_id,
            settings_path,
            cli_version: profile.cli_version,
            model_provider: profile.model_provider,
            model_config_id: profile.model_config_id,
            reasoning_config_id: profile.reasoning_config_id,
            catalog,
            env_model,
            env_reasoning,
            env_applied: false,
            cwd,
            session_ready,
            resume_error,
        })
    }

    fn retry_load(&mut self) {
        if self.session_ready {
            return;
        }
        let Some(client) = self.client.as_ref() else {
            return;
        };
        match client.session_load(self.session_id.clone(), &self.cwd) {
            Ok(session) => {
                if let Some(options) = session.config_options.as_deref() {
                    self.catalog.ingest_typed(options);
                }
                drain_notifications(client, &mut self.catalog, HANDSHAKE_DRAIN);
                self.session_ready = true;
                self.resume_error = None;
            }
            Err(error) => {
                drain_notifications(client, &mut self.catalog, HANDSHAKE_DRAIN);
                self.resume_error = Some(format!(
                    "failed to resume session {}: {error}",
                    self.thread_id()
                ));
            }
        }
    }

    fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    fn thread_id(&self) -> &str {
        self.session_id.0.as_ref()
    }

    fn kill_agent(&mut self) {
        if let Some(client) = &self.client {
            client.terminals().kill_all();
        }
        match self.child.try_wait() {
            Ok(Some(_)) => {}
            _ => {
                let _ = self.child.kill();
                let deadline = Instant::now() + CHILD_EXIT_WAIT;
                loop {
                    match self.child.try_wait() {
                        Ok(Some(_)) => break,
                        Ok(None) if Instant::now() < deadline => {
                            thread::sleep(Duration::from_millis(20));
                        }
                        _ => {
                            let _ = self.child.wait();
                            break;
                        }
                    }
                }
            }
        }
    }

    fn session_cancel(&self) -> crate::acp::Result<()> {
        let client = self.client.as_ref().ok_or_else(|| AcpError::Timeout {
            method: "session/cancel".to_string(),
        })?;
        client.session_cancel(self.session_id.clone())
    }

    fn spawn_prompt(
        &self,
        blocks: Vec<ContentBlock>,
    ) -> Receiver<crate::acp::Result<agent_client_protocol_schema::v1::PromptResponse>> {
        let (tx, rx) = mpsc::channel();
        let Some(client) = self.client.clone() else {
            let _ = tx.send(Err(AcpError::TransportClosed));
            return rx;
        };
        let session_id = self.session_id.clone();
        thread::spawn(move || {
            let _ = tx.send(client.session_prompt(session_id, blocks));
        });
        rx
    }

    fn apply_desired_config(
        &mut self,
        model: Option<String>,
        reasoning: Option<String>,
    ) -> std::result::Result<(), String> {
        let client = self
            .client
            .as_ref()
            .ok_or_else(|| "droid client missing".to_string())?;
        let mut desired_model = nonempty(model.as_deref()).map(str::to_owned);
        let mut desired_reasoning = nonempty(reasoning.as_deref()).map(str::to_owned);
        if !self.env_applied {
            self.env_applied = true;
            if desired_model.is_none() {
                desired_model = self.env_model.clone();
            }
            if desired_reasoning.is_none() {
                desired_reasoning = self.env_reasoning.clone();
            }
        }
        let session_id = self.session_id.clone();
        if let Some(value) = desired_model.as_deref() {
            apply_config_option(
                client,
                &mut self.catalog,
                &session_id,
                &self.model_config_id,
                value,
            )?;
        }
        if let Some(value) = desired_reasoning.as_deref() {
            apply_config_option(
                client,
                &mut self.catalog,
                &session_id,
                &self.reasoning_config_id,
                value,
            )?;
        }
        drain_notifications(client, &mut self.catalog, HANDSHAKE_DRAIN);
        Ok(())
    }
}

fn factory_dir() -> Option<PathBuf> {
    env::var_os("HOME").map(|home| PathBuf::from(home).join(".factory"))
}

fn last_session_path() -> Option<PathBuf> {
    factory_dir().map(|dir| dir.join(LAST_SESSION_FILE))
}

fn load_persisted_session_id() -> Option<String> {
    let path = last_session_path()?;
    let text = fs::read_to_string(path).ok()?;
    nonempty(Some(text.trim())).map(str::to_owned)
}

fn persist_last_session_id(id: &str) {
    let Some(dir) = factory_dir() else {
        return;
    };
    if let Err(error) = fs::create_dir_all(&dir) {
        eprintln!("failed to persist Droid session id: {error}");
        return;
    }
    let path = dir.join(LAST_SESSION_FILE);
    let mut file = match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&path)
    {
        Ok(file) => file,
        Err(error) => {
            eprintln!("failed to persist Droid session id: {error}");
            return;
        }
    };
    if let Err(error) = writeln!(file, "{id}") {
        eprintln!("failed to persist Droid session id: {error}");
    }
}

fn handshake_session(
    client: &AcpClient,
    cwd: &Path,
    resume: Option<&str>,
    catalog: &mut ConfigCatalog,
    child: &mut Child,
) -> Result<(SessionId, bool, Option<String>)> {
    if let Some(id) = nonempty(resume) {
        let session_id = SessionId::new(id.to_string());
        match client.session_load(session_id.clone(), cwd) {
            Ok(session) => {
                if let Some(options) = session.config_options.as_deref() {
                    catalog.ingest_typed(options);
                }
                drain_notifications(client, catalog, HANDSHAKE_DRAIN);
                Ok((session_id, true, None))
            }
            Err(error) => {
                drain_notifications(client, catalog, HANDSHAKE_DRAIN);
                if matches!(error, AcpError::TransportClosed) {
                    return Err(map_acp_error(error, child));
                }
                Ok((
                    session_id,
                    false,
                    Some(format!("failed to resume session {id}: {error}")),
                ))
            }
        }
    } else {
        match client.session_new(cwd) {
            Ok(session) => {
                if let Some(options) = session.config_options.as_deref() {
                    catalog.ingest_typed(options);
                }
                Ok((session.session_id, true, None))
            }
            Err(error) => Err(map_acp_error(error, child)),
        }
    }
}

fn drain_notifications(client: &AcpClient, catalog: &mut ConfigCatalog, window: Duration) {
    let deadline = Instant::now() + window;
    while Instant::now() < deadline {
        match client.try_recv_notification() {
            Some((method, params)) => catalog.ingest_update(&method, &params),
            None => thread::sleep(Duration::from_millis(5)),
        }
    }
    while let Some((method, params)) = client.try_recv_notification() {
        catalog.ingest_update(&method, &params);
    }
}

fn apply_notification<W: Write>(
    mapper: &mut DroidMapper,
    catalog: &mut ConfigCatalog,
    normalizer: &mut CodexTurnNormalizer,
    stdout: &mut W,
    method: &str,
    params: Value,
) -> Result<()> {
    catalog.ingest_update(method, &params);
    if method != "session/update" {
        return Ok(());
    }
    write_mapped(normalizer, stdout, mapper.map_update(&params))
}

fn write_mapped<W: Write>(
    normalizer: &mut CodexTurnNormalizer,
    stdout: &mut W,
    mapped: crate::acp::MappedUpdate,
) -> Result<()> {
    for event in mapped.events {
        for notification in normalizer.process_event(&event)? {
            write_notification(stdout, &notification)?;
        }
    }
    if let Some(plan) = mapped.plan {
        for notification in normalizer.emit_plan_updated(plan)? {
            write_notification(stdout, &notification)?;
        }
    }
    Ok(())
}

fn close_open<W: Write>(
    child: &mut DroidChild,
    normalizer: &mut CodexTurnNormalizer,
    stdout: &mut W,
) -> Result<()> {
    write_mapped(normalizer, stdout, child.mapper.abandon_open())?;
    for notification in normalizer.close_open_items()? {
        write_notification(stdout, &notification)?;
    }
    Ok(())
}

fn apply_config_option(
    client: &AcpClient,
    catalog: &mut ConfigCatalog,
    session_id: &SessionId,
    config_id: &str,
    value: &str,
) -> std::result::Result<(), String> {
    if !catalog.is_allowed(config_id, value) {
        return Err(format!("invalid {config_id} value '{value}'"));
    }
    if catalog.current(config_id) == Some(value) {
        return Ok(());
    }
    match client.session_set_config_option(
        session_id.clone(),
        config_id.to_string(),
        value.to_string(),
    ) {
        Ok(response) => {
            if !response.config_options.is_empty() {
                catalog.ingest_typed(&response.config_options);
            }
            catalog.set_current(config_id, value.to_string());
            Ok(())
        }
        Err(error) => Err(format!("failed to set {config_id} to '{value}': {error}")),
    }
}

fn fail_turn<W: Write>(
    normalizer: &mut CodexTurnNormalizer,
    stdout: &mut W,
    message: String,
) -> Result<()> {
    for notification in normalizer.close_open_items()? {
        write_notification(stdout, &notification)?;
    }
    for notification in normalizer.process_event(&NormalizedEvent::Error {
        message: message.clone(),
    })? {
        write_notification(stdout, &notification)?;
    }
    if let Some(notification) = normalizer.finish_turn(Some(message))? {
        write_notification(stdout, &notification)?;
    }
    Ok(())
}

fn write_notification<W: Write>(stdout: &mut W, notification: &ServerNotification) -> Result<()> {
    write_value(stdout, &notification_to_wire_value(notification)?)
}

fn map_acp_error(error: AcpError, child: &mut Child) -> HarnessServerError {
    if matches!(error, AcpError::TransportClosed) {
        return droid_exited(child);
    }
    HarnessServerError::Protocol(error.to_string())
}

fn droid_exited(child: &mut Child) -> HarnessServerError {
    match child.try_wait() {
        Ok(Some(status)) => HarnessServerError::DroidExited { status },
        Ok(None) => match child.wait() {
            Ok(status) => HarnessServerError::DroidExited { status },
            Err(error) => HarnessServerError::Io(error),
        },
        Err(error) => HarnessServerError::Io(error),
    }
}
