//! Factory Droid harness — drives `droid exec --output-format acp` as a
//! Centaur blocks-mode runtime.
//!
//! One long-lived Droid process per harness process. Lazy-spawned on the first
//! user block, then handshake (`initialize` → `session/new`) and one
//! `session/prompt` per turn. ACP `session/update` events are mapped through
//! `AcpMapper` into the shared `CodexTurnNormalizer`.
//!
//! Per-message model/reasoning, image blocks, interrupt, steer, resume, and
//! crash recovery live in later Droid runtime features.

use std::env;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command as ProcessCommand, Stdio};
use std::sync::Arc;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use agent_client_protocol_schema::v1::{ContentBlock, SessionId, StopReason, TextContent};
use codex_app_server_protocol::UserInput;
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::acp::{AcpAgentProfile, AcpClient, AcpError, AcpMapper, TerminalManager};

type DroidMapper = AcpMapper<Arc<TerminalManager>>;
use crate::server::{BlocksCommand, BlocksState, parse_blocks_line_with_state, write_blocks_error};
use crate::turn::{BridgeConfig, CodexTurnNormalizer};
use crate::util::write_value;
use crate::wire::notification_to_wire_value;
use crate::{HarnessServerError, Result};

const HANDSHAKE_DRAIN: Duration = Duration::from_millis(20);
const PROMPT_POLL: Duration = Duration::from_millis(20);
const CHILD_EXIT_WAIT: Duration = Duration::from_secs(2);

/// Entry point for `harness-server droid`.
pub fn run_droid_blocks_server() -> Result<()> {
    let mut stdout = io::stdout().lock();
    let mut droid: Option<DroidChild> = None;
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

    let mut turn = 0u64;
    while let Ok(input) = command_rx.recv() {
        let thread_id = droid
            .as_ref()
            .map(DroidChild::thread_id)
            .unwrap_or("droid")
            .to_owned();
        match input {
            Ok(BlocksCommand::User {
                input,
                client_user_message_id,
                model: _,
                provider: _,
                reasoning: _,
                trace_context: _,
            }) => {
                turn += 1;
                let result = ensure_child(&mut droid).and_then(|child| {
                    child.run_turn(&mut stdout, input, client_user_message_id, turn)
                });
                if let Err(error) = result {
                    eprintln!("Droid blocks turn failed: {error:#}");
                    write_blocks_error(
                        &mut stdout,
                        &thread_id,
                        &format!("turn-{turn}"),
                        error.to_string(),
                    )?;
                    if droid.as_mut().is_some_and(|child| !child.is_alive()) {
                        droid = None;
                    }
                }
            }
            Ok(BlocksCommand::Interrupt) => {
                eprintln!("Droid blocks interrupt ignored: no active turn runs");
            }
            Ok(BlocksCommand::AttachmentChunk) => {}
            Err(error) => {
                eprintln!("invalid Droid blocks input: {error}");
                write_blocks_error(&mut stdout, &thread_id, "input", error)?;
            }
        }
        while interrupt_rx.try_recv().is_ok() {}
    }
    Ok(())
}

fn ensure_child(droid: &mut Option<DroidChild>) -> Result<&mut DroidChild> {
    if droid.is_none() {
        *droid = Some(DroidChild::start()?);
    }
    Ok(droid.as_mut().expect("droid started"))
}

/// Build the Droid ACP profile (program, argv, env, display names).
pub fn droid_profile(cwd: &Path, settings_path: &Path) -> AcpAgentProfile {
    AcpAgentProfile {
        program: droid_program(),
        args: droid_argv(cwd, settings_path),
        extra_env: vec![(
            "FACTORY_DROID_AUTO_UPDATE_ENABLED".to_string(),
            "false".to_string(),
        )],
        model_config_id: "model".to_string(),
        reasoning_config_id: "reasoning_effort".to_string(),
        cli_version: "droid".to_string(),
        model_provider: "factory".to_string(),
    }
}

pub fn droid_program() -> PathBuf {
    env::var("DROID_BIN")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("droid"))
}

pub fn droid_argv(cwd: &Path, settings_path: &Path) -> Vec<String> {
    vec![
        "exec".to_string(),
        "--output-format".to_string(),
        "acp".to_string(),
        "--cwd".to_string(),
        cwd.to_string_lossy().into_owned(),
        "--settings".to_string(),
        settings_path.to_string_lossy().into_owned(),
    ]
}

pub fn settings_document(model: Option<&str>, reasoning: Option<&str>) -> Value {
    let mut session = Map::new();
    session.insert("autonomyLevel".to_string(), json!("high"));
    session.insert("autonomyMode".to_string(), json!("auto-high"));
    session.insert("interactionMode".to_string(), json!("auto"));
    if let Some(model) = nonempty(model) {
        session.insert("model".to_string(), json!(model));
    }
    if let Some(reasoning) = nonempty(reasoning) {
        session.insert("reasoningEffort".to_string(), json!(reasoning));
    }
    json!({
        "sessionDefaultSettings": session,
        "cloudSessionSync": false,
        "enableWarmup": false,
        "enableCompletionBell": false,
    })
}

fn env_settings_document() -> Value {
    settings_document(
        nonempty(env::var("DROID_MODEL").ok().as_deref()),
        nonempty(env::var("DROID_REASONING_EFFORT").ok().as_deref()),
    )
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

pub fn write_settings_file(path: &Path, document: &Value) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    serde_json::to_writer_pretty(&mut file, document)?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(())
}

fn generated_settings_path() -> PathBuf {
    env::temp_dir().join(format!("centaur-droid-settings-{}.json", Uuid::new_v4()))
}

struct DroidChild {
    child: Child,
    client: Option<AcpClient>,
    mapper: DroidMapper,
    session_id: SessionId,
    settings_path: PathBuf,
    cli_version: String,
    model_provider: String,
}

impl Drop for DroidChild {
    fn drop(&mut self) {
        if let Some(client) = self.client.take() {
            client.terminals().kill_all();
            drop(client);
        }
        let deadline = Instant::now() + CHILD_EXIT_WAIT;
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(20));
                }
                _ => {
                    let _ = self.child.kill();
                    let _ = self.child.wait();
                    break;
                }
            }
        }
        let _ = fs::remove_file(&self.settings_path);
    }
}

impl DroidChild {
    fn start() -> Result<Self> {
        let cwd = env::current_dir()?;
        let cwd = cwd.canonicalize().unwrap_or(cwd);
        let settings_path = generated_settings_path();
        write_settings_file(&settings_path, &env_settings_document())?;
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

        let client = AcpClient::from_rw(stdin, stdout, cwd.clone());
        if let Err(error) = client.initialize() {
            return Err(map_acp_error(error, &mut child));
        }
        drain_notifications(&client, HANDSHAKE_DRAIN);
        let session = match client.session_new(&cwd) {
            Ok(session) => session,
            Err(error) => return Err(map_acp_error(error, &mut child)),
        };
        drain_notifications(&client, HANDSHAKE_DRAIN);
        let mapper = AcpMapper::with_terminals(Arc::clone(client.terminals()));
        Ok(Self {
            child,
            client: Some(client),
            mapper,
            session_id: session.session_id,
            settings_path,
            cli_version: profile.cli_version,
            model_provider: profile.model_provider,
        })
    }

    fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    fn thread_id(&self) -> &str {
        self.session_id.0.as_ref()
    }

    fn run_turn<W: Write>(
        &mut self,
        stdout: &mut W,
        input: Vec<UserInput>,
        client_user_message_id: Option<String>,
        turn: u64,
    ) -> Result<()> {
        let mut config = BridgeConfig::new(self.thread_id().to_string(), format!("turn-{turn}"));
        config.cli_version = self.cli_version.clone();
        config.model_provider = self.model_provider.clone();
        let mut normalizer = CodexTurnNormalizer::new(config);

        for notification in normalizer.start_notifications(turn == 1)? {
            write_value(stdout, &notification_to_wire_value(&notification)?)?;
        }
        for notification in normalizer.emit_user_message(client_user_message_id, input.clone())? {
            write_value(stdout, &notification_to_wire_value(&notification)?)?;
        }

        let client = self
            .client
            .as_ref()
            .ok_or_else(|| HarnessServerError::Protocol("droid client missing".to_string()))?;
        let session_id = self.session_id.clone();
        let blocks = prompt_blocks(&input);
        let (tx, rx) = mpsc::channel();
        thread::scope(|scope| {
            scope.spawn(move || {
                let result = client.session_prompt(session_id, blocks);
                let _ = tx.send(result);
            });

            loop {
                match client.recv_notification_timeout(PROMPT_POLL) {
                    Ok((method, params)) => {
                        apply_notification(
                            &mut self.mapper,
                            &mut normalizer,
                            stdout,
                            &method,
                            params,
                        )?;
                    }
                    Err(RecvTimeoutError::Timeout) => match rx.try_recv() {
                        Ok(result) => {
                            drain_mapped(&mut self.mapper, &mut normalizer, stdout, client)?;
                            return finish_prompt(
                                &mut self.mapper,
                                &mut normalizer,
                                stdout,
                                result,
                                &mut self.child,
                            );
                        }
                        Err(mpsc::TryRecvError::Empty) => {}
                        Err(mpsc::TryRecvError::Disconnected) => {
                            return Err(droid_exited(&mut self.child));
                        }
                    },
                    Err(RecvTimeoutError::Disconnected) => {
                        return Err(droid_exited(&mut self.child));
                    }
                }
            }
        })
    }
}

fn drain_notifications(client: &AcpClient, window: Duration) {
    let deadline = Instant::now() + window;
    while Instant::now() < deadline {
        if client.try_recv_notification().is_none() {
            thread::sleep(Duration::from_millis(5));
        }
    }
    while client.try_recv_notification().is_some() {}
}

fn drain_mapped<W: Write>(
    mapper: &mut DroidMapper,
    normalizer: &mut CodexTurnNormalizer,
    stdout: &mut W,
    client: &AcpClient,
) -> Result<()> {
    while let Some((method, params)) = client.try_recv_notification() {
        apply_notification(mapper, normalizer, stdout, &method, params)?;
    }
    Ok(())
}

fn apply_notification<W: Write>(
    mapper: &mut DroidMapper,
    normalizer: &mut CodexTurnNormalizer,
    stdout: &mut W,
    method: &str,
    params: Value,
) -> Result<()> {
    if method != "session/update" {
        return Ok(());
    }
    let mapped = mapper.map_update(&params);
    for event in mapped.events {
        for notification in normalizer.process_event(&event)? {
            write_value(stdout, &notification_to_wire_value(&notification)?)?;
        }
    }
    if let Some(plan) = mapped.plan {
        for notification in normalizer.emit_plan_updated(plan)? {
            write_value(stdout, &notification_to_wire_value(&notification)?)?;
        }
    }
    Ok(())
}

fn finish_prompt<W: Write>(
    mapper: &mut DroidMapper,
    normalizer: &mut CodexTurnNormalizer,
    stdout: &mut W,
    result: crate::acp::Result<agent_client_protocol_schema::v1::PromptResponse>,
    child: &mut Child,
) -> Result<()> {
    match result {
        Ok(response) => {
            let mapped = mapper.finish("end_turn");
            for event in mapped.events {
                for notification in normalizer.process_event(&event)? {
                    write_value(stdout, &notification_to_wire_value(&notification)?)?;
                }
            }
            let failed = match response.stop_reason {
                StopReason::Refusal => Some("agent refused".to_string()),
                StopReason::Cancelled => {
                    if let Some(notification) = normalizer.finish_turn_interrupted()? {
                        write_value(stdout, &notification_to_wire_value(&notification)?)?;
                    }
                    return Ok(());
                }
                StopReason::EndTurn | StopReason::MaxTokens | StopReason::MaxTurnRequests => None,
                _ => None,
            };
            if let Some(notification) = normalizer.finish_turn(failed)? {
                write_value(stdout, &notification_to_wire_value(&notification)?)?;
            }
            Ok(())
        }
        Err(error) => {
            let message = error.to_string();
            for notification in
                normalizer.process_event(&crate::traits::NormalizedEvent::Error {
                    message: message.clone(),
                })?
            {
                write_value(stdout, &notification_to_wire_value(&notification)?)?;
            }
            if let Some(notification) = normalizer.finish_turn(Some(message.clone()))? {
                write_value(stdout, &notification_to_wire_value(&notification)?)?;
            }
            if matches!(error, AcpError::TransportClosed) {
                return Err(droid_exited(child));
            }
            Ok(())
        }
    }
}

fn prompt_blocks(input: &[UserInput]) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();
    for item in input {
        match item {
            UserInput::Text { text, .. } => {
                blocks.push(ContentBlock::Text(TextContent::new(text.clone())));
            }
            UserInput::Image { url, .. } => {
                blocks.push(ContentBlock::Text(TextContent::new(format!(
                    "[image: {url}]"
                ))));
            }
            UserInput::LocalImage { path, .. } => {
                blocks.push(ContentBlock::Text(TextContent::new(format!(
                    "[Attached image saved to {}]",
                    path.display()
                ))));
            }
            UserInput::Skill { name, path } => {
                blocks.push(ContentBlock::Text(TextContent::new(format!(
                    "[skill: {name} at {}]",
                    path.display()
                ))));
            }
            UserInput::Mention { name, path } => {
                blocks.push(ContentBlock::Text(TextContent::new(format!(
                    "[mention: {name} at {path}]"
                ))));
            }
        }
    }
    if blocks.is_empty() {
        blocks.push(ContentBlock::Text(TextContent::new("continue")));
    }
    blocks
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

#[cfg(test)]
mod tests {
    use super::{droid_argv, droid_profile, settings_document, write_settings_file};
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use uuid::Uuid;

    #[test]
    fn droid_argv_is_exec_acp_cwd_settings() {
        let cwd = PathBuf::from("/workspace");
        let settings = PathBuf::from("/tmp/settings.json");
        assert_eq!(
            droid_argv(&cwd, &settings),
            vec![
                "exec",
                "--output-format",
                "acp",
                "--cwd",
                "/workspace",
                "--settings",
                "/tmp/settings.json",
            ]
        );
    }

    #[test]
    fn settings_omits_model_and_reasoning_when_unset() {
        let value = settings_document(None, None);
        let session = &value["sessionDefaultSettings"];
        assert_eq!(session["autonomyLevel"], "high");
        assert_eq!(session["autonomyMode"], "auto-high");
        assert_eq!(session["interactionMode"], "auto");
        assert!(session.get("model").is_none());
        assert!(session.get("reasoningEffort").is_none());
        assert_eq!(value["cloudSessionSync"], false);
        assert_eq!(value["enableWarmup"], false);
        assert_eq!(value["enableCompletionBell"], false);
    }

    #[test]
    fn settings_includes_model_and_reasoning_when_set() {
        let value = settings_document(Some("gpt-5.4-mini-fast"), Some("low"));
        let session = &value["sessionDefaultSettings"];
        assert_eq!(session["model"], "gpt-5.4-mini-fast");
        assert_eq!(session["reasoningEffort"], "low");
    }

    #[test]
    fn settings_file_mode_is_0600() {
        let path =
            std::env::temp_dir().join(format!("droid-settings-test-{}.json", Uuid::new_v4()));
        write_settings_file(&path, &settings_document(None, None)).expect("write");
        let mode = fs::metadata(&path).expect("meta").permissions().mode() & 0o777;
        let _ = fs::remove_file(&path);
        assert_eq!(mode, 0o600);
    }

    #[test]
    fn profile_disables_auto_update_and_names_factory() {
        let cwd = PathBuf::from("/workspace");
        let settings = PathBuf::from("/tmp/settings.json");
        let profile = droid_profile(&cwd, &settings);
        assert_eq!(
            profile.extra_env,
            vec![(
                "FACTORY_DROID_AUTO_UPDATE_ENABLED".to_string(),
                "false".to_string()
            )]
        );
        assert_eq!(profile.cli_version, "droid");
        assert_eq!(profile.model_provider, "factory");
        assert_eq!(profile.model_config_id, "model");
        assert_eq!(profile.reasoning_config_id, "reasoning_effort");
        assert!(!profile.args.iter().any(|arg| arg.contains("fk-")));
    }
}
