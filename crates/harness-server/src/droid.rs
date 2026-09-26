//! Factory Droid harness — drives `droid exec --output-format acp` as a
//! Centaur blocks-mode runtime.
//!
//! One long-lived Droid process per harness process. Lazy-spawned on the first
//! user block, then handshake (`initialize` → `session/new`) and one
//! `session/prompt` per turn. ACP `session/update` events are mapped through
//! `AcpMapper` into the shared `CodexTurnNormalizer`.
//!
//! Per-message `model` / `reasoning` are applied with `session/set_config_option`
//! before each prompt. Local images become ACP image content blocks.

use std::collections::{HashMap, HashSet};
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

use agent_client_protocol_schema::v1::{
    ContentBlock, ImageContent, SessionConfigKind, SessionConfigOption, SessionConfigSelectOptions,
    SessionId, StopReason, TextContent,
};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use codex_app_server_protocol::UserInput;
use serde_json::{Map, Value, json};
use uuid::Uuid;

use crate::acp::{AcpAgentProfile, AcpClient, AcpError, AcpMapper, TerminalManager};
use crate::traits::NormalizedEvent;

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
                model,
                provider: _,
                reasoning,
                trace_context: _,
            }) => {
                turn += 1;
                let result = ensure_child(&mut droid).and_then(|child| {
                    child.run_turn(
                        &mut stdout,
                        input,
                        client_user_message_id,
                        model,
                        reasoning,
                        turn,
                    )
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

fn env_opt(name: &str) -> Option<String> {
    nonempty(env::var(name).ok().as_deref()).map(str::to_owned)
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

#[derive(Debug, Default, Clone)]
struct ConfigCatalog {
    options: HashMap<String, CatalogOption>,
}

#[derive(Debug, Clone, Default)]
struct CatalogOption {
    current: Option<String>,
    values: HashSet<String>,
}

impl ConfigCatalog {
    fn ingest_typed(&mut self, options: &[SessionConfigOption]) {
        for option in options {
            let id = option.id.to_string();
            match &option.kind {
                SessionConfigKind::Select(select) => {
                    let mut values = HashSet::new();
                    match &select.options {
                        SessionConfigSelectOptions::Ungrouped(entries) => {
                            for entry in entries {
                                values.insert(entry.value.to_string());
                            }
                        }
                        SessionConfigSelectOptions::Grouped(groups) => {
                            for group in groups {
                                for entry in &group.options {
                                    values.insert(entry.value.to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                    self.options.insert(
                        id,
                        CatalogOption {
                            current: Some(select.current_value.to_string()),
                            values,
                        },
                    );
                }
                SessionConfigKind::Boolean(_) => {}
                _ => {}
            }
        }
    }

    fn ingest_update(&mut self, method: &str, params: &Value) {
        if method != "session/update" {
            return;
        }
        let update = params.get("update").unwrap_or(params);
        if update.get("sessionUpdate").and_then(Value::as_str) != Some("config_option_update") {
            return;
        }
        let Some(options) = update.get("configOptions") else {
            return;
        };
        if let Ok(parsed) = serde_json::from_value::<Vec<SessionConfigOption>>(options.clone()) {
            self.ingest_typed(&parsed);
        }
    }

    fn current(&self, id: &str) -> Option<&str> {
        self.options
            .get(id)
            .and_then(|option| option.current.as_deref())
    }

    fn is_allowed(&self, id: &str, value: &str) -> bool {
        match self.options.get(id) {
            Some(option) if !option.values.is_empty() => option.values.contains(value),
            _ => true,
        }
    }

    fn set_current(&mut self, id: &str, value: String) {
        self.options.entry(id.to_string()).or_default().current = Some(value);
    }
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
    model_config_id: String,
    reasoning_config_id: String,
    catalog: ConfigCatalog,
    env_model: Option<String>,
    env_reasoning: Option<String>,
    env_applied: bool,
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
        let env_model = env_opt("DROID_MODEL");
        let env_reasoning = env_opt("DROID_REASONING_EFFORT");
        let settings_path = generated_settings_path();
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

        let client = AcpClient::from_rw(stdin, stdout, cwd.clone());
        if let Err(error) = client.initialize() {
            return Err(map_acp_error(error, &mut child));
        }
        let mut catalog = ConfigCatalog::default();
        drain_notifications(&client, &mut catalog, HANDSHAKE_DRAIN);
        let session = match client.session_new(&cwd) {
            Ok(session) => session,
            Err(error) => return Err(map_acp_error(error, &mut child)),
        };
        if let Some(options) = session.config_options.as_deref() {
            catalog.ingest_typed(options);
        }
        drain_notifications(&client, &mut catalog, HANDSHAKE_DRAIN);
        let mapper = AcpMapper::with_terminals(Arc::clone(client.terminals()));
        Ok(Self {
            child,
            client: Some(client),
            mapper,
            session_id: session.session_id,
            settings_path,
            cli_version: profile.cli_version,
            model_provider: profile.model_provider,
            model_config_id: profile.model_config_id,
            reasoning_config_id: profile.reasoning_config_id,
            catalog,
            env_model,
            env_reasoning,
            env_applied: false,
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
        model: Option<String>,
        reasoning: Option<String>,
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
        if let Some(value) = desired_model.as_deref()
            && let Err(message) = apply_config_option(
                client,
                &mut self.catalog,
                &session_id,
                &self.model_config_id,
                value,
            )
        {
            drain_notifications(client, &mut self.catalog, HANDSHAKE_DRAIN);
            return fail_turn(&mut normalizer, stdout, message);
        }
        if let Some(value) = desired_reasoning.as_deref()
            && let Err(message) = apply_config_option(
                client,
                &mut self.catalog,
                &session_id,
                &self.reasoning_config_id,
                value,
            )
        {
            drain_notifications(client, &mut self.catalog, HANDSHAKE_DRAIN);
            return fail_turn(&mut normalizer, stdout, message);
        }
        drain_notifications(client, &mut self.catalog, HANDSHAKE_DRAIN);

        let blocks = prompt_blocks(&input)?;
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
                            &mut self.catalog,
                            &mut normalizer,
                            stdout,
                            &method,
                            params,
                        )?;
                    }
                    Err(RecvTimeoutError::Timeout) => match rx.try_recv() {
                        Ok(result) => {
                            drain_mapped(
                                &mut self.mapper,
                                &mut self.catalog,
                                &mut normalizer,
                                stdout,
                                client,
                            )?;
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

fn drain_mapped<W: Write>(
    mapper: &mut DroidMapper,
    catalog: &mut ConfigCatalog,
    normalizer: &mut CodexTurnNormalizer,
    stdout: &mut W,
    client: &AcpClient,
) -> Result<()> {
    while let Some((method, params)) = client.try_recv_notification() {
        apply_notification(mapper, catalog, normalizer, stdout, &method, params)?;
    }
    Ok(())
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
    for notification in normalizer.process_event(&NormalizedEvent::Error {
        message: message.clone(),
    })? {
        write_value(stdout, &notification_to_wire_value(&notification)?)?;
    }
    if let Some(notification) = normalizer.finish_turn(Some(message))? {
        write_value(stdout, &notification_to_wire_value(&notification)?)?;
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
            for notification in normalizer.process_event(&NormalizedEvent::Error {
                message: message.clone(),
            })? {
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

fn prompt_blocks(input: &[UserInput]) -> Result<Vec<ContentBlock>> {
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
                let bytes = fs::read(path)?;
                blocks.push(ContentBlock::Image(ImageContent::new(
                    BASE64_STANDARD.encode(bytes),
                    mime_type_for_path(path),
                )));
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
    Ok(blocks)
}

fn mime_type_for_path(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("svg") => "image/svg+xml",
        _ => "image/png",
    }
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
    use agent_client_protocol_schema::v1::SessionConfigOption;
    use codex_app_server_protocol::UserInput;
    use serde_json::{Value, json};
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

    #[test]
    fn prompt_blocks_keep_notice_and_emit_image() {
        let path = std::env::temp_dir().join(format!("droid-image-test-{}.png", Uuid::new_v4()));
        let png: &[u8] = &[
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9C, 0x63, 0x00, 0x01, 0x00, 0x00, 0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        fs::write(&path, png).expect("write png");
        let notice = format!("[Attached image saved to {}]", path.display());
        let blocks = super::prompt_blocks(&[
            UserInput::Text {
                text: notice.clone(),
                text_elements: Vec::new(),
            },
            UserInput::LocalImage {
                path: path.clone(),
                detail: None,
            },
        ])
        .expect("blocks");
        let _ = fs::remove_file(&path);
        let json = serde_json::to_value(&blocks).expect("serialize");
        let arr = json.as_array().expect("blocks");
        assert!(
            arr.iter().any(|block| {
                block.get("type").and_then(Value::as_str) == Some("text")
                    && block.get("text").and_then(Value::as_str) == Some(notice.as_str())
            }),
            "kept notice: {json}"
        );
        let image = arr
            .iter()
            .find(|block| block.get("type").and_then(Value::as_str) == Some("image"))
            .expect("image block");
        assert_eq!(image["mimeType"], "image/png");
        assert_eq!(
            image["data"],
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAACklEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
        );
    }

    #[test]
    fn catalog_rejects_unadvertised_values_and_allows_current() {
        let mut catalog = super::ConfigCatalog::default();
        let options = serde_json::from_value::<Vec<SessionConfigOption>>(json!([{
            "id": "model",
            "name": "Model",
            "category": "model",
            "type": "select",
            "currentValue": "gpt-5.4-mini-fast",
            "options": [
                {"value": "gpt-5.4-mini-fast", "name": "mini"},
                {"value": "gpt-test", "name": "test"}
            ]
        }]))
        .expect("options");
        catalog.ingest_typed(&options);
        assert!(catalog.is_allowed("model", "gpt-5.4-mini-fast"));
        assert!(catalog.is_allowed("model", "gpt-test"));
        assert!(!catalog.is_allowed("model", "not-a-droid-model-xyzzy"));
        assert_eq!(catalog.current("model"), Some("gpt-5.4-mini-fast"));
        catalog.set_current("model", "gpt-test".to_string());
        assert_eq!(catalog.current("model"), Some("gpt-test"));
        assert!(catalog.is_allowed("reasoning_effort", "medium"));
    }
}
