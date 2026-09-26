//! ACP `session/update` → `NormalizedEvent` (plus plan) translation.
//!
//! Each update is parsed from `serde_json::Value`. Unknown `sessionUpdate`
//! tags are ignored and never fail the stream.

use std::collections::HashMap;
use std::sync::Arc;

use agent_client_protocol_schema::v1::TerminalId;
use codex_app_server_protocol::{TurnPlanStep, TurnPlanStepStatus};
use serde_json::{Value, json};

use super::terminal::{TerminalManager, TerminalSnapshot};
use crate::traits::{NormalizedContent, NormalizedEvent, NormalizedToolResult};

/// Looks up captured terminal output/exit so execute tools can become
/// `commandExecution` items with `aggregatedOutput` and `exitCode`.
pub trait TerminalLookup {
    fn lookup_terminal(&self, terminal_id: &str) -> Option<TerminalSnapshot>;
}

/// Lookup that always misses. Useful when no client terminals exist yet.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoTerminals;

impl TerminalLookup for NoTerminals {
    fn lookup_terminal(&self, _terminal_id: &str) -> Option<TerminalSnapshot> {
        None
    }
}

impl TerminalLookup for TerminalManager {
    fn lookup_terminal(&self, terminal_id: &str) -> Option<TerminalSnapshot> {
        self.snapshot(&TerminalId::new(terminal_id.to_string()))
    }
}

impl TerminalLookup for HashMap<String, TerminalSnapshot> {
    fn lookup_terminal(&self, terminal_id: &str) -> Option<TerminalSnapshot> {
        self.get(terminal_id).cloned()
    }
}

impl<T: TerminalLookup + ?Sized> TerminalLookup for &T {
    fn lookup_terminal(&self, terminal_id: &str) -> Option<TerminalSnapshot> {
        (**self).lookup_terminal(terminal_id)
    }
}

impl<T: TerminalLookup + ?Sized> TerminalLookup for Arc<T> {
    fn lookup_terminal(&self, terminal_id: &str) -> Option<TerminalSnapshot> {
        (**self).lookup_terminal(terminal_id)
    }
}

/// Events and optional plan produced from one ACP `session/update`.
#[derive(Debug, Default, Clone)]
pub struct MappedUpdate {
    pub events: Vec<NormalizedEvent>,
    pub plan: Option<Vec<TurnPlanStep>>,
}

impl MappedUpdate {
    pub fn is_empty(&self) -> bool {
        self.events.is_empty() && self.plan.is_none()
    }
}

#[derive(Debug, Clone)]
struct OpenAgent {
    item_id: String,
    text: String,
}

#[derive(Debug, Clone)]
struct OpenTool {
    execute: bool,
    terminal_id: Option<String>,
}

/// Stateful translator for one ACP session/turn.
pub struct AcpMapper<L> {
    terminals: L,
    next_message: u32,
    next_reasoning: u32,
    open_agent: Option<OpenAgent>,
    open_tools: HashMap<String, OpenTool>,
}

impl AcpMapper<NoTerminals> {
    pub fn new() -> Self {
        Self::with_terminals(NoTerminals)
    }
}

impl Default for AcpMapper<NoTerminals> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L: TerminalLookup> AcpMapper<L> {
    pub fn with_terminals(terminals: L) -> Self {
        Self {
            terminals,
            next_message: 0,
            next_reasoning: 0,
            open_agent: None,
            open_tools: HashMap::new(),
        }
    }

    /// Translate one ACP update. Accepts the update object, `{update: ...}`,
    /// or a full `session/update` JSON-RPC notification. Unknown tags yield
    /// an empty result and never error.
    pub fn map_update(&mut self, value: &Value) -> MappedUpdate {
        let Some(update) = session_update_object(value) else {
            return MappedUpdate::default();
        };
        let Some(tag) = update.get("sessionUpdate").and_then(Value::as_str) else {
            return MappedUpdate::default();
        };
        match tag {
            "agent_message_chunk" => self.on_agent_message(update),
            "agent_thought_chunk" => self.on_thought(update),
            "tool_call" => self.on_tool_call(update),
            "tool_call_update" => self.on_tool_call_update(update),
            "plan" => self.on_plan(update),
            "user_message_chunk"
            | "available_commands_update"
            | "current_mode_update"
            | "config_option_update"
            | "usage_update"
            | "session_info_update" => MappedUpdate::default(),
            _ => MappedUpdate::default(),
        }
    }

    /// Close an in-flight agent message using the prompt stop reason.
    /// Text before a tool call is already flushed as commentary; the last
    /// remaining agent message becomes `final_answer` via `end_turn`.
    pub fn finish(&mut self, stop_reason: &str) -> MappedUpdate {
        let mut events = Vec::new();
        let reason = match stop_reason {
            "tool_use" => "tool_use",
            _ => "end_turn",
        };
        self.flush_agent(&mut events, reason);
        MappedUpdate { events, plan: None }
    }

    fn on_agent_message(&mut self, update: &Value) -> MappedUpdate {
        let Some(text) = text_from_content_block(update) else {
            return MappedUpdate::default();
        };
        if text.is_empty() {
            return MappedUpdate::default();
        }
        let item_id = self.ensure_open_agent();
        if let Some(open) = &mut self.open_agent {
            open.text.push_str(&text);
        }
        MappedUpdate {
            events: vec![NormalizedEvent::AgentTextDelta {
                item_id,
                delta: text,
            }],
            plan: None,
        }
    }

    fn on_thought(&mut self, update: &Value) -> MappedUpdate {
        let mut events = Vec::new();
        self.flush_agent(&mut events, "tool_use");
        let Some(text) = text_from_content_block(update) else {
            return MappedUpdate { events, plan: None };
        };
        if text.is_empty() {
            return MappedUpdate { events, plan: None };
        }
        self.next_reasoning += 1;
        let item_id = format!("acp-reasoning-{}", self.next_reasoning);
        events.push(NormalizedEvent::AssistantMessage {
            partial: false,
            stop_reason: None,
            content: vec![NormalizedContent::ReasoningText { item_id, text }],
        });
        MappedUpdate { events, plan: None }
    }

    fn on_tool_call(&mut self, update: &Value) -> MappedUpdate {
        let Some(tool_call_id) = update.get("toolCallId").and_then(Value::as_str) else {
            return MappedUpdate::default();
        };
        let mut events = Vec::new();
        self.flush_agent(&mut events, "tool_use");

        let execute = is_execute_kind(update);
        let terminal_id = terminal_id_from_content(update);
        let (tool, arguments) = if execute {
            let command = execute_command(update);
            ("shell_command".to_string(), json!({"command": command}))
        } else {
            (
                other_tool_name(update),
                update.get("rawInput").cloned().unwrap_or_else(|| json!({})),
            )
        };

        self.open_tools.insert(
            tool_call_id.to_string(),
            OpenTool {
                execute,
                terminal_id: terminal_id.clone(),
            },
        );
        events.push(NormalizedEvent::AssistantMessage {
            partial: false,
            stop_reason: None,
            content: vec![NormalizedContent::ToolUse {
                raw_id: tool_call_id.to_string(),
                tool,
                arguments,
            }],
        });

        let status = status_of(update);
        if status == "completed" || status == "failed" {
            events.push(NormalizedEvent::ToolResults(vec![self.complete_tool(
                tool_call_id,
                update,
                status == "failed",
            )]));
        }

        MappedUpdate { events, plan: None }
    }

    fn on_tool_call_update(&mut self, update: &Value) -> MappedUpdate {
        let Some(tool_call_id) = update.get("toolCallId").and_then(Value::as_str) else {
            return MappedUpdate::default();
        };
        if let Some(terminal_id) = terminal_id_from_content(update) {
            let entry = self
                .open_tools
                .entry(tool_call_id.to_string())
                .or_insert(OpenTool {
                    execute: true,
                    terminal_id: None,
                });
            entry.terminal_id = Some(terminal_id);
            entry.execute = true;
        }

        let status = status_of(update);
        if status != "completed" && status != "failed" {
            return MappedUpdate::default();
        }
        MappedUpdate {
            events: vec![NormalizedEvent::ToolResults(vec![self.complete_tool(
                tool_call_id,
                update,
                status == "failed",
            )])],
            plan: None,
        }
    }

    fn on_plan(&mut self, update: &Value) -> MappedUpdate {
        let Some(entries) = update.get("entries").and_then(Value::as_array) else {
            return MappedUpdate::default();
        };
        let plan = entries.iter().filter_map(plan_step_from_entry).collect();
        MappedUpdate {
            events: Vec::new(),
            plan: Some(plan),
        }
    }

    fn complete_tool(
        &mut self,
        tool_call_id: &str,
        update: &Value,
        failed_update: bool,
    ) -> NormalizedToolResult {
        let open = self.open_tools.remove(tool_call_id);
        let mut execute = open.as_ref().is_some_and(|tool| tool.execute);
        let mut terminal_id = open.and_then(|tool| tool.terminal_id);
        if let Some(from_content) = terminal_id_from_content(update) {
            terminal_id = Some(from_content);
            execute = true;
        }

        if execute {
            let snapshot = terminal_id
                .as_deref()
                .and_then(|id| self.terminals.lookup_terminal(id));
            let exit_code = snapshot
                .as_ref()
                .and_then(|snap| snap.exit_code.and_then(|code| i32::try_from(code).ok()));
            let content = snapshot.map(|snap| snap.output).unwrap_or_default();
            let is_error = failed_update || exit_code.is_none_or(|code| code != 0);
            NormalizedToolResult {
                tool_use_id: tool_call_id.to_string(),
                content,
                is_error,
                exit_code,
            }
        } else {
            NormalizedToolResult {
                tool_use_id: tool_call_id.to_string(),
                content: tool_output_text(update),
                is_error: failed_update,
                exit_code: None,
            }
        }
    }

    fn ensure_open_agent(&mut self) -> String {
        if let Some(open) = &self.open_agent {
            return open.item_id.clone();
        }
        self.next_message += 1;
        let item_id = format!("acp-msg-{}", self.next_message);
        self.open_agent = Some(OpenAgent {
            item_id: item_id.clone(),
            text: String::new(),
        });
        item_id
    }

    fn flush_agent(&mut self, events: &mut Vec<NormalizedEvent>, stop_reason: &str) {
        let Some(open) = self.open_agent.take() else {
            return;
        };
        if open.text.is_empty() {
            return;
        }
        events.push(NormalizedEvent::AssistantMessage {
            partial: false,
            stop_reason: Some(stop_reason.to_string()),
            content: vec![NormalizedContent::AgentText {
                item_id: open.item_id,
                text: open.text,
            }],
        });
    }
}

fn session_update_object(value: &Value) -> Option<&Value> {
    if value.get("sessionUpdate").is_some() {
        return Some(value);
    }
    value
        .pointer("/params/update")
        .or_else(|| value.get("update"))
        .filter(|update| update.get("sessionUpdate").is_some())
}

fn text_from_content_block(update: &Value) -> Option<String> {
    let content = update.get("content")?;
    if let Some(kind) = content.get("type").and_then(Value::as_str)
        && kind != "text"
    {
        return None;
    }
    content
        .get("text")
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn is_execute_kind(update: &Value) -> bool {
    update
        .get("kind")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind.eq_ignore_ascii_case("execute"))
}

fn execute_command(update: &Value) -> String {
    update
        .pointer("/rawInput/command")
        .and_then(Value::as_str)
        .or_else(|| update.get("title").and_then(Value::as_str))
        .unwrap_or("")
        .to_string()
}

fn other_tool_name(update: &Value) -> String {
    update
        .get("kind")
        .and_then(Value::as_str)
        .filter(|kind| !kind.is_empty())
        .or_else(|| update.get("title").and_then(Value::as_str))
        .unwrap_or("other")
        .to_string()
}

fn status_of(update: &Value) -> &str {
    update.get("status").and_then(Value::as_str).unwrap_or("")
}

fn terminal_id_from_content(update: &Value) -> Option<String> {
    update
        .get("content")
        .and_then(Value::as_array)
        .and_then(|items| {
            items.iter().find_map(|item| {
                if item.get("type").and_then(Value::as_str) == Some("terminal") {
                    item.get("terminalId")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                } else {
                    None
                }
            })
        })
}

fn tool_output_text(update: &Value) -> String {
    let mut texts = Vec::new();
    if let Some(items) = update.get("content").and_then(Value::as_array) {
        for item in items {
            if item.get("type").and_then(Value::as_str) == Some("content")
                && let Some(text) = text_from_content_block(item)
            {
                texts.push(text);
            }
        }
    }
    if !texts.is_empty() {
        return texts.join("\n");
    }
    if let Some(text) = update.pointer("/rawOutput/text").and_then(Value::as_str) {
        return text.to_string();
    }
    match update.get("rawOutput") {
        Some(Value::String(text)) => text.clone(),
        Some(value) if !value.is_null() => serde_json::to_string(value).unwrap_or_default(),
        _ => String::new(),
    }
}

fn plan_step_from_entry(entry: &Value) -> Option<TurnPlanStep> {
    let step = entry.get("content").and_then(Value::as_str)?.to_string();
    let status = match entry.get("status").and_then(Value::as_str) {
        Some("in_progress") => TurnPlanStepStatus::InProgress,
        Some("completed") => TurnPlanStepStatus::Completed,
        _ => TurnPlanStepStatus::Pending,
    };
    Some(TurnPlanStep { step, status })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use serde_json::{Value, json};

    use super::AcpMapper;
    use crate::acp::terminal::TerminalSnapshot;
    use crate::traits::{NormalizedContent, NormalizedEvent};
    use crate::turn::{BridgeConfig, CodexTurnNormalizer};
    use crate::wire::{notification_to_jsonrpc, notification_to_wire_value};

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/acp")
            .join(name)
    }

    fn load_ndjson(name: &str) -> Vec<Value> {
        let text = std::fs::read_to_string(fixture_path(name)).unwrap();
        text.lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    fn session_updates(name: &str) -> Vec<Value> {
        load_ndjson(name)
            .into_iter()
            .filter(|value| value.get("method").and_then(Value::as_str) == Some("session/update"))
            .map(|value| value["params"]["update"].clone())
            .collect()
    }

    fn test_normalizer() -> CodexTurnNormalizer {
        let mut config = BridgeConfig::new("thread-1", "turn-1");
        config.cwd = PathBuf::from("/tmp");
        config.cli_version = "droid".to_string();
        config.model_provider = "factory".to_string();
        CodexTurnNormalizer::new(config)
    }

    fn methods_of(notifications: &[codex_app_server_protocol::ServerNotification]) -> Vec<String> {
        notifications
            .iter()
            .map(|notification| notification_to_jsonrpc(notification).unwrap().method)
            .collect()
    }

    fn params_of(notification: &codex_app_server_protocol::ServerNotification) -> Value {
        notification_to_jsonrpc(notification)
            .unwrap()
            .params
            .unwrap()
    }

    fn apply_mapped(
        normalizer: &mut CodexTurnNormalizer,
        mapped: super::MappedUpdate,
    ) -> Vec<codex_app_server_protocol::ServerNotification> {
        let mut out = Vec::new();
        if let Some(plan) = mapped.plan {
            out.extend(normalizer.emit_plan_updated(plan).unwrap());
        }
        for event in mapped.events {
            out.extend(normalizer.process_event(&event).unwrap());
        }
        out
    }

    #[test]
    fn unknown_session_update_tag_is_ignored() {
        let mut mapper = AcpMapper::new();
        let out = mapper.map_update(&json!({
            "sessionUpdate": "vendor_extension_xyz",
            "payload": {"foo": 1}
        }));
        assert!(out.is_empty());
        let out = mapper.map_update(&json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "sess",
                "update": {"sessionUpdate": "not_a_real_tag"}
            }
        }));
        assert!(out.is_empty());
        let finished = mapper.finish("end_turn");
        assert!(finished.is_empty());
    }

    #[test]
    fn ignored_session_update_tags_produce_no_output() {
        let mut mapper = AcpMapper::new();
        let tags = [
            "user_message_chunk",
            "available_commands_update",
            "current_mode_update",
            "config_option_update",
            "usage_update",
            "session_info_update",
        ];
        for tag in tags {
            let out = mapper.map_update(&json!({
                "sessionUpdate": tag,
                "content": {"type": "text", "text": "should be ignored"}
            }));
            assert!(out.is_empty(), "{tag} should be ignored");
        }
        for update in session_updates("session-load-replay.ndjson") {
            if update["sessionUpdate"] == "user_message_chunk" {
                assert!(mapper.map_update(&update).is_empty());
            }
        }
    }

    #[test]
    fn thought_chunks_emit_reasoning_item() {
        let thoughts: Vec<Value> = session_updates("plan.ndjson")
            .into_iter()
            .filter(|update| update["sessionUpdate"] == "agent_thought_chunk")
            .collect();
        assert_eq!(thoughts.len(), 3, "plan.ndjson should have three thoughts");

        let mut mapper = AcpMapper::new();
        let mut normalizer = test_normalizer();
        normalizer.start_notifications(false).unwrap();

        let mut notifications = Vec::new();
        for update in &thoughts {
            notifications.extend(apply_mapped(&mut normalizer, mapper.map_update(update)));
        }

        assert_eq!(
            methods_of(&notifications),
            vec![
                "item/started",
                "item/reasoning/textDelta",
                "item/completed",
                "item/started",
                "item/reasoning/textDelta",
                "item/completed",
                "item/started",
                "item/reasoning/textDelta",
                "item/completed",
            ]
        );

        for (index, update) in thoughts.iter().enumerate() {
            let expected = update["content"]["text"].as_str().unwrap();
            let started = params_of(&notifications[index * 3]);
            assert_eq!(started["item"]["type"], "reasoning");
            assert_eq!(
                started["item"]["id"],
                format!("acp-reasoning-{}", index + 1)
            );
            assert_ne!(started["item"]["type"], "agentMessage");

            let delta = params_of(&notifications[index * 3 + 1]);
            assert_eq!(delta["delta"], expected);
            assert_eq!(delta["itemId"], format!("acp-reasoning-{}", index + 1));
            assert_eq!(delta["contentIndex"], 0);

            let completed = params_of(&notifications[index * 3 + 2]);
            assert_eq!(completed["item"]["type"], "reasoning");
            assert_eq!(completed["item"]["content"][0], expected);
            assert_ne!(completed["item"]["type"], "agentMessage");
        }
    }

    #[test]
    fn plan_update_emits_turn_plan_updated() {
        let plans: Vec<Value> = session_updates("plan.ndjson")
            .into_iter()
            .filter(|update| update["sessionUpdate"] == "plan")
            .collect();
        assert_eq!(plans.len(), 3);

        let expected = [
            json!([
                {"step": "Read `note.txt`", "status": "inProgress"},
                {"step": "Search the current directory for `ALPHA`", "status": "pending"},
                {"step": "Write `result.txt` so it contains exactly `BETA`", "status": "pending"}
            ]),
            json!([
                {"step": "Read `note.txt`", "status": "completed"},
                {"step": "Search the current directory for `ALPHA`", "status": "completed"},
                {"step": "Write `result.txt` so it contains exactly `BETA`", "status": "inProgress"}
            ]),
            json!([
                {"step": "Read `note.txt`", "status": "completed"},
                {"step": "Search the current directory for `ALPHA`", "status": "completed"},
                {"step": "Write `result.txt` so it contains exactly `BETA`", "status": "completed"}
            ]),
        ];

        let mut mapper = AcpMapper::new();
        let mut normalizer = test_normalizer();
        normalizer.start_notifications(false).unwrap();

        for (update, expected_plan) in plans.iter().zip(expected) {
            let notifications = apply_mapped(&mut normalizer, mapper.map_update(update));
            assert_eq!(notifications.len(), 1);
            let wire = notification_to_wire_value(&notifications[0]).unwrap();
            assert_eq!(
                wire,
                json!({
                    "method": "turn/plan/updated",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "explanation": null,
                        "plan": expected_plan
                    }
                })
            );
        }
    }

    #[test]
    fn agent_message_chunks_stream_text_deltas_as_final_answer() {
        let updates: Vec<Value> = session_updates("simple-text-turn.ndjson")
            .into_iter()
            .filter(|update| update["sessionUpdate"] == "agent_message_chunk")
            .collect();
        assert_eq!(updates.len(), 1);

        let mut mapper = AcpMapper::new();
        let mut normalizer = test_normalizer();
        normalizer.start_notifications(false).unwrap();

        let mut notifications = Vec::new();
        for update in &updates {
            notifications.extend(apply_mapped(&mut normalizer, mapper.map_update(update)));
        }
        notifications.extend(apply_mapped(&mut normalizer, mapper.finish("end_turn")));

        assert_eq!(
            methods_of(&notifications),
            vec!["item/started", "item/agentMessage/delta", "item/completed",]
        );
        let delta = params_of(&notifications[1]);
        assert_eq!(delta["delta"], "PONG");
        let completed = params_of(&notifications[2]);
        assert_eq!(completed["item"]["type"], "agentMessage");
        assert_eq!(completed["item"]["text"], "PONG");
        assert_eq!(completed["item"]["phase"], "final_answer");
    }

    #[test]
    fn text_before_tool_call_is_commentary() {
        let mut mapper = AcpMapper::new();
        let mut normalizer = test_normalizer();
        normalizer.start_notifications(false).unwrap();

        let mut notifications = Vec::new();
        for update in session_updates("permission-request.ndjson") {
            notifications.extend(apply_mapped(&mut normalizer, mapper.map_update(&update)));
        }
        notifications.extend(apply_mapped(&mut normalizer, mapper.finish("end_turn")));

        let completed_messages: Vec<Value> = notifications
            .iter()
            .filter_map(|notification| {
                let rpc = notification_to_jsonrpc(notification).unwrap();
                if rpc.method != "item/completed" {
                    return None;
                }
                let params = rpc.params.unwrap();
                (params["item"]["type"] == "agentMessage").then_some(params["item"].clone())
            })
            .collect();
        assert_eq!(completed_messages.len(), 2);
        assert_eq!(completed_messages[0]["phase"], "commentary");
        assert!(
            completed_messages[0]["text"]
                .as_str()
                .unwrap()
                .contains("This will read a remote header")
        );
        assert_eq!(completed_messages[1]["phase"], "final_answer");
        assert_eq!(completed_messages[1]["text"], "DONE LLAMA9");
    }

    #[test]
    fn execute_tool_call_becomes_command_execution_with_terminal_output() {
        let mut terminals = HashMap::new();
        terminals.insert(
            "2273a33e-fade-4759-beb9-8bc0d8f31f16".to_string(),
            TerminalSnapshot {
                output: "hi\nAGENTS.md\nnote.txt\nresult.txt\nscratch.txt\n".to_string(),
                truncated: false,
                exit_code: Some(0),
                signal: None,
            },
        );
        let mut mapper = AcpMapper::with_terminals(terminals);
        let mut normalizer = test_normalizer();
        normalizer.start_notifications(false).unwrap();

        let mut notifications = Vec::new();
        for update in session_updates("execute-terminal.ndjson") {
            notifications.extend(apply_mapped(&mut normalizer, mapper.map_update(&update)));
        }

        let started = notifications.iter().find_map(|notification| {
            let rpc = notification_to_jsonrpc(notification).unwrap();
            (rpc.method == "item/started").then(|| rpc.params.unwrap())
        });
        let started = started.expect("command item started");
        assert_eq!(started["item"]["type"], "commandExecution");
        assert_eq!(started["item"]["command"], "echo hi; ls");
        assert_eq!(started["item"]["status"], "inProgress");

        let completed = notifications
            .iter()
            .rev()
            .find_map(|notification| {
                let rpc = notification_to_jsonrpc(notification).unwrap();
                if rpc.method != "item/completed" {
                    return None;
                }
                let params = rpc.params.unwrap();
                (params["item"]["type"] == "commandExecution").then_some(params)
            })
            .expect("command item completed");
        assert_eq!(completed["item"]["status"], "completed");
        assert_eq!(completed["item"]["exitCode"], 0);
        assert_eq!(
            completed["item"]["aggregatedOutput"],
            "hi\nAGENTS.md\nnote.txt\nresult.txt\nscratch.txt\n"
        );
    }

    fn completed_command_item(
        notifications: &[codex_app_server_protocol::ServerNotification],
    ) -> Value {
        notifications
            .iter()
            .rev()
            .find_map(|notification| {
                let rpc = notification_to_jsonrpc(notification).unwrap();
                if rpc.method != "item/completed" {
                    return None;
                }
                let item = rpc.params.unwrap()["item"].clone();
                (item["type"] == "commandExecution").then_some(item)
            })
            .expect("commandExecution item/completed")
    }

    #[test]
    fn execute_nonzero_exit_is_failed() {
        let mut terminals = HashMap::new();
        terminals.insert(
            "term-nonzero".to_string(),
            TerminalSnapshot {
                output: "FAIL-MARKER\n".to_string(),
                truncated: false,
                exit_code: Some(7),
                signal: None,
            },
        );
        let mut mapper = AcpMapper::with_terminals(terminals);
        let mut normalizer = test_normalizer();
        normalizer.start_notifications(false).unwrap();
        apply_mapped(
            &mut normalizer,
            mapper.map_update(&json!({
                "sessionUpdate": "tool_call",
                "toolCallId": "call_nonzero",
                "kind": "execute",
                "rawInput": {"command": "sh -c 'echo FAIL-MARKER; exit 7'"}
            })),
        );
        let notifications = apply_mapped(
            &mut normalizer,
            mapper.map_update(&json!({
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_nonzero",
                "status": "completed",
                "content": [{"type": "terminal", "terminalId": "term-nonzero"}]
            })),
        );
        let item = completed_command_item(&notifications);
        assert_eq!(item["status"], "failed");
        assert_eq!(item["exitCode"], 7);
        assert_eq!(item["aggregatedOutput"], "FAIL-MARKER\n");
    }

    #[test]
    fn execute_failed_update_is_failed_even_with_exit_zero() {
        let mut terminals = HashMap::new();
        terminals.insert(
            "term-failed".to_string(),
            TerminalSnapshot {
                output: "ok\n".to_string(),
                truncated: false,
                exit_code: Some(0),
                signal: None,
            },
        );
        let mut mapper = AcpMapper::with_terminals(terminals);
        let mut normalizer = test_normalizer();
        normalizer.start_notifications(false).unwrap();
        apply_mapped(
            &mut normalizer,
            mapper.map_update(&json!({
                "sessionUpdate": "tool_call",
                "toolCallId": "call_failed",
                "kind": "execute",
                "rawInput": {"command": "echo ok"}
            })),
        );
        let notifications = apply_mapped(
            &mut normalizer,
            mapper.map_update(&json!({
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_failed",
                "status": "failed",
                "content": [{"type": "terminal", "terminalId": "term-failed"}]
            })),
        );
        let item = completed_command_item(&notifications);
        assert_eq!(item["status"], "failed");
        assert_eq!(item["exitCode"], 0);
        assert_eq!(item["aggregatedOutput"], "ok\n");
    }

    #[test]
    fn non_execute_tool_call_becomes_dynamic_tool_call() {
        let mut mapper = AcpMapper::new();
        let mut normalizer = test_normalizer();
        normalizer.start_notifications(false).unwrap();

        let mut notifications = Vec::new();
        for update in session_updates("plan.ndjson") {
            if let Some("tool_call" | "tool_call_update") = update["sessionUpdate"].as_str() {
                notifications.extend(apply_mapped(&mut normalizer, mapper.map_update(&update)));
            }
        }

        let completed: Vec<Value> = notifications
            .iter()
            .filter_map(|notification| {
                let rpc = notification_to_jsonrpc(notification).unwrap();
                if rpc.method != "item/completed" {
                    return None;
                }
                Some(rpc.params.unwrap()["item"].clone())
            })
            .collect();

        let read = completed
            .iter()
            .find(|item| item["tool"] == "read")
            .expect("read tool");
        assert_eq!(read["type"], "dynamicToolCall");
        assert_eq!(read["status"], "completed");
        assert_eq!(read["success"], true);
        assert_eq!(read["arguments"]["file_path"], "/workspace/note.txt");
        assert_eq!(read["contentItems"][0]["text"], "ALPHA-NOTE");

        let search = completed
            .iter()
            .find(|item| item["tool"] == "search")
            .expect("search tool");
        assert_eq!(search["type"], "dynamicToolCall");
        assert_eq!(search["success"], true);
        assert!(
            search["contentItems"][0]["text"]
                .as_str()
                .unwrap()
                .contains("ALPHA-NOTE")
        );

        let other = completed
            .iter()
            .find(|item| item["tool"] == "other")
            .expect("other tool");
        assert_eq!(other["type"], "dynamicToolCall");
        assert_eq!(other["success"], true);
        assert!(
            other["arguments"]["input"]
                .as_str()
                .unwrap()
                .contains("result.txt")
        );
    }

    #[test]
    fn execute_falls_back_to_title_when_command_missing() {
        let mut mapper = AcpMapper::new();
        let mapped = mapper.map_update(&json!({
            "sessionUpdate": "tool_call",
            "toolCallId": "call_title",
            "title": "echo TITLE-FALLBACK",
            "kind": "execute",
            "rawInput": {"summary": "no command field"}
        }));
        match &mapped.events[..] {
            [NormalizedEvent::AssistantMessage { content: items, .. }] => match &items[..] {
                [
                    NormalizedContent::ToolUse {
                        tool, arguments, ..
                    },
                ] => {
                    assert_eq!(tool, "shell_command");
                    assert_eq!(arguments["command"], "echo TITLE-FALLBACK");
                }
                other => panic!("expected ToolUse, got {other:?}"),
            },
            other => panic!("expected one assistant message, got {other:?}"),
        }
    }
}
