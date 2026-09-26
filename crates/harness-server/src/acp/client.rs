//! Typed ACP client operations and incoming-request dispatch.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use agent_client_protocol_schema::ProtocolVersion;
use agent_client_protocol_schema::v1::{
    AGENT_METHOD_NAMES, CLIENT_METHOD_NAMES, CancelNotification, ClientCapabilities, ContentBlock,
    CreateTerminalRequest, Error, FileSystemCapabilities, Implementation, InitializeRequest,
    InitializeResponse, KillTerminalRequest, LoadSessionRequest, LoadSessionResponse,
    NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse, ReleaseTerminalRequest,
    RequestId, RequestPermissionRequest, SessionConfigId, SessionConfigValueId, SessionId,
    SetSessionConfigOptionRequest, SetSessionConfigOptionResponse, TerminalOutputRequest,
    WaitForTerminalExitRequest,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use super::permission::permission_response;
use super::terminal::TerminalManager;
use super::transport::{AcpTransport, IncomingMessage, SharedHandle};
use super::{AcpError, Result};

/// Handshake + RPC budget, matching the Hermes gateway timeout.
const RPC_TIMEOUT: Duration = Duration::from_secs(180);

/// Agent-specific launch and display settings. No Droid constants live here;
/// the Droid runtime fills this in.
#[derive(Debug, Clone)]
pub struct AcpAgentProfile {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub extra_env: Vec<(String, String)>,
    pub model_config_id: String,
    pub reasoning_config_id: String,
    pub cli_version: String,
    pub model_provider: String,
}

/// Generic ACP client: typed outbound methods plus client-side permission and
/// terminal handlers. Incoming requests run on worker threads so a long
/// `terminal/wait_for_exit` cannot stall the reader.
pub struct AcpClient {
    handle: SharedHandle,
    terminals: Arc<TerminalManager>,
    cancelled: Arc<AtomicBool>,
    notifications: Mutex<Receiver<(String, Value)>>,
}

impl AcpClient {
    pub fn from_rw(
        writer: impl Write + Send + 'static,
        reader: impl Read + Send + 'static,
        session_cwd: PathBuf,
    ) -> Self {
        Self::from_transport(AcpTransport::from_rw(writer, reader), session_cwd)
    }

    pub fn from_transport(transport: AcpTransport, session_cwd: PathBuf) -> Self {
        let (handle, incoming) = transport.into_parts();
        let terminals = Arc::new(TerminalManager::new(session_cwd));
        let cancelled = Arc::new(AtomicBool::new(false));
        let (notification_tx, notification_rx) = std::sync::mpsc::channel();
        let dispatch_handle = handle.clone();
        let dispatch_terminals = Arc::clone(&terminals);
        let dispatch_cancelled = Arc::clone(&cancelled);
        thread::spawn(move || {
            for message in incoming {
                match message {
                    IncomingMessage::Notification { method, params } => {
                        if notification_tx.send((method, params)).is_err() {
                            break;
                        }
                    }
                    IncomingMessage::Request { id, method, params } => {
                        let handle = dispatch_handle.clone();
                        let terminals = Arc::clone(&dispatch_terminals);
                        let cancelled = Arc::clone(&dispatch_cancelled);
                        thread::spawn(move || {
                            dispatch_request(&handle, &terminals, &cancelled, id, method, params);
                        });
                    }
                }
            }
        });
        Self {
            handle,
            terminals,
            cancelled,
            notifications: Mutex::new(notification_rx),
        }
    }

    pub fn terminals(&self) -> &Arc<TerminalManager> {
        &self.terminals
    }

    pub fn try_recv_notification(&self) -> Option<(String, Value)> {
        self.notifications
            .lock()
            .expect("acp notifications")
            .try_recv()
            .ok()
    }

    pub fn recv_notification_timeout(
        &self,
        timeout: Duration,
    ) -> std::result::Result<(String, Value), RecvTimeoutError> {
        self.notifications
            .lock()
            .expect("acp notifications")
            .recv_timeout(timeout)
    }

    pub fn initialize(&self) -> Result<InitializeResponse> {
        let capabilities =
            ClientCapabilities::new()
                .terminal(true)
                .fs(FileSystemCapabilities::new()
                    .read_text_file(false)
                    .write_text_file(false));
        let request = InitializeRequest::new(ProtocolVersion::V1)
            .client_capabilities(capabilities)
            .client_info(
                Implementation::new("centaur-harness", env!("CARGO_PKG_VERSION")).title("Centaur"),
            );
        self.rpc(AGENT_METHOD_NAMES.initialize, request)
    }

    pub fn session_new(&self, cwd: impl Into<PathBuf>) -> Result<NewSessionResponse> {
        let cwd = cwd.into();
        self.terminals.set_session_cwd(cwd.clone());
        self.rpc(
            AGENT_METHOD_NAMES.session_new,
            NewSessionRequest::new(cwd).mcp_servers(Vec::new()),
        )
    }

    pub fn session_load(
        &self,
        session_id: impl Into<SessionId>,
        cwd: impl Into<PathBuf>,
    ) -> Result<LoadSessionResponse> {
        let cwd = cwd.into();
        self.terminals.set_session_cwd(cwd.clone());
        self.rpc(
            AGENT_METHOD_NAMES.session_load,
            LoadSessionRequest::new(session_id, cwd).mcp_servers(Vec::new()),
        )
    }

    pub fn session_set_config_option(
        &self,
        session_id: impl Into<SessionId>,
        config_id: impl Into<SessionConfigId>,
        value: impl Into<SessionConfigValueId>,
    ) -> Result<SetSessionConfigOptionResponse> {
        let request = SetSessionConfigOptionRequest::new(
            session_id,
            config_id,
            agent_client_protocol_schema::v1::SessionConfigOptionValue::value_id(value),
        );
        self.rpc(AGENT_METHOD_NAMES.session_set_config_option, request)
    }

    pub fn session_prompt(
        &self,
        session_id: impl Into<SessionId>,
        prompt: Vec<ContentBlock>,
    ) -> Result<PromptResponse> {
        self.cancelled.store(false, Ordering::SeqCst);
        self.rpc(
            AGENT_METHOD_NAMES.session_prompt,
            PromptRequest::new(session_id, prompt),
        )
    }

    pub fn session_cancel(&self, session_id: impl Into<SessionId>) -> Result<()> {
        self.cancelled.store(true, Ordering::SeqCst);
        let params = serde_json::to_value(CancelNotification::new(session_id))?;
        self.handle
            .send_notification(AGENT_METHOD_NAMES.session_cancel, params)
    }

    fn rpc<P: Serialize, R: DeserializeOwned>(&self, method: &str, params: P) -> Result<R> {
        let params = serde_json::to_value(params)?;
        let result = self.handle.request(method, params, RPC_TIMEOUT)?;
        Ok(serde_json::from_value(result)?)
    }
}

impl Drop for AcpClient {
    fn drop(&mut self) {
        self.terminals.kill_all();
    }
}

fn dispatch_request(
    handle: &SharedHandle,
    terminals: &TerminalManager,
    cancelled: &AtomicBool,
    id: RequestId,
    method: String,
    params: Value,
) {
    let result = handle_request(terminals, cancelled, &method, params);
    let _ = handle.send_response(id, result);
}

fn handle_request(
    terminals: &TerminalManager,
    cancelled: &AtomicBool,
    method: &str,
    params: Value,
) -> std::result::Result<Value, Error> {
    if method == CLIENT_METHOD_NAMES.session_request_permission {
        let request: RequestPermissionRequest = serde_json::from_value(params)?;
        let response = permission_response(&request.options, cancelled.load(Ordering::SeqCst));
        return Ok(serde_json::to_value(response)?);
    }
    if method == CLIENT_METHOD_NAMES.terminal_create {
        let request: CreateTerminalRequest = serde_json::from_value(params)?;
        let response = terminals.create(&request).map_err(terminal_error)?;
        return Ok(serde_json::to_value(response)?);
    }
    if method == CLIENT_METHOD_NAMES.terminal_output {
        let request: TerminalOutputRequest = serde_json::from_value(params)?;
        let response = terminals
            .output(&request.terminal_id)
            .map_err(terminal_error)?;
        return Ok(serde_json::to_value(response)?);
    }
    if method == CLIENT_METHOD_NAMES.terminal_wait_for_exit {
        let request: WaitForTerminalExitRequest = serde_json::from_value(params)?;
        let response = terminals
            .wait_for_exit(&request.terminal_id)
            .map_err(terminal_error)?;
        return Ok(serde_json::to_value(response)?);
    }
    if method == CLIENT_METHOD_NAMES.terminal_kill {
        let request: KillTerminalRequest = serde_json::from_value(params)?;
        let response = terminals
            .kill(&request.terminal_id)
            .map_err(terminal_error)?;
        return Ok(serde_json::to_value(response)?);
    }
    if method == CLIENT_METHOD_NAMES.terminal_release {
        let request: ReleaseTerminalRequest = serde_json::from_value(params)?;
        let response = terminals
            .release(&request.terminal_id)
            .map_err(terminal_error)?;
        return Ok(serde_json::to_value(response)?);
    }
    Err(Error::method_not_found())
}

fn terminal_error(error: AcpError) -> Error {
    match error {
        AcpError::UnknownTerminal(id) => Error::invalid_params()
            .data(json!({"terminalId": id, "message": "unknown terminal id"})),
        AcpError::Json(error) => Error::invalid_params().data(error.to_string()),
        other => Error::internal_error().data(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;
    use std::time::{Duration, Instant};

    use agent_client_protocol_schema::v1::{ContentBlock, SessionId, TextContent};
    use serde_json::{Value, json};

    use super::AcpClient;

    struct AgentEnd {
        writer: UnixStream,
        reader: BufReader<UnixStream>,
    }

    fn pair() -> (AcpClient, AgentEnd) {
        let (client_stream, agent) = UnixStream::pair().expect("unix pair");
        agent
            .set_read_timeout(Some(Duration::from_secs(8)))
            .expect("timeout");
        let reader = agent.try_clone().expect("clone agent");
        let writer = client_stream.try_clone().expect("clone");
        let client =
            AcpClient::from_rw(writer, client_stream, std::env::current_dir().expect("cwd"));
        (
            client,
            AgentEnd {
                writer: agent,
                reader: BufReader::new(reader),
            },
        )
    }

    fn write_line(agent: &mut AgentEnd, value: &Value) {
        serde_json::to_writer(&mut agent.writer, value).expect("write");
        agent.writer.write_all(b"\n").expect("newline");
        agent.writer.flush().expect("flush");
    }

    fn read_line(agent: &mut AgentEnd) -> Value {
        let mut line = String::new();
        agent.reader.read_line(&mut line).expect("read");
        serde_json::from_str(line.trim()).expect("json")
    }

    #[test]
    fn initialize_advertises_terminal_and_not_fs() {
        let (client, mut agent) = pair();
        std::thread::scope(|scope| {
            let job = scope.spawn(|| client.initialize());
            let request = read_line(&mut agent);
            assert_eq!(request["method"], "initialize");
            assert_eq!(request["params"]["protocolVersion"], 1);
            assert_eq!(request["params"]["clientCapabilities"]["terminal"], true);
            assert_eq!(
                request["params"]["clientCapabilities"]["fs"]["readTextFile"],
                false
            );
            assert_eq!(
                request["params"]["clientCapabilities"]["fs"]["writeTextFile"],
                false
            );
            write_line(
                &mut agent,
                &json!({
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"protocolVersion": 1}
                }),
            );
            job.join().expect("join").expect("initialize");
        });
    }

    #[test]
    fn session_methods_and_cancel_notification() {
        let (client, mut agent) = pair();
        std::thread::scope(|scope| {
            let job = scope.spawn(|| {
                let cwd = std::env::current_dir().expect("cwd");
                client.session_new(&cwd)?;
                client.session_load(SessionId::new("sess-1"), &cwd)?;
                client.session_set_config_option(SessionId::new("sess-1"), "model", "gpt-test")?;
                client.session_prompt(
                    SessionId::new("sess-1"),
                    vec![ContentBlock::Text(TextContent::new("hi"))],
                )?;
                client.session_cancel(SessionId::new("sess-1"))
            });

            let new_req = read_line(&mut agent);
            assert_eq!(new_req["method"], "session/new");
            assert_eq!(new_req["params"]["mcpServers"], json!([]));
            assert!(new_req["params"]["cwd"].is_string());
            write_line(
                &mut agent,
                &json!({"jsonrpc":"2.0","id": new_req["id"], "result": {"sessionId": "sess-1"}}),
            );

            let load_req = read_line(&mut agent);
            assert_eq!(load_req["method"], "session/load");
            assert_eq!(load_req["params"]["sessionId"], "sess-1");
            assert_eq!(load_req["params"]["mcpServers"], json!([]));
            write_line(
                &mut agent,
                &json!({"jsonrpc":"2.0","id": load_req["id"], "result": {}}),
            );

            let config_req = read_line(&mut agent);
            assert_eq!(config_req["method"], "session/set_config_option");
            assert_eq!(config_req["params"]["sessionId"], "sess-1");
            assert_eq!(config_req["params"]["configId"], "model");
            assert_eq!(config_req["params"]["value"], "gpt-test");
            write_line(
                &mut agent,
                &json!({"jsonrpc":"2.0","id": config_req["id"], "result": {"configOptions": []}}),
            );

            let prompt_req = read_line(&mut agent);
            assert_eq!(prompt_req["method"], "session/prompt");
            write_line(
                &mut agent,
                &json!({"jsonrpc":"2.0","id": prompt_req["id"], "result": {"stopReason": "end_turn"}}),
            );

            let cancel = read_line(&mut agent);
            assert_eq!(cancel["method"], "session/cancel");
            assert!(cancel.get("id").is_none());
            assert_eq!(cancel["params"]["sessionId"], "sess-1");

            job.join().expect("join").expect("session methods");
        });
    }

    #[test]
    fn unknown_incoming_method_returns_32601() {
        let (_client, mut agent) = pair();
        write_line(
            &mut agent,
            &json!({
                "jsonrpc": "2.0",
                "id": 7,
                "method": "no/such/method",
                "params": {}
            }),
        );
        let response = read_line(&mut agent);
        assert_eq!(response["id"], 7);
        assert_eq!(response["error"]["code"], -32601);
    }

    #[test]
    fn incoming_request_id_zero_is_answered() {
        let (_client, mut agent) = pair();
        write_line(
            &mut agent,
            &json!({
                "jsonrpc": "2.0",
                "id": 0,
                "method": "session/request_permission",
                "params": {
                    "sessionId": "s",
                    "options": [
                        {"optionId": "proceed_once", "name": "Allow", "kind": "allow_once"},
                        {"optionId": "proceed_always", "name": "Always", "kind": "allow_always"},
                        {"optionId": "cancel", "name": "No", "kind": "reject_once"}
                    ],
                    "toolCall": {"toolCallId": "call_1"}
                }
            }),
        );
        let response = read_line(&mut agent);
        assert_eq!(response["id"], 0);
        assert_eq!(
            response["result"]["outcome"],
            json!({"outcome": "selected", "optionId": "proceed_always"})
        );
    }

    #[test]
    fn wait_for_exit_does_not_block_the_reader() {
        let (_client, mut agent) = pair();
        write_line(
            &mut agent,
            &json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "terminal/create",
                "params": {"sessionId": "s", "command": "sleep 2"}
            }),
        );
        let created = read_line(&mut agent);
        let terminal_id = created["result"]["terminalId"]
            .as_str()
            .expect("id")
            .to_string();

        write_line(
            &mut agent,
            &json!({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "terminal/wait_for_exit",
                "params": {"sessionId": "s", "terminalId": terminal_id}
            }),
        );

        let started = Instant::now();
        write_line(
            &mut agent,
            &json!({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/request_permission",
                "params": {
                    "sessionId": "s",
                    "options": [
                        {"optionId": "proceed_once", "name": "Allow", "kind": "allow_once"}
                    ],
                    "toolCall": {"toolCallId": "call_2"}
                }
            }),
        );
        let permission = read_line(&mut agent);
        assert!(
            started.elapsed() < Duration::from_millis(800),
            "reader was blocked by wait_for_exit: {:?}",
            started.elapsed()
        );
        assert_eq!(permission["id"], 3);
        assert_eq!(permission["result"]["outcome"]["optionId"], "proceed_once");
    }
}
