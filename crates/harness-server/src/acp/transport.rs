//! NDJSON JSON-RPC 2.0 framing over a child agent's stdin/stdout.
//!
//! Outbound requests use our own id counter and an outbound-id map. Incoming
//! agent requests are delivered on a separate channel so colliding numeric ids
//! never cross. Id `0` is a real request, never a notification.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use agent_client_protocol_schema::v1::{Error, JsonRpcMessage, Notification, Request, RequestId};
use serde::Serialize;
use serde_json::Value;

use super::{AcpError, Result};

/// A parsed JSON-RPC 2.0 line.
#[derive(Debug, Clone, PartialEq)]
pub enum ClassifiedMessage {
    Request {
        id: RequestId,
        method: String,
        params: Value,
    },
    Response {
        id: RequestId,
        result: std::result::Result<Value, Error>,
    },
    Notification {
        method: String,
        params: Value,
    },
}

/// Incoming agent → client messages that are not responses to our requests.
#[derive(Debug, Clone, PartialEq)]
pub enum IncomingMessage {
    Request {
        id: RequestId,
        method: String,
        params: Value,
    },
    Notification {
        method: String,
        params: Value,
    },
}

struct Shared {
    writer: Mutex<Box<dyn Write + Send>>,
    next_outbound_id: AtomicI64,
    outbound: Mutex<HashMap<RequestId, Sender<std::result::Result<Value, Error>>>>,
}

/// JSON-RPC NDJSON transport with a mutex-guarded writer and a reader thread.
pub struct AcpTransport {
    shared: Arc<Shared>,
    incoming: Receiver<IncomingMessage>,
}

impl AcpTransport {
    /// Wrap already-piped child stdio (or any equivalent pair of streams).
    pub fn from_rw(
        writer: impl Write + Send + 'static,
        reader: impl Read + Send + 'static,
    ) -> Self {
        let shared = Arc::new(Shared {
            writer: Mutex::new(Box::new(writer)),
            next_outbound_id: AtomicI64::new(0),
            outbound: Mutex::new(HashMap::new()),
        });
        let (incoming_tx, incoming_rx) = mpsc::channel();
        spawn_reader(Arc::clone(&shared), incoming_tx, reader);
        Self {
            shared,
            incoming: incoming_rx,
        }
    }

    pub fn incoming(&self) -> &Receiver<IncomingMessage> {
        &self.incoming
    }

    pub fn into_parts(self) -> (SharedHandle, Receiver<IncomingMessage>) {
        (
            SharedHandle {
                shared: self.shared,
            },
            self.incoming,
        )
    }

    pub fn handle(&self) -> SharedHandle {
        SharedHandle {
            shared: Arc::clone(&self.shared),
        }
    }

    pub fn send_request(
        &self,
        method: &str,
        params: Value,
    ) -> Result<(RequestId, Receiver<std::result::Result<Value, Error>>)> {
        self.handle().send_request(method, params)
    }

    pub fn send_notification(&self, method: &str, params: Value) -> Result<()> {
        self.handle().send_notification(method, params)
    }

    pub fn send_response(
        &self,
        id: RequestId,
        result: std::result::Result<Value, Error>,
    ) -> Result<()> {
        self.handle().send_response(id, result)
    }

    pub fn request(&self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        self.handle().request(method, params, timeout)
    }
}

/// Cloneable write/request handle used by dispatcher worker threads.
#[derive(Clone)]
pub struct SharedHandle {
    shared: Arc<Shared>,
}

impl SharedHandle {
    pub fn send_request(
        &self,
        method: &str,
        params: Value,
    ) -> Result<(RequestId, Receiver<std::result::Result<Value, Error>>)> {
        let id = RequestId::Number(self.shared.next_outbound_id.fetch_add(1, Ordering::Relaxed));
        let (tx, rx) = mpsc::channel();
        self.shared
            .outbound
            .lock()
            .expect("acp outbound map")
            .insert(id.clone(), tx);
        let request = JsonRpcMessage::wrap(Request {
            id: id.clone(),
            method: method.into(),
            params: Some(params),
        });
        match self.write_message(&request) {
            Ok(()) => Ok((id.clone(), rx)),
            Err(error) => {
                self.shared
                    .outbound
                    .lock()
                    .expect("acp outbound map")
                    .remove(&id);
                Err(error)
            }
        }
    }

    pub fn send_notification(&self, method: &str, params: Value) -> Result<()> {
        let notification = JsonRpcMessage::wrap(Notification {
            method: method.into(),
            params: Some(params),
        });
        self.write_message(&notification)
    }

    pub fn send_response(
        &self,
        id: RequestId,
        result: std::result::Result<Value, Error>,
    ) -> Result<()> {
        let response =
            JsonRpcMessage::wrap(agent_client_protocol_schema::v1::Response::new(id, result));
        self.write_message(&response)
    }

    pub fn request(&self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        let (_id, rx) = self.send_request(method, params)?;
        match rx.recv_timeout(timeout) {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(error)) => Err(error.into()),
            Err(RecvTimeoutError::Timeout) => Err(AcpError::Timeout {
                method: method.to_string(),
            }),
            Err(RecvTimeoutError::Disconnected) => Err(AcpError::TransportClosed),
        }
    }

    fn write_message<T: Serialize>(&self, message: &T) -> Result<()> {
        let mut writer = self.shared.writer.lock().expect("acp writer");
        serde_json::to_writer(&mut *writer, message)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        Ok(())
    }
}

fn spawn_reader(
    shared: Arc<Shared>,
    incoming_tx: Sender<IncomingMessage>,
    reader: impl Read + Send + 'static,
) {
    thread::spawn(move || {
        let reader = BufReader::new(reader);
        for raw in reader.lines() {
            let line = match raw {
                Ok(line) => line,
                Err(_) => break,
            };
            match classify_line(&line) {
                Some(ClassifiedMessage::Request { id, method, params }) => {
                    if incoming_tx
                        .send(IncomingMessage::Request { id, method, params })
                        .is_err()
                    {
                        break;
                    }
                }
                Some(ClassifiedMessage::Notification { method, params }) => {
                    if incoming_tx
                        .send(IncomingMessage::Notification { method, params })
                        .is_err()
                    {
                        break;
                    }
                }
                Some(ClassifiedMessage::Response { id, result }) => {
                    let waiter = shared
                        .outbound
                        .lock()
                        .expect("acp outbound map")
                        .remove(&id);
                    if let Some(waiter) = waiter {
                        let _ = waiter.send(result);
                    }
                }
                None => {
                    if !line.trim().is_empty() {
                        let preview: String = line.chars().take(200).collect();
                        eprintln!("acp transport: skipping unparseable line: {preview}");
                    }
                }
            }
        }
    });
}

/// Classify one NDJSON line. Returns `None` for empty or unparseable input.
pub fn classify_line(line: &str) -> Option<ClassifiedMessage> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let value: Value = serde_json::from_str(trimmed).ok()?;
    let object = value.as_object()?;
    let method = object
        .get("method")
        .and_then(Value::as_str)
        .map(str::to_string);
    let params = object.get("params").cloned().unwrap_or(Value::Null);
    let id_value = object.get("id").cloned();
    let has_id = id_value.is_some();
    let has_result = object.contains_key("result");
    let has_error = object.contains_key("error");

    if let Some(method) = method {
        if has_id {
            let id = serde_json::from_value(id_value?).ok()?;
            return Some(ClassifiedMessage::Request { id, method, params });
        }
        return Some(ClassifiedMessage::Notification { method, params });
    }

    if has_id && (has_result || has_error) {
        let id = serde_json::from_value(id_value?).ok()?;
        if has_error {
            let error = serde_json::from_value(object.get("error")?.clone()).ok()?;
            return Some(ClassifiedMessage::Response {
                id,
                result: Err(error),
            });
        }
        return Some(ClassifiedMessage::Response {
            id,
            result: Ok(object.get("result").cloned().unwrap_or(Value::Null)),
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;
    use std::time::Duration;

    use agent_client_protocol_schema::v1::RequestId;
    use serde_json::json;

    use super::{AcpTransport, ClassifiedMessage, IncomingMessage, classify_line};

    struct AgentEnd {
        writer: UnixStream,
        reader: BufReader<UnixStream>,
    }

    fn pair() -> (AcpTransport, AgentEnd) {
        let (client, agent) = UnixStream::pair().expect("unix pair");
        agent
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("read timeout");
        let reader = agent.try_clone().expect("clone agent");
        let writer = client.try_clone().expect("clone");
        (
            AcpTransport::from_rw(writer, client),
            AgentEnd {
                writer: agent,
                reader: BufReader::new(reader),
            },
        )
    }

    fn write_line(agent: &mut AgentEnd, value: &serde_json::Value) {
        serde_json::to_writer(&mut agent.writer, value).expect("write");
        agent.writer.write_all(b"\n").expect("newline");
        agent.writer.flush().expect("flush");
    }

    fn read_line(agent: &mut AgentEnd) -> serde_json::Value {
        let mut line = String::new();
        agent.reader.read_line(&mut line).expect("read");
        serde_json::from_str(line.trim()).expect("json")
    }

    #[test]
    fn classify_request_response_notification() {
        let request = classify_line(
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1}}"#,
        );
        assert!(matches!(
            request,
            Some(ClassifiedMessage::Request { id: RequestId::Number(1), method, .. })
                if method == "initialize"
        ));

        let notification = classify_line(
            r#"{"jsonrpc":"2.0","method":"session/cancel","params":{"sessionId":"s"}}"#,
        );
        assert!(matches!(
            notification,
            Some(ClassifiedMessage::Notification { method, .. }) if method == "session/cancel"
        ));

        let response = classify_line(r#"{"jsonrpc":"2.0","id":1,"result":{"ok":true}}"#);
        assert!(matches!(
            response,
            Some(ClassifiedMessage::Response { id: RequestId::Number(1), result: Ok(value) })
                if value == json!({"ok": true})
        ));
    }

    #[test]
    fn incoming_request_id_zero_is_a_request() {
        let classified = classify_line(
            r#"{"jsonrpc":"2.0","id":0,"method":"session/request_permission","params":{}}"#,
        );
        assert!(matches!(
            classified,
            Some(ClassifiedMessage::Request { id: RequestId::Number(0), method, .. })
                if method == "session/request_permission"
        ));
    }

    #[test]
    fn unknown_lines_are_skipped() {
        assert!(classify_line("").is_none());
        assert!(classify_line("   ").is_none());
        assert!(classify_line("this is not json").is_none());
        assert!(classify_line(r#"{"foo":1}"#).is_none());
        assert!(classify_line("{").is_none());
    }

    #[test]
    fn outbound_ids_start_at_zero() {
        let (transport, mut agent) = pair();
        let (id, _rx) = transport
            .send_request("initialize", json!({"protocolVersion": 1}))
            .expect("send");
        assert_eq!(id, RequestId::Number(0));
        let frame = read_line(&mut agent);
        assert_eq!(frame["id"], json!(0));
        assert_eq!(frame["method"], json!("initialize"));
        assert_eq!(frame["jsonrpc"], json!("2.0"));
    }

    #[test]
    fn colliding_agent_and_client_ids_are_not_confused() {
        let (transport, mut agent) = pair();
        let (outbound_id, outbound_rx) = transport
            .send_request("session/prompt", json!({"sessionId": "s"}))
            .expect("send");
        assert_eq!(outbound_id, RequestId::Number(0));
        let _sent = read_line(&mut agent);

        write_line(
            &mut agent,
            &json!({
                "jsonrpc": "2.0",
                "id": 0,
                "method": "session/request_permission",
                "params": {"sessionId": "s", "options": []}
            }),
        );
        let incoming = transport
            .incoming()
            .recv_timeout(Duration::from_secs(2))
            .expect("incoming request");
        match incoming {
            IncomingMessage::Request { id, method, .. } => {
                assert_eq!(id, RequestId::Number(0));
                assert_eq!(method, "session/request_permission");
            }
            other => panic!("expected incoming request, got {other:?}"),
        }
        assert!(
            outbound_rx.recv_timeout(Duration::from_millis(50)).is_err(),
            "inbound request must not complete the outbound waiter"
        );

        write_line(
            &mut agent,
            &json!({
                "jsonrpc": "2.0",
                "id": 0,
                "result": {"stopReason": "end_turn"}
            }),
        );
        let result = outbound_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("outbound result")
            .expect("ok");
        assert_eq!(result, json!({"stopReason": "end_turn"}));
    }

    #[test]
    fn unparseable_lines_are_skipped_and_stream_continues() {
        let (transport, mut agent) = pair();
        agent
            .writer
            .write_all(b"not-json\n{\"foo\":1}\n")
            .expect("garbage");
        write_line(
            &mut agent,
            &json!({"jsonrpc":"2.0","method":"session/update","params":{"sessionId":"s"}}),
        );
        let incoming = transport
            .incoming()
            .recv_timeout(Duration::from_secs(2))
            .expect("notification after garbage");
        assert!(matches!(
            incoming,
            IncomingMessage::Notification { method, .. } if method == "session/update"
        ));
    }

    #[test]
    fn writer_serializes_notifications_without_id() {
        let (transport, mut agent) = pair();
        transport
            .send_notification("session/cancel", json!({"sessionId": "s"}))
            .expect("notify");
        let frame = read_line(&mut agent);
        assert_eq!(frame["method"], json!("session/cancel"));
        assert!(frame.get("id").is_none());
        assert_eq!(frame["jsonrpc"], json!("2.0"));
    }
}
