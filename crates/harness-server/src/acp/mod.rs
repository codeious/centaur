//! Generic Agent Client Protocol (ACP) client.
//!
//! This module is agent-agnostic: Droid-specific command lines, settings, and
//! env vars live in the Droid runtime, not here. Concurrency follows the rest
//! of harness-server (std threads + `mpsc`), with no async runtime.

mod client;
mod permission;
mod terminal;
mod transport;

pub use client::{AcpAgentProfile, AcpClient};
pub use permission::{permission_response, select_permission_option};
pub use terminal::{TerminalManager, TerminalSnapshot};
pub use transport::{AcpTransport, ClassifiedMessage, classify_line};

use agent_client_protocol_schema::v1::Error;
use serde_json::Value;
use thiserror::Error;

/// Errors from the generic ACP client, transport, or terminal manager.
#[derive(Debug, Error)]
pub enum AcpError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("ACP JSON-RPC error {code}: {message}")]
    Rpc {
        code: i32,
        message: String,
        data: Option<Value>,
    },
    #[error("ACP transport closed")]
    TransportClosed,
    #[error("timed out waiting for ACP response to {method}")]
    Timeout { method: String },
    #[error("unknown terminal {0}")]
    UnknownTerminal(String),
}

impl From<Error> for AcpError {
    fn from(error: Error) -> Self {
        Self::Rpc {
            code: error.code.into(),
            message: error.message,
            data: error.data,
        }
    }
}

pub type Result<T> = std::result::Result<T, AcpError>;
