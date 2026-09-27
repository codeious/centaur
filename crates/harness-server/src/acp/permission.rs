//! Auto-approve `session/request_permission`.

use agent_client_protocol_schema::v1::{
    PermissionOption, PermissionOptionKind, RequestPermissionOutcome, RequestPermissionResponse,
    SelectedPermissionOutcome,
};

/// Prefer `allow_always`, else `allow_once`.
pub fn select_permission_option(options: &[PermissionOption]) -> Option<&PermissionOption> {
    options
        .iter()
        .find(|option| option.kind == PermissionOptionKind::AllowAlways)
        .or_else(|| {
            options
                .iter()
                .find(|option| option.kind == PermissionOptionKind::AllowOnce)
        })
}

/// Build the JSON-RPC result for a permission request.
///
/// When the prompt was cancelled while this request was pending, ACP requires
/// `{outcome:{outcome:"cancelled"}}` instead of a selected option.
pub fn permission_response(
    options: &[PermissionOption],
    cancelled: bool,
) -> RequestPermissionResponse {
    if cancelled {
        return RequestPermissionResponse::new(RequestPermissionOutcome::Cancelled);
    }
    match select_permission_option(options) {
        Some(option) => RequestPermissionResponse::new(RequestPermissionOutcome::Selected(
            SelectedPermissionOutcome::new(option.option_id.clone()),
        )),
        None => RequestPermissionResponse::new(RequestPermissionOutcome::Cancelled),
    }
}

#[cfg(test)]
mod tests {
    use agent_client_protocol_schema::v1::{
        PermissionOption, PermissionOptionKind, RequestPermissionOutcome,
    };

    use super::{permission_response, select_permission_option};

    fn option(id: &str, kind: PermissionOptionKind) -> PermissionOption {
        PermissionOption::new(id.to_string(), id.to_string(), kind)
    }

    #[test]
    fn prefers_allow_always_over_allow_once() {
        let options = vec![
            option("proceed_once", PermissionOptionKind::AllowOnce),
            option("proceed_always", PermissionOptionKind::AllowAlways),
            option("cancel", PermissionOptionKind::RejectOnce),
        ];
        let selected = select_permission_option(&options).expect("option");
        assert_eq!(selected.option_id.0.as_ref(), "proceed_always");

        let response = permission_response(&options, false);
        match response.outcome {
            RequestPermissionOutcome::Selected(selected) => {
                assert_eq!(selected.option_id.0.as_ref(), "proceed_always");
            }
            other => panic!("expected selected, got {other:?}"),
        }
    }

    #[test]
    fn falls_back_to_allow_once() {
        let options = vec![
            option("cancel", PermissionOptionKind::RejectOnce),
            option("proceed_once", PermissionOptionKind::AllowOnce),
        ];
        let selected = select_permission_option(&options).expect("option");
        assert_eq!(selected.option_id.0.as_ref(), "proceed_once");
    }

    #[test]
    fn cancelled_prompt_returns_cancelled_outcome() {
        let options = vec![
            option("proceed_always", PermissionOptionKind::AllowAlways),
            option("proceed_once", PermissionOptionKind::AllowOnce),
        ];
        let response = permission_response(&options, true);
        assert!(matches!(
            response.outcome,
            RequestPermissionOutcome::Cancelled
        ));
        let encoded = serde_json::to_value(&response).expect("serialize");
        assert_eq!(
            encoded
                .get("outcome")
                .and_then(|value| value.get("outcome")),
            Some(&serde_json::Value::String("cancelled".into()))
        );
    }

    #[test]
    fn selected_outcome_wire_shape() {
        let options = vec![option("proceed_always", PermissionOptionKind::AllowAlways)];
        let encoded =
            serde_json::to_value(permission_response(&options, false)).expect("serialize");
        assert_eq!(
            encoded,
            serde_json::json!({
                "outcome": {"outcome": "selected", "optionId": "proceed_always"}
            })
        );
    }
}
