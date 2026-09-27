use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use crate::acp::AcpAgentProfile;
use crate::{HarnessServerError, Result};

const CUSTOM_MODELS_ENV: &str = "DROID_CUSTOM_MODELS";
const ALLOWED_PROVIDERS: &[&str] = &["anthropic", "openai", "generic-chat-completion-api"];

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

pub fn settings_document(
    model: Option<&str>,
    reasoning: Option<&str>,
    custom_models: Option<Value>,
) -> Value {
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
    let mut document = Map::new();
    document.insert("sessionDefaultSettings".to_string(), Value::Object(session));
    document.insert("cloudSessionSync".to_string(), json!(false));
    document.insert("enableWarmup".to_string(), json!(false));
    document.insert("enableCompletionBell".to_string(), json!(false));
    if let Some(custom_models) = custom_models {
        document.insert("customModels".to_string(), custom_models);
    }
    Value::Object(document)
}

pub(super) fn custom_models_from_env() -> Result<Option<Value>> {
    match env_opt(CUSTOM_MODELS_ENV) {
        Some(raw) => parse_custom_models(&raw).map(Some),
        None => Ok(None),
    }
}

pub(super) fn parse_custom_models(raw: &str) -> Result<Value> {
    let value: Value = serde_json::from_str(raw)
        .map_err(|_| custom_models_error(String::from("value is not valid JSON")))?;
    let Some(entries) = value.as_array() else {
        return Err(custom_models_error(String::from("value is not an array")));
    };
    for (index, entry) in entries.iter().enumerate() {
        let Some(object) = entry.as_object() else {
            return Err(custom_models_error(format!(
                "entry {index} is not an object"
            )));
        };
        nonempty_object_string(object, index, "model")?;
        nonempty_object_string(object, index, "baseUrl")?;
        let provider = nonempty_object_string(object, index, "provider")?;
        if !ALLOWED_PROVIDERS.contains(&provider) {
            return Err(custom_models_error(format!(
                "entry {index} field provider is not supported"
            )));
        }
        if object.contains_key("apiKeyHelper") {
            return Err(custom_models_error(format!(
                "entry {index} field apiKeyHelper is not allowed"
            )));
        }
        if let Some(api_key) = object.get("apiKey")
            && !api_key.as_str().is_some_and(is_env_ref)
        {
            return Err(custom_models_error(format!(
                "entry {index} field apiKey is not an environment reference"
            )));
        }
    }
    Ok(value)
}

fn nonempty_object_string<'a>(
    object: &'a Map<String, Value>,
    index: usize,
    field: &'static str,
) -> Result<&'a str> {
    match object.get(field) {
        None => Err(custom_models_error(format!(
            "entry {index} field {field} is missing"
        ))),
        Some(Value::String(value)) => {
            if value.trim().is_empty() {
                Err(custom_models_error(format!(
                    "entry {index} field {field} is empty"
                )))
            } else {
                Ok(value.as_str())
            }
        }
        Some(_) => Err(custom_models_error(format!(
            "entry {index} field {field} is not a string"
        ))),
    }
}

fn is_env_ref(value: &str) -> bool {
    let Some(name) = value
        .strip_prefix("${")
        .and_then(|rest| rest.strip_suffix('}'))
    else {
        return false;
    };
    let mut chars = name.chars();
    match chars.next() {
        Some(first) if first.is_ascii_uppercase() => {
            chars.all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
        }
        _ => false,
    }
}

fn custom_models_error(reason: String) -> HarnessServerError {
    HarnessServerError::InvalidDroidCustomModels { reason }
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

pub(super) fn env_opt(name: &str) -> Option<String> {
    nonempty(env::var(name).ok().as_deref()).map(str::to_owned)
}

pub(super) fn nonempty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::{
        droid_argv, droid_profile, parse_custom_models, settings_document, write_settings_file,
    };
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;
    use uuid::Uuid;

    const CLIPROXY_ENTRY: &str = r#"[{
        "model": "grok-4.6",
        "id": "custom:cliproxy-grok-4.6",
        "baseUrl": "https://llm.example.com/v1",
        "apiKey": "${CLIPROXY_API_KEY}",
        "provider": "openai"
    }]"#;

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
        let value = settings_document(None, None, None);
        let session = &value["sessionDefaultSettings"];
        assert_eq!(session["autonomyLevel"], "high");
        assert_eq!(session["autonomyMode"], "auto-high");
        assert_eq!(session["interactionMode"], "auto");
        assert!(session.get("model").is_none());
        assert!(session.get("reasoningEffort").is_none());
        assert!(value.get("customModels").is_none());
        assert_eq!(value["cloudSessionSync"], false);
        assert_eq!(value["enableWarmup"], false);
        assert_eq!(value["enableCompletionBell"], false);
    }

    #[test]
    fn settings_includes_model_and_reasoning_when_set() {
        let value = settings_document(Some("gpt-5.4-mini-fast"), Some("low"), None);
        let session = &value["sessionDefaultSettings"];
        assert_eq!(session["model"], "gpt-5.4-mini-fast");
        assert_eq!(session["reasoningEffort"], "low");
        assert!(value.get("customModels").is_none());
    }

    #[test]
    fn settings_includes_custom_models_when_set() {
        let models = parse_custom_models(CLIPROXY_ENTRY).expect("valid custom models");
        let value = settings_document(None, None, Some(models.clone()));
        assert_eq!(value["customModels"], models);
        assert_eq!(value["customModels"][0]["apiKey"], "${CLIPROXY_API_KEY}");
        assert_eq!(value["customModels"][0]["id"], "custom:cliproxy-grok-4.6");
    }

    #[test]
    fn custom_models_accept_cliproxy_env_ref() {
        let models = parse_custom_models(CLIPROXY_ENTRY).expect("valid custom models");
        assert_eq!(models[0]["provider"], "openai");
        assert_eq!(models[0]["apiKey"], "${CLIPROXY_API_KEY}");
    }

    #[test]
    fn custom_models_preserve_anthropic_auth_mode() {
        let models = parse_custom_models(
            r#"[{"model":"claude","baseUrl":"https://llm.example.com","provider":"anthropic","authMode":"bearer","apiKey":"${CLIPROXY_API_KEY}"}]"#,
        )
        .expect("valid custom models");
        assert_eq!(models[0]["authMode"], "bearer");
        assert_eq!(models[0]["provider"], "anthropic");
    }

    #[test]
    fn custom_models_reject_literal_api_key() {
        let err = parse_custom_models(
            r#"[{"model":"m","baseUrl":"https://llm.example.com","provider":"openai","apiKey":"sk-literal"}]"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("entry 0"));
        assert!(err.contains("apiKey"));
        assert!(!err.contains("sk-literal"));
    }

    #[test]
    fn custom_models_reject_api_key_helper() {
        let err = parse_custom_models(
            r#"[{"model":"m","baseUrl":"https://llm.example.com","provider":"openai","apiKeyHelper":"echo secret"}]"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("entry 0"));
        assert!(err.contains("apiKeyHelper"));
        assert!(!err.contains("echo secret"));
    }

    #[test]
    fn custom_models_reject_non_array() {
        let err = parse_custom_models(r#"{"model":"m"}"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("not an array"));
        assert!(!err.contains("\"model\""));
    }

    #[test]
    fn custom_models_reject_unsupported_provider() {
        let err = parse_custom_models(
            r#"[{"model":"m","baseUrl":"https://llm.example.com","provider":"azure"}]"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("entry 0"));
        assert!(err.contains("provider"));
        assert!(!err.contains("azure"));
    }

    #[test]
    fn settings_file_mode_is_0600() {
        let path =
            std::env::temp_dir().join(format!("droid-settings-test-{}.json", Uuid::new_v4()));
        write_settings_file(&path, &settings_document(None, None, None)).expect("write");
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
