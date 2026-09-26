use std::env;
use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use crate::Result;
use crate::acp::AcpAgentProfile;

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
