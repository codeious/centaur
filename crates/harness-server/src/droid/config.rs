use std::collections::{HashMap, HashSet};

use agent_client_protocol_schema::v1::{
    SessionConfigKind, SessionConfigOption, SessionConfigSelectOptions,
};
use serde_json::Value;

#[derive(Debug, Default, Clone)]
pub(super) struct ConfigCatalog {
    options: HashMap<String, CatalogOption>,
}

#[derive(Debug, Clone, Default)]
struct CatalogOption {
    current: Option<String>,
    values: HashSet<String>,
}

impl ConfigCatalog {
    pub(super) fn ingest_typed(&mut self, options: &[SessionConfigOption]) {
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

    pub(super) fn ingest_update(&mut self, method: &str, params: &Value) {
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

    pub(super) fn current(&self, id: &str) -> Option<&str> {
        self.options
            .get(id)
            .and_then(|option| option.current.as_deref())
    }

    pub(super) fn is_allowed(&self, id: &str, value: &str) -> bool {
        match self.options.get(id) {
            Some(option) if !option.values.is_empty() => option.values.contains(value),
            _ => true,
        }
    }

    pub(super) fn set_current(&mut self, id: &str, value: String) {
        self.options.entry(id.to_string()).or_default().current = Some(value);
    }
}

#[cfg(test)]
mod tests {
    use super::ConfigCatalog;
    use agent_client_protocol_schema::v1::SessionConfigOption;
    use serde_json::json;

    #[test]
    fn catalog_rejects_unadvertised_values_and_allows_current() {
        let mut catalog = ConfigCatalog::default();
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
