use std::fs;
use std::path::Path;

use agent_client_protocol_schema::v1::{ContentBlock, ImageContent, TextContent};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use codex_app_server_protocol::UserInput;

use crate::Result;

pub(super) fn prompt_blocks(input: &[UserInput]) -> Result<Vec<ContentBlock>> {
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

#[cfg(test)]
mod tests {
    use super::prompt_blocks;
    use codex_app_server_protocol::UserInput;
    use serde_json::Value;
    use std::fs;
    use uuid::Uuid;

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
        let blocks = prompt_blocks(&[
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
}
