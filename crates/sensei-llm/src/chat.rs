//! Hand-rendered Gemma 4 chat format.

/// Format a system + user message pair in the Gemma 4 instruction format.
///
/// Gemma 4 replaced Gemma 2/3's `<start_of_turn>role ... <end_of_turn>`
/// markers with `<|turn>{role}\n{content}<turn|>\n` turns, gained a native
/// system role, and ends with a `<|turn>model\n` generation prompt (see
/// `common_chat_params_init_gemma4` in llama.cpp's `common/chat.cpp`).
/// llama.cpp's C-side `llama_chat_apply_template` heuristics do not know
/// this template (only its Jinja engine does, which llama-cpp-2 does not
/// expose), so the non-thinking, no-tools path of the GGUF's Jinja template
/// is rendered here by hand:
///
/// ```text
/// <|turn>system
/// {system}<turn|>
/// <|turn>user
/// {user}<turn|>
/// <|turn>model
/// ```
///
/// Content is trimmed so stray whitespace never distorts the turn markers.
/// The BOS token is intentionally omitted — `ModelManager` tokenizes with
/// `AddBos::Always`.
pub fn format_chat(system: &str, user: &str) -> String {
    format!(
        "<|turn>system\n{}<turn|>\n<|turn>user\n{}<turn|>\n<|turn>model\n",
        system.trim(),
        user.trim(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_gemma4_markers() {
        let prompt = format_chat("system text", "user text");
        assert_eq!(
            prompt,
            "<|turn>system\nsystem text<turn|>\n<|turn>user\nuser text<turn|>\n<|turn>model\n"
        );
    }

    #[test]
    fn trims_content() {
        let prompt = format_chat("  system text \n", "\n user text  ");
        assert!(prompt.contains("<|turn>system\nsystem text<turn|>"));
        assert!(prompt.contains("<|turn>user\nuser text<turn|>"));
    }

    #[test]
    fn ends_with_model_turn_and_omits_bos() {
        let prompt = format_chat("s", "u");
        assert!(prompt.ends_with("<|turn>model\n"));
        // No Gemma 3 markers and no BOS — AddBos::Always handles BOS.
        assert!(!prompt.contains("<start_of_turn>"));
        assert!(!prompt.contains("<bos>"));
    }
}
