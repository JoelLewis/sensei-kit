/// Identifies a downloadable GGUF model and how to verify it.
///
/// App-facing metadata (display name, RAM requirements, model picker ids)
/// stays app-side; this is only what the download and load paths need.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    /// HuggingFace repo id, e.g. `unsloth/gemma-4-E2B-it-GGUF`.
    pub repo_id: &'static str,
    /// GGUF filename within the repo.
    pub filename: &'static str,
    /// Exact size in bytes as published on HuggingFace. Used to report a
    /// download total before the server responds and to sanity-check the
    /// downloaded file.
    pub size_bytes: u64,
    /// Expected SHA256 of the GGUF file (lowercase hex). `None` skips
    /// verification.
    pub sha256: Option<&'static str>,
}

/// Gemma 4 E2B instruct, Q4_K_M quantization — the model both Sensei apps
/// ship with.
///
/// llama.cpp embeds the tokenizer in the GGUF, so no separate tokenizer
/// download is needed.
pub const GEMMA4_E2B_Q4: ModelSpec = ModelSpec {
    repo_id: "unsloth/gemma-4-E2B-it-GGUF",
    filename: "gemma-4-E2B-it-Q4_K_M.gguf",
    size_bytes: 3_106_736_256,
    sha256: Some("9378bc471710229ef165709b62e34bfb62231420ddaf6d729e727305b5b8672d"),
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_spec_is_reasonable() {
        assert_eq!(GEMMA4_E2B_Q4.repo_id, "unsloth/gemma-4-E2B-it-GGUF");
        assert!(GEMMA4_E2B_Q4.filename.ends_with("Q4_K_M.gguf"));
        assert_eq!(GEMMA4_E2B_Q4.size_bytes, 3_106_736_256);
        assert_eq!(GEMMA4_E2B_Q4.sha256.unwrap().len(), 64);
        assert!(
            GEMMA4_E2B_Q4
                .sha256
                .unwrap()
                .chars()
                .all(|c| c.is_ascii_hexdigit())
        );
    }
}
