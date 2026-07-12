/// Errors from the LLM subsystem.
#[derive(Debug, Clone, thiserror::Error)]
pub enum LlmError {
    /// No model loaded yet. Never returned by this crate directly; provided
    /// for app state machines that gate commands on a loaded model.
    #[error("model not loaded")]
    ModelNotLoaded,

    #[error("model file not found: {0}")]
    ModelNotFound(String),

    #[error("inference failed: {0}")]
    Inference(String),

    #[error("model download failed: {0}")]
    Download(String),

    #[error("generation cancelled")]
    Cancelled,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_are_stable() {
        assert_eq!(LlmError::ModelNotLoaded.to_string(), "model not loaded");
        assert_eq!(
            LlmError::ModelNotFound("/x/y.gguf".into()).to_string(),
            "model file not found: /x/y.gguf"
        );
        assert_eq!(
            LlmError::Inference("boom".into()).to_string(),
            "inference failed: boom"
        );
        assert_eq!(
            LlmError::Download("net".into()).to_string(),
            "model download failed: net"
        );
        assert_eq!(LlmError::Cancelled.to_string(), "generation cancelled");
    }

    #[test]
    fn implements_std_error() {
        fn takes_error(_: &dyn std::error::Error) {}
        takes_error(&LlmError::Cancelled);
    }
}
