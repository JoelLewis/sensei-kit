//! Shared local-LLM coaching runtime for the Sensei apps.
//!
//! Unifies the llama.cpp + Gemma 4 stacks previously duplicated in
//! ChessMentor (`mentor-llm`) and GoSensei (`gosensei-llm`):
//!
//! - [`ModelSpec`] / [`GEMMA4_E2B_Q4`] — which GGUF to fetch and how to
//!   verify it.
//! - [`ModelStore`] (feature `download`) — hf-hub download with byte
//!   progress, sha256 + size verification, and two-tier resolution
//!   (bundled resources vs. user downloads).
//! - [`ModelManager`] (feature `llm`) — blocking load + generation with
//!   streaming token callback, optional GBNF grammar constraint, and
//!   cooperative cancellation.
//! - [`chat::format_chat`] — the hand-rendered Gemma 4 `<|turn|>` chat
//!   format shared by both apps.
//! - [`ChannelFilter`] — strips `<|channel>...<channel|>` thought blocks
//!   from streamed output.
//!
//! Prompt *content* (system prompts, payload rendering) and domain GBNF
//! grammars stay app-side; this crate only provides the format-level
//! plumbing.

pub mod chat;
pub mod error;
pub mod filter;
pub mod options;
pub mod spec;

#[cfg(feature = "llm")]
pub mod model;
#[cfg(feature = "download")]
pub mod store;

pub use chat::format_chat;
pub use error::LlmError;
pub use filter::ChannelFilter;
pub use options::{GenerateOptions, SamplerConfig};
pub use spec::{GEMMA4_E2B_Q4, ModelSpec};

#[cfg(feature = "llm")]
pub use model::{ModelManager, device_name};
#[cfg(feature = "download")]
pub use store::ModelStore;
