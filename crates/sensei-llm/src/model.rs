//! Model loading and generation via llama.cpp.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use tracing::{debug, info, warn};

use crate::error::LlmError;
use crate::filter::ChannelFilter;
use crate::options::GenerateOptions;

/// Number of GPU layers to offload plus a display name for the device.
///
/// macOS builds always have Metal compiled in and use it by default;
/// `SENSEI_LLM_DEVICE=cpu` forces CPU inference everywhere.
fn gpu_config() -> (u32, &'static str) {
    if matches!(std::env::var("SENSEI_LLM_DEVICE").as_deref(), Ok("cpu")) {
        info!("SENSEI_LLM_DEVICE=cpu — forcing CPU inference");
        return (0, "cpu");
    }
    compiled_gpu_config()
}

#[cfg(any(target_os = "macos", feature = "metal"))]
fn compiled_gpu_config() -> (u32, &'static str) {
    (u32::MAX, "metal")
}

#[cfg(all(not(any(target_os = "macos", feature = "metal")), feature = "cuda"))]
fn compiled_gpu_config() -> (u32, &'static str) {
    (u32::MAX, "cuda")
}

#[cfg(not(any(target_os = "macos", feature = "metal", feature = "cuda")))]
fn compiled_gpu_config() -> (u32, &'static str) {
    (0, "cpu")
}

/// Display name for the compute device inference will use.
pub fn device_name() -> &'static str {
    gpu_config().1
}

/// Manages a loaded LLM model. `LlamaModel` is `Send+Sync`, so this can live
/// in an `Arc`. `LlamaContext` is `!Send`, so each generation call creates a
/// fresh context. Clone is cheap — both fields are `Arc`.
#[derive(Clone, Debug)]
pub struct ModelManager {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
}

// LlamaBackend and LlamaModel are Send+Sync in llama-cpp-2
unsafe impl Send for ModelManager {}
unsafe impl Sync for ModelManager {}

impl ModelManager {
    /// Load a GGUF model from disk. Blocking — call from `spawn_blocking`.
    pub fn load(model_path: &Path) -> Result<Self, LlmError> {
        if !model_path.exists() {
            return Err(LlmError::ModelNotFound(model_path.display().to_string()));
        }

        let (n_gpu_layers, device) = gpu_config();
        info!("Loading LLM model from {} ({device})", model_path.display());

        let backend =
            LlamaBackend::init().map_err(|e| LlmError::Inference(format!("backend init: {e}")))?;

        let model_params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| LlmError::Inference(format!("model load: {e}")))?;

        info!(
            "LLM model loaded successfully ({} params)",
            model.n_params()
        );

        Ok(Self {
            backend: Arc::new(backend),
            model: Arc::new(model),
        })
    }

    /// Generate text from a formatted prompt. Blocking — call from
    /// `spawn_blocking`.
    pub fn generate(&self, prompt: &str, opts: &GenerateOptions) -> Result<String, LlmError> {
        self.generate_streaming(prompt, opts, |_| {})
    }

    /// Generate text with per-token streaming callback. Blocking.
    pub fn generate_streaming(
        &self,
        prompt: &str,
        opts: &GenerateOptions,
        on_token: impl FnMut(&str),
    ) -> Result<String, LlmError> {
        self.generate_cancellable(prompt, opts, on_token, || false)
    }

    /// Generate text with streaming callback and cooperative cancellation.
    /// Blocking.
    ///
    /// `is_cancelled` is polled between tokens; returns [`LlmError::Cancelled`]
    /// when it reports true. Creates a fresh `LlamaContext` for thread safety.
    pub fn generate_cancellable(
        &self,
        prompt: &str,
        opts: &GenerateOptions,
        mut on_token: impl FnMut(&str),
        is_cancelled: impl Fn() -> bool,
    ) -> Result<String, LlmError> {
        let n_ctx = NonZeroU32::new(opts.n_ctx)
            .ok_or_else(|| LlmError::Inference("n_ctx must be non-zero".to_string()))?;
        let ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));

        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| LlmError::Inference(format!("context creation: {e}")))?;

        // Tokenize prompt
        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| LlmError::Inference(format!("tokenization: {e}")))?;

        debug!("Prompt tokenized: {} tokens", tokens.len());

        if tokens.is_empty() {
            return Err(LlmError::Inference("empty prompt".to_string()));
        }
        if tokens.len() >= opts.n_ctx as usize {
            return Err(LlmError::Inference(
                "prompt exceeds context window".to_string(),
            ));
        }

        // Process prompt tokens in a batch
        let mut batch = LlamaBatch::new(opts.n_ctx as usize, 1);
        let last_idx = (tokens.len() - 1) as i32;
        for (i, token) in tokens.into_iter().enumerate() {
            let pos = i as i32;
            batch
                .add(token, pos, &[0], pos == last_idx)
                .map_err(|e| LlmError::Inference(format!("batch add: {e}")))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| LlmError::Inference(format!("prompt decode: {e}")))?;

        let mut sampler = self.build_sampler(opts)?;

        let mut output = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut filter = opts.filter_channels.then(ChannelFilter::new);
        let n_start = batch.n_tokens();

        for n_cur in n_start..n_start + opts.max_tokens as i32 {
            if is_cancelled() {
                return Err(LlmError::Cancelled);
            }

            // `sample` already accepts the token into the chain's internal
            // state (grammar, penalties). Accepting again would advance the
            // grammar twice and abort inside llama.cpp.
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);

            if self.model.is_eog_token(token) {
                debug!("End of generation token reached");
                break;
            }

            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| LlmError::Inference(format!("token decode: {e}")))?;

            let visible = match filter.as_mut() {
                Some(filter) => filter.push(&piece),
                None => piece,
            };
            if !visible.is_empty() {
                on_token(&visible);
                output.push_str(&visible);
            }

            batch.clear();
            batch
                .add(token, n_cur, &[0], true)
                .map_err(|e| LlmError::Inference(format!("batch add gen: {e}")))?;

            ctx.decode(&mut batch)
                .map_err(|e| LlmError::Inference(format!("decode gen: {e}")))?;
        }

        if let Some(filter) = filter {
            let tail = filter.finish();
            if !tail.is_empty() {
                on_token(&tail);
                output.push_str(&tail);
            }
        }

        if output.is_empty() {
            warn!("LLM generated empty output");
        }

        Ok(output.trim().to_string())
    }

    /// Build the sampler chain: `[grammar →] temp → top_k → min_p → top_p →
    /// penalties → dist`.
    ///
    /// The grammar sampler runs first so truncation samplers only ever see
    /// grammar-legal tokens and can never empty the candidate set.
    fn build_sampler(&self, opts: &GenerateOptions) -> Result<LlamaSampler, LlmError> {
        let s = &opts.sampler;
        let mut samplers = Vec::with_capacity(7);

        if let Some(ref grammar) = opts.grammar {
            samplers.push(
                LlamaSampler::grammar(&self.model, grammar, &opts.grammar_root)
                    .map_err(|e| LlmError::Inference(format!("grammar init: {e}")))?,
            );
        }
        samplers.push(LlamaSampler::temp(s.temperature));
        if let Some(top_k) = s.top_k {
            samplers.push(LlamaSampler::top_k(top_k));
        }
        if let Some(min_p) = s.min_p {
            samplers.push(LlamaSampler::min_p(min_p, 1));
        }
        samplers.extend([
            LlamaSampler::top_p(s.top_p, 1),
            LlamaSampler::penalties(s.repeat_penalty_last_n, s.repeat_penalty, 0.0, 0.0),
            LlamaSampler::dist(s.seed),
        ]);

        Ok(LlamaSampler::chain_simple(samplers))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn load_missing_file_returns_error() {
        let result = ModelManager::load(&PathBuf::from("/nonexistent/model.gguf"));
        assert!(matches!(result, Err(LlmError::ModelNotFound(_))));
    }

    #[test]
    fn device_name_is_known_value() {
        assert!(matches!(device_name(), "cpu" | "metal" | "cuda"));
    }

    /// Real-model integration test: downloads Gemma 4 E2B Q4_K_M (~2.9 GiB)
    /// on first run, loads it, and verifies grammar-constrained generation
    /// plus streaming/channel-filter behavior.
    ///
    /// Run explicitly with:
    /// `cargo test -p sensei-llm --features llm,download --release -- --ignored real_model --nocapture`
    /// Set `SENSEI_LLM_MODEL_DIR` to reuse an existing model directory.
    #[test]
    #[ignore = "downloads and loads the ~2.9 GiB Gemma 4 E2B model"]
    #[cfg(feature = "download")]
    fn real_model_constrained_generation() {
        use crate::chat::format_chat;
        use crate::spec::GEMMA4_E2B_Q4;
        use crate::store::ModelStore;

        let models_dir = std::env::var_os("SENSEI_LLM_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("sensei-llm-real-model"));

        let store = ModelStore::new(models_dir);
        let model_path = store.download(&GEMMA4_E2B_Q4, |_, _| {}).expect("download");
        eprintln!("inference device: {}", device_name());

        let t0 = std::time::Instant::now();
        let manager = ModelManager::load(&model_path).expect("model load");
        eprintln!("load took {:.1}s", t0.elapsed().as_secs_f32());

        // Unconstrained generation with streaming.
        let prompt = format_chat(
            "You are a concise coach.",
            "Reply with one short sentence: why does practice matter?",
        );
        let mut streamed = String::new();
        let text = manager
            .generate_streaming(&prompt, &GenerateOptions::new(128), |t| {
                streamed.push_str(t);
            })
            .expect("generation should succeed");
        eprintln!("unconstrained output: {text}");
        assert!(text.len() > 20, "suspiciously short output: {text:?}");
        assert!(
            !text.contains("<|channel>") && !text.contains("<channel|>"),
            "output leaked channel markers: {text:?}"
        );
        assert_eq!(streamed.trim(), text, "streamed text should match final");

        // Grammar-constrained generation with a crate-agnostic test grammar.
        let grammar = "root ::= \"Answer: \" [a-zA-Z ,.]{1,200}\n";
        let opts = GenerateOptions::new(64).with_grammar(grammar);
        let constrained = manager
            .generate(&prompt, &opts)
            .expect("constrained generation");
        eprintln!("constrained output: {constrained}");
        assert!(
            constrained.starts_with("Answer: "),
            "grammar should force the prefix, got: {constrained}"
        );
    }
}
