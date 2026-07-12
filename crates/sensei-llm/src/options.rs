//! Generation and sampling configuration.

/// Sampler chain parameters.
///
/// Defaults are the conservative coaching settings from ChessMentor
/// (low temperature, top-k truncation). GoSensei-style sampling is
/// `temperature: 0.6, top_k: None, min_p: Some(0.05), repeat_penalty: 1.15`.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerConfig {
    pub temperature: f32,
    /// `None` skips the top-k stage.
    pub top_k: Option<i32>,
    /// `None` skips the min-p stage.
    pub min_p: Option<f32>,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub repeat_penalty_last_n: i32,
    /// RNG seed for the final distribution sampler. Fixed by default so
    /// coaching output is reproducible for a given prompt.
    pub seed: u32,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.3,
            top_k: Some(50),
            min_p: None,
            top_p: 0.9,
            repeat_penalty: 1.1,
            repeat_penalty_last_n: 64,
            seed: 42,
        }
    }
}

/// Options for a single [`crate::model::ModelManager`] generation call.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerateOptions {
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
    /// Context window size. Each generation creates a fresh context; the
    /// prompt must tokenize to fewer than this many tokens.
    pub n_ctx: u32,
    pub sampler: SamplerConfig,
    /// Optional GBNF grammar constraining decoding. Domain grammars are
    /// owned by the apps; this crate only wires the grammar sampler in
    /// (first in the chain, so truncation samplers only ever see
    /// grammar-legal tokens and can never empty the candidate set).
    pub grammar: Option<String>,
    /// Name of the grammar's root rule (llama.cpp convention is `root`).
    pub grammar_root: String,
    /// Strip `<|channel>...<channel|>` thought blocks from output and
    /// streamed tokens. On by default; harmless for grammar-constrained
    /// output whose grammar cannot produce channel markers.
    pub filter_channels: bool,
}

impl GenerateOptions {
    pub fn new(max_tokens: u32) -> Self {
        Self {
            max_tokens,
            n_ctx: 1024,
            sampler: SamplerConfig::default(),
            grammar: None,
            grammar_root: "root".to_string(),
            filter_channels: true,
        }
    }

    /// Constrain decoding with a GBNF grammar whose root rule is `root`.
    pub fn with_grammar(mut self, grammar: impl Into<String>) -> Self {
        self.grammar = Some(grammar.into());
        self
    }

    pub fn with_sampler(mut self, sampler: SamplerConfig) -> Self {
        self.sampler = sampler;
        self
    }
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self::new(256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_sampler_matches_documented_values() {
        let s = SamplerConfig::default();
        assert_eq!(s.temperature, 0.3);
        assert_eq!(s.top_k, Some(50));
        assert_eq!(s.min_p, None);
        assert_eq!(s.top_p, 0.9);
        assert_eq!(s.repeat_penalty, 1.1);
        assert_eq!(s.repeat_penalty_last_n, 64);
        assert_eq!(s.seed, 42);
    }

    #[test]
    fn default_options_are_unconstrained_and_filtered() {
        let opts = GenerateOptions::default();
        assert_eq!(opts.max_tokens, 256);
        assert_eq!(opts.n_ctx, 1024);
        assert!(opts.grammar.is_none());
        assert_eq!(opts.grammar_root, "root");
        assert!(opts.filter_channels);
    }

    #[test]
    fn with_grammar_sets_grammar() {
        let opts = GenerateOptions::new(64).with_grammar("root ::= \"ok\"");
        assert_eq!(opts.grammar.as_deref(), Some("root ::= \"ok\""));
        assert_eq!(opts.grammar_root, "root");
    }

    #[test]
    fn with_sampler_overrides_defaults() {
        let gosensei = SamplerConfig {
            temperature: 0.6,
            top_k: None,
            min_p: Some(0.05),
            top_p: 0.9,
            repeat_penalty: 1.15,
            repeat_penalty_last_n: 64,
            seed: 1234,
        };
        let opts = GenerateOptions::new(200).with_sampler(gosensei.clone());
        assert_eq!(opts.sampler, gosensei);
    }
}
