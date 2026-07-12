# Migrating the apps onto sensei-kit

Follow-up work, one PR per app. Nothing in this repo depends on either app;
each app swaps its local LLM crate for `sensei-llm` and keeps its domain
layer.

## Dependency line (both apps)

In the app's workspace `Cargo.toml` (or `src-tauri/Cargo.toml`):

```toml
sensei-llm = { git = "https://github.com/joelelewis/sensei-kit", rev = "<sha>", features = ["download", "llm"] }
```

- Pin `rev` (or a tag once releases are tagged) — do not track `main`.
- The apps' existing `llm` cargo feature can forward to sensei-llm's
  features if the "build without llama.cpp" dev path should survive:
  `llm = ["sensei-llm/llm", "sensei-llm/download"]` with
  `default-features = false` on the dependency.
- GPU: macOS gets Metal automatically. A Linux/Windows CUDA build adds
  `features = ["cuda"]`. Chess's `llm-metal`/`llm-cuda` features map to
  `metal`/`cuda`.

## Shared renames

| Old (chess / go) | New |
| --- | --- |
| `LlmError::InferenceError` / `LlmError::InferenceFailed` | `LlmError::Inference` |
| `LlmError::DownloadError` / `LlmError::DownloadFailed` | `LlmError::Download` |
| `CHESS_MENTOR_DEVICE=cpu` | `SENSEI_LLM_DEVICE=cpu` |
| chess `ModelConfig` / go repo+filename consts | `ModelSpec` / `GEMMA4_E2B_Q4` |

Generation methods now take `&GenerateOptions` instead of positional
`max_tokens` / grammar arguments — each app constructs options with its own
`SamplerConfig` (values below).

## teach-chess (`crates/mentor-llm`)

**Delete** (now provided by sensei-llm):

- `src/download.rs` → `ModelStore` + `ModelSpec`
- `src/model.rs` → `ModelManager`
- `src/stream.rs` → `ChannelFilter`
- `src/error.rs` → `LlmError`

**Stays app-side** (move into `src-tauri` or keep a slim `mentor-coaching`
crate): `src/prompts.rs` (system prompts, `build_prompt`,
`build_game_summary_prompt` — call `sensei_llm::format_chat` instead of the
local `format_chat`) and `src/types.rs` (`PlayerLevel`).

**Call-site changes:**

- `ModelStore::new(app_data_dir, resource_dir)` →
  `ModelStore::new(app_data_dir.join("models"))` (+
  `.with_resource_dir(dir)` if bundled). The implicit `join("models")`
  moved app-side.
- `ModelConfig` UI metadata (`id`, `display_name`, `ram_requirement_mb`,
  `get_config`) is app policy — keep a small app-side table that pairs a
  `ModelSpec` with its display metadata. File size is now exact bytes
  (`size_bytes: 3_106_736_256`), not `file_size_mb: 2963`.
- Sampler settings move into an app-side constant:
  `SamplerConfig { temperature: 0.3, top_k: Some(50), min_p: None, top_p: 0.9, repeat_penalty: 1.1, repeat_penalty_last_n: 64, seed: 42 }`
  (these are sensei-llm's defaults, so `GenerateOptions::new(max_tokens)`
  already matches; state it explicitly anyway). `n_ctx` stays 1024
  (the default).

**Behavior changes to verify:**

- `sampler.accept(token)` after `sample()` is gone (sample already accepts;
  the old double-accept skewed repeat penalties). Coaching text for a fixed
  seed will differ slightly from before.
- Download progress callbacks are now throttled to ~8 MiB granularity and
  fire an immediate `(0, size_bytes)` report.
- `format_chat` now trims system/user content (from gosensei-llm).

## teach-go (`crates/gosensei-llm`)

**Delete** (now provided by sensei-llm):

- `src/download.rs` → `ModelStore` + `GEMMA4_E2B_Q4`
- `src/model.rs` → `ModelManager` (including `apply_chat_template`, which
  was already a thin alias for the chat formatter)
- `src/error.rs` → `LlmError` — but see the tricky note on `Timeout` /
  `ParseError` below.

**Stays app-side** (keep the crate, slimmed, or fold into `src-tauri`):
`src/prompt.rs` (`SYSTEM_PROMPT`, `build_user_prompt`, `rank_to_display` —
use `sensei_llm::format_chat`), `src/grammar.rs` (`coaching_grammar` — the
domain GBNF), `src/parse.rs` (tag parsing, `KNOWN_ERROR_CLASSES`,
sanitization), `src/types.rs` (`CoachingPayload` etc.).

**Call-site changes:**

- `ensure_model(model_dir, cb)` →
  `ModelStore::new(model_dir).download(&GEMMA4_E2B_Q4, cb)`.
- `generate_streaming_with_grammar(prompt, 200, &grammar, cb)` →
  `manager.generate_streaming(prompt, &GenerateOptions::new(200).with_grammar(grammar).with_sampler(GO_SAMPLER), cb)`
  where `GO_SAMPLER` is
  `SamplerConfig { temperature: 0.6, top_k: None, min_p: Some(0.05), top_p: 0.9, repeat_penalty: 1.15, repeat_penalty_last_n: 64, seed: 1234 }`,
  and set `n_ctx: 512` to keep the old context size (default is 1024 —
  keeping 512 preserves the old prompt-length guard and memory profile).
- `grammar::GRAMMAR_ROOT` can be dropped if it stays `"root"` (the
  `GenerateOptions` default).

**Tricky bits (flagged):**

1. **Model path layout changes.** gosensei stored the GGUF flat at
   `{model_dir}/{filename}` and used `model_dir` itself as the hf-hub cache;
   sensei-llm stores at `{models_dir}/{repo_id}/{filename}` with the cache
   in `{models_dir}/hf-cache`. Existing installs would re-download ~2.9 GiB.
   The migration should move/symlink the old flat file into the new layout
   on first run (or check the legacy path before calling `download`).
2. **Error variants `Timeout(u64)` and `ParseError(String)`** were
   app-level concerns living in the crate error. They move to a gosensei
   app error type that wraps `sensei_llm::LlmError` (`thiserror` +
   `#[from]`).
3. **Channel filtering is now on by default.** gosensei previously streamed
   raw pieces. Its grammar output (`<classification>...` tags) passes
   through the filter untouched — the filter only strips `<|channel>` /
   `<channel|>` markers, which the grammar forbids anyway — but partial
   `<` prefixes are briefly buffered, so streamed chunk boundaries shift.
   Set `filter_channels: false` to keep byte-identical streaming, or keep
   the default for the extra safety net on unconstrained calls.
4. **Cancellation is newly available** to gosensei — the Tauri command
   layer can adopt the chess-style cancel flag via `generate_cancellable`
   instead of relying on timeouts alone.
5. **Download size mismatch is still only a warning**, but sha256
   verification (from chess) is now enforced for `GEMMA4_E2B_Q4` — a
   corrupted download deletes the file and errors instead of failing later
   at llama.cpp load.

## Stays app-side in both apps

- Prompt text and payload rendering (chess `prompts.rs`, go `prompt.rs`)
- Domain grammars (go `grammar.rs`) and output parsing (go `parse.rs`)
- Response caching
- The Tauri command layer, event emission, and app state (`ModelNotLoaded`
  remains available in `LlmError` for those state machines)
- Skill/rank modeling (chess `PlayerLevel`, go rank display)
