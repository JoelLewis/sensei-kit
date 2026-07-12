# sensei-kit

Shared Rust workspace for the local-LLM coaching stacks of the Sensei Tauri
apps (ChessMentor / `teach-chess` and GoSensei / `teach-go`). Both apps run
the same model — Gemma 4 E2B instruct, Q4_K_M GGUF from
`unsloth/gemma-4-E2B-it-GGUF` — through llama.cpp, with the same
hand-rendered Gemma 4 `<|turn|>` chat format. This workspace holds the one
copy of that plumbing; the apps keep only what is genuinely theirs (prompt
text, domain grammars, output parsing, Tauri command layer).

## Crates

### `sensei-llm`

| API | Purpose |
| --- | --- |
| `ModelSpec`, `GEMMA4_E2B_Q4` | Which GGUF to fetch (repo, filename, exact byte size, sha256) |
| `ModelStore` | hf-hub download with throttled byte-progress callback, sha256 + size verification, bundled-resource resolution (`download` feature) |
| `ModelManager` | Load a GGUF; generate with streaming token callback, optional GBNF grammar constraint, cooperative cancellation (`llm` feature) |
| `GenerateOptions`, `SamplerConfig` | Per-call generation tuning (each app keeps its own sampler settings) |
| `format_chat` | The Gemma 4 `<|turn>role ... <turn|>` chat format |
| `ChannelFilter` | Strips `<|channel>...<channel|>` thought blocks from streamed output |
| `LlmError` | Shared error type (`thiserror`) |

### Feature flags

Default features are empty — the pure-Rust helpers (`format_chat`,
`ChannelFilter`, `ModelSpec`, options, errors) compile with no native or
network dependencies.

| Feature | Adds |
| --- | --- |
| `download` | `ModelStore` (hf-hub + sha2). Pure Rust, no native build. |
| `llm` | `ModelManager` (llama-cpp-2 + encoding_rs). Compiles llama.cpp — cmake required. |
| `metal` | `llm` + Metal backend. (macOS builds always compile Metal in, even without this flag.) |
| `cuda` | `llm` + CUDA backend. |

## Consuming from an app

There is no crates.io release; apps consume the crate as a git dependency.
Pin a rev (or tag) so app builds are reproducible:

```toml
[dependencies]
sensei-llm = { git = "https://github.com/joelelewis/sensei-kit", rev = "<sha>", features = ["download", "llm"] }
```

Until the repo is published, a local path dependency works for development:

```toml
sensei-llm = { path = "../sensei-kit/crates/sensei-llm", features = ["download", "llm"] }
```

Typical flow (all calls are blocking — run them under `spawn_blocking`):

```rust,ignore
use sensei_llm::{format_chat, GenerateOptions, ModelManager, ModelStore, GEMMA4_E2B_Q4};

let store = ModelStore::new(app_data_dir.join("models"));
let path = store.download(&GEMMA4_E2B_Q4, |done, total| emit_progress(done, total))?;

let manager = ModelManager::load(&path)?;
let prompt = format_chat(APP_SYSTEM_PROMPT, &render_user_payload(&payload));
let opts = GenerateOptions::new(200).with_grammar(app_domain_grammar());
let text = manager.generate_cancellable(&prompt, &opts, |tok| stream(tok), || cancelled())?;
```

`SENSEI_LLM_DEVICE=cpu` forces CPU inference regardless of compiled GPU
backends.

## Development

```bash
cargo test --workspace                                        # default features
cargo test --workspace --features download                    # + ModelStore
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

A real-model integration test (downloads ~2.9 GiB, runs constrained
generation) is `#[ignore]`d:

```bash
cargo test -p sensei-llm --features llm,download --release -- --ignored real_model --nocapture
```

See `MIGRATION.md` for the plan to convert each app onto this workspace.
