//! Model download and on-disk resolution.

use std::path::{Path, PathBuf};

use tracing::{info, warn};

use crate::error::LlmError;
use crate::spec::ModelSpec;

/// Report at most every 8 MiB to avoid flooding progress event channels.
const PROGRESS_GRANULARITY_BYTES: u64 = 8 * 1024 * 1024;

/// Forwards hf-hub download progress to a `(downloaded, total)` callback,
/// throttled to [`PROGRESS_GRANULARITY_BYTES`].
struct ThrottledProgress<F: FnMut(u64, u64)> {
    on_progress: F,
    downloaded: u64,
    total: u64,
    last_reported: u64,
}

impl<F: FnMut(u64, u64)> hf_hub::api::Progress for ThrottledProgress<F> {
    fn init(&mut self, size: usize, _filename: &str) {
        self.total = size as u64;
        (self.on_progress)(0, self.total);
    }

    fn update(&mut self, size: usize) {
        self.downloaded += size as u64;
        if self.downloaded - self.last_reported >= PROGRESS_GRANULARITY_BYTES {
            self.last_reported = self.downloaded;
            (self.on_progress)(self.downloaded, self.total);
        }
    }

    fn finish(&mut self) {
        (self.on_progress)(self.total.max(self.downloaded), self.total);
    }
}

/// Resolves and downloads model files.
///
/// Supports two-tier resolution: bundled models in an optional resource
/// directory (read-only, shipped with the app) and user-downloaded models in
/// `models_dir` (writable). Downloaded files live at
/// `{models_dir}/{repo_id}/{filename}`.
#[derive(Debug, Clone)]
pub struct ModelStore {
    models_dir: PathBuf,
    resource_dir: Option<PathBuf>,
}

impl ModelStore {
    /// A store rooted at `models_dir` (e.g. `{app_data_dir}/models`).
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            models_dir: models_dir.into(),
            resource_dir: None,
        }
    }

    /// Also resolve models bundled in a read-only resource directory
    /// (checked before `models_dir`).
    pub fn with_resource_dir(mut self, resource_dir: impl Into<PathBuf>) -> Self {
        self.resource_dir = Some(resource_dir.into());
        self
    }

    /// Path where a model's GGUF file is located.
    /// Checks the bundled resource dir first, then the models dir.
    pub fn model_path(&self, spec: &ModelSpec) -> PathBuf {
        if let Some(ref res) = self.resource_dir {
            let bundled = res.join(spec.filename);
            if bundled.exists() {
                return bundled;
            }
        }
        self.models_dir.join(spec.repo_id).join(spec.filename)
    }

    /// Check if a model's GGUF file is available (bundled or downloaded).
    ///
    /// Verifies the file exists and its size is within 10% of the expected
    /// size, so truncated downloads never count as available.
    pub fn is_available(&self, spec: &ModelSpec) -> bool {
        match std::fs::metadata(self.model_path(spec)) {
            Ok(metadata) => metadata.len() >= spec.size_bytes * 9 / 10,
            Err(_) => false,
        }
    }

    /// Whether the model is available from the bundled resource directory.
    pub fn is_bundled(&self, spec: &ModelSpec) -> bool {
        self.resource_dir
            .as_ref()
            .is_some_and(|res| res.join(spec.filename).exists())
    }

    /// Download a model GGUF from HuggingFace Hub.
    ///
    /// Progress is reported via callback as `(bytes_downloaded, total_bytes)`,
    /// throttled to roughly one report per 8 MiB. The first report fires
    /// immediately with the published size so UIs can show a total before the
    /// server responds. After download the file's SHA256 is verified against
    /// [`ModelSpec::sha256`] (mismatches delete the file and error) and its
    /// size is sanity-checked against [`ModelSpec::size_bytes`] (mismatches
    /// only warn — llama.cpp validates GGUF integrity at load time, so a size
    /// drift usually means upstream requantization).
    ///
    /// This is a blocking operation — call from `spawn_blocking`.
    pub fn download(
        &self,
        spec: &ModelSpec,
        on_progress: impl FnMut(u64, u64),
    ) -> Result<PathBuf, LlmError> {
        info!("Downloading model from {}/{}", spec.repo_id, spec.filename);

        self.download_with_fetcher(spec, on_progress, |on_progress, total| {
            let api = hf_hub::api::sync::ApiBuilder::new()
                .with_cache_dir(self.models_dir.join("hf-cache"))
                .build()
                .map_err(|e| LlmError::Download(format!("HF API init: {e}")))?;

            let repo = api.model(spec.repo_id.to_string());

            // Signal download start with the published size until the server
            // tells us more.
            on_progress(0, total);

            let progress = ThrottledProgress {
                on_progress,
                downloaded: 0,
                total,
                last_reported: 0,
            };

            repo.download_with_progress(spec.filename, progress)
                .map_err(|e| LlmError::Download(format!("download: {e}")))
        })
    }

    fn download_with_fetcher<F>(
        &self,
        spec: &ModelSpec,
        mut on_progress: impl FnMut(u64, u64),
        fetch: F,
    ) -> Result<PathBuf, LlmError>
    where
        F: FnOnce(&mut dyn FnMut(u64, u64), u64) -> Result<PathBuf, LlmError>,
    {
        let target = self.models_dir.join(spec.repo_id).join(spec.filename);

        if target.exists() {
            if cached_model_is_valid(&target, spec) {
                info!("Model already cached at {}", target.display());
                return Ok(target);
            }

            warn!("Removing invalid cached model at {}", target.display());
            std::fs::remove_file(&target)
                .map_err(|e| LlmError::Download(format!("remove invalid model: {e}")))?;
        }

        let target_dir = target
            .parent()
            .expect("model target path always has a parent");
        std::fs::create_dir_all(target_dir)
            .map_err(|e| LlmError::Download(format!("create model dir: {e}")))?;

        let downloaded_path = fetch(&mut on_progress, spec.size_bytes)?;

        info!("Model downloaded to {}", downloaded_path.display());

        if let Some(expected_hash) = spec.sha256 {
            info!("Verifying SHA256 hash for {}", spec.filename);
            if let Err(e) = verify_hash(&downloaded_path, expected_hash) {
                let _ = std::fs::remove_file(&downloaded_path);
                return Err(e);
            }
            info!("SHA256 hash verified for {}", spec.filename);
        }

        if let Ok(meta) = std::fs::metadata(&downloaded_path)
            && meta.len() != spec.size_bytes
        {
            warn!(
                "Downloaded model size {} differs from expected {}",
                meta.len(),
                spec.size_bytes
            );
        }

        // hf-hub caches files in its own snapshot structure; link to a
        // temporary stable path and atomically rename it into place so an
        // interrupted materialization cannot leave a plausible final file.
        if downloaded_path != target {
            let temporary = target.with_file_name(format!(".{}.part", spec.filename));
            let _ = std::fs::remove_file(&temporary);
            if let Err(error) = link_or_copy(&downloaded_path, &temporary).and_then(|_| {
                std::fs::rename(&temporary, &target).map_err(|e| {
                    LlmError::Download(format!("rename downloaded model into place: {e}"))
                })
            }) {
                let _ = std::fs::remove_file(&temporary);
                return Err(error);
            }
        }

        info!("Model download complete: {}", spec.filename);
        Ok(target)
    }
}

fn cached_model_is_valid(path: &Path, spec: &ModelSpec) -> bool {
    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };

    metadata.len() == spec.size_bytes
        && spec
            .sha256
            .is_none_or(|expected_hash| verify_hash(path, expected_hash).is_ok())
}

#[cfg(unix)]
fn link_or_copy(src: &Path, dest: &Path) -> Result<(), LlmError> {
    std::os::unix::fs::symlink(src, dest).map_err(|e| LlmError::Download(format!("symlink: {e}")))
}

#[cfg(not(unix))]
fn link_or_copy(src: &Path, dest: &Path) -> Result<(), LlmError> {
    std::fs::copy(src, dest)
        .map(|_| ())
        .map_err(|e| LlmError::Download(format!("copy: {e}")))
}

/// Verify the SHA256 hash of a file matches the expected value.
///
/// Reads the file in 8KB chunks to avoid loading large files into memory.
fn verify_hash(path: &Path, expected_hex: &str) -> Result<(), LlmError> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut file = std::fs::File::open(path).map_err(|e| {
        LlmError::Download(format!("Failed to open file for hash verification: {e}"))
    })?;

    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let bytes_read = file.read(&mut buffer).map_err(|e| {
            LlmError::Download(format!("Failed to read file for hash verification: {e}"))
        })?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    let actual_hex = format!("{:x}", hasher.finalize());
    if actual_hex != expected_hex {
        return Err(LlmError::Download(format!(
            "SHA256 hash mismatch for {}: expected {expected_hex}, got {actual_hex}. \
             The file may be corrupted — please delete it and re-download.",
            path.display()
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::GEMMA4_E2B_Q4;

    const TEST_SPEC: ModelSpec = ModelSpec {
        repo_id: "test/repo",
        filename: "model.gguf",
        size_bytes: 1024 * 1024,
        sha256: None,
    };

    const HASHED_TEST_SPEC: ModelSpec = ModelSpec {
        repo_id: "test/repo",
        filename: "hashed-model.gguf",
        size_bytes: 11,
        sha256: Some("b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"),
    };

    #[test]
    fn model_path_without_resource_dir() {
        let store = ModelStore::new("/tmp/test-models");
        let path = store.model_path(&GEMMA4_E2B_Q4);
        assert!(path.to_str().unwrap().contains("unsloth"));
        assert!(path.to_str().unwrap().ends_with(".gguf"));
    }

    #[test]
    fn is_available_returns_false_for_missing_files() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path());
        assert!(!store.is_available(&GEMMA4_E2B_Q4));
    }

    #[test]
    fn is_available_returns_false_for_truncated_model() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path());

        let model_path = store.model_path(&TEST_SPEC);
        std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();

        // Way below 90% of 1 MiB.
        std::fs::write(&model_path, b"tiny").unwrap();

        assert!(!store.is_available(&TEST_SPEC));
    }

    #[test]
    fn is_available_returns_true_for_adequate_size() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path());

        let model_path = store.model_path(&TEST_SPEC);
        std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();

        let data = vec![0u8; 1024 * 1024];
        std::fs::write(&model_path, &data).unwrap();

        assert!(store.is_available(&TEST_SPEC));
    }

    #[test]
    fn download_truncated_cached_file_is_not_trusted() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path());

        let model_path = store.model_path(&TEST_SPEC);
        std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        std::fs::write(&model_path, b"truncated").unwrap();

        let mut calls = 0u32;
        let result = store.download(&TEST_SPEC, |_, _| calls += 1);
        assert!(result.is_err(), "an invalid cache must trigger a download");
        assert!(calls > 0, "an invalid cache must report download progress");
        assert!(
            !model_path.exists(),
            "failed download must not leave a target"
        );
    }

    #[test]
    fn download_returns_complete_cached_path_without_network() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path());

        let model_path = store.model_path(&TEST_SPEC);
        std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        std::fs::write(&model_path, vec![0u8; TEST_SPEC.size_bytes as usize]).unwrap();

        let mut calls = 0u32;
        let result = store.download(&TEST_SPEC, |_, _| calls += 1);
        assert_eq!(result.unwrap(), model_path);
        assert_eq!(calls, 0, "complete cached hit should not report progress");
    }

    #[test]
    fn download_same_size_cached_file_with_wrong_hash_is_replaced() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path());
        let model_path = store.model_path(&HASHED_TEST_SPEC);
        std::fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        std::fs::write(&model_path, b"wrong bytes").unwrap();

        let fetched_path = tmp.path().join("fetched.gguf");
        std::fs::write(&fetched_path, b"hello world").unwrap();
        let mut fetches = 0;
        let result = store.download_with_fetcher(
            &HASHED_TEST_SPEC,
            |_, _| {},
            |_, _| {
                fetches += 1;
                Ok(fetched_path.clone())
            },
        );

        assert_eq!(result.unwrap(), model_path);
        assert_eq!(fetches, 1);
        assert_eq!(std::fs::read(model_path).unwrap(), b"hello world");
    }

    #[test]
    fn interrupted_download_never_leaves_file_at_final_path() {
        let tmp = tempfile::tempdir().unwrap();
        let store = ModelStore::new(tmp.path());
        let model_path = store.model_path(&TEST_SPEC);

        let result = store.download_with_fetcher(
            &TEST_SPEC,
            |_, _| {},
            |_, _| {
                let partial_path = model_path.with_file_name(".model.gguf.part");
                std::fs::create_dir_all(partial_path.parent().unwrap()).unwrap();
                std::fs::write(partial_path, b"partial download").unwrap();
                Err(LlmError::Download("interrupted download".into()))
            },
        );

        assert!(result.is_err());
        assert!(!model_path.exists());
    }

    #[test]
    fn verify_hash_succeeds_for_correct_hash() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("test.bin");
        std::fs::write(&file_path, b"hello world").unwrap();

        // SHA256 of "hello world"
        let expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
        assert!(verify_hash(&file_path, expected).is_ok());
    }

    #[test]
    fn verify_hash_fails_for_wrong_hash() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("test.bin");
        std::fs::write(&file_path, b"hello world").unwrap();

        let wrong = "0000000000000000000000000000000000000000000000000000000000000000";
        let result = verify_hash(&file_path, wrong);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("hash mismatch"));
    }

    #[test]
    fn bundled_resource_dir_preferred_when_exists() {
        let tmp = tempfile::tempdir().unwrap();
        let res_dir = tmp.path().join("resources");
        std::fs::create_dir_all(&res_dir).unwrap();
        std::fs::write(res_dir.join(TEST_SPEC.filename), b"bundled").unwrap();

        let store = ModelStore::new(tmp.path().join("models")).with_resource_dir(&res_dir);
        assert_eq!(
            store.model_path(&TEST_SPEC),
            res_dir.join(TEST_SPEC.filename)
        );
        assert!(store.is_bundled(&TEST_SPEC));
    }

    #[test]
    fn is_bundled_false_without_resource_dir() {
        let store = ModelStore::new("/tmp/test-models");
        assert!(!store.is_bundled(&GEMMA4_E2B_Q4));
    }

    #[test]
    fn is_bundled_false_when_files_missing() {
        let store = ModelStore::new("/tmp/test-models").with_resource_dir("/tmp/nonexistent");
        assert!(!store.is_bundled(&GEMMA4_E2B_Q4));
    }

    #[test]
    fn throttle_reports_first_and_final_progress() {
        let mut reports: Vec<(u64, u64)> = Vec::new();
        {
            use hf_hub::api::Progress;
            let mut p = ThrottledProgress {
                on_progress: |d, t| reports.push((d, t)),
                downloaded: 0,
                total: 0,
                last_reported: 0,
            };
            p.init(100 * 1024 * 1024, "model.gguf");
            // 1 MiB chunks: only every 8th chunk should report.
            for _ in 0..100 {
                p.update(1024 * 1024);
            }
            p.finish();
        }

        let total = 100 * 1024 * 1024;
        assert_eq!(reports.first(), Some(&(0, total)));
        assert_eq!(reports.last(), Some(&(total, total)));
        // init + 12 throttled updates (8, 16, ..., 96 MiB) + finish.
        assert_eq!(reports.len(), 14);
    }
}
