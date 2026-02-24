use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use arrow_array::{Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::connect;
use lancedb::query::{ExecutableQuery, QueryBase};
use notify::{recommended_watcher, EventKind, RecursiveMode, Watcher};
use rusqlite::{params, params_from_iter, Connection};
use tauri::{Emitter, Manager, State};
use exif::{In, Reader as ExifReader, Tag};

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
fn open_file(path: String) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(["/C", "start", ""])
            .arg(path)
            .spawn()
            .map_err(|err| format!("No se pudo abrir el archivo: {err}"))?;

        Ok(())
    }

    #[cfg(not(target_os = "windows"))]
    {
        Err("Abrir archivo solo está implementado para Windows en esta fase".to_string())
    }
}

#[derive(Serialize)]
struct SearchResultItem {
    title: String,
    path: String,
    snippet: String,
}

#[derive(Serialize)]
struct ImageMetadata {
    path: String,
    width: Option<u32>,
    height: Option<u32>,
    format: Option<String>,
    date_taken: Option<String>,
    orientation: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
struct IndexedFileItem {
    title: String,
    path: String,
    search_key: String,
    size_bytes: u64,
    modified_unix_secs: u64,
    content_excerpt: Option<String>,
}

#[tauri::command]
fn get_image_metadata(path: String) -> Result<ImageMetadata, String> {
    let image_path = PathBuf::from(&path);

    if !image_path.exists() {
        return Err("La imagen no existe".to_string());
    }

    if !is_image_path(&image_path) {
        return Err("El archivo no es una imagen soportada".to_string());
    }

    Ok(read_image_metadata(&image_path))
}

#[derive(Clone, Serialize, Deserialize)]
struct IndexSnapshot {
    files: Vec<IndexedFileItem>,
    roots: Vec<String>,
    indexed_at: String,
}

#[derive(Serialize)]
struct IndexStatus {
    has_index: bool,
    indexed_files: usize,
    indexed_at: Option<String>,
    roots: Vec<String>,
}

#[derive(Clone, Serialize)]
struct IndexProgressEvent {
    phase: String,
    message: String,
    scanned_files: usize,
    indexed_files: usize,
    done: bool,
}

#[derive(Clone, Serialize, Deserialize, Default)]
struct IndexDiagnostics {
    scanned_files: usize,
    indexed_files: usize,
    pdf_scanned: usize,
    pdf_indexed: usize,
    pdf_failed: usize,
    pdf_failed_examples: Vec<String>,
    last_error: Option<String>,
    updated_at: Option<String>,
    canceled: bool,
    lancedb_synced: bool,
    pdf_fallback_used: usize,
}

struct BuildIndexOutcome {
    files: Vec<IndexedFileItem>,
    scanned_files: usize,
    pdf_scanned: usize,
    pdf_indexed: usize,
    pdf_failed: usize,
    pdf_failed_examples: Vec<String>,
    canceled: bool,
    pdf_fallback_used: usize,
}

#[derive(Clone, Serialize, Deserialize, Default)]
struct AiProviderConfig {
    provider: String,
    base_url: String,
    embedding_model: String,
    api_key: String,
}

#[derive(Serialize)]
struct AiProviderStatus {
    configured: bool,
    provider: String,
    base_url: String,
    embedding_model: String,
    api_key_hint: Option<String>,
}

#[derive(Clone)]
struct ChunkCandidate {
    title: String,
    path: String,
    chunk_text: String,
    lexical_score: f32,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingDataItem>,
}

#[derive(Deserialize)]
struct EmbeddingDataItem {
    embedding: Vec<f32>,
}

#[derive(Clone)]
struct IndexingSettings {
    roots: Vec<String>,
    excluded_extensions: Vec<String>,
    excluded_folders: Vec<String>,
    max_file_size_mb: u64,
}

struct FileWatcherRuntime {
    stop_flag: Arc<AtomicBool>,
    thread: std::thread::JoinHandle<()>,
}

#[derive(Clone, Serialize, Default)]
struct FileWatcherStatus {
    running: bool,
    roots: Vec<String>,
    pending_events: bool,
    debounce_ms: u64,
    last_event_at: Option<String>,
    last_reindex_at: Option<String>,
    last_error: Option<String>,
}

struct AppState {
    index: Mutex<Option<IndexSnapshot>>,
    diagnostics: Mutex<Option<IndexDiagnostics>>,
    cancel_indexing: AtomicBool,
    ai_config: Mutex<Option<AiProviderConfig>>,
    index_settings: Mutex<IndexingSettings>,
    watcher_runtime: Mutex<Option<FileWatcherRuntime>>,
    watcher_status: Mutex<FileWatcherStatus>,
    is_indexing: AtomicBool,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            index: Mutex::new(None),
            diagnostics: Mutex::new(None),
            cancel_indexing: AtomicBool::new(false),
            ai_config: Mutex::new(None),
            index_settings: Mutex::new(default_indexing_settings()),
            watcher_runtime: Mutex::new(None),
            watcher_status: Mutex::new(FileWatcherStatus::default()),
            is_indexing: AtomicBool::new(false),
        }
    }
}

#[tauri::command]
fn cancel_indexing(state: State<'_, AppState>) -> Result<(), String> {
    state.cancel_indexing.store(true, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
fn get_index_diagnostics(state: State<'_, AppState>) -> IndexDiagnostics {
    let guard = match state.diagnostics.lock() {
        Ok(value) => value,
        Err(_) => return IndexDiagnostics::default(),
    };

    guard.clone().unwrap_or_default()
}

#[tauri::command]
fn get_file_watcher_status(state: State<'_, AppState>) -> FileWatcherStatus {
    state
        .watcher_status
        .lock()
        .map(|value| value.clone())
        .unwrap_or_default()
}

#[tauri::command]
fn start_file_watcher(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    roots: Option<Vec<String>>,
    excluded_extensions: Option<Vec<String>>,
    excluded_folders: Option<Vec<String>>,
    max_file_size_mb: Option<u64>,
    debounce_ms: Option<u64>,
) -> Result<FileWatcherStatus, String> {
    if state
        .watcher_runtime
        .lock()
        .map(|guard| guard.is_some())
        .unwrap_or(false)
    {
        return get_existing_watcher_status(state.inner());
    }

    let settings = merge_indexing_settings(
        state.inner(),
        roots,
        excluded_extensions,
        excluded_folders,
        max_file_size_mb,
    )?;

    let debounce_value = debounce_ms.unwrap_or(1200).clamp(300, 30_000);
    let root_paths = resolve_roots(Some(settings.roots.clone()));

    if root_paths.is_empty() {
        return Err("No hay carpetas válidas para vigilar".to_string());
    }

    update_watcher_status(state.inner(), |status| {
        status.running = true;
        status.roots = root_paths
            .iter()
            .map(|path| path.to_string_lossy().to_string())
            .collect();
        status.pending_events = false;
        status.debounce_ms = debounce_value;
        status.last_error = None;
    });

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_thread = stop_flag.clone();
    let app_handle = app.clone();
    let watched_roots = root_paths;

    let thread = std::thread::spawn(move || {
        run_file_watcher_loop(&app_handle, watched_roots, debounce_value, stop_flag_thread);
    });

    {
        let mut guard = state
            .watcher_runtime
            .lock()
            .map_err(|_| "No se pudo actualizar watcher runtime".to_string())?;
        *guard = Some(FileWatcherRuntime { stop_flag, thread });
    }

    get_existing_watcher_status(state.inner())
}

#[tauri::command]
fn stop_file_watcher(state: State<'_, AppState>) -> Result<FileWatcherStatus, String> {
    let runtime = {
        let mut guard = state
            .watcher_runtime
            .lock()
            .map_err(|_| "No se pudo acceder al watcher runtime".to_string())?;
        guard.take()
    };

    if let Some(runtime) = runtime {
        runtime.stop_flag.store(true, Ordering::SeqCst);
        let _ = runtime.thread.join();
    }

    update_watcher_status(state.inner(), |status| {
        status.running = false;
        status.pending_events = false;
    });

    get_existing_watcher_status(state.inner())
}

#[tauri::command]
fn configure_ai_provider(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    api_key: String,
    embedding_model: Option<String>,
    base_url: Option<String>,
) -> Result<AiProviderStatus, String> {
    let trimmed_key = api_key.trim().to_string();
    if trimmed_key.is_empty() {
        return Err("API key vacía".to_string());
    }

    let config = AiProviderConfig {
        provider: "openrouter-compatible".to_string(),
        base_url: base_url
            .unwrap_or_else(|| "https://openrouter.ai/api/v1/embeddings".to_string())
            .trim()
            .to_string(),
        embedding_model: embedding_model
            .unwrap_or_else(|| "text-embedding-3-small".to_string())
            .trim()
            .to_string(),
        api_key: trimmed_key,
    };

    save_ai_provider_config(&app, &config)?;

    {
        let mut guard = state
            .ai_config
            .lock()
            .map_err(|_| "No se pudo actualizar la config de IA".to_string())?;
        *guard = Some(config.clone());
    }

    Ok(ai_provider_status_from_config(Some(&config)))
}

#[tauri::command]
fn get_ai_provider_status(state: State<'_, AppState>) -> AiProviderStatus {
    let guard = state.ai_config.lock().ok();
    let maybe_cfg = guard.and_then(|value| value.as_ref().cloned());
    ai_provider_status_from_config(maybe_cfg.as_ref())
}

#[tauri::command]
async fn semantic_search(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    query: String,
    limit: Option<usize>,
    roots: Option<Vec<String>>,
    excluded_extensions: Option<Vec<String>>,
    excluded_folders: Option<Vec<String>>,
    max_file_size_mb: Option<u64>,
) -> Result<Vec<SearchResultItem>, String> {
    let clean_query = query.trim();
    if clean_query.is_empty() {
        return Ok(Vec::new());
    }

    let search_roots = resolve_roots(roots);
    let exclusions = normalize_extensions(excluded_extensions);
    let folder_exclusions = normalize_folder_rules(excluded_folders);
    let max_file_size_bytes = max_size_bytes(max_file_size_mb);

    let mut candidates = match load_chunk_candidates_from_lancedb(
        &app,
        clean_query,
        &search_roots,
        &exclusions,
        &folder_exclusions,
        max_file_size_bytes,
        80,
    )
    .await
    {
        Ok(value) if !value.is_empty() => value,
        _ => load_chunk_candidates(
            &app,
            clean_query,
            &search_roots,
            &exclusions,
            &folder_exclusions,
            max_file_size_bytes,
            80,
        )?,
    };

    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    let ai_cfg = state
        .ai_config
        .lock()
        .ok()
        .and_then(|value| value.as_ref().cloned());

    let max_results = limit.unwrap_or(20).clamp(1, 50);

    if let Some(config) = ai_cfg {
        let mut inputs = Vec::with_capacity(candidates.len() + 1);
        inputs.push(clean_query.to_string());
        for candidate in &candidates {
            let trimmed = candidate.chunk_text.chars().take(1_500).collect::<String>();
            inputs.push(trimmed);
        }

        let vectors = request_embeddings(&config, &inputs).await?;
        if vectors.len() == inputs.len() {
            let query_vector = &vectors[0];
            let mut scored = Vec::new();

            for (index, candidate) in candidates.drain(..).enumerate() {
                let semantic = cosine_similarity(query_vector, &vectors[index + 1]);
                let blended = semantic * 0.72 + candidate.lexical_score * 0.28;
                scored.push((blended, candidate));
            }

            scored.sort_by(|left, right| right.0.total_cmp(&left.0));

            let results = scored
                .into_iter()
                .take(max_results)
                .map(|(score, item)| SearchResultItem {
                    title: item.title,
                    path: item.path,
                    snippet: format!(
                        "Semántico {:.2} · {}",
                        score,
                        build_chunk_snippet(&item.chunk_text, clean_query)
                    ),
                })
                .collect();

            return Ok(results);
        }
    }

    let mut lexical = candidates
        .into_iter()
        .map(|item| SearchResultItem {
            title: item.title,
            path: item.path,
            snippet: format!(
                "Léxico {:.2} · {}",
                item.lexical_score,
                build_chunk_snippet(&item.chunk_text, clean_query)
            ),
        })
        .collect::<Vec<_>>();

    lexical.truncate(max_results);
    Ok(lexical)
}

#[tauri::command]
fn get_index_status(state: State<'_, AppState>) -> IndexStatus {
    let guard = match state.index.lock() {
        Ok(value) => value,
        Err(_) => {
            return IndexStatus {
                has_index: false,
                indexed_files: 0,
                indexed_at: None,
                roots: Vec::new(),
            }
        }
    };

    if let Some(snapshot) = guard.as_ref() {
        return IndexStatus {
            has_index: true,
            indexed_files: snapshot.files.len(),
            indexed_at: Some(snapshot.indexed_at.clone()),
            roots: snapshot.roots.clone(),
        };
    }

    IndexStatus {
        has_index: false,
        indexed_files: 0,
        indexed_at: None,
        roots: Vec::new(),
    }
}

#[tauri::command]
fn start_indexing(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    roots: Option<Vec<String>>,
    excluded_extensions: Option<Vec<String>>,
    excluded_folders: Option<Vec<String>>,
    max_file_size_mb: Option<u64>,
) -> Result<IndexStatus, String> {
    let settings = merge_indexing_settings(
        state.inner(),
        roots,
        excluded_extensions,
        excluded_folders,
        max_file_size_mb,
    )?;

    execute_indexing(&app, state.inner(), settings, "manual")
}

#[tauri::command]
fn search_stub(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    query: &str,
    roots: Option<Vec<String>>,
    excluded_extensions: Option<Vec<String>>,
    excluded_folders: Option<Vec<String>>,
    max_file_size_mb: Option<u64>,
) -> Vec<SearchResultItem> {
    let clean_query = query.trim();

    if clean_query.is_empty() {
        return Vec::new();
    }

    let search_roots = resolve_roots(roots);
    let exclusions = normalize_extensions(excluded_extensions);
    let folder_exclusions = normalize_folder_rules(excluded_folders);
    let max_file_size_bytes = max_size_bytes(max_file_size_mb);

    if let Ok(db_results) = search_in_chunk_db(
        &app,
        clean_query,
        &search_roots,
        &exclusions,
        &folder_exclusions,
        max_file_size_bytes,
        30,
    ) {
        if !db_results.is_empty() {
            return db_results;
        }
    }

    if let Some(index_results) = search_in_index(
        state,
        clean_query,
        &search_roots,
        &exclusions,
        &folder_exclusions,
        max_file_size_bytes,
    ) {
        return index_results;
    }

    search_local_files(
        clean_query,
        &search_roots,
        &exclusions,
        &folder_exclusions,
        max_file_size_bytes,
    )
}

fn execute_indexing(
    app: &tauri::AppHandle,
    state: &AppState,
    settings: IndexingSettings,
    reason: &str,
) -> Result<IndexStatus, String> {
    if state
        .is_indexing
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("Ya hay una indexación en curso".to_string());
    }

    let result = (|| {
        state.cancel_indexing.store(false, Ordering::SeqCst);

        emit_index_progress(
            app,
            "start",
            "Iniciando indexación...",
            0,
            0,
            false,
        );

        let resolved_roots = resolve_roots(Some(settings.roots.clone()));
        let exclusions = normalize_extensions(Some(settings.excluded_extensions.clone()));
        let folder_exclusions = normalize_folder_rules(Some(settings.excluded_folders.clone()));
        let max_file_size_bytes = max_size_bytes(Some(settings.max_file_size_mb));

        let previous_items = state
            .index
            .lock()
            .ok()
            .and_then(|guard| guard.as_ref().map(|snapshot| snapshot.files.clone()));

        let build_outcome = build_index_files(
            app,
            state,
            &resolved_roots,
            &exclusions,
            &folder_exclusions,
            max_file_size_bytes,
            previous_items.as_deref(),
        );

        let diagnostics = IndexDiagnostics {
            scanned_files: build_outcome.scanned_files,
            indexed_files: build_outcome.files.len(),
            pdf_scanned: build_outcome.pdf_scanned,
            pdf_indexed: build_outcome.pdf_indexed,
            pdf_failed: build_outcome.pdf_failed,
            pdf_failed_examples: build_outcome.pdf_failed_examples,
            last_error: if build_outcome.canceled {
                Some("Indexación cancelada por el usuario".to_string())
            } else {
                None
            },
            updated_at: Some(now_timestamp_string()),
            canceled: build_outcome.canceled,
            lancedb_synced: false,
            pdf_fallback_used: build_outcome.pdf_fallback_used,
        };

        if let Ok(mut guard) = state.diagnostics.lock() {
            *guard = Some(diagnostics);
        }

        if build_outcome.canceled {
            return Err("Indexación cancelada por el usuario".to_string());
        }

        let snapshot = IndexSnapshot {
            files: build_outcome.files,
            roots: resolved_roots
                .iter()
                .map(|root| root.to_string_lossy().to_string())
                .collect(),
            indexed_at: now_timestamp_string(),
        };

        if let Err(err) = rebuild_chunk_db(app, &snapshot.files) {
            if let Ok(mut guard) = state.diagnostics.lock() {
                if let Some(current) = guard.as_mut() {
                    current.last_error = Some(format!("Indexado hecho, pero BDD de chunks falló: {err}"));
                    current.updated_at = Some(now_timestamp_string());
                }
            }
        }

        match rebuild_lancedb_index(app, &snapshot.files) {
            Ok(()) => {
                if let Ok(mut guard) = state.diagnostics.lock() {
                    if let Some(current) = guard.as_mut() {
                        current.lancedb_synced = true;
                    }
                }
            }
            Err(err) => {
                if let Ok(mut guard) = state.diagnostics.lock() {
                    if let Some(current) = guard.as_mut() {
                        current.last_error = Some(format!(
                            "Indexado listo, SQLite OK pero LanceDB falló: {err}"
                        ));
                        current.lancedb_synced = false;
                    }
                }
            }
        }

        save_index_snapshot(app, &snapshot);

        {
            let mut guard = state
                .index
                .lock()
                .map_err(|_| "No se pudo bloquear el estado de indexación".to_string())?;
            *guard = Some(snapshot.clone());
        }

        if reason == "watcher" {
            update_watcher_status(state, |status| {
                status.last_reindex_at = Some(now_timestamp_string());
                status.last_error = None;
            });
        }

        Ok(IndexStatus {
            has_index: true,
            indexed_files: snapshot.files.len(),
            indexed_at: Some(snapshot.indexed_at),
            roots: snapshot.roots,
        })
    })();

    state.is_indexing.store(false, Ordering::SeqCst);
    result
}

fn merge_indexing_settings(
    state: &AppState,
    roots: Option<Vec<String>>,
    excluded_extensions: Option<Vec<String>>,
    excluded_folders: Option<Vec<String>>,
    max_file_size_mb: Option<u64>,
) -> Result<IndexingSettings, String> {
    let mut guard = state
        .index_settings
        .lock()
        .map_err(|_| "No se pudo leer config de indexación".to_string())?;

    if let Some(roots) = roots {
        guard.roots = roots;
    }

    if let Some(exts) = excluded_extensions {
        guard.excluded_extensions = exts;
    }

    if let Some(folders) = excluded_folders {
        guard.excluded_folders = folders;
    }

    if let Some(max_mb) = max_file_size_mb {
        guard.max_file_size_mb = max_mb.max(1);
    }

    Ok(guard.clone())
}

fn default_indexing_settings() -> IndexingSettings {
    IndexingSettings {
        roots: Vec::new(),
        excluded_extensions: vec!["mkv".to_string(), "mp4".to_string(), "zip".to_string()],
        excluded_folders: vec![
            "node_modules".to_string(),
            ".git".to_string(),
            "target".to_string(),
            "AppData".to_string(),
        ],
        max_file_size_mb: 128,
    }
}

fn get_existing_watcher_status(state: &AppState) -> Result<FileWatcherStatus, String> {
    state
        .watcher_status
        .lock()
        .map(|value| value.clone())
        .map_err(|_| "No se pudo leer estado del watcher".to_string())
}

fn update_watcher_status<F>(state: &AppState, mut updater: F)
where
    F: FnMut(&mut FileWatcherStatus),
{
    if let Ok(mut guard) = state.watcher_status.lock() {
        updater(&mut guard);
    }
}

fn run_file_watcher_loop(
    app: &tauri::AppHandle,
    roots: Vec<PathBuf>,
    debounce_ms: u64,
    stop_flag: Arc<AtomicBool>,
) {
    let (tx, rx) = std::sync::mpsc::channel::<Result<notify::Event, notify::Error>>();
    let app_handle = app.clone();

    let mut watcher = match recommended_watcher(move |result| {
        let _ = tx.send(result);
    }) {
        Ok(value) => value,
        Err(err) => {
            let app_state = app_handle.state::<AppState>();
            update_watcher_status(app_state.inner(), |status| {
                status.running = false;
                status.last_error = Some(format!("Watcher no pudo iniciar: {err}"));
            });
            return;
        }
    };

    for root in &roots {
        if let Err(err) = watcher.watch(root, RecursiveMode::Recursive) {
            let app_state = app_handle.state::<AppState>();
            update_watcher_status(app_state.inner(), |status| {
                status.last_error = Some(format!("No se pudo vigilar {}: {err}", root.to_string_lossy()));
            });
        }
    }

    let mut pending_events = false;
    let mut last_event_time: Option<Instant> = None;

    while !stop_flag.load(Ordering::Relaxed) {
        match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(Ok(event)) => {
                if !is_relevant_watch_event(&event.kind) {
                    continue;
                }

                pending_events = true;
                last_event_time = Some(Instant::now());

                let app_state = app_handle.state::<AppState>();
                update_watcher_status(app_state.inner(), |status| {
                    status.pending_events = true;
                    status.last_event_at = Some(now_timestamp_string());
                });
            }
            Ok(Err(err)) => {
                let app_state = app_handle.state::<AppState>();
                update_watcher_status(app_state.inner(), |status| {
                    status.last_error = Some(format!("Evento watcher inválido: {err}"));
                });
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                if pending_events
                    && last_event_time
                        .map(|value| value.elapsed() >= Duration::from_millis(debounce_ms))
                        .unwrap_or(false)
                {
                    pending_events = false;
                    last_event_time = None;

                    let app_state = app_handle.state::<AppState>();
                    update_watcher_status(app_state.inner(), |status| {
                        status.pending_events = false;
                    });

                    let settings = app_state
                        .index_settings
                        .lock()
                        .map(|value| value.clone())
                        .unwrap_or_else(|_| default_indexing_settings());

                    if let Err(err) = execute_indexing(&app_handle, app_state.inner(), settings, "watcher") {
                        update_watcher_status(app_state.inner(), |status| {
                            status.last_error = Some(err.clone());
                        });
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    let app_state = app_handle.state::<AppState>();
    update_watcher_status(app_state.inner(), |status| {
        status.running = false;
        status.pending_events = false;
    });
}

fn is_relevant_watch_event(kind: &EventKind) -> bool {
    matches!(
        kind,
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) | EventKind::Any
    )
}

fn search_in_index(
    state: State<'_, AppState>,
    query: &str,
    roots: &[PathBuf],
    excluded_extensions: &[String],
    excluded_folders: &[String],
    max_file_size_bytes: u64,
) -> Option<Vec<SearchResultItem>> {
    const MAX_RESULTS: usize = 30;

    let guard = state.index.lock().ok()?;
    let snapshot = guard.as_ref()?;
    let query_tokens = tokenize_query(query);

    if query_tokens.is_empty() {
        return Some(Vec::new());
    }

    let mut results = Vec::new();

    for item in &snapshot.files {
        if results.len() >= MAX_RESULTS {
            break;
        }

        let path = PathBuf::from(&item.path);
        if !matches_roots(&path, roots) {
            continue;
        }

        if should_skip_custom_dir(&path, excluded_folders) {
            continue;
        }

        if is_excluded_file(&path, excluded_extensions) {
            continue;
        }

        if item.size_bytes > max_file_size_bytes {
            continue;
        }

        let is_match = query_tokens
            .iter()
            .all(|token| item.search_key.contains(token));

        if !is_match {
            continue;
        }

        results.push(SearchResultItem {
            title: item.title.clone(),
            path: item.path.clone(),
            snippet: build_index_snippet(item, &query_tokens),
        });
    }

    Some(results)
}

fn search_local_files(
    query: &str,
    roots: &[PathBuf],
    excluded_extensions: &[String],
    excluded_folders: &[String],
    max_file_size_bytes: u64,
) -> Vec<SearchResultItem> {
    const MAX_RESULTS: usize = 30;
    const MAX_DEPTH: usize = 8;

    let query_tokens = tokenize_query(query);

    if query_tokens.is_empty() {
        return Vec::new();
    }

    let max_scanned_files = if roots.iter().any(|path| is_drive_root(path)) {
        250_000
    } else {
        40_000
    };

    let mut results = Vec::new();
    let mut scanned_files = 0usize;
    let mut stack: Vec<(PathBuf, usize)> = roots
        .iter()
        .cloned()
        .into_iter()
        .map(|path| (path, 0usize))
        .collect();

    while let Some((current_dir, depth)) = stack.pop() {
        if results.len() >= MAX_RESULTS || scanned_files >= max_scanned_files {
            break;
        }

        if depth > MAX_DEPTH || should_skip_dir(&current_dir) || should_skip_custom_dir(&current_dir, excluded_folders) {
            continue;
        }

        let read_dir = match fs::read_dir(&current_dir) {
            Ok(dir) => dir,
            Err(_) => continue,
        };

        let mut subdirs = Vec::new();

        for entry in read_dir.flatten() {
            if results.len() >= MAX_RESULTS || scanned_files >= max_scanned_files {
                break;
            }

            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(kind) => kind,
                Err(_) => continue,
            };

            if file_type.is_dir() {
                if !should_skip_dir(&path) && !should_skip_custom_dir(&path, excluded_folders) {
                    subdirs.push(path);
                }
                continue;
            }

            if !file_type.is_file() {
                continue;
            }

            if is_excluded_file(&path, excluded_extensions) {
                continue;
            }

            let file_size = entry
                .metadata()
                .ok()
                .map(|value| value.len())
                .unwrap_or_default();

            if file_size > max_file_size_bytes {
                continue;
            }

            scanned_files += 1;

            let file_name = match path.file_name().and_then(|name| name.to_str()) {
                Some(name) => name,
                None => continue,
            };

            let lowered_name = file_name.to_lowercase();
            let is_match = query_tokens.iter().all(|token| lowered_name.contains(token));

            if !is_match {
                continue;
            }

            results.push(SearchResultItem {
                title: file_name.to_string(),
                path: path.to_string_lossy().to_string(),
                snippet: "Coincidencia por nombre de archivo (búsqueda local inicial).".to_string(),
            });
        }

        subdirs.sort_by_key(|dir| directory_priority(dir, &query_tokens));
        for dir in subdirs {
            stack.push((dir, depth + 1));
        }
    }

    results
}

fn build_index_files(
    app: &tauri::AppHandle,
    state: &AppState,
    roots: &[PathBuf],
    excluded_extensions: &[String],
    excluded_folders: &[String],
    max_file_size_bytes: u64,
    previous_items: Option<&[IndexedFileItem]>,
) -> BuildIndexOutcome {
    const MAX_INDEXED_FILES: usize = 400_000;
    const MAX_DEPTH: usize = 14;

    let previous_lookup = previous_items
        .unwrap_or(&[])
        .iter()
        .cloned()
        .map(|item| (item.path.clone(), item))
        .collect::<HashMap<String, IndexedFileItem>>();

    let mut indexed = Vec::new();
    let mut scanned_files = 0usize;
    let mut pdf_scanned = 0usize;
    let mut pdf_indexed = 0usize;
    let mut pdf_failed = 0usize;
    let mut pdf_fallback_used = 0usize;
    let mut pdf_failed_examples: Vec<String> = Vec::new();
    let mut stack: Vec<(PathBuf, usize)> = roots
        .iter()
        .cloned()
        .map(|path| (path, 0usize))
        .collect();

    emit_index_progress(
        app,
        "scan",
        "Escaneando carpetas...",
        scanned_files,
        indexed.len(),
        false,
    );

    while let Some((current_dir, depth)) = stack.pop() {
        if state.cancel_indexing.load(Ordering::Relaxed) {
            emit_index_progress(
                app,
                "cancelled",
                "Indexación cancelada",
                scanned_files,
                indexed.len(),
                true,
            );

            return BuildIndexOutcome {
                files: indexed,
                scanned_files,
                pdf_scanned,
                pdf_indexed,
                pdf_failed,
                pdf_failed_examples,
                canceled: true,
                pdf_fallback_used,
            };
        }

        if indexed.len() >= MAX_INDEXED_FILES {
            break;
        }

        if depth > MAX_DEPTH || should_skip_dir(&current_dir) || should_skip_custom_dir(&current_dir, excluded_folders) {
            continue;
        }

        let read_dir = match fs::read_dir(&current_dir) {
            Ok(dir) => dir,
            Err(_) => continue,
        };

        for entry in read_dir.flatten() {
            if state.cancel_indexing.load(Ordering::Relaxed) {
                emit_index_progress(
                    app,
                    "cancelled",
                    "Indexación cancelada",
                    scanned_files,
                    indexed.len(),
                    true,
                );

                return BuildIndexOutcome {
                    files: indexed,
                    scanned_files,
                    pdf_scanned,
                    pdf_indexed,
                    pdf_failed,
                    pdf_failed_examples,
                    canceled: true,
                    pdf_fallback_used,
                };
            }

            if indexed.len() >= MAX_INDEXED_FILES {
                break;
            }

            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(kind) => kind,
                Err(_) => continue,
            };

            if file_type.is_dir() {
                if !should_skip_dir(&path) && !should_skip_custom_dir(&path, excluded_folders) {
                    stack.push((path, depth + 1));
                }
                continue;
            }

            if !file_type.is_file() || is_excluded_file(&path, excluded_extensions) {
                continue;
            }

            scanned_files += 1;

            let extension = path
                .extension()
                .and_then(|value| value.to_str())
                .map(|value| value.to_lowercase())
                .unwrap_or_default();

            let is_pdf = extension == "pdf";

            if is_pdf {
                pdf_scanned += 1;
            }

            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };

            let metadata = match entry.metadata() {
                Ok(value) => value,
                Err(_) => continue,
            };

            let size_bytes = metadata.len();

            if size_bytes > max_file_size_bytes {
                continue;
            }

            let modified_unix_secs = metadata
                .modified()
                .ok()
                .and_then(|value| value.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|value| value.as_secs())
                .unwrap_or_default();

            let full_path = path.to_string_lossy().to_string();

            if let Some(previous) = previous_lookup.get(&full_path) {
                if previous.size_bytes == size_bytes
                    && previous.modified_unix_secs == modified_unix_secs
                    && !needs_content_refresh(&path, previous)
                {
                    if is_pdf {
                        if previous.content_excerpt.is_some() {
                            pdf_indexed += 1;
                        } else {
                            pdf_failed += 1;
                        }
                    }

                    indexed.push(previous.clone());
                    continue;
                }
            }

            let content_excerpt = if is_pdf {
                match extract_pdf_text(&path) {
                    Some((text, used_fallback)) => {
                        if used_fallback {
                            pdf_fallback_used += 1;
                        }

                        let normalized = normalize_text_for_index(&text, 120_000);
                        if normalized.is_empty() {
                            pdf_failed += 1;
                            if pdf_failed_examples.len() < 5 {
                                pdf_failed_examples.push(format!(
                                    "{} (sin texto extraíble)",
                                    full_path
                                ));
                            }
                            None
                        } else {
                            pdf_indexed += 1;
                            Some(normalized)
                        }
                    }
                    None => {
                        pdf_failed += 1;
                        if pdf_failed_examples.len() < 5 {
                            pdf_failed_examples.push(format!(
                                "{} (error al extraer)",
                                full_path
                            ));
                        }
                        None
                    }
                }
            } else {
                extract_indexable_text(&path)
            };
            let mut search_key = file_name.to_lowercase();

            if let Some(content) = &content_excerpt {
                search_key.push(' ');
                search_key.push_str(&content.to_lowercase());
            }

            indexed.push(IndexedFileItem {
                title: file_name.to_string(),
                path: full_path,
                search_key,
                size_bytes,
                modified_unix_secs,
                content_excerpt,
            });

            if scanned_files % 1500 == 0 {
                emit_index_progress(
                    app,
                    "scan",
                    "Indexando archivos...",
                    scanned_files,
                    indexed.len(),
                    false,
                );
            }
        }
    }

    emit_index_progress(
        app,
        "done",
        "Indexación finalizada",
        scanned_files,
        indexed.len(),
        true,
    );

    BuildIndexOutcome {
        files: indexed,
        scanned_files,
        pdf_scanned,
        pdf_indexed,
        pdf_failed,
        pdf_failed_examples,
        canceled: false,
        pdf_fallback_used,
    }
}

fn emit_index_progress(
    app: &tauri::AppHandle,
    phase: &str,
    message: &str,
    scanned_files: usize,
    indexed_files: usize,
    done: bool,
) {
    let payload = IndexProgressEvent {
        phase: phase.to_string(),
        message: message.to_string(),
        scanned_files,
        indexed_files,
        done,
    };

    let _ = app.emit("index-progress", payload);
}

fn needs_content_refresh(path: &Path, previous: &IndexedFileItem) -> bool {
    if previous.content_excerpt.is_some() {
        return false;
    }

    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_lowercase())
        .unwrap_or_default();

    matches!(
        extension.as_str(),
        "txt" | "md" | "pdf" | "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "tiff"
    )
}

fn extract_indexable_text(path: &Path) -> Option<String> {
    const MAX_READ_BYTES: usize = 32 * 1024;

    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_lowercase())?;

    if is_image_extension(&extension) {
        return extract_image_metadata_text(path);
    }

    if extension != "txt" && extension != "md" {
        return None;
    }

    let bytes = fs::read(path).ok()?;
    let slice = &bytes[..bytes.len().min(MAX_READ_BYTES)];
    let content = String::from_utf8_lossy(slice);
    let normalized = normalize_text_for_index(&content, 12_000);

    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

fn extract_image_metadata_text(path: &Path) -> Option<String> {
    let metadata = read_image_metadata(path);

    let mut parts = vec!["imagen".to_string()];

    if let Some(format) = metadata.format {
        parts.push(format.to_lowercase());
    }

    if let (Some(width), Some(height)) = (metadata.width, metadata.height) {
        parts.push(format!("{}x{}", width, height));
        parts.push(format!("{} {}", width, height));
    }

    if let Some(date_taken) = metadata.date_taken {
        parts.push(date_taken.to_lowercase());
    }

    if let Some(orientation) = metadata.orientation {
        parts.push(orientation.to_lowercase());
    }

    let joined = parts.join(" ");
    let normalized = normalize_text_for_index(&joined, 600);
    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

fn read_image_metadata(path: &Path) -> ImageMetadata {
    let dimensions = image::image_dimensions(path).ok();
    let format = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|value| value.to_uppercase());

    let mut date_taken = None;
    let mut orientation = None;

    if let Ok(file) = File::open(path) {
        let mut reader = BufReader::new(file);
        if let Ok(exif) = ExifReader::new().read_from_container(&mut reader) {
            if let Some(field) = exif.get_field(Tag::DateTimeOriginal, In::PRIMARY) {
                date_taken = Some(field.display_value().with_unit(&exif).to_string());
            } else if let Some(field) = exif.get_field(Tag::DateTime, In::PRIMARY) {
                date_taken = Some(field.display_value().with_unit(&exif).to_string());
            }

            if let Some(field) = exif.get_field(Tag::Orientation, In::PRIMARY) {
                orientation = Some(field.display_value().with_unit(&exif).to_string());
            }
        }
    }

    ImageMetadata {
        path: path.to_string_lossy().to_string(),
        width: dimensions.map(|value| value.0),
        height: dimensions.map(|value| value.1),
        format,
        date_taken,
        orientation,
    }
}

fn is_image_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|value| is_image_extension(&value.to_lowercase()))
        .unwrap_or(false)
}

fn is_image_extension(ext: &str) -> bool {
    matches!(ext, "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "tiff")
}

fn extract_pdf_text(path: &Path) -> Option<(String, bool)> {
    let primary = std::panic::catch_unwind(|| pdf_extract::extract_text(path))
        .ok()
        .and_then(|result| result.ok())
        .filter(|text| !text.trim().is_empty());

    if let Some(text) = primary {
        return Some((text, false));
    }

    extract_pdf_text_lopdf(path).map(|text| (text, true))
}

fn extract_pdf_text_lopdf(path: &Path) -> Option<String> {
    let document = lopdf::Document::load(path).ok()?;
    let pages = document.get_pages();
    if pages.is_empty() {
        return None;
    }

    let mut combined = String::new();
    let mut page_numbers = pages.keys().copied().collect::<Vec<u32>>();
    page_numbers.sort_unstable();

    for chunk in page_numbers.chunks(12) {
        let chunk_vec = chunk.to_vec();
        if let Ok(text) = document.extract_text(&chunk_vec) {
            if !text.trim().is_empty() {
                combined.push_str(&text);
                combined.push('\n');
            }
        }

        if combined.len() > 450_000 {
            break;
        }
    }

    let trimmed = combined.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn normalize_text_for_index(content: &str, max_chars: usize) -> String {
    let compact = content.split_whitespace().collect::<Vec<_>>().join(" ");
    compact.chars().take(max_chars).collect()
}

fn build_index_snippet(item: &IndexedFileItem, query_tokens: &[String]) -> String {
    if let Some(content) = &item.content_excerpt {
        let lowered = content.to_lowercase();
        let first_match = query_tokens
            .iter()
            .filter_map(|token| lowered.find(token))
            .min();

        if let Some(match_pos) = first_match {
            let start_char = lowered[..match_pos].chars().count().saturating_sub(40);
            let total_chars = content.chars().count();
            let take_chars = 170usize;
            let end_char = (start_char + take_chars).min(total_chars);

            let excerpt = content
                .chars()
                .skip(start_char)
                .take(end_char.saturating_sub(start_char))
                .collect::<String>();

            return format!("Coincidencia en nombre/contenido indexado: {}", excerpt);
        }

        return format!("Coincidencia en nombre/contenido indexado: {}", content);
    }

    "Coincidencia por nombre desde índice local (más rápido).".to_string()
}

fn semantic_db_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    let path = app.path().app_data_dir().ok()?.join("semantic-chunks.db");

    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    Some(path)
}

fn lancedb_dir_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    let path = app.path().app_data_dir().ok()?.join("semantic-lancedb");
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::create_dir_all(&path);
    Some(path)
}

fn rebuild_lancedb_index(app: &tauri::AppHandle, items: &[IndexedFileItem]) -> Result<(), String> {
    tauri::async_runtime::block_on(rebuild_lancedb_index_async(app, items))
}

async fn rebuild_lancedb_index_async(app: &tauri::AppHandle, items: &[IndexedFileItem]) -> Result<(), String> {
    let db_path = lancedb_dir_path(app).ok_or_else(|| "No se pudo resolver path de LanceDB".to_string())?;
    let uri = db_path.to_string_lossy().to_string();

    let db = connect(&uri)
        .execute()
        .await
        .map_err(|err| format!("No se pudo conectar LanceDB: {err}"))?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("path", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, false),
        Field::new("chunk_index", DataType::Int64, false),
        Field::new("chunk_text", DataType::Utf8, false),
        Field::new("search_key", DataType::Utf8, false),
        Field::new("size_bytes", DataType::Int64, false),
        Field::new("modified_unix_secs", DataType::Int64, false),
        Field::new("updated_unix_secs", DataType::Int64, false),
    ]));

    let mut paths = Vec::<String>::new();
    let mut titles = Vec::<String>::new();
    let mut chunk_indexes = Vec::<i64>::new();
    let mut chunk_texts = Vec::<String>::new();
    let mut search_keys = Vec::<String>::new();
    let mut sizes = Vec::<i64>::new();
    let mut modified = Vec::<i64>::new();
    let mut updated = Vec::<i64>::new();
    let now_unix = now_timestamp_string().parse::<i64>().unwrap_or_default();

    for item in items {
        let source_text = item
            .content_excerpt
            .as_ref()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| item.title.clone());

        let mut chunks = split_text_chunks(&source_text, 900, 180);
        if chunks.is_empty() {
            chunks.push(item.title.clone());
        }

        for (index, chunk) in chunks.into_iter().enumerate() {
            paths.push(item.path.clone());
            titles.push(item.title.clone());
            chunk_indexes.push(index as i64);
            search_keys.push(format!("{} {}", item.title.to_lowercase(), chunk.to_lowercase()));
            chunk_texts.push(chunk);
            sizes.push(item.size_bytes as i64);
            modified.push(item.modified_unix_secs as i64);
            updated.push(now_unix);
        }
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(paths)),
            Arc::new(StringArray::from(titles)),
            Arc::new(Int64Array::from(chunk_indexes)),
            Arc::new(StringArray::from(chunk_texts)),
            Arc::new(StringArray::from(search_keys)),
            Arc::new(Int64Array::from(sizes)),
            Arc::new(Int64Array::from(modified)),
            Arc::new(Int64Array::from(updated)),
        ],
    )
    .map_err(|err| format!("No se pudo construir batch Arrow para LanceDB: {err}"))?;

    let reader = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema.clone());

    let _ = db.drop_table("chunks").await;

    db.create_table("chunks", Box::new(reader))
        .execute()
        .await
        .map_err(|err| format!("No se pudo crear tabla chunks en LanceDB: {err}"))?;

    Ok(())
}

fn open_semantic_connection(app: &tauri::AppHandle) -> Result<Connection, String> {
    let path = semantic_db_path(app).ok_or_else(|| "No se pudo resolver ruta de BDD".to_string())?;
    let conn = Connection::open(path).map_err(|err| format!("No se pudo abrir la BDD local: {err}"))?;
    ensure_semantic_schema(&conn)?;
    Ok(conn)
}

fn ensure_semantic_schema(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS chunks (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          path TEXT NOT NULL,
          title TEXT NOT NULL,
          chunk_index INTEGER NOT NULL,
          chunk_text TEXT NOT NULL,
          search_key TEXT NOT NULL,
          size_bytes INTEGER NOT NULL,
          modified_unix_secs INTEGER NOT NULL,
          updated_unix_secs INTEGER NOT NULL,
          UNIQUE(path, chunk_index)
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_search_key ON chunks(search_key);
        CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
        ",
    )
    .map_err(|err| format!("No se pudo preparar schema de chunks: {err}"))
}

fn rebuild_chunk_db(app: &tauri::AppHandle, items: &[IndexedFileItem]) -> Result<(), String> {
    let mut conn = open_semantic_connection(app)?;
    let tx = conn
        .transaction()
        .map_err(|err| format!("No se pudo abrir transacción de chunks: {err}"))?;

    tx.execute("DELETE FROM chunks", [])
        .map_err(|err| format!("No se pudo limpiar chunks previos: {err}"))?;

    let now_unix = now_timestamp_string().parse::<u64>().unwrap_or_default();

    for item in items {
        let source_text = item
            .content_excerpt
            .as_ref()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| item.title.clone());

        let mut chunks = split_text_chunks(&source_text, 900, 180);
        if chunks.is_empty() {
            chunks.push(item.title.clone());
        }

        for (index, chunk) in chunks.into_iter().enumerate() {
            let lowered_chunk = chunk.to_lowercase();
            let search_key = format!("{} {}", item.title.to_lowercase(), lowered_chunk);

            tx.execute(
                "INSERT INTO chunks (path, title, chunk_index, chunk_text, search_key, size_bytes, modified_unix_secs, updated_unix_secs)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    item.path,
                    item.title,
                    index as i64,
                    chunk,
                    search_key,
                    item.size_bytes as i64,
                    item.modified_unix_secs as i64,
                    now_unix as i64
                ],
            )
            .map_err(|err| format!("No se pudo insertar chunk en BDD: {err}"))?;
        }
    }

    tx.commit()
        .map_err(|err| format!("No se pudo cerrar transacción de chunks: {err}"))
}

fn split_text_chunks(content: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let chars = content.chars().collect::<Vec<_>>();
    if chars.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    let safe_overlap = overlap.min(chunk_size.saturating_sub(1));

    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        let chunk = chars[start..end].iter().collect::<String>();
        let normalized = normalize_text_for_index(&chunk, chunk_size + 200);

        if !normalized.is_empty() {
            chunks.push(normalized);
        }

        if end == chars.len() {
            break;
        }

        start = end.saturating_sub(safe_overlap);
    }

    chunks
}

fn search_in_chunk_db(
    app: &tauri::AppHandle,
    query: &str,
    roots: &[PathBuf],
    excluded_extensions: &[String],
    excluded_folders: &[String],
    max_file_size_bytes: u64,
    max_results: usize,
) -> Result<Vec<SearchResultItem>, String> {
    let candidates = load_chunk_candidates(
        app,
        query,
        roots,
        excluded_extensions,
        excluded_folders,
        max_file_size_bytes,
        max_results,
    )?;

    Ok(candidates
        .into_iter()
        .take(max_results)
        .map(|item| SearchResultItem {
            title: item.title,
            path: item.path,
            snippet: build_chunk_snippet(&item.chunk_text, query),
        })
        .collect())
}

fn load_chunk_candidates(
    app: &tauri::AppHandle,
    query: &str,
    roots: &[PathBuf],
    excluded_extensions: &[String],
    excluded_folders: &[String],
    max_file_size_bytes: u64,
    max_results: usize,
) -> Result<Vec<ChunkCandidate>, String> {
    let tokens = tokenize_query(query);
    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    let conn = open_semantic_connection(app)?;
    let mut sql = String::from(
        "SELECT title, path, chunk_text, search_key, size_bytes FROM chunks",
    );

    let clauses = tokens
        .iter()
        .map(|_| "search_key LIKE ?")
        .collect::<Vec<_>>();

    if !clauses.is_empty() {
        sql.push_str(" WHERE ");
        sql.push_str(&clauses.join(" OR "));
    }

    sql.push_str(" ORDER BY updated_unix_secs DESC LIMIT ?");

    let mut params_vec = tokens
        .iter()
        .map(|token| format!("%{}%", token))
        .collect::<Vec<String>>();
    params_vec.push((max_results.saturating_mul(12).max(80)).to_string());

    let mut stmt = conn
        .prepare(&sql)
        .map_err(|err| format!("No se pudo preparar consulta de chunks: {err}"))?;

    let rows = stmt
        .query_map(params_from_iter(params_vec.iter()), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
            ))
        })
        .map_err(|err| format!("No se pudo ejecutar consulta de chunks: {err}"))?;

    let lowered_query = query.to_lowercase();
    let mut by_path = HashMap::<String, ChunkCandidate>::new();

    for row in rows.flatten() {
        let (title, path, chunk_text, search_key, size_bytes_raw) = row;
        let size_bytes = size_bytes_raw.max(0) as u64;

        if size_bytes > max_file_size_bytes {
            continue;
        }

        let path_buf = PathBuf::from(&path);
        if !matches_roots(&path_buf, roots)
            || should_skip_custom_dir(&path_buf, excluded_folders)
            || is_excluded_file(&path_buf, excluded_extensions)
        {
            continue;
        }

        let mut hits = 0f32;
        for token in &tokens {
            if search_key.contains(token) {
                hits += 1.0;
            }
        }

        if lowered_query.len() > 2 && search_key.contains(&lowered_query) {
            hits += 0.75;
        }

        if hits <= 0.0 {
            continue;
        }

        let lexical_score = (hits / tokens.len() as f32).min(1.6);
        let current = by_path.get(&path).map(|value| value.lexical_score).unwrap_or(-1.0);

        if lexical_score > current {
            by_path.insert(
                path.clone(),
                ChunkCandidate {
                    title,
                    path,
                    chunk_text,
                    lexical_score,
                },
            );
        }
    }

    let mut values = by_path.into_values().collect::<Vec<_>>();
    values.sort_by(|left, right| right.lexical_score.total_cmp(&left.lexical_score));
    values.truncate(max_results.max(1));
    Ok(values)
}

async fn load_chunk_candidates_from_lancedb(
    app: &tauri::AppHandle,
    query: &str,
    roots: &[PathBuf],
    excluded_extensions: &[String],
    excluded_folders: &[String],
    max_file_size_bytes: u64,
    max_results: usize,
) -> Result<Vec<ChunkCandidate>, String> {
    let tokens = tokenize_query(query);
    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    let db_path = lancedb_dir_path(app).ok_or_else(|| "No se pudo resolver path de LanceDB".to_string())?;
    let uri = db_path.to_string_lossy().to_string();

    let db = connect(&uri)
        .execute()
        .await
        .map_err(|err| format!("No se pudo conectar LanceDB para buscar: {err}"))?;

    let table = match db.open_table("chunks").execute().await {
        Ok(value) => value,
        Err(_) => return Ok(Vec::new()),
    };

    let where_clause = tokens
        .iter()
        .map(|token| format!("search_key LIKE '%{}%'", token.replace('\'', "''")))
        .collect::<Vec<_>>()
        .join(" OR ");

    let mut query_builder = table
        .query()
        .limit(max_results.saturating_mul(14).max(120));

    if !where_clause.is_empty() {
        query_builder = query_builder.only_if(where_clause);
    }

    let mut stream = query_builder
        .execute()
        .await
        .map_err(|err| format!("No se pudo ejecutar query LanceDB: {err}"))?;

    let lowered_query = query.to_lowercase();
    let mut by_path = HashMap::<String, ChunkCandidate>::new();

    while let Some(batch) = stream
        .try_next()
        .await
        .map_err(|err| format!("No se pudo leer stream de LanceDB: {err}"))?
    {
        let title_col = batch
            .column_by_name("title")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| "LanceDB: columna title inválida".to_string())?;
        let path_col = batch
            .column_by_name("path")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| "LanceDB: columna path inválida".to_string())?;
        let chunk_text_col = batch
            .column_by_name("chunk_text")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| "LanceDB: columna chunk_text inválida".to_string())?;
        let search_key_col = batch
            .column_by_name("search_key")
            .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            .ok_or_else(|| "LanceDB: columna search_key inválida".to_string())?;
        let size_col = batch
            .column_by_name("size_bytes")
            .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
            .ok_or_else(|| "LanceDB: columna size_bytes inválida".to_string())?;

        for row in 0..batch.num_rows() {
            if title_col.is_null(row)
                || path_col.is_null(row)
                || chunk_text_col.is_null(row)
                || search_key_col.is_null(row)
                || size_col.is_null(row)
            {
                continue;
            }

            let title = title_col.value(row).to_string();
            let path = path_col.value(row).to_string();
            let chunk_text = chunk_text_col.value(row).to_string();
            let search_key = search_key_col.value(row).to_string();
            let size_bytes = size_col.value(row).max(0) as u64;

            if size_bytes > max_file_size_bytes {
                continue;
            }

            let path_buf = PathBuf::from(&path);
            if !matches_roots(&path_buf, roots)
                || should_skip_custom_dir(&path_buf, excluded_folders)
                || is_excluded_file(&path_buf, excluded_extensions)
            {
                continue;
            }

            let mut hits = 0f32;
            for token in &tokens {
                if search_key.contains(token) {
                    hits += 1.0;
                }
            }

            if lowered_query.len() > 2 && search_key.contains(&lowered_query) {
                hits += 0.75;
            }

            if hits <= 0.0 {
                continue;
            }

            let lexical_score = (hits / tokens.len() as f32).min(1.6);
            let current = by_path.get(&path).map(|value| value.lexical_score).unwrap_or(-1.0);

            if lexical_score > current {
                by_path.insert(
                    path.clone(),
                    ChunkCandidate {
                        title,
                        path,
                        chunk_text,
                        lexical_score,
                    },
                );
            }
        }
    }

    let mut values = by_path.into_values().collect::<Vec<_>>();
    values.sort_by(|left, right| right.lexical_score.total_cmp(&left.lexical_score));
    values.truncate(max_results.max(1));
    Ok(values)
}

fn build_chunk_snippet(content: &str, query: &str) -> String {
    if content.is_empty() {
        return "Coincidencia semántica/lexical en chunk indexado.".to_string();
    }

    let lowered_content = content.to_lowercase();
    let tokens = tokenize_query(query);

    let position = tokens
        .iter()
        .filter_map(|token| lowered_content.find(token))
        .min()
        .unwrap_or(0);

    let start_char = lowered_content[..position].chars().count().saturating_sub(60);
    let snippet = content
        .chars()
        .skip(start_char)
        .take(220)
        .collect::<String>();

    format!("Coincidencia en chunk: {}", snippet)
}

fn ai_config_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    let path = app.path().app_data_dir().ok()?.join("ai-config.json");

    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    Some(path)
}

fn save_ai_provider_config(app: &tauri::AppHandle, config: &AiProviderConfig) -> Result<(), String> {
    let path = ai_config_path(app).ok_or_else(|| "No se pudo resolver ruta de config IA".to_string())?;
    let serialized = serde_json::to_string(config)
        .map_err(|err| format!("No se pudo serializar config IA: {err}"))?;
    fs::write(path, serialized).map_err(|err| format!("No se pudo guardar config IA: {err}"))
}

fn load_ai_provider_config(app: &tauri::AppHandle) -> Option<AiProviderConfig> {
    let path = ai_config_path(app)?;
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str::<AiProviderConfig>(&content).ok()
}

fn ai_provider_status_from_config(config: Option<&AiProviderConfig>) -> AiProviderStatus {
    if let Some(cfg) = config {
        return AiProviderStatus {
            configured: true,
            provider: cfg.provider.clone(),
            base_url: cfg.base_url.clone(),
            embedding_model: cfg.embedding_model.clone(),
            api_key_hint: Some(mask_api_key(&cfg.api_key)),
        };
    }

    AiProviderStatus {
        configured: false,
        provider: "openrouter-compatible".to_string(),
        base_url: "https://openrouter.ai/api/v1/embeddings".to_string(),
        embedding_model: "text-embedding-3-small".to_string(),
        api_key_hint: None,
    }
}

fn mask_api_key(value: &str) -> String {
    let chars = value.chars().collect::<Vec<_>>();
    if chars.len() <= 8 {
        return "********".to_string();
    }

    let prefix = chars.iter().take(4).collect::<String>();
    let suffix = chars.iter().skip(chars.len() - 4).collect::<String>();
    format!("{}...{}", prefix, suffix)
}

async fn request_embeddings(config: &AiProviderConfig, input: &[String]) -> Result<Vec<Vec<f32>>, String> {
    let payload = serde_json::json!({
        "model": config.embedding_model,
        "input": input,
    });

    let client = reqwest::Client::new();
    let response = client
        .post(&config.base_url)
        .header("Authorization", format!("Bearer {}", config.api_key))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
        .map_err(|err| format!("No se pudo llamar API de embeddings: {err}"))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format!(
            "Embeddings respondieron {}: {}",
            status.as_u16(),
            body.chars().take(220).collect::<String>()
        ));
    }

    let parsed = response
        .json::<EmbeddingResponse>()
        .await
        .map_err(|err| format!("Respuesta de embeddings inválida: {err}"))?;

    Ok(parsed
        .data
        .into_iter()
        .map(|item| item.embedding)
        .collect())
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.is_empty() || right.is_empty() || left.len() != right.len() {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;

    for (a, b) in left.iter().zip(right.iter()) {
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }

    if left_norm <= 0.0 || right_norm <= 0.0 {
        return 0.0;
    }

    dot / (left_norm.sqrt() * right_norm.sqrt())
}

fn tokenize_query(query: &str) -> Vec<String> {
    query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn matches_roots(path: &Path, roots: &[PathBuf]) -> bool {
    if roots.is_empty() {
        return true;
    }

    roots.iter().any(|root| path.starts_with(root))
}

fn is_drive_root(path: &Path) -> bool {
    match path.to_str() {
        Some(raw) => {
            let normalized = raw.replace('/', "\\");
            normalized.len() == 3
                && normalized.as_bytes().get(1) == Some(&b':')
                && normalized.as_bytes().get(2) == Some(&b'\\')
        }
        None => false,
    }
}

fn directory_priority(path: &Path, query_tokens: &[String]) -> i32 {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return 100;
    };

    let lowered = name.to_lowercase();
    let has_match = query_tokens.iter().any(|token| lowered.contains(token));

    if has_match {
        0
    } else {
        1
    }
}

fn resolve_roots(roots: Option<Vec<String>>) -> Vec<PathBuf> {
    let candidate_roots: Vec<PathBuf> = roots
        .unwrap_or_default()
        .into_iter()
        .map(PathBuf::from)
        .filter(|path| path.exists() && path.is_dir())
        .collect();

    if candidate_roots.is_empty() {
        return default_roots();
    }

    candidate_roots
}

fn normalize_extensions(excluded_extensions: Option<Vec<String>>) -> Vec<String> {
    excluded_extensions
        .unwrap_or_default()
        .into_iter()
        .map(|ext| ext.trim().trim_start_matches('.').to_lowercase())
        .filter(|ext| !ext.is_empty())
        .collect()
}

fn normalize_folder_rules(excluded_folders: Option<Vec<String>>) -> Vec<String> {
    excluded_folders
        .unwrap_or_default()
        .into_iter()
        .map(|name| name.trim().to_lowercase())
        .filter(|name| !name.is_empty())
        .collect()
}

fn should_skip_custom_dir(path: &Path, excluded_folders: &[String]) -> bool {
    if excluded_folders.is_empty() {
        return false;
    }

    let lowered_path = path.to_string_lossy().to_lowercase();
    excluded_folders
        .iter()
        .any(|rule| lowered_path.contains(&format!("\\{}\\", rule)) || lowered_path.ends_with(&format!("\\{}", rule)))
}

fn max_size_bytes(max_file_size_mb: Option<u64>) -> u64 {
    const DEFAULT_MB: u64 = 128;

    let mb = max_file_size_mb.unwrap_or(DEFAULT_MB).max(1);
    mb.saturating_mul(1024 * 1024)
}

fn is_excluded_file(path: &Path, excluded_extensions: &[String]) -> bool {
    if excluded_extensions.is_empty() {
        return false;
    }

    let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };

    let normalized = ext.to_lowercase();
    excluded_extensions.iter().any(|rule| rule == &normalized)
}

fn default_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Ok(user_profile) = std::env::var("USERPROFILE") {
        let user_root = PathBuf::from(user_profile);

        for folder in ["Documents", "Desktop", "Downloads"] {
            let path = user_root.join(folder);
            if path.exists() {
                roots.push(path);
            }
        }

        if roots.is_empty() {
            roots.push(user_root);
        }
    }

    if roots.is_empty() {
        if let Ok(current) = std::env::current_dir() {
            roots.push(current);
        }
    }

    roots
}

fn should_skip_dir(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some("node_modules") | Some("target") | Some(".git")
    )
}

fn now_timestamp_string() -> String {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(duration) => format!("{}", duration.as_secs()),
        Err(_) => "0".to_string(),
    }
}

fn index_cache_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    let path = app.path().app_data_dir().ok()?.join("index-cache.json");

    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    Some(path)
}

fn save_index_snapshot(app: &tauri::AppHandle, snapshot: &IndexSnapshot) {
    let Some(path) = index_cache_path(app) else {
        return;
    };

    let Ok(serialized) = serde_json::to_string(snapshot) else {
        return;
    };

    let _ = fs::write(path, serialized);
}

fn load_index_snapshot(app: &tauri::AppHandle) -> Option<IndexSnapshot> {
    let path = index_cache_path(app)?;
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str::<IndexSnapshot>(&content).ok()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState::default())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            if let Some(snapshot) = load_index_snapshot(app.handle()) {
                let snapshot_roots = snapshot.roots.clone();

                if let Ok(mut guard) = app.state::<AppState>().index.lock() {
                    *guard = Some(snapshot);
                }

                if let Ok(mut settings) = app.state::<AppState>().index_settings.lock() {
                    settings.roots = snapshot_roots;
                }
            }

            if let Some(ai_config) = load_ai_provider_config(app.handle()) {
                if let Ok(mut guard) = app.state::<AppState>().ai_config.lock() {
                    *guard = Some(ai_config);
                }
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            open_file,
            get_image_metadata,
            search_stub,
            semantic_search,
            start_indexing,
            get_index_status,
            get_index_diagnostics,
            cancel_indexing,
            start_file_watcher,
            stop_file_watcher,
            get_file_watcher_status,
            configure_ai_provider,
            get_ai_provider_status
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
