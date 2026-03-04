use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use notify::{recommended_watcher, EventKind, RecursiveMode, Watcher};
use rusqlite::{params, params_from_iter, Connection};
use tauri::{Emitter, Manager, State};

const MAX_TEXT_READ_BYTES: usize = 512 * 1024;
const MAX_DOC_XML_READ_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Serialize)]
struct SearchResultItem {
    title: String,
    path: String,
    snippet: String,
    match_reason: String,
    origin: String,
}

#[derive(Serialize)]
struct FileTextPreview {
    available: bool,
    source: String,
    text: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct IndexedFileItem {
    title: String,
    path: String,
    search_key: String,
    size_bytes: u64,
    modified_unix_secs: u64,
    content_excerpt: Option<String>,
    #[serde(default)]
    content_hash: Option<String>,
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
    chunk_db_synced: bool,
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
    last_event_kind: Option<String>,
    pending_event_count: u64,
    total_event_count: u64,
    last_batch_event_count: u64,
    last_batch_reason: Option<String>,
    last_reindex_at: Option<String>,
    last_error: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
struct ExportableAppConfig {
    version: u32,
    roots: Vec<String>,
    excluded_extensions: Vec<String>,
    excluded_folders: Vec<String>,
    max_file_size_mb: u64,
}

struct AppState {
    index: Mutex<Option<IndexSnapshot>>,
    diagnostics: Mutex<Option<IndexDiagnostics>>,
    cancel_indexing: AtomicBool,
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
            index_settings: Mutex::new(default_indexing_settings()),
            watcher_runtime: Mutex::new(None),
            watcher_status: Mutex::new(FileWatcherStatus::default()),
            is_indexing: AtomicBool::new(false),
        }
    }
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
        Err("Abrir archivo solo está implementado para Windows".to_string())
    }
}

#[tauri::command]
fn get_file_text_preview(path: String, max_chars: Option<usize>) -> Result<FileTextPreview, String> {
    let file_path = PathBuf::from(&path);
    if !file_path.exists() {
        return Err("El archivo no existe".to_string());
    }

    let limit = max_chars.unwrap_or(14_000).clamp(500, 80_000);

    let extension = file_path
        .extension()
        .and_then(|v| v.to_str())
        .map(|v| v.to_lowercase())
        .unwrap_or_default();

    let maybe_text = match extension.as_str() {
        "pdf" => extract_pdf_text(&file_path).map(|(v, _)| v),
        "docx" | "docm" => extract_docx_text(&file_path),
        "odt" | "ods" | "odp" => extract_odt_text(&file_path),
        "rtf" => extract_rtf_text(&file_path),
        "pptx" | "pptm" => extract_pptx_text(&file_path),
        "xlsx" | "xlsm" => extract_xlsx_text(&file_path),
        _ => extract_generic_text(&file_path, &extension),
    };

    if let Some(content) = maybe_text {
        let normalized = normalize_text_for_index(&content, limit);
        return Ok(FileTextPreview {
            available: !normalized.is_empty(),
            source: if extension.is_empty() { "generic".into() } else { extension },
            text: normalized,
        });
    }

    Ok(FileTextPreview {
        available: false,
        source: "unsupported".to_string(),
        text: "No se pudo extraer texto de este archivo.".to_string(),
    })
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
    Ok(search_internal(
        &app,
        state.inner(),
        query.trim(),
        limit.unwrap_or(30).clamp(1, 50),
        roots,
        excluded_extensions,
        excluded_folders,
        max_file_size_mb,
    ))
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
    search_internal(
        &app,
        state.inner(),
        query.trim(),
        30,
        roots,
        excluded_extensions,
        excluded_folders,
        max_file_size_mb,
    )
}

fn search_internal(
    app: &tauri::AppHandle,
    state: &AppState,
    query: &str,
    max_results: usize,
    roots: Option<Vec<String>>,
    excluded_extensions: Option<Vec<String>>,
    excluded_folders: Option<Vec<String>>,
    max_file_size_mb: Option<u64>,
) -> Vec<SearchResultItem> {
    if query.is_empty() {
        return Vec::new();
    }

    let search_roots = resolve_roots(roots);
    let exclusions = normalize_extensions(excluded_extensions);
    let folder_exclusions = normalize_folder_rules(excluded_folders);
    let max_file_size_bytes = max_size_bytes(max_file_size_mb);

    if let Ok(db_results) = search_in_chunk_db(
        app,
        query,
        &search_roots,
        &exclusions,
        &folder_exclusions,
        max_file_size_bytes,
        max_results,
    ) {
        if !db_results.is_empty() {
            return db_results;
        }
    }

    if let Some(index_results) = search_in_index(
        state,
        query,
        &search_roots,
        &exclusions,
        &folder_exclusions,
        max_file_size_bytes,
        max_results,
    ) {
        if !index_results.is_empty() {
            return index_results;
        }
    }

    search_local_files(
        query,
        &search_roots,
        &exclusions,
        &folder_exclusions,
        max_file_size_bytes,
        max_results,
    )
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
fn get_index_status(state: State<'_, AppState>) -> IndexStatus {
    current_index_status(state.inner())
}

#[tauri::command]
fn get_index_diagnostics(state: State<'_, AppState>) -> IndexDiagnostics {
    state
        .diagnostics
        .lock()
        .ok()
        .and_then(|v| v.clone())
        .unwrap_or_default()
}

#[tauri::command]
fn cancel_indexing(state: State<'_, AppState>) -> Result<(), String> {
    state.cancel_indexing.store(true, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
fn clear_index_data(app: tauri::AppHandle, state: State<'_, AppState>) -> Result<IndexStatus, String> {
    clear_index_data_internal(&app, state.inner())
}

#[tauri::command]
fn forget_index_root(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    root: String,
    reindex: Option<bool>,
) -> Result<IndexStatus, String> {
    let normalized = root.trim().to_lowercase();
    if normalized.is_empty() {
        return Err("Ruta de carpeta vacía".to_string());
    }

    let updated_settings = {
        let mut settings = state
            .index_settings
            .lock()
            .map_err(|_| "No se pudo actualizar configuración de indexado".to_string())?;

        let before = settings.roots.len();
        settings.roots.retain(|v| v.trim().to_lowercase() != normalized);
        if settings.roots.len() == before {
            return Err("La carpeta no estaba en la configuración actual".to_string());
        }
        settings.clone()
    };

    if updated_settings.roots.is_empty() {
        return clear_index_data_internal(&app, state.inner());
    }

    if reindex.unwrap_or(true) {
        execute_indexing(&app, state.inner(), updated_settings, "manual")
    } else {
        Ok(current_index_status(state.inner()))
    }
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
        .map(|v| v.is_some())
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
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        status.pending_events = false;
        status.debounce_ms = debounce_value;
        status.last_error = None;
    });

    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_thread = stop_flag.clone();
    let app_handle = app.clone();

    let thread = std::thread::spawn(move || {
        run_file_watcher_loop(&app_handle, root_paths, debounce_value, stop_flag_thread);
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
fn clear_file_watcher_error(state: State<'_, AppState>) -> Result<FileWatcherStatus, String> {
    update_watcher_status(state.inner(), |status| {
        status.last_error = None;
    });
    get_existing_watcher_status(state.inner())
}

#[tauri::command]
fn trigger_watcher_reindex(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<FileWatcherStatus, String> {
    let settings = state
        .index_settings
        .lock()
        .map(|v| v.clone())
        .map_err(|_| "No se pudo leer configuración de indexado".to_string())?;

    execute_indexing(&app, state.inner(), settings, "watcher-manual")?;

    update_watcher_status(state.inner(), |status| {
        status.last_reindex_at = Some(now_timestamp_string());
        status.pending_events = false;
        status.pending_event_count = 0;
        status.last_batch_reason = Some("manual".to_string());
    });

    get_existing_watcher_status(state.inner())
}

#[tauri::command]
fn get_file_watcher_status(state: State<'_, AppState>) -> FileWatcherStatus {
    state
        .watcher_status
        .lock()
        .map(|v| v.clone())
        .unwrap_or_default()
}

#[tauri::command]
fn export_app_config_to_file(
    state: State<'_, AppState>,
    path: String,
    include_secrets: Option<bool>,
) -> Result<String, String> {
    let _ = include_secrets;

    let settings = state
        .index_settings
        .lock()
        .map_err(|_| "No se pudo leer configuración".to_string())?
        .clone();

    let payload = ExportableAppConfig {
        version: 1,
        roots: settings.roots,
        excluded_extensions: settings.excluded_extensions,
        excluded_folders: settings.excluded_folders,
        max_file_size_mb: settings.max_file_size_mb,
    };

    let serialized = serde_json::to_string_pretty(&payload)
        .map_err(|err| format!("No se pudo serializar configuración: {err}"))?;

    let target = PathBuf::from(path);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("No se pudo crear carpeta destino: {err}"))?;
    }

    fs::write(&target, serialized).map_err(|err| format!("No se pudo exportar configuración: {err}"))?;
    Ok(target.to_string_lossy().to_string())
}

#[tauri::command]
fn import_app_config_from_file(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    path: String,
    reindex: Option<bool>,
) -> Result<IndexStatus, String> {
    let source = PathBuf::from(path);
    let content = fs::read_to_string(&source)
        .map_err(|err| format!("No se pudo leer configuración: {err}"))?;

    let parsed = serde_json::from_str::<ExportableAppConfig>(&content)
        .map_err(|err| format!("JSON inválido: {err}"))?;

    if parsed.version != 1 {
        return Err("Versión de configuración no soportada".to_string());
    }

    let sanitized = IndexingSettings {
        roots: parsed.roots,
        excluded_extensions: parsed.excluded_extensions,
        excluded_folders: parsed.excluded_folders,
        max_file_size_mb: parsed.max_file_size_mb.max(1),
    };

    {
        let mut settings = state
            .index_settings
            .lock()
            .map_err(|_| "No se pudo aplicar configuración".to_string())?;
        *settings = sanitized.clone();
    }

    if reindex.unwrap_or(true) && !sanitized.roots.is_empty() {
        execute_indexing(&app, state.inner(), sanitized, "manual")
    } else {
        Ok(current_index_status(state.inner()))
    }
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
        emit_index_progress(app, "start", "Iniciando indexación...", 0, 0, false);

        let resolved_roots = resolve_roots(Some(settings.roots.clone()));
        let exclusions = normalize_extensions(Some(settings.excluded_extensions.clone()));
        let folder_exclusions = normalize_folder_rules(Some(settings.excluded_folders.clone()));
        let max_file_size_bytes = max_size_bytes(Some(settings.max_file_size_mb));

        let previous_items = state
            .index
            .lock()
            .ok()
            .and_then(|v| v.as_ref().map(|s| s.files.clone()));

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
            chunk_db_synced: true,
            pdf_fallback_used: build_outcome.pdf_fallback_used,
        };

        if let Ok(mut d) = state.diagnostics.lock() {
            *d = Some(diagnostics);
        }

        if build_outcome.canceled {
            return Err("Indexación cancelada por el usuario".to_string());
        }

        let snapshot = IndexSnapshot {
            files: build_outcome.files,
            roots: resolved_roots
                .iter()
                .map(|v| v.to_string_lossy().to_string())
                .collect(),
            indexed_at: now_timestamp_string(),
        };

        rebuild_chunk_db(app, &snapshot.files)?;
        save_index_snapshot(app, &snapshot);

        {
            let mut guard = state
                .index
                .lock()
                .map_err(|_| "No se pudo guardar estado de indexación".to_string())?;
            *guard = Some(snapshot.clone());
        }

        if reason.starts_with("watcher") {
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

fn build_index_files(
    app: &tauri::AppHandle,
    state: &AppState,
    roots: &[PathBuf],
    excluded_extensions: &[String],
    excluded_folders: &[String],
    max_file_size_bytes: u64,
    previous_items: Option<&[IndexedFileItem]>,
) -> BuildIndexOutcome {
    const MAX_INDEXED_FILES: usize = 450_000;
    const MAX_DEPTH: usize = 14;

    let previous_lookup: HashMap<String, IndexedFileItem> = previous_items
        .unwrap_or(&[])
        .iter()
        .cloned()
        .map(|item| (item.path.clone(), item))
        .collect();

    let mut indexed = Vec::new();
    let mut scanned_files = 0usize;
    let mut pdf_scanned = 0usize;
    let mut pdf_indexed = 0usize;
    let mut pdf_failed = 0usize;
    let mut pdf_fallback_used = 0usize;
    let mut pdf_failed_examples: Vec<String> = Vec::new();

    let mut stack: Vec<(PathBuf, usize)> = roots.iter().cloned().map(|v| (v, 0usize)).collect();

    emit_index_progress(app, "scan", "Escaneando carpetas...", 0, 0, false);

    while let Some((current_dir, depth)) = stack.pop() {
        if state.cancel_indexing.load(Ordering::Relaxed) {
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

        if depth > MAX_DEPTH || should_skip_dir(&current_dir) || should_skip_custom_dir(&current_dir, excluded_folders) {
            continue;
        }

        let read_dir = match fs::read_dir(&current_dir) {
            Ok(v) => v,
            Err(_) => continue,
        };

        for entry in read_dir.flatten() {
            if indexed.len() >= MAX_INDEXED_FILES {
                break;
            }
            if state.cancel_indexing.load(Ordering::Relaxed) {
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

            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(v) => v,
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

            let metadata = match entry.metadata() {
                Ok(v) => v,
                Err(_) => continue,
            };

            let size_bytes = metadata.len();
            if size_bytes > max_file_size_bytes {
                continue;
            }

            let modified_unix_secs = metadata
                .modified()
                .ok()
                .and_then(|v| v.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|v| v.as_secs())
                .unwrap_or_default();

            let extension = path
                .extension()
                .and_then(|v| v.to_str())
                .map(|v| v.to_lowercase())
                .unwrap_or_default();

            let is_pdf = extension == "pdf";
            if is_pdf {
                pdf_scanned += 1;
            }

            let file_name = match path.file_name().and_then(|v| v.to_str()) {
                Some(v) => v,
                None => continue,
            };

            let full_path = path.to_string_lossy().to_string();
            let content_hash = compute_index_content_hash(&path, &extension);

            if let Some(previous) = previous_lookup.get(&full_path) {
                if previous.size_bytes == size_bytes
                    && previous.modified_unix_secs == modified_unix_secs
                    && previous.content_hash == content_hash
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

            let (extracted, used_pdf_fallback) = if is_pdf {
                match extract_pdf_text(&path) {
                    Some((pdf_text, used_fallback)) => {
                        let normalized = normalize_text_for_index(&pdf_text, 140_000);
                        if normalized.is_empty() {
                            (None, used_fallback)
                        } else {
                            (Some(normalized), used_fallback)
                        }
                    }
                    None => (None, false),
                }
            } else {
                (extract_indexable_text(&path), false)
            };

            if is_pdf {
                if extracted.is_some() {
                    pdf_indexed += 1;
                    if used_pdf_fallback {
                        pdf_fallback_used = pdf_fallback_used.saturating_add(1);
                    }
                } else {
                    pdf_failed += 1;
                    if pdf_failed_examples.len() < 6 {
                        pdf_failed_examples.push(full_path.clone());
                    }
                }
            }

            let mut search_key = format!(
                "{} {} {} {} {}",
                file_name.to_lowercase(),
                full_path.to_lowercase(),
                extension,
                size_bytes,
                modified_unix_secs
            );

            if let Some(content) = &extracted {
                search_key.push(' ');
                search_key.push_str(&content.to_lowercase());
            }

            indexed.push(IndexedFileItem {
                title: file_name.to_string(),
                path: full_path,
                search_key,
                size_bytes,
                modified_unix_secs,
                content_excerpt: extracted,
                content_hash,
            });

            if scanned_files % 1500 == 0 {
                emit_index_progress(
                    app,
                    "scan",
                    "Indexando archivos de texto y metadatos...",
                    scanned_files,
                    indexed.len(),
                    false,
                );
            }
        }
    }

    emit_index_progress(app, "done", "Indexación finalizada", scanned_files, indexed.len(), true);

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

fn search_in_index(
    state: &AppState,
    query: &str,
    roots: &[PathBuf],
    excluded_extensions: &[String],
    excluded_folders: &[String],
    max_file_size_bytes: u64,
    max_results: usize,
) -> Option<Vec<SearchResultItem>> {
    let tokens = tokenize_query(query);
    if tokens.is_empty() {
        return Some(Vec::new());
    }

    let guard = state.index.lock().ok()?;
    let snapshot = guard.as_ref()?;

    let mut scored: Vec<(&IndexedFileItem, f32)> = Vec::new();

    for item in &snapshot.files {
        let path = PathBuf::from(&item.path);
        if !matches_roots(&path, roots)
            || should_skip_custom_dir(&path, excluded_folders)
            || is_excluded_file(&path, excluded_extensions)
            || item.size_bytes > max_file_size_bytes
        {
            continue;
        }

        let score = compute_lexical_score(&item.search_key, &tokens, &query.to_lowercase());
        if score > 0.0 {
            scored.push((item, score));
        }
    }

    scored.sort_by(|a, b| b.1.total_cmp(&a.1));

    Some(
        scored
            .into_iter()
            .take(max_results)
            .map(|(item, score)| SearchResultItem {
                title: item.title.clone(),
                path: item.path.clone(),
                snippet: build_index_snippet(item, &tokens),
                match_reason: format!("Coincidencia léxica en índice local (score {:.2}).", score),
                origin: "local-index".to_string(),
            })
            .collect(),
    )
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
    let tokens = tokenize_query(query);
    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    let conn = open_semantic_connection(app)?;

    let clauses: Vec<&str> = tokens.iter().map(|_| "search_key LIKE ?").collect();
    let mut sql = String::from(
        "SELECT title, path, chunk_text, search_key, size_bytes, modified_unix_secs FROM chunks",
    );
    if !clauses.is_empty() {
        sql.push_str(" WHERE ");
        sql.push_str(&clauses.join(" OR "));
    }
    sql.push_str(" ORDER BY updated_unix_secs DESC LIMIT ?");

    let mut params_vec: Vec<String> = tokens.iter().map(|t| format!("%{}%", t)).collect();
    params_vec.push((max_results.saturating_mul(16)).max(80).to_string());

    let mut stmt = conn
        .prepare(&sql)
        .map_err(|err| format!("No se pudo preparar consulta local: {err}"))?;

    let rows = stmt
        .query_map(params_from_iter(params_vec.iter()), |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
            ))
        })
        .map_err(|err| format!("No se pudo ejecutar consulta local: {err}"))?;

    let lowered_query = query.to_lowercase();
    let mut best_by_path = HashMap::<String, (String, String, f32)>::new();

    for row in rows.flatten() {
        let (title, path, chunk_text, search_key, size_raw, _) = row;
        let size_bytes = size_raw.max(0) as u64;
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

        let score = compute_lexical_score(&search_key, &tokens, &lowered_query);
        if score <= 0.0 {
            continue;
        }

        let current = best_by_path.get(&path).map(|v| v.2).unwrap_or(-1.0);
        if score > current {
            best_by_path.insert(path.clone(), (title, chunk_text, score));
        }
    }

    let mut values: Vec<(String, String, String, f32)> = best_by_path
        .into_iter()
        .map(|(path, (title, chunk, score))| (title, path, chunk, score))
        .collect();

    values.sort_by(|a, b| b.3.total_cmp(&a.3));

    Ok(values
        .into_iter()
        .take(max_results)
        .map(|(title, path, chunk, score)| SearchResultItem {
            title,
            path,
            snippet: build_chunk_snippet(&chunk, query),
            match_reason: format!("Coincidencia léxica local en contenido (score {:.2}).", score),
            origin: "local-chunk".to_string(),
        })
        .collect())
}

fn search_local_files(
    query: &str,
    roots: &[PathBuf],
    excluded_extensions: &[String],
    excluded_folders: &[String],
    max_file_size_bytes: u64,
    max_results: usize,
) -> Vec<SearchResultItem> {
    const MAX_DEPTH: usize = 8;

    let query_tokens = tokenize_query(query);
    if query_tokens.is_empty() {
        return Vec::new();
    }

    let max_scanned_files = if roots.iter().any(|p| is_drive_root(p)) {
        240_000
    } else {
        50_000
    };

    let mut results = Vec::new();
    let mut scanned_files = 0usize;
    let mut stack: Vec<(PathBuf, usize)> = roots.iter().cloned().map(|v| (v, 0usize)).collect();

    while let Some((current_dir, depth)) = stack.pop() {
        if results.len() >= max_results || scanned_files >= max_scanned_files {
            break;
        }

        if depth > MAX_DEPTH || should_skip_dir(&current_dir) || should_skip_custom_dir(&current_dir, excluded_folders) {
            continue;
        }

        let read_dir = match fs::read_dir(&current_dir) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let mut subdirs = Vec::new();

        for entry in read_dir.flatten() {
            if results.len() >= max_results || scanned_files >= max_scanned_files {
                break;
            }

            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(v) => v,
                Err(_) => continue,
            };

            if file_type.is_dir() {
                if !should_skip_dir(&path) && !should_skip_custom_dir(&path, excluded_folders) {
                    subdirs.push(path);
                }
                continue;
            }

            if !file_type.is_file() || is_excluded_file(&path, excluded_extensions) {
                continue;
            }

            let file_size = entry.metadata().ok().map(|m| m.len()).unwrap_or_default();
            if file_size > max_file_size_bytes {
                continue;
            }

            scanned_files += 1;

            let file_name = match path.file_name().and_then(|v| v.to_str()) {
                Some(v) => v,
                None => continue,
            };

            let lowered_name = file_name.to_lowercase();
            if query_tokens.iter().all(|t| lowered_name.contains(t)) {
                results.push(SearchResultItem {
                    title: file_name.to_string(),
                    path: path.to_string_lossy().to_string(),
                    snippet: "Coincidencia por nombre de archivo (búsqueda local).".to_string(),
                    match_reason: "El nombre del archivo contiene todos los términos de búsqueda.".to_string(),
                    origin: "local-filename".to_string(),
                });
                continue;
            }

            let extension = path
                .extension()
                .and_then(|v| v.to_str())
                .map(|v| v.to_lowercase())
                .unwrap_or_default();

            if !is_supported_text_extension(&extension) {
                continue;
            }

            if file_size > 2 * 1024 * 1024 {
                continue;
            }

            if let Some(content) = extract_generic_text(&path, &extension) {
                let lowered = content.to_lowercase();
                if query_tokens.iter().all(|t| lowered.contains(t)) {
                    results.push(SearchResultItem {
                        title: file_name.to_string(),
                        path: path.to_string_lossy().to_string(),
                        snippet: build_local_file_snippet(&content, &query_tokens),
                        match_reason: "Coincidencia por contenido (sin índice activo).".to_string(),
                        origin: "local-content-fallback".to_string(),
                    });
                }
            }
        }

        subdirs.sort_by_key(|d| directory_priority(d, &query_tokens));
        for dir in subdirs {
            stack.push((dir, depth + 1));
        }
    }

    results
}

fn rebuild_chunk_db(app: &tauri::AppHandle, items: &[IndexedFileItem]) -> Result<(), String> {
    let mut conn = open_semantic_connection(app)?;
    ensure_semantic_schema(&conn)?;

    let tx = conn
        .transaction()
        .map_err(|err| format!("No se pudo abrir transacción local: {err}"))?;

    tx.execute("DELETE FROM chunks", [])
        .map_err(|err| format!("No se pudo limpiar chunks: {err}"))?;

    let now_unix = now_timestamp_string().parse::<u64>().unwrap_or_default() as i64;

    for item in items {
        let base = item
            .content_excerpt
            .as_ref()
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| item.title.clone());

        let mut chunks = split_text_chunks(&base, 900, 180);
        if chunks.is_empty() {
            chunks.push(item.title.clone());
        }

        for (chunk_index, chunk_text) in chunks.into_iter().enumerate() {
            let search_key = format!("{} {}", item.title.to_lowercase(), chunk_text.to_lowercase());

            tx.execute(
                "INSERT INTO chunks (path, title, chunk_index, chunk_text, search_key, size_bytes, modified_unix_secs, updated_unix_secs)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    item.path,
                    item.title,
                    chunk_index as i64,
                    chunk_text,
                    search_key,
                    item.size_bytes as i64,
                    item.modified_unix_secs as i64,
                    now_unix,
                ],
            )
            .map_err(|err| format!("No se pudo insertar chunk: {err}"))?;
        }
    }

    tx.commit()
        .map_err(|err| format!("No se pudo cerrar transacción local: {err}"))
}

fn open_semantic_connection(app: &tauri::AppHandle) -> Result<Connection, String> {
    let db_path = semantic_db_path(app).ok_or_else(|| "No se pudo resolver ruta de BDD".to_string())?;
    Connection::open(db_path).map_err(|err| format!("No se pudo abrir BDD local: {err}"))
}

fn ensure_semantic_schema(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS chunks (
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
        CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);",
    )
    .map_err(|err| format!("No se pudo crear schema local: {err}"))
}

fn get_existing_watcher_status(state: &AppState) -> Result<FileWatcherStatus, String> {
    state
        .watcher_status
        .lock()
        .map(|v| v.clone())
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
        Ok(v) => v,
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
    let mut pending_event_count = 0u64;
    let mut pending_priority = 1u8;
    let mut last_event_kind = "modify".to_string();
    let mut last_event_time: Option<Instant> = None;

    while !stop_flag.load(Ordering::Relaxed) {
        match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(Ok(event)) => {
                if !is_relevant_watch_event(&event.kind) {
                    continue;
                }

                let (event_kind, priority) = watcher_event_kind_and_priority(&event.kind);
                pending_events = true;
                pending_event_count = pending_event_count.saturating_add(1);
                pending_priority = pending_priority.max(priority);
                last_event_kind = event_kind.clone();
                last_event_time = Some(Instant::now());

                let app_state = app_handle.state::<AppState>();
                update_watcher_status(app_state.inner(), |status| {
                    status.pending_events = true;
                    status.pending_event_count = pending_event_count;
                    status.total_event_count = status.total_event_count.saturating_add(1);
                    status.last_event_at = Some(now_timestamp_string());
                    status.last_event_kind = Some(event_kind.clone());
                });
            }
            Ok(Err(err)) => {
                let app_state = app_handle.state::<AppState>();
                update_watcher_status(app_state.inner(), |status| {
                    status.last_error = Some(format!("Evento watcher inválido: {err}"));
                });
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                let effective_debounce_ms = match pending_priority {
                    3 => debounce_ms.min(450),
                    2 => debounce_ms.min(700),
                    _ => debounce_ms,
                }
                .max(250);

                if pending_events
                    && last_event_time
                        .map(|v| v.elapsed() >= Duration::from_millis(effective_debounce_ms))
                        .unwrap_or(false)
                {
                    let app_state = app_handle.state::<AppState>();

                    if app_state.is_indexing.load(Ordering::Relaxed) {
                        update_watcher_status(app_state.inner(), |status| {
                            status.pending_events = true;
                            status.last_batch_reason = Some("esperando-indexación-en-curso".to_string());
                        });
                        continue;
                    }

                    let batch_count = pending_event_count;
                    let batch_kind = last_event_kind.clone();
                    pending_events = false;
                    pending_event_count = 0;
                    pending_priority = 1;
                    last_event_time = None;

                    update_watcher_status(app_state.inner(), |status| {
                        status.pending_events = false;
                        status.pending_event_count = 0;
                        status.last_batch_event_count = batch_count;
                        status.last_batch_reason = Some(format!("event:{}", batch_kind));
                    });

                    let settings = app_state
                        .index_settings
                        .lock()
                        .map(|v| v.clone())
                        .unwrap_or_else(|_| default_indexing_settings());

                    if let Err(err) = execute_indexing(&app_handle, app_state.inner(), settings, "watcher") {
                        let err_message = err.clone();
                        update_watcher_status(app_state.inner(), |status| {
                            status.last_error = Some(err_message.clone());
                        });
                    } else {
                        update_watcher_status(app_state.inner(), |status| {
                            status.last_reindex_at = Some(now_timestamp_string());
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
    matches!(kind, EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) | EventKind::Any)
}

fn watcher_event_kind_and_priority(kind: &EventKind) -> (String, u8) {
    match kind {
        EventKind::Remove(_) => ("remove".to_string(), 3),
        EventKind::Create(_) => ("create".to_string(), 2),
        EventKind::Modify(_) => ("modify".to_string(), 1),
        EventKind::Any => ("any".to_string(), 1),
        _ => ("other".to_string(), 1),
    }
}

fn clear_index_data_internal(app: &tauri::AppHandle, state: &AppState) -> Result<IndexStatus, String> {
    state.cancel_indexing.store(true, Ordering::SeqCst);

    if let Ok(mut g) = state.index.lock() {
        *g = None;
    }
    if let Ok(mut g) = state.diagnostics.lock() {
        *g = Some(IndexDiagnostics {
            updated_at: Some(now_timestamp_string()),
            ..IndexDiagnostics::default()
        });
    }

    if let Ok(mut s) = state.index_settings.lock() {
        s.roots.clear();
    }

    if let Some(path) = index_cache_path(app) {
        let _ = fs::remove_file(path);
    }
    if let Some(path) = semantic_db_path(app) {
        let _ = fs::remove_file(path);
    }

    Ok(current_index_status(state))
}

fn current_index_status(state: &AppState) -> IndexStatus {
    let guard = match state.index.lock() {
        Ok(v) => v,
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

fn merge_indexing_settings(
    state: &AppState,
    roots: Option<Vec<String>>,
    excluded_extensions: Option<Vec<String>>,
    excluded_folders: Option<Vec<String>>,
    max_file_size_mb: Option<u64>,
) -> Result<IndexingSettings, String> {
    let mut settings = state
        .index_settings
        .lock()
        .map_err(|_| "No se pudo leer configuración de indexado".to_string())?;

    if let Some(v) = roots {
        settings.roots = v;
    }
    if let Some(v) = excluded_extensions {
        settings.excluded_extensions = v;
    }
    if let Some(v) = excluded_folders {
        settings.excluded_folders = v;
    }
    if let Some(v) = max_file_size_mb {
        settings.max_file_size_mb = v.max(1);
    }

    Ok(settings.clone())
}

fn default_indexing_settings() -> IndexingSettings {
    IndexingSettings {
        roots: Vec::new(),
        excluded_extensions: Vec::new(),
        excluded_folders: vec!["node_modules".into(), ".git".into(), "target".into(), "AppData".into()],
        max_file_size_mb: 128,
    }
}

fn extract_indexable_text(path: &Path) -> Option<String> {
    let extension = path
        .extension()
        .and_then(|v| v.to_str())
        .map(|v| v.to_lowercase())
        .unwrap_or_default();

    let content = match extension.as_str() {
        "pdf" => extract_pdf_text(path).map(|(v, _)| v),
        "docx" | "docm" => extract_docx_text(path),
        "odt" | "ods" | "odp" => extract_odt_text(path),
        "rtf" => extract_rtf_text(path),
        "pptx" | "pptm" => extract_pptx_text(path),
        "xlsx" | "xlsm" => extract_xlsx_text(path),
        _ => extract_generic_text(path, &extension),
    }?;

    let normalized = normalize_text_for_index(&content, 140_000);
    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

fn extract_generic_text(path: &Path, extension: &str) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let slice = &bytes[..bytes.len().min(MAX_TEXT_READ_BYTES)];

    if !is_supported_text_extension(extension) && !is_probably_text_content(slice) {
        return None;
    }

    let decoded = decode_text_bytes(slice);
    if decoded.trim().is_empty() {
        None
    } else {
        Some(decoded)
    }
}

fn extract_pdf_text(path: &Path) -> Option<(String, bool)> {
    let primary = std::panic::catch_unwind(|| pdf_extract::extract_text(path))
        .ok()
        .and_then(|v| v.ok())
        .filter(|v| !v.trim().is_empty());

    if let Some(text) = primary {
        return Some((text, false));
    }

    extract_pdf_text_lopdf(path).map(|v| (v, true))
}

fn extract_pdf_text_lopdf(path: &Path) -> Option<String> {
    let document = lopdf::Document::load(path).ok()?;
    let pages = document.get_pages();
    if pages.is_empty() {
        return None;
    }

    let mut page_numbers: Vec<u32> = pages.keys().copied().collect();
    page_numbers.sort_unstable();

    let mut combined = String::new();

    for chunk in page_numbers.chunks(10) {
        let chunk_vec = chunk.to_vec();
        if let Ok(text) = document.extract_text(&chunk_vec) {
            if !text.trim().is_empty() {
                combined.push_str(&text);
                combined.push('\n');
            }
        }
        if combined.len() > 420_000 {
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

fn extract_docx_text(path: &Path) -> Option<String> {
    let file = File::open(path).ok()?;
    let mut archive = zip::ZipArchive::new(file).ok()?;

    let xml_entries = [
        "word/document.xml",
        "word/header1.xml",
        "word/header2.xml",
        "word/header3.xml",
        "word/footer1.xml",
        "word/footer2.xml",
        "word/footer3.xml",
        "word/footnotes.xml",
        "word/endnotes.xml",
        "word/comments.xml",
    ];

    let mut combined = String::new();

    for entry in xml_entries {
        if let Ok(mut xml) = archive.by_name(entry) {
            if xml.size() > MAX_DOC_XML_READ_BYTES {
                continue;
            }
            let mut content = String::new();
            if xml.read_to_string(&mut content).is_ok() {
                let text = extract_text_from_xml(&content);
                if !text.trim().is_empty() {
                    if !combined.is_empty() {
                        combined.push('\n');
                    }
                    combined.push_str(&text);
                }
            }
        }
    }

    if combined.trim().is_empty() {
        None
    } else {
        Some(combined)
    }
}

fn extract_odt_text(path: &Path) -> Option<String> {
    extract_odf_package_text(path)
}

fn extract_odf_package_text(path: &Path) -> Option<String> {
    let file = File::open(path).ok()?;
    let mut archive = zip::ZipArchive::new(file).ok()?;
    let mut xml = archive.by_name("content.xml").ok()?;

    if xml.size() > MAX_DOC_XML_READ_BYTES {
        return None;
    }

    let mut content = String::new();
    xml.read_to_string(&mut content).ok()?;

    let text = extract_text_from_xml(&content);
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

fn extract_pptx_text(path: &Path) -> Option<String> {
    let file = File::open(path).ok()?;
    let mut archive = zip::ZipArchive::new(file).ok()?;
    let mut combined = String::new();

    for index in 0..archive.len() {
        let mut entry = match archive.by_index(index) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let name = entry.name().to_string();
        if !name.starts_with("ppt/slides/slide") || !name.ends_with(".xml") {
            continue;
        }
        if entry.size() > MAX_DOC_XML_READ_BYTES {
            continue;
        }
        let mut xml = String::new();
        if entry.read_to_string(&mut xml).is_ok() {
            let text = extract_text_from_xml(&xml);
            if !text.trim().is_empty() {
                if !combined.is_empty() {
                    combined.push('\n');
                }
                combined.push_str(&text);
            }
        }
    }

    if combined.trim().is_empty() {
        None
    } else {
        Some(combined)
    }
}

fn extract_xlsx_text(path: &Path) -> Option<String> {
    let file = File::open(path).ok()?;
    let mut archive = zip::ZipArchive::new(file).ok()?;
    let mut combined = String::new();

    if let Ok(mut shared) = archive.by_name("xl/sharedStrings.xml") {
        if shared.size() <= MAX_DOC_XML_READ_BYTES {
            let mut xml = String::new();
            if shared.read_to_string(&mut xml).is_ok() {
                let text = extract_text_from_xml(&xml);
                if !text.trim().is_empty() {
                    combined.push_str(&text);
                    combined.push('\n');
                }
            }
        }
    }

    for index in 0..archive.len() {
        let mut entry = match archive.by_index(index) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let name = entry.name().to_string();
        if !name.starts_with("xl/worksheets/") || !name.ends_with(".xml") {
            continue;
        }
        if entry.size() > MAX_DOC_XML_READ_BYTES {
            continue;
        }
        let mut xml = String::new();
        if entry.read_to_string(&mut xml).is_ok() {
            let text = extract_text_from_xml(&xml);
            if !text.trim().is_empty() {
                combined.push_str(&text);
                combined.push('\n');
            }
        }
    }

    if combined.trim().is_empty() {
        None
    } else {
        Some(combined)
    }
}

fn extract_rtf_text(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let slice = &bytes[..bytes.len().min(MAX_TEXT_READ_BYTES)];
    let content = String::from_utf8_lossy(slice);

    let mut output = String::new();
    let chars: Vec<char> = content.chars().collect();
    let mut index = 0usize;

    while index < chars.len() {
        let current = chars[index];
        if current == '\\' {
            index += 1;
            if index >= chars.len() {
                break;
            }

            let escaped = chars[index];
            if escaped == '\\' || escaped == '{' || escaped == '}' {
                output.push(escaped);
                index += 1;
                continue;
            }

            while index < chars.len() && chars[index].is_alphabetic() {
                index += 1;
            }
            if index < chars.len() && (chars[index] == '-' || chars[index].is_ascii_digit()) {
                index += 1;
                while index < chars.len() && chars[index].is_ascii_digit() {
                    index += 1;
                }
            }
            if index < chars.len() && chars[index] == ' ' {
                index += 1;
            }
            continue;
        }

        if current != '{' && current != '}' {
            output.push(current);
        }
        index += 1;
    }

    if output.trim().is_empty() {
        None
    } else {
        Some(output)
    }
}

fn extract_text_from_xml(xml: &str) -> String {
    let mut output = String::new();
    let mut inside_tag = false;
    let mut tag_buffer = String::new();

    for ch in xml.chars() {
        if inside_tag {
            if ch == '>' {
                let lowered = tag_buffer.trim().to_lowercase();
                if lowered.starts_with("w:p")
                    || lowered.starts_with("/w:p")
                    || lowered.starts_with("text:p")
                    || lowered.starts_with("/text:p")
                    || lowered.starts_with("text:h")
                    || lowered.starts_with("/text:h")
                    || lowered.starts_with("a:p")
                    || lowered.starts_with("/a:p")
                    || lowered.starts_with("a:br")
                    || lowered.starts_with("text:line-break")
                    || lowered.starts_with("w:br")
                {
                    output.push('\n');
                } else if lowered.starts_with("w:tab") || lowered.starts_with("text:tab") {
                    output.push('\t');
                }
                tag_buffer.clear();
                inside_tag = false;
                continue;
            }
            tag_buffer.push(ch);
            continue;
        }

        if ch == '<' {
            inside_tag = true;
            continue;
        }

        output.push(ch);
    }

    decode_xml_entities(&output)
}

fn decode_xml_entities(value: &str) -> String {
    value
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&#10;", "\n")
        .replace("&#9;", "\t")
}

fn is_supported_text_extension(extension: &str) -> bool {
    matches!(
        extension,
        "txt"
            | "md"
            | "markdown"
            | "rst"
            | "adoc"
            | "text"
            | "log"
            | "csv"
            | "tsv"
            | "json"
            | "jsonl"
            | "yaml"
            | "yml"
            | "toml"
            | "ini"
            | "cfg"
            | "conf"
            | "env"
            | "properties"
            | "xml"
            | "html"
            | "htm"
            | "css"
            | "scss"
            | "sass"
            | "less"
            | "js"
            | "jsx"
            | "ts"
            | "tsx"
            | "mjs"
            | "cjs"
            | "py"
            | "pyi"
            | "ipynb"
            | "java"
            | "kt"
            | "kts"
            | "scala"
            | "go"
            | "rs"
            | "c"
            | "h"
            | "cpp"
            | "cc"
            | "cxx"
            | "hpp"
            | "cs"
            | "swift"
            | "m"
            | "mm"
            | "php"
            | "rb"
            | "pl"
            | "lua"
            | "r"
            | "jl"
            | "dart"
            | "sh"
            | "bash"
            | "zsh"
            | "ps1"
            | "psm1"
            | "cmd"
            | "bat"
            | "sql"
            | "graphql"
            | "gql"
            | "vue"
            | "svelte"
            | "dockerfile"
            | "makefile"
            | "gitignore"
            | "gitattributes"
            | "editorconfig"
            | "lock"
            | "tex"
            | "bib"
            | "srt"
            | "vtt"
            | "po"
            | "docx"
            | "docm"
            | "odt"
            | "ods"
            | "odp"
            | "rtf"
            | "pdf"
            | "pptx"
            | "pptm"
            | "xlsx"
            | "xlsm"
            | "tf"
            | "tfvars"
            | "hcl"
    )
}

fn is_probably_text_content(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }

    if bytes.starts_with(&[0xFF, 0xFE]) || bytes.starts_with(&[0xFE, 0xFF]) {
        return true;
    }

    let mut control_count = 0usize;
    let mut nul_count = 0usize;

    for byte in bytes {
        if *byte == 0 {
            nul_count += 1;
        }
        if (*byte < 0x09) || (*byte > 0x0D && *byte < 0x20) {
            control_count += 1;
        }
    }

    let len = bytes.len().max(1);
    let control_ratio = control_count as f32 / len as f32;
    let nul_ratio = nul_count as f32 / len as f32;

    control_ratio < 0.03 && nul_ratio < 0.01
}

fn decode_text_bytes(bytes: &[u8]) -> String {
    if bytes.starts_with(&[0xFF, 0xFE]) {
        let mut units = Vec::with_capacity(bytes.len() / 2);
        for chunk in bytes[2..].chunks_exact(2) {
            units.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }
        return String::from_utf16_lossy(&units);
    }

    if bytes.starts_with(&[0xFE, 0xFF]) {
        let mut units = Vec::with_capacity(bytes.len() / 2);
        for chunk in bytes[2..].chunks_exact(2) {
            units.push(u16::from_be_bytes([chunk[0], chunk[1]]));
        }
        return String::from_utf16_lossy(&units);
    }

    String::from_utf8_lossy(bytes).to_string()
}

fn normalize_text_for_index(content: &str, max_chars: usize) -> String {
    let compact: String = content.split_whitespace().collect::<Vec<_>>().join(" ");
    compact.chars().take(max_chars).collect()
}

fn tokenize_query(query: &str) -> Vec<String> {
    query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn compute_lexical_score(search_key: &str, tokens: &[String], lowered_query: &str) -> f32 {
    if tokens.is_empty() {
        return 0.0;
    }

    let mut hits = 0.0;
    for token in tokens {
        if search_key.contains(token) {
            hits += 1.0;
        }
    }

    if lowered_query.len() > 2 && search_key.contains(lowered_query) {
        hits += 0.75;
    }

    if hits <= 0.0 {
        return 0.0;
    }

    (hits / tokens.len() as f32).min(1.6)
}

fn build_chunk_snippet(content: &str, query: &str) -> String {
    if content.is_empty() {
        return "Coincidencia léxica en chunk indexado.".to_string();
    }

    let lowered_content = content.to_lowercase();
    let tokens = tokenize_query(query);
    let position = tokens
        .iter()
        .filter_map(|token| lowered_content.find(token))
        .min()
        .unwrap_or(0);

    let start_char = lowered_content[..position].chars().count().saturating_sub(60);
    let snippet: String = content.chars().skip(start_char).take(220).collect();

    format!("Coincidencia en chunk: {}", snippet)
}

fn build_local_file_snippet(content: &str, query_tokens: &[String]) -> String {
    if content.is_empty() {
        return "Coincidencia por contenido local.".to_string();
    }

    let lowered = content.to_lowercase();
    let position = query_tokens
        .iter()
        .filter_map(|token| lowered.find(token))
        .min()
        .unwrap_or(0);

    let start_char = lowered[..position].chars().count().saturating_sub(55);
    let snippet: String = content.chars().skip(start_char).take(200).collect();
    format!("Coincidencia en contenido local: {}", normalize_text_for_index(&snippet, 200))
}

fn build_index_snippet(item: &IndexedFileItem, query_tokens: &[String]) -> String {
    if let Some(content) = &item.content_excerpt {
        let lowered = content.to_lowercase();
        let first_match = query_tokens.iter().filter_map(|t| lowered.find(t)).min();

        if let Some(match_pos) = first_match {
            let start_char = lowered[..match_pos].chars().count().saturating_sub(40);
            let total_chars = content.chars().count();
            let end_char = (start_char + 170).min(total_chars);
            let excerpt: String = content
                .chars()
                .skip(start_char)
                .take(end_char.saturating_sub(start_char))
                .collect();
            return format!("Coincidencia en contenido: {}", excerpt);
        }

        return format!(
            "Coincidencia en contenido: {}",
            content.chars().take(170).collect::<String>()
        );
    }

    "Coincidencia por nombre desde índice local.".to_string()
}

fn split_text_chunks(content: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    let chars: Vec<char> = content.chars().collect();
    if chars.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    let safe_overlap = overlap.min(chunk_size.saturating_sub(1));

    while start < chars.len() {
        let end = (start + chunk_size).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
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

fn needs_content_refresh(path: &Path, previous: &IndexedFileItem) -> bool {
    if previous.content_excerpt.is_some() {
        return false;
    }

    let extension = path
        .extension()
        .and_then(|v| v.to_str())
        .map(|v| v.to_lowercase())
        .unwrap_or_default();

    extension == "pdf"
        || extension == "docx"
        || extension == "docm"
        || extension == "odt"
        || extension == "ods"
        || extension == "odp"
        || extension == "rtf"
        || extension == "pptx"
        || extension == "pptm"
        || extension == "xlsx"
        || extension == "xlsm"
        || is_supported_text_extension(&extension)
}

fn compute_index_content_hash(path: &Path, extension: &str) -> Option<String> {
    const SAMPLE_BYTES: usize = 24 * 1024;

    let mut file = File::open(path).ok()?;
    let mut buffer = vec![0u8; SAMPLE_BYTES];
    let read_bytes = file.read(&mut buffer).ok()?;
    buffer.truncate(read_bytes);

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    extension.hash(&mut hasher);
    buffer.hash(&mut hasher);

    Some(format!("{:016x}", hasher.finish()))
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

fn resolve_roots(roots: Option<Vec<String>>) -> Vec<PathBuf> {
    let candidate_roots: Vec<PathBuf> = roots
        .unwrap_or_default()
        .into_iter()
        .map(PathBuf::from)
        .filter(|p| p.exists() && p.is_dir())
        .collect();

    if candidate_roots.is_empty() {
        default_roots()
    } else {
        candidate_roots
    }
}

fn normalize_extensions(excluded_extensions: Option<Vec<String>>) -> Vec<String> {
    excluded_extensions
        .unwrap_or_default()
        .into_iter()
        .map(|value| value.trim().trim_start_matches('.').to_lowercase())
        .filter(|value| !value.is_empty())
        .collect()
}

fn normalize_folder_rules(excluded_folders: Option<Vec<String>>) -> Vec<String> {
    excluded_folders
        .unwrap_or_default()
        .into_iter()
        .map(|value| value.trim().to_lowercase())
        .filter(|value| !value.is_empty())
        .collect()
}

fn matches_roots(path: &Path, roots: &[PathBuf]) -> bool {
    if roots.is_empty() {
        return true;
    }
    roots.iter().any(|root| path.starts_with(root))
}

fn should_skip_dir(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|n| n.to_str()),
        Some("node_modules") | Some("target") | Some(".git")
    )
}

fn should_skip_custom_dir(path: &Path, excluded_folders: &[String]) -> bool {
    if excluded_folders.is_empty() {
        return false;
    }

    let lowered = path.to_string_lossy().to_lowercase();
    excluded_folders.iter().any(|rule| {
        lowered.contains(&format!("\\{}\\", rule)) || lowered.ends_with(&format!("\\{}", rule))
    })
}

fn is_excluded_file(path: &Path, excluded_extensions: &[String]) -> bool {
    if excluded_extensions.is_empty() {
        return false;
    }

    let extension = match path.extension().and_then(|v| v.to_str()) {
        Some(v) => v.to_lowercase(),
        None => return false,
    };

    excluded_extensions.iter().any(|rule| rule == &extension)
}

fn max_size_bytes(max_file_size_mb: Option<u64>) -> u64 {
    let mb = max_file_size_mb.unwrap_or(128).max(1);
    mb.saturating_mul(1024 * 1024)
}

fn default_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Ok(user_profile) = std::env::var("USERPROFILE") {
        let base = PathBuf::from(user_profile);
        for folder in ["Documents", "Desktop", "Downloads"] {
            let path = base.join(folder);
            if path.exists() {
                roots.push(path);
            }
        }
        if roots.is_empty() {
            roots.push(base);
        }
    }

    if roots.is_empty() {
        if let Ok(current) = std::env::current_dir() {
            roots.push(current);
        }
    }

    roots
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
    let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
        return 100;
    };

    let lowered = name.to_lowercase();
    if query_tokens.iter().any(|t| lowered.contains(t)) {
        0
    } else {
        1
    }
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

fn semantic_db_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    let path = app.path().app_data_dir().ok()?.join("semantic-chunks.db");
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
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
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
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            open_file,
            get_file_text_preview,
            semantic_search,
            search_stub,
            start_indexing,
            get_index_status,
            get_index_diagnostics,
            cancel_indexing,
            clear_index_data,
            forget_index_root,
            start_file_watcher,
            stop_file_watcher,
            clear_file_watcher_error,
            trigger_watcher_reindex,
            get_file_watcher_status,
            export_app_config_to_file,
            import_app_config_from_file,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
