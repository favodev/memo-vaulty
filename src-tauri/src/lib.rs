use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use tauri::{Emitter, Manager, State};

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

#[derive(Clone, Serialize, Deserialize)]
struct IndexedFileItem {
    title: String,
    path: String,
    search_key: String,
    size_bytes: u64,
    modified_unix_secs: u64,
    content_excerpt: Option<String>,
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

#[derive(Default)]
struct AppState {
    index: Mutex<Option<IndexSnapshot>>,
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
    emit_index_progress(
        &app,
        "start",
        "Iniciando indexación...",
        0,
        0,
        false,
    );

    let resolved_roots = resolve_roots(roots);
    let exclusions = normalize_extensions(excluded_extensions);
    let folder_exclusions = normalize_folder_rules(excluded_folders);
    let max_file_size_bytes = max_size_bytes(max_file_size_mb);
    let previous_items = state
        .index
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|snapshot| snapshot.files.clone()));

    let indexed_files = build_index_files(
        &app,
        &resolved_roots,
        &exclusions,
        &folder_exclusions,
        max_file_size_bytes,
        previous_items.as_deref(),
    );

    let snapshot = IndexSnapshot {
        files: indexed_files,
        roots: resolved_roots
            .iter()
            .map(|root| root.to_string_lossy().to_string())
            .collect(),
        indexed_at: now_timestamp_string(),
    };

    save_index_snapshot(&app, &snapshot);

    {
        let mut guard = state
            .index
            .lock()
            .map_err(|_| "No se pudo bloquear el estado de indexación".to_string())?;
        *guard = Some(snapshot.clone());
    }

    Ok(IndexStatus {
        has_index: true,
        indexed_files: snapshot.files.len(),
        indexed_at: Some(snapshot.indexed_at),
        roots: snapshot.roots,
    })
}

#[tauri::command]
fn search_stub(
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
    roots: &[PathBuf],
    excluded_extensions: &[String],
    excluded_folders: &[String],
    max_file_size_bytes: u64,
    previous_items: Option<&[IndexedFileItem]>,
) -> Vec<IndexedFileItem> {
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
                    indexed.push(previous.clone());
                    continue;
                }
            }

            let content_excerpt = extract_indexable_text(&path);
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

    indexed
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

    matches!(extension.as_str(), "txt" | "md" | "pdf")
}

fn extract_indexable_text(path: &Path) -> Option<String> {
    const MAX_READ_BYTES: usize = 32 * 1024;

    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_lowercase())?;

    if extension == "pdf" {
        let text = extract_pdf_text(path)?;
        let normalized = normalize_text_for_index(&text, 120_000);
        return if normalized.is_empty() {
            None
        } else {
            Some(normalized)
        };
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

fn extract_pdf_text(path: &Path) -> Option<String> {
    let extracted = std::panic::catch_unwind(|| pdf_extract::extract_text(path))
        .ok()?
        .ok()?;

    if extracted.trim().is_empty() {
        None
    } else {
        Some(extracted)
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
                if let Ok(mut guard) = app.state::<AppState>().index.lock() {
                    *guard = Some(snapshot);
                }
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            open_file,
            search_stub,
            start_indexing,
            get_index_status
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
