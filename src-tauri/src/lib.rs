use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[derive(Serialize)]
struct SearchResultItem {
    title: String,
    path: String,
    snippet: String,
}

#[tauri::command]
fn search_stub(
    query: &str,
    roots: Option<Vec<String>>,
    excluded_extensions: Option<Vec<String>>,
) -> Vec<SearchResultItem> {
    let clean_query = query.trim();

    if clean_query.is_empty() {
        return Vec::new();
    }

    let search_roots = resolve_roots(roots);
    let exclusions = normalize_extensions(excluded_extensions);

    search_local_files(clean_query, &search_roots, &exclusions)
}

fn search_local_files(
    query: &str,
    roots: &[PathBuf],
    excluded_extensions: &[String],
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

        if depth > MAX_DEPTH || should_skip_dir(&current_dir) {
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
                if !should_skip_dir(&path) {
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

fn tokenize_query(query: &str) -> Vec<String> {
    query
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(ToOwned::to_owned)
        .collect()
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![greet, search_stub])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
