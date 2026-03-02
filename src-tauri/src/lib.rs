use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use base64::Engine;
use std::fs::File;
use std::fs;
use std::io::{BufReader, Read};
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
use tauri::path::BaseDirectory;
use tauri::{Emitter, Manager, State};
use exif::{In, Reader as ExifReader, Tag};
use tract_onnx::prelude::*;
use sysinfo::System;
use tokenizers::Tokenizer;

const TEXT_EMBEDDING_CACHE_MAX_ITEMS: usize = 3500;
const CLIP_IMAGE_CACHE_MAX_ITEMS: usize = 2200;
const CLOUD_RETRY_ATTEMPTS: usize = 3;
const SEMANTIC_SCHEMA_VERSION: i64 = 2;
const OCR_MAX_IMAGE_BYTES: u64 = 3 * 1024 * 1024;
const MAX_TEXT_READ_BYTES: usize = 512 * 1024;
const MAX_DOC_XML_READ_BYTES: u64 = 3 * 1024 * 1024;

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
    match_reason: String,
    origin: String,
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

#[derive(Serialize)]
struct FileTextPreview {
    available: bool,
    source: String,
    text: String,
}

#[derive(Serialize)]
struct FileVisualPreview {
    available: bool,
    mime: String,
    data_url: Option<String>,
    size_bytes: u64,
    reason: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
struct RagSourceItem {
    #[serde(default)]
    ref_id: String,
    title: String,
    path: String,
    snippet: String,
    score: f32,
}

#[derive(Serialize)]
struct RagAnswerResponse {
    answer: String,
    grounded: bool,
    mode: String,
    sources: Vec<RagSourceItem>,
}

#[derive(Clone, Serialize, Deserialize)]
struct ChatHistoryItem {
    id: String,
    timestamp: String,
    query: String,
    answer: String,
    #[serde(default)]
    grounded: bool,
    #[serde(default = "default_mode_value")]
    mode: String,
    #[serde(default)]
    sources: Vec<RagSourceItem>,
}

#[derive(Clone, Serialize, Deserialize)]
struct AuditLogEntry {
    timestamp: String,
    event: String,
    detail: String,
}

#[derive(Default)]
struct RuntimeMetricsState {
    semantic_calls: u64,
    semantic_total_ms: u64,
    last_semantic_ms: u64,
    rag_calls: u64,
    rag_total_ms: u64,
    last_rag_ms: u64,
    indexing_runs: u64,
    indexing_total_ms: u64,
    last_index_ms: u64,
    embedding_cache_hits: u64,
    embedding_cache_misses: u64,
}

#[derive(Serialize)]
struct RuntimeMetrics {
    semantic_calls: u64,
    semantic_avg_ms: f32,
    rag_calls: u64,
    rag_avg_ms: f32,
    indexing_runs: u64,
    indexing_avg_ms: f32,
    last_index_ms: u64,
    embedding_cache_hits: u64,
    embedding_cache_misses: u64,
    embedding_cache_hit_rate: f32,
    embedding_cache_items: usize,
    last_semantic_ms: u64,
    last_rag_ms: u64,
}

#[derive(Serialize)]
struct PerformanceTelemetry {
    memory_total_gb: f32,
    memory_available_gb: f32,
    memory_pressure_pct: f32,
    cpu_usage_pct: f32,
    pressure_level: String,
    semantic_last_ms: u64,
    rag_last_ms: u64,
    indexing_last_ms: u64,
    adaptive_enabled: bool,
    throttling_factor: f32,
}

#[derive(Serialize)]
struct EmbeddingCacheStatus {
    enabled: bool,
    items: usize,
    hits: u64,
    misses: u64,
    hit_rate: f32,
}

#[derive(Serialize)]
struct HardwareProfile {
    cpu_cores: usize,
    cpu_brand: String,
    total_memory_gb: f32,
    recommended_mode: String,
    recommended_top_k: usize,
    recommended_max_file_size_mb: u64,
    note: String,
}

#[derive(Serialize)]
struct SearchBenchmarkResult {
    query: String,
    iterations: usize,
    avg_ms: f32,
    p95_ms: u64,
    best_ms: u64,
    worst_ms: u64,
    last_result_count: usize,
    candidate_limit: usize,
    images_only: bool,
    backend: String,
}

#[derive(Serialize)]
struct SearchColdHotBenchmarkResult {
    query: String,
    iterations: usize,
    cold_avg_ms: f32,
    cold_p95_ms: u64,
    hot_avg_ms: f32,
    hot_p95_ms: u64,
    speedup_percent: f32,
    candidate_count: usize,
    backend: String,
}

#[derive(Serialize)]
struct AppliedPerformanceProfile {
    cpu_cores: usize,
    total_memory_gb: f32,
    recommended_mode: String,
    recommended_top_k: usize,
    recommended_max_file_size_mb: u64,
    applied_max_file_size_mb: u64,
    note: String,
}

#[derive(Serialize)]
struct ClipImageCacheStatus {
    items: usize,
}

#[derive(Serialize)]
struct SemanticSchemaInfo {
    schema_version: i64,
    chunk_count: u64,
    db_path: String,
}

#[derive(Serialize)]
struct EmbeddingModelIndexStatus {
    current_embedding_model: String,
    indexed_embedding_model: String,
    requires_reindex: bool,
}

#[derive(Clone, Serialize, Deserialize)]
struct ClipOnnxConfig {
    enabled: bool,
    image_model_path: String,
    text_model_path: String,
    tokenizer_path: String,
    input_size: u32,
    max_length: usize,
}

#[derive(Serialize)]
struct ClipOnnxStatus {
    configured: bool,
    enabled: bool,
    image_model_path: String,
    text_model_path: String,
    tokenizer_path: String,
    input_size: u32,
    max_length: usize,
}

#[derive(Serialize)]
struct ClipValidationStatus {
    configured: bool,
    tokenizer_ok: bool,
    text_model_ok: bool,
    image_model_ok: bool,
    text_inference_ok: bool,
    image_inference_ok: bool,
    text_dim: Option<usize>,
    image_dim: Option<usize>,
    sample_image_path: Option<String>,
    message: String,
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

#[tauri::command]
fn get_file_text_preview(path: String, max_chars: Option<usize>) -> Result<FileTextPreview, String> {
    let file_path = PathBuf::from(&path);
    if !file_path.exists() {
        return Err("El archivo no existe".to_string());
    }

    let limit = max_chars.unwrap_or(14_000).clamp(500, 80_000);

    let extension = file_path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_lowercase())
        .unwrap_or_default();

    if extension == "pdf" {
        if let Some((text, used_fallback)) = extract_pdf_text(&file_path) {
            let normalized = normalize_text_for_index(&text, limit);
            return Ok(FileTextPreview {
                available: !normalized.is_empty(),
                source: if used_fallback {
                    "pdf-fallback".to_string()
                } else {
                    "pdf-extract".to_string()
                },
                text: normalized,
            });
        }

        return Ok(FileTextPreview {
            available: false,
            source: "pdf".to_string(),
            text: "No se pudo extraer texto de este PDF.".to_string(),
        });
    }

    if extension == "txt" || extension == "md" {
        let bytes = fs::read(&file_path).map_err(|err| format!("No se pudo leer archivo: {err}"))?;
        let slice = &bytes[..bytes.len().min(limit.saturating_mul(2))];
        let text = String::from_utf8_lossy(slice);
        let normalized = normalize_text_for_index(&text, limit);

        return Ok(FileTextPreview {
            available: !normalized.is_empty(),
            source: "text-file".to_string(),
            text: normalized,
        });
    }

    Ok(FileTextPreview {
        available: false,
        source: "unsupported".to_string(),
        text: "Vista textual no disponible para este tipo de archivo.".to_string(),
    })
}

#[tauri::command]
fn get_file_visual_preview(path: String, max_mb: Option<u64>) -> Result<FileVisualPreview, String> {
    let file_path = PathBuf::from(&path);
    if !file_path.exists() {
        return Err("El archivo no existe".to_string());
    }

    let extension = file_path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_lowercase())
        .unwrap_or_default();

    let mime = match extension.as_str() {
        "pdf" => "application/pdf",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "tiff" | "tif" => "image/tiff",
        _ => {
            return Ok(FileVisualPreview {
                available: false,
                mime: String::new(),
                data_url: None,
                size_bytes: 0,
                reason: Some("Tipo no soportado para preview visual".to_string()),
            })
        }
    }
    .to_string();

    let metadata = fs::metadata(&file_path)
        .map_err(|err| format!("No se pudo leer metadata del archivo: {err}"))?;
    let size_bytes = metadata.len();
    let max_size_mb = max_mb.unwrap_or(16).clamp(2, 64);
    let max_bytes = max_size_mb.saturating_mul(1024 * 1024);

    if size_bytes > max_bytes {
        return Ok(FileVisualPreview {
            available: false,
            mime,
            data_url: None,
            size_bytes,
            reason: Some(format!(
                "Archivo demasiado grande para preview visual ({} MB > {} MB)",
                (size_bytes as f64 / (1024.0 * 1024.0)).round(),
                max_size_mb
            )),
        });
    }

    let bytes = fs::read(&file_path).map_err(|err| format!("No se pudo leer archivo visual: {err}"))?;
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);

    Ok(FileVisualPreview {
        available: true,
        mime: mime.clone(),
        data_url: Some(format!("data:{};base64,{}", mime, encoded)),
        size_bytes,
        reason: None,
    })
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
    #[serde(default = "default_chat_base_url_value")]
    chat_base_url: String,
    #[serde(default = "default_chat_model_value")]
    chat_model: String,
    api_key: String,
}

#[derive(Serialize)]
struct AiProviderStatus {
    configured: bool,
    provider: String,
    base_url: String,
    embedding_model: String,
    chat_base_url: String,
    chat_model: String,
    api_key_hint: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
struct ExportableAppConfig {
    version: u32,
    roots: Vec<String>,
    excluded_extensions: Vec<String>,
    excluded_folders: Vec<String>,
    max_file_size_mb: u64,
    ai_provider: Option<AiProviderConfig>,
    #[serde(default)]
    performance_runtime: Option<PerformanceRuntimeConfig>,
}

#[derive(Clone, Serialize, Deserialize)]
struct PerformanceRuntimeConfig {
    adaptive_enabled: bool,
}

impl Default for PerformanceRuntimeConfig {
    fn default() -> Self {
        Self {
            adaptive_enabled: true,
        }
    }
}

#[derive(Clone, Default)]
struct PerformanceRuntimeState {
    config: PerformanceRuntimeConfig,
    last_decision: String,
    last_candidate_limit: usize,
    last_rag_top_k: usize,
}

#[derive(Serialize)]
struct PerformanceRuntimeStatus {
    adaptive_enabled: bool,
    last_decision: String,
    last_candidate_limit: usize,
    last_rag_top_k: usize,
}

#[derive(Clone)]
struct ChunkCandidate {
    title: String,
    path: String,
    chunk_text: String,
    lexical_score: f32,
    modified_unix_secs: u64,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingDataItem>,
}

#[derive(Deserialize)]
struct EmbeddingDataItem {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<ChatChoiceItem>,
}

#[derive(Deserialize)]
struct ChatChoiceItem {
    message: ChatMessageItem,
}

#[derive(Deserialize)]
struct ChatMessageItem {
    content: serde_json::Value,
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

struct AppState {
    index: Mutex<Option<IndexSnapshot>>,
    diagnostics: Mutex<Option<IndexDiagnostics>>,
    cancel_indexing: AtomicBool,
    ai_config: Mutex<Option<AiProviderConfig>>,
    index_settings: Mutex<IndexingSettings>,
    watcher_runtime: Mutex<Option<FileWatcherRuntime>>,
    watcher_status: Mutex<FileWatcherStatus>,
    is_indexing: AtomicBool,
    audit_logs: Mutex<Vec<AuditLogEntry>>,
    runtime_metrics: Mutex<RuntimeMetricsState>,
    clip_config: Mutex<Option<ClipOnnxConfig>>,
    clip_image_cache: Mutex<HashMap<String, Vec<f32>>>,
    text_embedding_cache: Mutex<HashMap<String, Vec<f32>>>,
    performance_runtime: Mutex<PerformanceRuntimeState>,
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
            audit_logs: Mutex::new(Vec::new()),
            runtime_metrics: Mutex::new(RuntimeMetricsState::default()),
            clip_config: Mutex::new(None),
            clip_image_cache: Mutex::new(HashMap::new()),
            text_embedding_cache: Mutex::new(HashMap::new()),
            performance_runtime: Mutex::new(PerformanceRuntimeState {
                config: PerformanceRuntimeConfig::default(),
                last_decision: "sin decisiones aún".to_string(),
                last_candidate_limit: 80,
                last_rag_top_k: 4,
            }),
        }
    }
}

#[tauri::command]
fn get_performance_runtime_status(state: State<'_, AppState>) -> PerformanceRuntimeStatus {
    state
        .performance_runtime
        .lock()
        .map(|runtime| PerformanceRuntimeStatus {
            adaptive_enabled: runtime.config.adaptive_enabled,
            last_decision: runtime.last_decision.clone(),
            last_candidate_limit: runtime.last_candidate_limit,
            last_rag_top_k: runtime.last_rag_top_k,
        })
        .unwrap_or(PerformanceRuntimeStatus {
            adaptive_enabled: true,
            last_decision: "sin estado".to_string(),
            last_candidate_limit: 80,
            last_rag_top_k: 4,
        })
}

#[tauri::command]
fn configure_performance_runtime(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    adaptive_enabled: Option<bool>,
) -> Result<PerformanceRuntimeStatus, String> {
    {
        let mut runtime = state
            .performance_runtime
            .lock()
            .map_err(|_| "No se pudo actualizar runtime performance".to_string())?;
        if let Some(value) = adaptive_enabled {
            runtime.config.adaptive_enabled = value;
        }
        save_performance_runtime_config(&app, &runtime.config)?;
    }

    append_audit_log(
        state.inner(),
        "performance.runtime.updated",
        format!(
            "adaptive_enabled={}",
            adaptive_enabled.unwrap_or(true)
        ),
    );

    Ok(get_performance_runtime_status(state))
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
fn get_runtime_metrics(state: State<'_, AppState>) -> RuntimeMetrics {
    let guard = match state.runtime_metrics.lock() {
        Ok(value) => value,
        Err(_) => {
            return RuntimeMetrics {
                semantic_calls: 0,
                semantic_avg_ms: 0.0,
                rag_calls: 0,
                rag_avg_ms: 0.0,
                indexing_runs: 0,
                indexing_avg_ms: 0.0,
                last_index_ms: 0,
                embedding_cache_hits: 0,
                embedding_cache_misses: 0,
                embedding_cache_hit_rate: 0.0,
                embedding_cache_items: 0,
                last_semantic_ms: 0,
                last_rag_ms: 0,
            }
        }
    };

    let cache_items = state
        .text_embedding_cache
        .lock()
        .map(|cache| cache.len())
        .unwrap_or(0);

    RuntimeMetrics {
        semantic_calls: guard.semantic_calls,
        semantic_avg_ms: compute_avg_ms(guard.semantic_total_ms, guard.semantic_calls),
        rag_calls: guard.rag_calls,
        rag_avg_ms: compute_avg_ms(guard.rag_total_ms, guard.rag_calls),
        indexing_runs: guard.indexing_runs,
        indexing_avg_ms: compute_avg_ms(guard.indexing_total_ms, guard.indexing_runs),
        last_index_ms: guard.last_index_ms,
        embedding_cache_hits: guard.embedding_cache_hits,
        embedding_cache_misses: guard.embedding_cache_misses,
        embedding_cache_hit_rate: compute_hit_rate(guard.embedding_cache_hits, guard.embedding_cache_misses),
        embedding_cache_items: cache_items,
        last_semantic_ms: guard.last_semantic_ms,
        last_rag_ms: guard.last_rag_ms,
    }
}

#[tauri::command]
fn get_performance_telemetry(state: State<'_, AppState>) -> PerformanceTelemetry {
    let adaptive_enabled = state
        .performance_runtime
        .lock()
        .map(|runtime| runtime.config.adaptive_enabled)
        .unwrap_or(true);

    let metrics = state
        .runtime_metrics
        .lock()
        .map(|value| RuntimeMetricsState {
            semantic_calls: value.semantic_calls,
            semantic_total_ms: value.semantic_total_ms,
            last_semantic_ms: value.last_semantic_ms,
            rag_calls: value.rag_calls,
            rag_total_ms: value.rag_total_ms,
            last_rag_ms: value.last_rag_ms,
            indexing_runs: value.indexing_runs,
            indexing_total_ms: value.indexing_total_ms,
            last_index_ms: value.last_index_ms,
            embedding_cache_hits: value.embedding_cache_hits,
            embedding_cache_misses: value.embedding_cache_misses,
        })
        .unwrap_or_default();

    let snapshot = runtime_pressure_snapshot();
    let pressure_level = runtime_pressure_level(snapshot.memory_pressure_pct).to_string();

    PerformanceTelemetry {
        memory_total_gb: snapshot.total_memory_gb,
        memory_available_gb: snapshot.available_memory_gb,
        memory_pressure_pct: snapshot.memory_pressure_pct,
        cpu_usage_pct: runtime_cpu_usage_pct(),
        pressure_level,
        semantic_last_ms: metrics.last_semantic_ms,
        rag_last_ms: metrics.last_rag_ms,
        indexing_last_ms: metrics.last_index_ms,
        adaptive_enabled,
        throttling_factor: compute_runtime_throttling_factor(snapshot.memory_pressure_pct),
    }
}

#[tauri::command]
fn get_semantic_schema_info(app: tauri::AppHandle) -> Result<SemanticSchemaInfo, String> {
    let conn = open_semantic_connection(&app)?;
    let schema_version = conn
        .query_row(
            "SELECT value_int FROM metadata WHERE key = 'schema_version' LIMIT 1",
            [],
            |row| row.get::<_, i64>(0),
        )
        .unwrap_or(SEMANTIC_SCHEMA_VERSION);

    let chunk_count = conn
        .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get::<_, i64>(0))
        .unwrap_or(0)
        .max(0) as u64;

    let db_path = semantic_db_path(&app)
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_default();

    Ok(SemanticSchemaInfo {
        schema_version,
        chunk_count,
        db_path,
    })
}

#[tauri::command]
fn get_embedding_model_index_status(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<EmbeddingModelIndexStatus, String> {
    let current_model = state
        .ai_config
        .lock()
        .ok()
        .and_then(|cfg| cfg.as_ref().map(|value| value.embedding_model.clone()))
        .unwrap_or_else(|| "local-lexical".to_string());

    let conn = open_semantic_connection(&app)?;
    let indexed_model = conn
        .query_row(
            "SELECT value_text FROM metadata WHERE key = 'embedding_model' LIMIT 1",
            [],
            |row| row.get::<_, String>(0),
        )
        .unwrap_or_else(|_| "unknown".to_string());

    Ok(EmbeddingModelIndexStatus {
        current_embedding_model: current_model.clone(),
        indexed_embedding_model: indexed_model.clone(),
        requires_reindex: indexed_model != current_model,
    })
}

#[tauri::command]
fn sync_embedding_model_index(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    reindex: Option<bool>,
) -> Result<EmbeddingModelIndexStatus, String> {
    let should_reindex = reindex.unwrap_or(true);
    if should_reindex {
        let settings = state
            .index_settings
            .lock()
            .map_err(|_| "No se pudo leer configuración de indexado".to_string())?
            .clone();

        if !settings.roots.is_empty() {
            let _ = execute_indexing(&app, state.inner(), settings, "model-version-sync")?;
        }
    } else {
        persist_embedding_model_version(&app, state.inner())?;
    }

    get_embedding_model_index_status(app, state)
}

#[tauri::command]
fn get_embedding_cache_status(state: State<'_, AppState>) -> EmbeddingCacheStatus {
    let (hits, misses) = state
        .runtime_metrics
        .lock()
        .map(|metrics| (metrics.embedding_cache_hits, metrics.embedding_cache_misses))
        .unwrap_or((0, 0));
    let items = state
        .text_embedding_cache
        .lock()
        .map(|cache| cache.len())
        .unwrap_or(0);

    EmbeddingCacheStatus {
        enabled: true,
        items,
        hits,
        misses,
        hit_rate: compute_hit_rate(hits, misses),
    }
}

#[tauri::command]
fn clear_embedding_cache(state: State<'_, AppState>) -> Result<(), String> {
    if let Ok(mut cache) = state.text_embedding_cache.lock() {
        cache.clear();
    }

    if let Ok(mut metrics) = state.runtime_metrics.lock() {
        metrics.embedding_cache_hits = 0;
        metrics.embedding_cache_misses = 0;
    }

    append_audit_log(state.inner(), "embeddings.cache.clear", "cache limpiada".to_string());
    Ok(())
}

#[tauri::command]
fn get_hardware_profile() -> HardwareProfile {
    detect_hardware_profile()
}

#[tauri::command]
fn apply_hardware_profile_defaults(state: State<'_, AppState>) -> Result<AppliedPerformanceProfile, String> {
    let profile = detect_hardware_profile();

    let applied_max = {
        let mut settings = state
            .index_settings
            .lock()
            .map_err(|_| "No se pudo actualizar perfil de rendimiento".to_string())?;
        settings.max_file_size_mb = profile.recommended_max_file_size_mb;
        settings.max_file_size_mb
    };

    append_audit_log(
        state.inner(),
        "performance.profile.applied",
        format!(
            "mode={} top_k={} max_file_mb={}",
            profile.recommended_mode, profile.recommended_top_k, applied_max
        ),
    );

    Ok(AppliedPerformanceProfile {
        cpu_cores: profile.cpu_cores,
        total_memory_gb: profile.total_memory_gb,
        recommended_mode: profile.recommended_mode,
        recommended_top_k: profile.recommended_top_k,
        recommended_max_file_size_mb: profile.recommended_max_file_size_mb,
        applied_max_file_size_mb: applied_max,
        note: profile.note,
    })
}

#[tauri::command]
async fn run_local_semantic_benchmark(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    query: String,
    iterations: Option<usize>,
    candidate_limit: Option<usize>,
    roots: Option<Vec<String>>,
    excluded_extensions: Option<Vec<String>>,
    excluded_folders: Option<Vec<String>>,
    max_file_size_mb: Option<u64>,
    images_only: Option<bool>,
) -> Result<SearchBenchmarkResult, String> {
    let clean_query = query.trim();
    if clean_query.is_empty() {
        return Err("La consulta de benchmark no puede estar vacía".to_string());
    }

    let loops = iterations.unwrap_or(8).clamp(3, 40);
    let cap = candidate_limit.unwrap_or(80).clamp(20, 200);
    let search_roots = resolve_roots(roots);
    let exclusions = normalize_extensions(excluded_extensions);
    let folder_exclusions = normalize_folder_rules(excluded_folders);
    let max_file_size_bytes = max_size_bytes(max_file_size_mb);
    let images_only = images_only.unwrap_or(false);

    let mut durations_ms = Vec::<u64>::with_capacity(loops);
    let mut last_result_count = 0usize;
    let mut lancedb_successes = 0usize;

    for _ in 0..loops {
        let started = Instant::now();
        let candidates = match load_chunk_candidates_from_lancedb(
            &app,
            clean_query,
            &search_roots,
            &exclusions,
            &folder_exclusions,
            max_file_size_bytes,
            cap,
            images_only,
        )
        .await
        {
            Ok(value) if !value.is_empty() => {
                lancedb_successes = lancedb_successes.saturating_add(1);
                value
            }
            _ => load_chunk_candidates(
                &app,
                clean_query,
                &search_roots,
                &exclusions,
                &folder_exclusions,
                max_file_size_bytes,
                cap,
                images_only,
            )?,
        };

        last_result_count = candidates.len().min(20);
        durations_ms.push(started.elapsed().as_millis() as u64);
    }

    durations_ms.sort_unstable();
    let best_ms = durations_ms.first().copied().unwrap_or(0);
    let worst_ms = durations_ms.last().copied().unwrap_or(0);
    let total = durations_ms.iter().copied().sum::<u64>();
    let avg_ms = if loops == 0 { 0.0 } else { total as f32 / loops as f32 };
    let p95_index = ((durations_ms.len() as f32 * 0.95).ceil() as usize).saturating_sub(1);
    let p95_ms = durations_ms
        .get(p95_index.min(durations_ms.len().saturating_sub(1)))
        .copied()
        .unwrap_or(0);

    let backend = if lancedb_successes == loops {
        "lancedb"
    } else if lancedb_successes == 0 {
        "sqlite-fallback"
    } else {
        "mixed"
    };

    append_audit_log(
        state.inner(),
        "benchmark.semantic.local",
        format!(
            "iterations={} avg_ms={:.2} p95_ms={} backend={} images_only={}",
            loops, avg_ms, p95_ms, backend, images_only
        ),
    );

    Ok(SearchBenchmarkResult {
        query: clean_query.to_string(),
        iterations: loops,
        avg_ms,
        p95_ms,
        best_ms,
        worst_ms,
        last_result_count,
        candidate_limit: cap,
        images_only,
        backend: backend.to_string(),
    })
}

#[tauri::command]
async fn run_semantic_cold_hot_benchmark(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    query: String,
    iterations: Option<usize>,
    candidate_limit: Option<usize>,
    roots: Option<Vec<String>>,
    excluded_extensions: Option<Vec<String>>,
    excluded_folders: Option<Vec<String>>,
    max_file_size_mb: Option<u64>,
    images_only: Option<bool>,
) -> Result<SearchColdHotBenchmarkResult, String> {
    let clean_query = query.trim();
    if clean_query.is_empty() {
        return Err("La consulta de benchmark no puede estar vacía".to_string());
    }

    let config = state
        .ai_config
        .lock()
        .ok()
        .and_then(|value| value.as_ref().cloned())
        .ok_or_else(|| "Configura proveedor de embeddings para benchmark cold/hot".to_string())?;

    let loops = iterations.unwrap_or(6).clamp(2, 20);
    let cap = candidate_limit.unwrap_or(60).clamp(20, 160);
    let search_roots = resolve_roots(roots);
    let exclusions = normalize_extensions(excluded_extensions);
    let folder_exclusions = normalize_folder_rules(excluded_folders);
    let max_file_size_bytes = max_size_bytes(max_file_size_mb);
    let images_only = images_only.unwrap_or(false);

    let (candidates, backend) = match load_chunk_candidates_from_lancedb(
        &app,
        clean_query,
        &search_roots,
        &exclusions,
        &folder_exclusions,
        max_file_size_bytes,
        cap,
        images_only,
    )
    .await
    {
        Ok(value) if !value.is_empty() => (value, "lancedb".to_string()),
        _ => (
            load_chunk_candidates(
                &app,
                clean_query,
                &search_roots,
                &exclusions,
                &folder_exclusions,
                max_file_size_bytes,
                cap,
                images_only,
            )?,
            "sqlite-fallback".to_string(),
        ),
    };

    if candidates.is_empty() {
        return Err("No hay candidatos para benchmark cold/hot".to_string());
    }

    let mut inputs = Vec::with_capacity(candidates.len() + 1);
    inputs.push(clean_query.to_string());
    for candidate in &candidates {
        inputs.push(candidate.chunk_text.chars().take(1_200).collect::<String>());
    }

    let mut cold_ms = Vec::<u64>::with_capacity(loops);
    let mut hot_ms = Vec::<u64>::with_capacity(loops);

    for _ in 0..loops {
        if let Ok(mut cache) = state.text_embedding_cache.lock() {
            cache.clear();
        }

        let cold_started = Instant::now();
        let _ = request_embeddings(state.inner(), &config, &inputs).await?;
        cold_ms.push(cold_started.elapsed().as_millis() as u64);

        let hot_started = Instant::now();
        let _ = request_embeddings(state.inner(), &config, &inputs).await?;
        hot_ms.push(hot_started.elapsed().as_millis() as u64);
    }

    let cold_avg = compute_avg_from_samples(&cold_ms);
    let hot_avg = compute_avg_from_samples(&hot_ms);
    let cold_p95 = percentile_ms(&cold_ms, 0.95);
    let hot_p95 = percentile_ms(&hot_ms, 0.95);
    let speedup = if cold_avg <= 0.0 {
        0.0
    } else {
        ((cold_avg - hot_avg) / cold_avg * 100.0).max(0.0)
    };

    append_audit_log(
        state.inner(),
        "benchmark.semantic.cold_hot",
        format!(
            "iterations={} cold_avg={:.2} hot_avg={:.2} speedup={:.1}% backend={}",
            loops, cold_avg, hot_avg, speedup, backend
        ),
    );

    Ok(SearchColdHotBenchmarkResult {
        query: clean_query.to_string(),
        iterations: loops,
        cold_avg_ms: cold_avg,
        cold_p95_ms: cold_p95,
        hot_avg_ms: hot_avg,
        hot_p95_ms: hot_p95,
        speedup_percent: speedup,
        candidate_count: candidates.len(),
        backend,
    })
}

#[tauri::command]
fn get_clip_image_cache_status(state: State<'_, AppState>) -> ClipImageCacheStatus {
    let items = state
        .clip_image_cache
        .lock()
        .map(|cache| cache.len())
        .unwrap_or(0);
    ClipImageCacheStatus { items }
}

#[tauri::command]
fn clear_clip_image_cache(state: State<'_, AppState>) -> Result<(), String> {
    if let Ok(mut cache) = state.clip_image_cache.lock() {
        cache.clear();
    }

    append_audit_log(state.inner(), "clip.cache.clear", "cache visual limpiada".to_string());
    Ok(())
}

#[tauri::command]
fn get_audit_logs(state: State<'_, AppState>, limit: Option<usize>) -> Vec<AuditLogEntry> {
    let max = limit.unwrap_or(50).clamp(1, 300);
    state
        .audit_logs
        .lock()
        .map(|logs| logs.iter().rev().take(max).cloned().collect::<Vec<_>>())
        .unwrap_or_default()
}

#[tauri::command]
fn clear_audit_logs(state: State<'_, AppState>) -> Result<(), String> {
    let mut guard = state
        .audit_logs
        .lock()
        .map_err(|_| "No se pudo limpiar auditoría".to_string())?;
    guard.clear();
    Ok(())
}

#[tauri::command]
fn export_audit_logs_to_file(
    state: State<'_, AppState>,
    path: String,
    limit: Option<usize>,
) -> Result<String, String> {
    let max = limit.unwrap_or(300).clamp(1, 1000);
    let logs = state
        .audit_logs
        .lock()
        .map(|value| value.iter().rev().take(max).cloned().collect::<Vec<_>>())
        .map_err(|_| "No se pudo leer auditoría".to_string())?;

    let target = PathBuf::from(path);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("No se pudo crear carpeta destino: {err}"))?;
    }

    let serialized = serde_json::to_string_pretty(&logs)
        .map_err(|err| format!("No se pudo serializar auditoría: {err}"))?;
    fs::write(&target, serialized)
        .map_err(|err| format!("No se pudo exportar auditoría: {err}"))?;

    Ok(target.to_string_lossy().to_string())
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
fn clear_file_watcher_error(state: State<'_, AppState>) -> Result<FileWatcherStatus, String> {
    update_watcher_status(state.inner(), |status| {
        status.last_error = None;
    });

    get_existing_watcher_status(state.inner())
}

#[tauri::command]
fn trigger_watcher_reindex(app: tauri::AppHandle, state: State<'_, AppState>) -> Result<FileWatcherStatus, String> {
    let settings = state
        .index_settings
        .lock()
        .map(|value| value.clone())
        .map_err(|_| "No se pudo leer configuración de indexado".to_string())?;

    execute_indexing(&app, state.inner(), settings, "watcher-manual")?;

    update_watcher_status(state.inner(), |status| {
        status.last_reindex_at = Some(now_timestamp_string());
        status.pending_events = false;
        status.pending_event_count = 0;
        status.last_batch_reason = Some("manual".to_string());
    });

    append_audit_log(state.inner(), "watcher.reindex.manual", "trigger manual desde UI".to_string());

    get_existing_watcher_status(state.inner())
}

#[tauri::command]
fn configure_ai_provider(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    provider: Option<String>,
    api_key: String,
    embedding_model: Option<String>,
    base_url: Option<String>,
    chat_model: Option<String>,
    chat_base_url: Option<String>,
) -> Result<AiProviderStatus, String> {
    let provider = provider
        .unwrap_or_else(|| "openrouter-compatible".to_string())
        .trim()
        .to_lowercase();
    let is_local_provider = provider == "ollama-local";

    let trimmed_key = api_key.trim().to_string();
    if trimmed_key.is_empty() && !is_local_provider {
        return Err("API key vacía".to_string());
    }

    let config = AiProviderConfig {
        provider: if is_local_provider {
            "ollama-local".to_string()
        } else {
            "openrouter-compatible".to_string()
        },
        base_url: base_url
            .unwrap_or_else(|| {
                if is_local_provider {
                    default_ollama_embeddings_url_value()
                } else {
                    "https://openrouter.ai/api/v1/embeddings".to_string()
                }
            })
            .trim()
            .to_string(),
        embedding_model: embedding_model
            .unwrap_or_else(|| {
                if is_local_provider {
                    default_ollama_model_value()
                } else {
                    "text-embedding-3-small".to_string()
                }
            })
            .trim()
            .to_string(),
        chat_base_url: chat_base_url
            .unwrap_or_else(|| {
                if is_local_provider {
                    default_ollama_chat_url_value()
                } else {
                    "https://openrouter.ai/api/v1/chat/completions".to_string()
                }
            })
            .trim()
            .to_string(),
        chat_model: chat_model
            .unwrap_or_else(|| {
                if is_local_provider {
                    default_ollama_chat_model_value()
                } else {
                    "gpt-4o-mini".to_string()
                }
            })
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

    append_audit_log(
        state.inner(),
        "ai.config.updated",
        format!(
            "provider={} embedding_model={} chat_model={} key={}",
            config.provider,
            config.embedding_model,
            config.chat_model,
            if config.api_key.is_empty() {
                "sin-api-key".to_string()
            } else {
                mask_api_key(&config.api_key)
            }
        ),
    );

    Ok(ai_provider_status_from_config(Some(&config)))
}

#[tauri::command]
fn get_ai_provider_status(state: State<'_, AppState>) -> AiProviderStatus {
    let guard = state.ai_config.lock().ok();
    let maybe_cfg = guard.and_then(|value| value.as_ref().cloned());
    ai_provider_status_from_config(maybe_cfg.as_ref())
}

#[tauri::command]
fn configure_clip_onnx(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    image_model_path: String,
    text_model_path: String,
    tokenizer_path: String,
    input_size: Option<u32>,
    max_length: Option<usize>,
    enabled: Option<bool>,
) -> Result<ClipOnnxStatus, String> {
    let image_path = PathBuf::from(image_model_path.trim());
    let text_path = PathBuf::from(text_model_path.trim());
    let tokenizer = PathBuf::from(tokenizer_path.trim());

    if !image_path.exists() || !text_path.exists() || !tokenizer.exists() {
        return Err("Rutas de CLIP/ONNX inválidas: verifica modelo imagen, modelo texto y tokenizer".to_string());
    }

    let config = ClipOnnxConfig {
        enabled: enabled.unwrap_or(true),
        image_model_path: image_path.to_string_lossy().to_string(),
        text_model_path: text_path.to_string_lossy().to_string(),
        tokenizer_path: tokenizer.to_string_lossy().to_string(),
        input_size: input_size.unwrap_or(224).clamp(128, 512),
        max_length: max_length.unwrap_or(77).clamp(16, 256),
    };

    save_clip_config(&app, &config)?;

    {
        let mut guard = state
            .clip_config
            .lock()
            .map_err(|_| "No se pudo actualizar config CLIP".to_string())?;
        *guard = Some(config.clone());
    }

    if let Ok(mut cache) = state.clip_image_cache.lock() {
        cache.clear();
    }

    append_audit_log(
        state.inner(),
        "clip.config.updated",
        format!(
            "enabled={} input_size={} max_length={}",
            config.enabled, config.input_size, config.max_length
        ),
    );

    Ok(clip_status_from_config(Some(&config)))
}

#[tauri::command]
fn get_clip_onnx_status(app: tauri::AppHandle, state: State<'_, AppState>) -> ClipOnnxStatus {
    let stored = state
        .clip_config
        .lock()
        .ok()
        .and_then(|value| value.as_ref().cloned());

    if let Some(config) = stored {
        return clip_status_from_config(Some(&config));
    }

    if let Some(detected) = detect_clip_config(&app) {
        if let Ok(mut guard) = state.clip_config.lock() {
            *guard = Some(detected.clone());
        }
        return clip_status_from_config(Some(&detected));
    }

    clip_status_from_config(None)
}

#[tauri::command]
fn clip_text_to_image_search(
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

    let mut config = state
        .clip_config
        .lock()
        .ok()
        .and_then(|value| value.as_ref().cloned())
        .filter(|value| value.enabled);

    if config.is_none() {
        config = detect_clip_config(&app).filter(|value| value.enabled);
        if let Some(found) = config.clone() {
            if let Ok(mut guard) = state.clip_config.lock() {
                *guard = Some(found.clone());
            }
            let _ = save_clip_config(&app, &found);
        }
    }

    let search_roots = resolve_roots(roots);
    let exclusions = normalize_extensions(excluded_extensions);
    let folder_exclusions = normalize_folder_rules(excluded_folders);
    let max_file_size_bytes = max_size_bytes(max_file_size_mb);
    let max_results = limit.unwrap_or(20).clamp(1, 50);
    let adaptive_enabled = state
        .performance_runtime
        .lock()
        .map(|runtime| runtime.config.adaptive_enabled)
        .unwrap_or(true);
    let pressure = runtime_pressure_snapshot();
    let inspect_limit = if adaptive_enabled {
        adaptive_clip_inspect_limit(pressure.memory_pressure_pct)
    } else {
        900
    };

    let snapshot_items = state
        .index
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|snapshot| snapshot.files.clone()))
        .unwrap_or_default();

    if snapshot_items.is_empty() {
        return Err("No hay índice local para buscar imágenes".to_string());
    }

    let Some(config) = config else {
        let fallback = fallback_image_search(
            &snapshot_items,
            clean_query,
            &search_roots,
            &exclusions,
            &folder_exclusions,
            max_file_size_bytes,
            max_results,
            "CLIP no configurado",
        );
        append_audit_log(
            state.inner(),
            "search.clip_onnx.fallback",
            format!("reason=no-config results={}", fallback.len()),
        );
        return Ok(fallback);
    };

    let query_embedding = match run_clip_text_embedding(&config, clean_query) {
        Ok(value) if !value.is_empty() => value,
        _ => {
            let fallback = fallback_image_search(
                &snapshot_items,
                clean_query,
                &search_roots,
                &exclusions,
                &folder_exclusions,
                max_file_size_bytes,
                max_results,
                "inferencia CLIP texto falló",
            );
            append_audit_log(
                state.inner(),
                "search.clip_onnx.fallback",
                format!("reason=text-embed-fail results={}", fallback.len()),
            );
            return Ok(fallback);
        }
    };

    let mut scored = Vec::<(f32, IndexedFileItem)>::new();
    let mut inspected = 0usize;

    for item in snapshot_items {
        let path_buf = PathBuf::from(&item.path);
        if !is_image_path(&path_buf) {
            continue;
        }

        if !matches_roots(&path_buf, &search_roots)
            || should_skip_custom_dir(&path_buf, &folder_exclusions)
            || is_excluded_file(&path_buf, &exclusions)
            || item.size_bytes > max_file_size_bytes
        {
            continue;
        }

        inspected += 1;
        if inspected > inspect_limit {
            break;
        }

        let image_embedding = load_or_compute_clip_image_embedding(state.inner(), &config, &item.path)?;
        if image_embedding.is_empty() {
            continue;
        }

        let score = cosine_similarity(&query_embedding, &image_embedding);
        scored.push((score, item));
    }

    scored.sort_by(|left, right| right.0.total_cmp(&left.0));

    let results = scored
        .into_iter()
        .take(max_results)
        .map(|(score, item)| SearchResultItem {
            title: item.title,
            path: item.path,
            snippet: format!("CLIP/ONNX {:.3} · búsqueda texto→imagen local", score),
            match_reason: format!("Similaridad CLIP entre texto y embedding visual (score {:.3})", score),
            origin: "local-clip-onnx".to_string(),
        })
        .collect::<Vec<_>>();

    append_audit_log(
        state.inner(),
        "search.clip_onnx",
        format!(
            "query_len={} inspected={} results={} inspect_limit={} pressure={:.1}%",
            clean_query.chars().count(),
            inspected,
            results.len(),
            inspect_limit,
            pressure.memory_pressure_pct
        ),
    );

    Ok(results)
}

fn fallback_image_search(
    items: &[IndexedFileItem],
    clean_query: &str,
    search_roots: &[PathBuf],
    exclusions: &[String],
    folder_exclusions: &[String],
    max_file_size_bytes: u64,
    max_results: usize,
    reason: &str,
) -> Vec<SearchResultItem> {
    let tokens = tokenize_query(clean_query);
    let mut scored = Vec::<(f32, &IndexedFileItem)>::new();

    for item in items {
        let path_buf = PathBuf::from(&item.path);
        if !is_image_path(&path_buf) {
            continue;
        }

        if !matches_roots(&path_buf, search_roots)
            || should_skip_custom_dir(&path_buf, folder_exclusions)
            || is_excluded_file(&path_buf, exclusions)
            || item.size_bytes > max_file_size_bytes
        {
            continue;
        }

        let lowered = format!("{} {}", item.title.to_lowercase(), item.path.to_lowercase());
        let token_hits = tokens
            .iter()
            .filter(|token| lowered.contains(token.as_str()))
            .count() as f32;
        let base = if clean_query.is_empty() { 0.1 } else { token_hits / (tokens.len().max(1) as f32) };
        if base > 0.0 {
            scored.push((base, item));
        }
    }

    scored.sort_by(|left, right| right.0.total_cmp(&left.0));

    scored
        .into_iter()
        .take(max_results)
        .map(|(score, item)| SearchResultItem {
            title: item.title.clone(),
            path: item.path.clone(),
            snippet: format!(
                "Fallback imagen {:.2} · {}",
                score,
                reason
            ),
            match_reason: format!(
                "Coincidencia léxica en nombre/ruta de imagen (score {:.2})",
                score
            ),
            origin: "local-image-lexical-fallback".to_string(),
        })
        .collect()
}

#[tauri::command]
fn validate_clip_onnx_setup(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    sample_image_path: Option<String>,
) -> ClipValidationStatus {
    let config = state
        .clip_config
        .lock()
        .ok()
        .and_then(|value| value.as_ref().cloned())
        .or_else(|| detect_clip_config(&app));

    let Some(config) = config else {
        return ClipValidationStatus {
            configured: false,
            tokenizer_ok: false,
            text_model_ok: false,
            image_model_ok: false,
            text_inference_ok: false,
            image_inference_ok: false,
            text_dim: None,
            image_dim: None,
            sample_image_path: None,
            message: "CLIP no configurado: faltan rutas de modelo/tokenizer".to_string(),
        };
    };

    let tokenizer_ok = Tokenizer::from_file(&config.tokenizer_path).is_ok();
    let text_model_ok = tract_onnx::onnx().model_for_path(&config.text_model_path).is_ok();
    let image_model_ok = tract_onnx::onnx().model_for_path(&config.image_model_path).is_ok();

    let mut text_inference_ok = false;
    let mut text_dim = None;
    if tokenizer_ok && text_model_ok {
        if let Ok(text_vec) = run_clip_text_embedding(&config, "prueba clip") {
            if !text_vec.is_empty() {
                text_inference_ok = true;
                text_dim = Some(text_vec.len());
            }
        }
    }

    let auto_sample = state
        .index
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|snapshot| snapshot.files.clone()))
        .unwrap_or_default()
        .into_iter()
        .map(|item| item.path)
        .find(|path| is_image_path(&PathBuf::from(path)));

    let sample_path = sample_image_path
        .filter(|path| !path.trim().is_empty())
        .or(auto_sample);

    let mut image_inference_ok = false;
    let mut image_dim = None;
    if image_model_ok {
        if let Some(path) = sample_path.as_ref() {
            if let Ok(image_vec) = run_clip_image_embedding(&config, path) {
                if !image_vec.is_empty() {
                    image_inference_ok = true;
                    image_dim = Some(image_vec.len());
                }
            }
        }
    }

    let message = if tokenizer_ok && text_model_ok && image_model_ok && text_inference_ok {
        if sample_path.is_some() {
            if image_inference_ok {
                "CLIP validado: tokenizer/modelos OK e inferencia texto+imagen operativa".to_string()
            } else {
                "CLIP parcial: texto OK, pero inferencia de imagen falló con la muestra".to_string()
            }
        } else {
            "CLIP validado: texto OK; falta imagen de muestra para validar encoder visual".to_string()
        }
    } else {
        "CLIP inválido: revisa rutas/modelos/tokenizer y compatibilidad ONNX".to_string()
    };

    append_audit_log(
        state.inner(),
        "clip.validate",
        format!(
            "tokenizer_ok={} text_model_ok={} image_model_ok={} text_inference_ok={} image_inference_ok={}",
            tokenizer_ok, text_model_ok, image_model_ok, text_inference_ok, image_inference_ok
        ),
    );

    ClipValidationStatus {
        configured: true,
        tokenizer_ok,
        text_model_ok,
        image_model_ok,
        text_inference_ok,
        image_inference_ok,
        text_dim,
        image_dim,
        sample_image_path: sample_path,
        message,
    }
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
    images_only: Option<bool>,
) -> Result<Vec<SearchResultItem>, String> {
    let started_at = Instant::now();
    let clean_query = query.trim();
    if clean_query.is_empty() {
        return Ok(Vec::new());
    }

    let search_roots = resolve_roots(roots);
    let exclusions = normalize_extensions(excluded_extensions);
    let folder_exclusions = normalize_folder_rules(excluded_folders);
    let max_file_size_bytes = max_size_bytes(max_file_size_mb);
    let images_only = images_only.unwrap_or(false);
    let adaptive_enabled = state
        .performance_runtime
        .lock()
        .map(|runtime| runtime.config.adaptive_enabled)
        .unwrap_or(true);
    let hardware = detect_hardware_profile();
    let pressure = runtime_pressure_snapshot();
    let candidate_limit = if adaptive_enabled {
        adaptive_candidate_limit(clean_query, images_only, &hardware, pressure.memory_pressure_pct)
    } else {
        80
    };

    let decision = if adaptive_enabled {
        format!(
            "adaptive:on cores={} ram={:.1}GB query_tokens={} images_only={} candidate_limit={}",
            hardware.cpu_cores,
            hardware.total_memory_gb,
            tokenize_query(clean_query).len(),
            images_only,
            candidate_limit,
        )
    } else {
        format!(
            "adaptive:off candidate_limit={} images_only={}",
            candidate_limit,
            images_only
        )
    };
    update_performance_decision(state.inner(), decision, Some(candidate_limit), None);

    let mut candidates = match load_chunk_candidates_from_lancedb(
        &app,
        clean_query,
        &search_roots,
        &exclusions,
        &folder_exclusions,
        max_file_size_bytes,
        candidate_limit,
        images_only,
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
            candidate_limit,
            images_only,
        )?,
    };

    if candidates.is_empty() {
        record_runtime_metric(state.inner(), "semantic", started_at.elapsed().as_millis() as u64);
        append_audit_log(
            state.inner(),
            "search.semantic",
            format!("results=0 images_only={images_only} query_len={}", clean_query.chars().count()),
        );
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

        let vectors = request_embeddings(state.inner(), &config, &inputs).await?;
        if vectors.len() == inputs.len() {
            let query_vector = &vectors[0];
            let mut scored = Vec::new();

            for (index, candidate) in candidates.drain(..).enumerate() {
                let semantic = cosine_similarity(query_vector, &vectors[index + 1]);
                let recency = compute_recency_signal(candidate.modified_unix_secs);
                let blended = semantic * 0.66 + candidate.lexical_score * 0.26 + recency * 0.08;
                scored.push((blended, candidate));
            }

            scored.sort_by(|left, right| right.0.total_cmp(&left.0));

            let results: Vec<SearchResultItem> = scored
                .into_iter()
                .take(max_results)
                .map(|(score, item)| {
                    let path = item.path;
                    let title = item.title;
                    let chunk_text = item.chunk_text;

                    SearchResultItem {
                        title,
                        path: path.clone(),
                        snippet: format!(
                            "Semántico {:.2} · {}",
                            score,
                            build_chunk_snippet(&chunk_text, clean_query)
                        ),
                        match_reason: format!(
                            "{} (score {:.2}).",
                            semantic_reason_for_path(&path, "Similaridad semántica alta con la consulta"),
                            score,
                        ),
                        origin: "cloud-semantic".to_string(),
                    }
                })
                .collect();

            record_runtime_metric(state.inner(), "semantic", started_at.elapsed().as_millis() as u64);
            append_audit_log(
                state.inner(),
                "search.semantic",
                format!("results={} mode=cloud images_only={images_only}", results.len()),
            );

            return Ok(results);
        }
    }

    let mut lexical = candidates
        .into_iter()
        .map(|item| {
            let path = item.path;
            let title = item.title;
            let chunk_text = item.chunk_text;
            let lexical_score = item.lexical_score;

            SearchResultItem {
                title,
                path: path.clone(),
                snippet: format!(
                    "Léxico {:.2} · {}",
                    lexical_score,
                    build_chunk_snippet(&chunk_text, clean_query)
                ),
                match_reason: format!(
                    "{} (score {:.2}).",
                    lexical_reason_for_path(&path, "Coincidencia léxica local en chunks indexados"),
                    lexical_score
                ),
                origin: "local-lexical".to_string(),
            }
        })
        .collect::<Vec<_>>();

    lexical.truncate(max_results);
    record_runtime_metric(state.inner(), "semantic", started_at.elapsed().as_millis() as u64);
    append_audit_log(
        state.inner(),
        "search.semantic",
        format!("results={} mode=local images_only={images_only}", lexical.len()),
    );
    Ok(lexical)
}

#[tauri::command]
async fn answer_with_local_context(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    query: String,
    roots: Option<Vec<String>>,
    excluded_extensions: Option<Vec<String>>,
    excluded_folders: Option<Vec<String>>,
    max_file_size_mb: Option<u64>,
    top_k: Option<usize>,
    mode: Option<String>,
    strict_grounding: Option<bool>,
    min_score: Option<f32>,
) -> Result<RagAnswerResponse, String> {
    let started_at = Instant::now();
    let clean_query = query.trim();
    if clean_query.is_empty() {
        return Ok(RagAnswerResponse {
            answer: "Escribe una consulta para poder responder con evidencia local.".to_string(),
            grounded: false,
            mode: "local".to_string(),
            sources: Vec::new(),
        });
    }

    let requested_mode = mode
        .unwrap_or_else(|| "auto".to_string())
        .trim()
        .to_lowercase();
    let requested_mode = if requested_mode == "cloud" || requested_mode == "local" || requested_mode == "auto" {
        requested_mode
    } else {
        "auto".to_string()
    };

    let search_roots = resolve_roots(roots);
    let exclusions = normalize_extensions(excluded_extensions);
    let folder_exclusions = normalize_folder_rules(excluded_folders);
    let max_file_size_bytes = max_size_bytes(max_file_size_mb);
    let adaptive_enabled = state
        .performance_runtime
        .lock()
        .map(|runtime| runtime.config.adaptive_enabled)
        .unwrap_or(true);
    let hardware = detect_hardware_profile();
    let pressure = runtime_pressure_snapshot();
    let adaptive_top_k = adaptive_rag_top_k(&hardware);
    let max_sources = top_k
        .unwrap_or(if adaptive_enabled { adaptive_top_k } else { 4 })
        .clamp(1, 8);
    let strict_grounding = strict_grounding.unwrap_or(true);
    let min_score = min_score.unwrap_or(0.55).clamp(0.0, 1.6);
    let rag_candidate_cap = max_sources
        .saturating_mul(if adaptive_enabled { 5 } else { 4 })
        .max(20)
        .min(if pressure.memory_pressure_pct >= 88.0 { 30 } else { 60 });

    update_performance_decision(
        state.inner(),
        format!(
            "rag adaptive={} top_k={} candidate_cap={} strict={} min_score={:.2}",
            adaptive_enabled,
            max_sources,
            rag_candidate_cap,
            strict_grounding,
            min_score
        ),
        None,
        Some(max_sources),
    );

    let candidates = match load_chunk_candidates_from_lancedb(
        &app,
        clean_query,
        &search_roots,
        &exclusions,
        &folder_exclusions,
        max_file_size_bytes,
        rag_candidate_cap,
        false,
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
            rag_candidate_cap,
            false,
        )?,
    };

    if candidates.is_empty() {
        let response = RagAnswerResponse {
            answer: "No tengo evidencia suficiente en tu índice local para responder esta consulta. Prueba reindexar o cambiar términos.".to_string(),
            grounded: false,
            mode: "local".to_string(),
            sources: Vec::new(),
        };

        let _ = append_chat_history(
            &app,
            &ChatHistoryItem {
                id: now_millis_string(),
                timestamp: now_timestamp_string(),
                query: clean_query.to_string(),
                answer: response.answer.clone(),
                grounded: false,
                mode: response.mode.clone(),
                sources: Vec::new(),
            },
        );

        return Ok(response);
    }

    let selected = candidates
        .into_iter()
        .take(max_sources)
        .collect::<Vec<_>>();

    let sources = selected
        .iter()
        .enumerate()
        .map(|(index, item)| RagSourceItem {
            ref_id: format!("S{}", index + 1),
            title: item.title.clone(),
            path: item.path.clone(),
            snippet: build_chunk_snippet(&item.chunk_text, clean_query),
            score: item.lexical_score,
        })
        .collect::<Vec<_>>();

    let strongest_score = selected
        .iter()
        .map(|item| item.lexical_score)
        .fold(0.0f32, f32::max);

    if strict_grounding && strongest_score < min_score {
        let response = RagAnswerResponse {
            answer: format!(
                "No tengo evidencia local suficiente para responder con confianza (score {:.2} < {:.2}). Ajusta la consulta, aumenta k o reindexa.",
                strongest_score,
                min_score
            ),
            grounded: false,
            mode: "local".to_string(),
            sources,
        };

        let _ = append_chat_history(
            &app,
            &ChatHistoryItem {
                id: now_millis_string(),
                timestamp: now_timestamp_string(),
                query: clean_query.to_string(),
                answer: response.answer.clone(),
                grounded: response.grounded,
                mode: response.mode.clone(),
                sources: response.sources.clone(),
            },
        );

        return Ok(response);
    }

    let local_answer = build_local_grounded_answer(clean_query, &selected);

    let ai_cfg = state
        .ai_config
        .lock()
        .ok()
        .and_then(|value| value.as_ref().cloned());

    let mut final_answer = local_answer.clone();
    let mut used_mode = "local".to_string();

    let should_try_cloud = requested_mode == "cloud" || requested_mode == "auto";

    if should_try_cloud {
        if let Some(config) = ai_cfg {
            let context = selected
                .iter()
                .enumerate()
                .map(|(index, item)| {
                    format!(
                        "[Fuente {}] {}\nRuta: {}\nContenido: {}",
                        index + 1,
                        item.title,
                        item.path,
                        normalize_text_for_index(&item.chunk_text, 900)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n");

            match request_chat_answer(&config, clean_query, &context).await {
                Ok(answer) if !answer.trim().is_empty() => {
                    if cloud_answer_has_citations(&answer) {
                        final_answer = answer;
                        used_mode = "cloud".to_string();
                    } else {
                        final_answer = format!(
                            "El proveedor cloud respondió sin citas inline [S#]. Fallback local:\n{}",
                            local_answer
                        );
                    }
                }
                Ok(_) => {
                    if requested_mode == "cloud" {
                        final_answer = format!(
                            "El proveedor cloud no devolvió contenido útil. Fallback local:\n{}",
                            local_answer
                        );
                    }
                }
                Err(err) => {
                    if requested_mode == "cloud" {
                        final_answer = format!(
                            "No se pudo responder en modo cloud ({err}). Fallback local:\n{}",
                            local_answer
                        );
                    }
                }
            }
        } else if requested_mode == "cloud" {
            final_answer = format!(
                "No hay proveedor cloud configurado; se usa fallback local.\n{}",
                local_answer
            );
        }
    }

    let response = RagAnswerResponse {
        answer: final_answer,
        grounded: true,
        mode: used_mode,
        sources,
    };

    let _ = append_chat_history(
        &app,
        &ChatHistoryItem {
            id: now_millis_string(),
            timestamp: now_timestamp_string(),
            query: clean_query.to_string(),
            answer: response.answer.clone(),
            grounded: response.grounded,
            mode: response.mode.clone(),
            sources: response.sources.clone(),
        },
    );

    record_runtime_metric(state.inner(), "rag", started_at.elapsed().as_millis() as u64);
    append_audit_log(
        state.inner(),
        "rag.answer",
        format!(
            "mode={} grounded={} sources={} strict={} min_score={:.2}",
            response.mode,
            response.grounded,
            response.sources.len(),
            strict_grounding,
            min_score
        ),
    );

    Ok(response)
}

#[tauri::command]
fn get_chat_history(app: tauri::AppHandle, limit: Option<usize>) -> Result<Vec<ChatHistoryItem>, String> {
    let history = load_chat_history(&app).unwrap_or_default();
    let max = limit.unwrap_or(20).clamp(1, 100);
    Ok(history
        .into_iter()
        .rev()
        .take(max)
        .collect::<Vec<_>>())
}

#[tauri::command]
fn clear_chat_history(app: tauri::AppHandle) -> Result<(), String> {
    let path = chat_history_path(&app).ok_or_else(|| "No se pudo resolver ruta de historial".to_string())?;
    fs::write(path, "[]").map_err(|err| format!("No se pudo limpiar historial: {err}"))
}

#[tauri::command]
fn export_chat_history_to_file(
    app: tauri::AppHandle,
    path: String,
    limit: Option<usize>,
) -> Result<String, String> {
    let max = limit.unwrap_or(100).clamp(1, 300);
    let history = load_chat_history(&app).unwrap_or_default();
    let selected = history
        .into_iter()
        .rev()
        .take(max)
        .collect::<Vec<_>>();

    let target = PathBuf::from(path);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("No se pudo crear carpeta de exportación: {err}"))?;
    }

    let payload = serde_json::to_string_pretty(&selected)
        .map_err(|err| format!("No se pudo serializar historial: {err}"))?;

    fs::write(&target, payload)
        .map_err(|err| format!("No se pudo exportar historial: {err}"))?;

    Ok(target.to_string_lossy().to_string())
}

#[tauri::command]
fn get_index_status(state: State<'_, AppState>) -> IndexStatus {
    current_index_status(state.inner())
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
    let trimmed = root.trim();
    if trimmed.is_empty() {
        return Err("Ruta de carpeta vacía".to_string());
    }

    let normalized = trimmed.to_lowercase();

    let updated_settings = {
        let mut settings = state
            .index_settings
            .lock()
            .map_err(|_| "No se pudo actualizar configuración de indexado".to_string())?;

        let before_len = settings.roots.len();
        settings
            .roots
            .retain(|value| value.trim().to_lowercase() != normalized);

        if settings.roots.len() == before_len {
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
fn export_app_config_to_file(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    path: String,
    include_secrets: Option<bool>,
) -> Result<String, String> {
    let settings = state
        .index_settings
        .lock()
        .map_err(|_| "No se pudo leer configuración de indexado".to_string())?
        .clone();

    let include = include_secrets.unwrap_or(false);
    let ai_provider = if include {
        state
            .ai_config
            .lock()
            .ok()
            .and_then(|cfg| cfg.as_ref().cloned())
    } else {
        None
    };

    let payload = ExportableAppConfig {
        version: 1,
        roots: settings.roots,
        excluded_extensions: settings.excluded_extensions,
        excluded_folders: settings.excluded_folders,
        max_file_size_mb: settings.max_file_size_mb,
        ai_provider,
        performance_runtime: state
            .performance_runtime
            .lock()
            .ok()
            .map(|runtime| runtime.config.clone()),
    };

    let serialized = serde_json::to_string_pretty(&payload)
        .map_err(|err| format!("No se pudo serializar configuración: {err}"))?;

    let target = PathBuf::from(path);
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("No se pudo crear carpeta destino: {err}"))?;
    }

    fs::write(&target, serialized).map_err(|err| format!("No se pudo exportar configuración: {err}"))?;

    let _ = app;
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
        .map_err(|err| format!("No se pudo leer configuración importada: {err}"))?;

    let parsed = serde_json::from_str::<ExportableAppConfig>(&content)
        .map_err(|err| format!("JSON de configuración inválido: {err}"))?;

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
            .map_err(|_| "No se pudo aplicar configuración importada".to_string())?;
        *settings = sanitized.clone();
    }

    if let Some(ai_cfg) = parsed.ai_provider {
        if !ai_cfg.api_key.trim().is_empty() {
            save_ai_provider_config(&app, &ai_cfg)?;
            if let Ok(mut guard) = state.ai_config.lock() {
                *guard = Some(ai_cfg);
            }
        }
    }

    if let Some(perf_cfg) = parsed.performance_runtime {
        if let Ok(mut runtime) = state.performance_runtime.lock() {
            runtime.config.adaptive_enabled = perf_cfg.adaptive_enabled;
            let _ = save_performance_runtime_config(&app, &runtime.config);
        }
    }

    if reindex.unwrap_or(true) {
        if sanitized.roots.is_empty() {
            return Ok(current_index_status(state.inner()));
        }

        execute_indexing(&app, state.inner(), sanitized, "manual")
    } else {
        Ok(current_index_status(state.inner()))
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
    images_only: Option<bool>,
) -> Vec<SearchResultItem> {
    let clean_query = query.trim();

    if clean_query.is_empty() {
        return Vec::new();
    }

    let search_roots = resolve_roots(roots);
    let exclusions = normalize_extensions(excluded_extensions);
    let folder_exclusions = normalize_folder_rules(excluded_folders);
    let max_file_size_bytes = max_size_bytes(max_file_size_mb);
    let images_only = images_only.unwrap_or(false);

    if let Ok(db_results) = search_in_chunk_db(
        &app,
        clean_query,
        &search_roots,
        &exclusions,
        &folder_exclusions,
        max_file_size_bytes,
        30,
        images_only,
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
        images_only,
    ) {
        return index_results;
    }

    search_local_files(
        clean_query,
        &search_roots,
        &exclusions,
        &folder_exclusions,
        max_file_size_bytes,
        images_only,
    )
}

fn execute_indexing(
    app: &tauri::AppHandle,
    state: &AppState,
    settings: IndexingSettings,
    reason: &str,
) -> Result<IndexStatus, String> {
    let started_at = Instant::now();
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

        let _ = persist_embedding_model_version(app, state);

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

    let elapsed_ms = started_at.elapsed().as_millis() as u64;
    record_runtime_metric(state, "indexing", elapsed_ms);

    match &result {
        Ok(status) => {
            append_audit_log(
                state,
                "indexing.completed",
                format!(
                    "reason={reason} indexed_files={} roots={} elapsed_ms={elapsed_ms}",
                    status.indexed_files,
                    status.roots.len()
                ),
            );
        }
        Err(err) => {
            append_audit_log(
                state,
                "indexing.failed",
                format!("reason={reason} elapsed_ms={elapsed_ms} error={}", sanitize_for_log(err)),
            );
        }
    }

    result
}

fn current_index_status(state: &AppState) -> IndexStatus {
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

fn clear_index_data_internal(app: &tauri::AppHandle, state: &AppState) -> Result<IndexStatus, String> {
    state.cancel_indexing.store(true, Ordering::SeqCst);

    {
        let mut index_guard = state
            .index
            .lock()
            .map_err(|_| "No se pudo limpiar snapshot de índice".to_string())?;
        *index_guard = None;
    }

    {
        let mut diagnostics_guard = state
            .diagnostics
            .lock()
            .map_err(|_| "No se pudo limpiar diagnóstico".to_string())?;
        *diagnostics_guard = Some(IndexDiagnostics {
            updated_at: Some(now_timestamp_string()),
            ..IndexDiagnostics::default()
        });
    }

    if let Ok(mut settings) = state.index_settings.lock() {
        settings.roots.clear();
    }

    if let Ok(mut watcher) = state.watcher_status.lock() {
        watcher.pending_events = false;
        watcher.roots.clear();
    }

    if let Some(path) = index_cache_path(app) {
        let _ = fs::remove_file(path);
    }

    if let Some(path) = semantic_db_path(app) {
        let _ = fs::remove_file(path);
    }

    if let Some(path) = lancedb_dir_path(app) {
        let _ = fs::remove_dir_all(&path);
        let _ = fs::create_dir_all(&path);
    }

    if let Ok(mut text_cache) = state.text_embedding_cache.lock() {
        text_cache.clear();
    }
    if let Ok(mut clip_cache) = state.clip_image_cache.lock() {
        clip_cache.clear();
    }
    if let Ok(mut metrics) = state.runtime_metrics.lock() {
        metrics.embedding_cache_hits = 0;
        metrics.embedding_cache_misses = 0;
    }

    append_audit_log(
        state,
        "indexing.cleared",
        "snapshot+sqlite+lancedb+cache reset".to_string(),
    );

    Ok(current_index_status(state))
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
                    status.last_error = Some(sanitize_for_log(&format!("Evento watcher inválido: {err}")));
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
                        .map(|value| value.elapsed() >= Duration::from_millis(effective_debounce_ms))
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
                        .map(|value| value.clone())
                        .unwrap_or_else(|_| default_indexing_settings());

                    if let Err(err) = execute_indexing(&app_handle, app_state.inner(), settings, "watcher") {
                        update_watcher_status(app_state.inner(), |status| {
                            status.last_error = Some(sanitize_for_log(&err));
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
    matches!(
        kind,
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) | EventKind::Any
    )
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

fn search_in_index(
    state: State<'_, AppState>,
    query: &str,
    roots: &[PathBuf],
    excluded_extensions: &[String],
    excluded_folders: &[String],
    max_file_size_bytes: u64,
    images_only: bool,
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

        if images_only && !is_image_path(&path) {
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
            match_reason: if is_image_path(&path) {
                "Coincidencia de tokens sobre metadata de imagen indexada localmente.".to_string()
            } else {
                "Coincidencia de tokens en índice local persistente.".to_string()
            },
            origin: "local-index".to_string(),
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
    images_only: bool,
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

            if images_only && !is_image_path(&path) {
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
                match_reason: "El nombre del archivo contiene todos los términos de búsqueda.".to_string(),
                origin: "local-filename".to_string(),
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
                extract_indexable_text(&path, state)
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
                content_hash,
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

    extension == "pdf"
        || is_image_extension(&extension)
        || is_supported_text_extension(&extension)
}

fn compute_index_content_hash(path: &Path, extension: &str) -> Option<String> {
    const SAMPLE_BYTES: usize = 24 * 1024;

    let mut file = File::open(path).ok()?;
    let mut buffer = vec![0_u8; SAMPLE_BYTES];
    let read_bytes = file.read(&mut buffer).ok()?;
    buffer.truncate(read_bytes);

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    extension.hash(&mut hasher);
    buffer.hash(&mut hasher);

    Some(format!("{:016x}", hasher.finish()))
}

fn extract_indexable_text(path: &Path, state: &AppState) -> Option<String> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_lowercase())?;

    if is_image_extension(&extension) {
        let metadata_text = extract_image_metadata_text(path);
        let ocr_text = extract_image_ocr_text(path, state);

        return match (metadata_text, ocr_text) {
            (Some(metadata), Some(ocr)) => {
                let merged = format!("{} {}", metadata, ocr);
                let normalized = normalize_text_for_index(&merged, 4_600);
                if normalized.is_empty() {
                    None
                } else {
                    Some(normalized)
                }
            }
            (Some(metadata), None) => Some(metadata),
            (None, Some(ocr)) => Some(ocr),
            (None, None) => None,
        };
    }

    let normalized = match extension.as_str() {
        "docx" => extract_docx_text(path)
            .map(|content| normalize_text_for_index(&content, 120_000)),
        "odt" => extract_odt_text(path)
            .map(|content| normalize_text_for_index(&content, 120_000)),
        "rtf" => extract_rtf_text(path)
            .map(|content| normalize_text_for_index(&content, 120_000)),
        _ => extract_generic_text(path, &extension)
            .map(|content| normalize_text_for_index(&content, 120_000)),
    };

    let normalized = match normalized {
        Some(value) => value,
        None => return None,
    };

    if normalized.is_empty() { None } else { Some(normalized) }
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
            | "odt"
            | "rtf"
    )
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
    let control_ratio = (control_count as f32) / (len as f32);
    let nul_ratio = (nul_count as f32) / (len as f32);

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

fn extract_docx_text(path: &Path) -> Option<String> {
    let file = File::open(path).ok()?;
    let mut archive = zip::ZipArchive::new(file).ok()?;
    let mut combined = String::new();

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

    for entry in xml_entries {
        if let Ok(mut xml) = archive.by_name(entry) {
            if xml.size() > MAX_DOC_XML_READ_BYTES {
                continue;
            }

            let mut buffer = String::new();
            if xml.read_to_string(&mut buffer).is_ok() {
                if !combined.is_empty() {
                    combined.push('\n');
                }
                combined.push_str(&extract_text_from_xml(&buffer));
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
    let file = File::open(path).ok()?;
    let mut archive = zip::ZipArchive::new(file).ok()?;
    let mut xml = archive.by_name("content.xml").ok()?;
    if xml.size() > MAX_DOC_XML_READ_BYTES {
        return None;
    }

    let mut content = String::new();
    xml.read_to_string(&mut content).ok()?;
    let extracted = extract_text_from_xml(&content);
    if extracted.trim().is_empty() {
        None
    } else {
        Some(extracted)
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

fn extract_rtf_text(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let slice = &bytes[..bytes.len().min(MAX_TEXT_READ_BYTES)];
    let content = String::from_utf8_lossy(slice);

    let mut output = String::new();
    let chars = content.chars().collect::<Vec<_>>();
    let mut index = 0usize;

    while index < chars.len() {
        let ch = chars[index];

        if ch == '\\' {
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

        if ch != '{' && ch != '}' {
            output.push(ch);
        }

        index += 1;
    }

    if output.trim().is_empty() {
        None
    } else {
        Some(output)
    }
}

fn extract_image_ocr_text(path: &Path, state: &AppState) -> Option<String> {
    let ai_config = state
        .ai_config
        .lock()
        .ok()
        .and_then(|value| value.as_ref().cloned())?;

    if ai_config.provider != "ollama-local" && ai_config.api_key.trim().is_empty() {
        return None;
    }

    let metadata = fs::metadata(path).ok()?;
    if metadata.len() > OCR_MAX_IMAGE_BYTES {
        return None;
    }

    let ocr = tauri::async_runtime::block_on(request_image_ocr_text(&ai_config, path)).ok()?;
    let normalized = normalize_text_for_index(&ocr, 4_000);

    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

fn image_mime_for_path(path: &Path) -> Option<&'static str> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_lowercase())?;

    match extension.as_str() {
        "png" => Some("image/png"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "webp" => Some("image/webp"),
        "gif" => Some("image/gif"),
        "bmp" => Some("image/bmp"),
        "tiff" | "tif" => Some("image/tiff"),
        _ => None,
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

fn semantic_reason_for_path(path: &str, base: &str) -> String {
    let path_buf = PathBuf::from(path);
    if is_image_path(&path_buf) {
        format!("{} sobre metadata de imagen", base)
    } else {
        base.to_string()
    }
}

fn lexical_reason_for_path(path: &str, base: &str) -> String {
    let path_buf = PathBuf::from(path);
    if is_image_path(&path_buf) {
        format!("{} sobre metadata de imagen", base)
    } else {
        base.to_string()
    }
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

fn persist_embedding_model_version(app: &tauri::AppHandle, state: &AppState) -> Result<(), String> {
    let embedding_model = state
        .ai_config
        .lock()
        .ok()
        .and_then(|cfg| cfg.as_ref().map(|value| value.embedding_model.clone()))
        .unwrap_or_else(|| "local-lexical".to_string());

    let conn = open_semantic_connection(app)?;
    let now_unix = now_timestamp_string().parse::<u64>().unwrap_or_default() as i64;
    conn.execute(
        "INSERT INTO metadata (key, value_text, updated_unix_secs)
         VALUES ('embedding_model', ?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value_text = excluded.value_text, updated_unix_secs = excluded.updated_unix_secs",
        params![embedding_model, now_unix],
    )
    .map_err(|err| format!("No se pudo persistir versión de embedding model: {err}"))?;

    Ok(())
}

fn ensure_semantic_schema(conn: &Connection) -> Result<(), String> {
    conn.execute_batch(
        "
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value_text TEXT,
                    value_int INTEGER,
                    updated_unix_secs INTEGER NOT NULL
                );

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
    .map_err(|err| format!("No se pudo preparar schema de chunks: {err}"))?;

    let now_unix = now_timestamp_string().parse::<u64>().unwrap_or_default() as i64;
    conn.execute(
        "INSERT INTO metadata (key, value_int, updated_unix_secs)
         VALUES ('schema_version', ?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value_int = excluded.value_int, updated_unix_secs = excluded.updated_unix_secs",
        params![SEMANTIC_SCHEMA_VERSION, now_unix],
    )
    .map_err(|err| format!("No se pudo persistir schema version: {err}"))?;

    Ok(())
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

fn compute_lexical_score(search_key: &str, tokens: &[String], lowered_query: &str) -> f32 {
    if tokens.is_empty() {
        return 0.0;
    }

    let mut hits = 0f32;
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

fn search_in_chunk_db(
    app: &tauri::AppHandle,
    query: &str,
    roots: &[PathBuf],
    excluded_extensions: &[String],
    excluded_folders: &[String],
    max_file_size_bytes: u64,
    max_results: usize,
    images_only: bool,
) -> Result<Vec<SearchResultItem>, String> {
    let candidates = load_chunk_candidates(
        app,
        query,
        roots,
        excluded_extensions,
        excluded_folders,
        max_file_size_bytes,
        max_results,
        images_only,
    )?;

    Ok(candidates
        .into_iter()
        .take(max_results)
        .map(|item| {
            let path = item.path;
            let title = item.title;
            let chunk_text = item.chunk_text;
            let lexical_score = item.lexical_score;

            SearchResultItem {
                title,
                path: path.clone(),
                snippet: build_chunk_snippet(&chunk_text, query),
                match_reason: format!(
                    "{} (score {:.2}).",
                    lexical_reason_for_path(&path, "Coincidencia léxica local en contenido indexado"),
                    lexical_score
                ),
                origin: "local-chunk".to_string(),
            }
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
    images_only: bool,
) -> Result<Vec<ChunkCandidate>, String> {
    let tokens = tokenize_query(query);
    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    let conn = open_semantic_connection(app)?;
    let mut sql = String::from(
        "SELECT title, path, chunk_text, search_key, size_bytes, modified_unix_secs FROM chunks",
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
                row.get::<_, i64>(5)?,
            ))
        })
        .map_err(|err| format!("No se pudo ejecutar consulta de chunks: {err}"))?;

    let lowered_query = query.to_lowercase();
    let mut by_path = HashMap::<String, ChunkCandidate>::new();

    for row in rows.flatten() {
        let (title, path, chunk_text, search_key, size_bytes_raw, modified_raw) = row;
        let size_bytes = size_bytes_raw.max(0) as u64;
        let modified_unix_secs = modified_raw.max(0) as u64;

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

        if images_only && !is_image_path(&path_buf) {
            continue;
        }

        let lexical_score = compute_lexical_score(&search_key, &tokens, &lowered_query);
        if lexical_score <= 0.0 {
            continue;
        }

        let current = by_path.get(&path).map(|value| value.lexical_score).unwrap_or(-1.0);

        if lexical_score > current {
            by_path.insert(
                path.clone(),
                ChunkCandidate {
                    title,
                    path,
                    chunk_text,
                    lexical_score,
                    modified_unix_secs,
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
    images_only: bool,
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
        let modified_col = batch
            .column_by_name("modified_unix_secs")
            .and_then(|col| col.as_any().downcast_ref::<Int64Array>())
            .ok_or_else(|| "LanceDB: columna modified_unix_secs inválida".to_string())?;

        for row in 0..batch.num_rows() {
            if title_col.is_null(row)
                || path_col.is_null(row)
                || chunk_text_col.is_null(row)
                || search_key_col.is_null(row)
                || size_col.is_null(row)
                || modified_col.is_null(row)
            {
                continue;
            }

            let title = title_col.value(row).to_string();
            let path = path_col.value(row).to_string();
            let chunk_text = chunk_text_col.value(row).to_string();
            let search_key = search_key_col.value(row).to_string();
            let size_bytes = size_col.value(row).max(0) as u64;
            let modified_unix_secs = modified_col.value(row).max(0) as u64;

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

            if images_only && !is_image_path(&path_buf) {
                continue;
            }

            let lexical_score = compute_lexical_score(&search_key, &tokens, &lowered_query);
            if lexical_score <= 0.0 {
                continue;
            }

            let current = by_path.get(&path).map(|value| value.lexical_score).unwrap_or(-1.0);

            if lexical_score > current {
                by_path.insert(
                    path.clone(),
                    ChunkCandidate {
                        title,
                        path,
                        chunk_text,
                        lexical_score,
                        modified_unix_secs,
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

fn build_local_grounded_answer(query: &str, candidates: &[ChunkCandidate]) -> String {
    if candidates.is_empty() {
        return "No tengo evidencia local suficiente para responder con confianza.".to_string();
    }

    let mut lines = Vec::new();
    lines.push("Respuesta basada en tus archivos locales:".to_string());

    for (index, candidate) in candidates.iter().take(3).enumerate() {
        let summary = build_compact_evidence_line(&candidate.chunk_text, query);
        lines.push(format!("- {} [S{}]", summary, index + 1));
    }

    lines.push("Si necesitas más precisión, abre las fuentes citadas [S1..] y vuelve a preguntar con términos más específicos.".to_string());
    lines.join("\n")
}

fn cloud_answer_has_citations(answer: &str) -> bool {
    let lowered = answer.to_lowercase();
    lowered.contains("[s1") || lowered.contains("[s2") || lowered.contains("[s3") || lowered.contains("[s4")
}

fn build_compact_evidence_line(content: &str, query: &str) -> String {
    if content.trim().is_empty() {
        return "Se detectó una coincidencia relevante en el contenido indexado.".to_string();
    }

    let lowered_content = content.to_lowercase();
    let tokens = tokenize_query(query);
    let position = tokens
        .iter()
        .filter_map(|token| lowered_content.find(token))
        .min()
        .unwrap_or(0);

    let start_char = lowered_content[..position].chars().count().saturating_sub(36);
    let excerpt = content
        .chars()
        .skip(start_char)
        .take(180)
        .collect::<String>();

    normalize_text_for_index(&excerpt, 180)
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
            chat_base_url: cfg.chat_base_url.clone(),
            chat_model: cfg.chat_model.clone(),
            api_key_hint: if cfg.api_key.trim().is_empty() {
                None
            } else {
                Some(mask_api_key(&cfg.api_key))
            },
        };
    }

    AiProviderStatus {
        configured: false,
        provider: "openrouter-compatible".to_string(),
        base_url: "https://openrouter.ai/api/v1/embeddings".to_string(),
        embedding_model: "text-embedding-3-small".to_string(),
        chat_base_url: default_chat_base_url_value(),
        chat_model: default_chat_model_value(),
        api_key_hint: None,
    }
}

fn default_mode_value() -> String {
    "local".to_string()
}

fn clip_status_from_config(config: Option<&ClipOnnxConfig>) -> ClipOnnxStatus {
    if let Some(cfg) = config {
        return ClipOnnxStatus {
            configured: true,
            enabled: cfg.enabled,
            image_model_path: cfg.image_model_path.clone(),
            text_model_path: cfg.text_model_path.clone(),
            tokenizer_path: cfg.tokenizer_path.clone(),
            input_size: cfg.input_size,
            max_length: cfg.max_length,
        };
    }

    ClipOnnxStatus {
        configured: false,
        enabled: false,
        image_model_path: String::new(),
        text_model_path: String::new(),
        tokenizer_path: String::new(),
        input_size: 224,
        max_length: 77,
    }
}

fn load_or_compute_clip_image_embedding(
    state: &AppState,
    config: &ClipOnnxConfig,
    path: &str,
) -> Result<Vec<f32>, String> {
    if let Ok(cache) = state.clip_image_cache.lock() {
        if let Some(value) = cache.get(path) {
            return Ok(value.clone());
        }
    }

    let embedding = run_clip_image_embedding(config, path)?;

    if let Ok(mut cache) = state.clip_image_cache.lock() {
        bounded_cache_insert(
            &mut cache,
            path.to_string(),
            embedding.clone(),
            CLIP_IMAGE_CACHE_MAX_ITEMS,
        );
    }

    Ok(embedding)
}

fn run_clip_text_embedding(config: &ClipOnnxConfig, query: &str) -> Result<Vec<f32>, String> {
    let tokenizer = Tokenizer::from_file(&config.tokenizer_path)
        .map_err(|err| format!("No se pudo cargar tokenizer CLIP: {err}"))?;
    let encoding = tokenizer
        .encode(query, true)
        .map_err(|err| format!("No se pudo tokenizar consulta CLIP: {err}"))?;

    let mut ids = encoding
        .get_ids()
        .iter()
        .map(|value| *value as i64)
        .collect::<Vec<i64>>();
    let mut mask = encoding
        .get_attention_mask()
        .iter()
        .map(|value| *value as i64)
        .collect::<Vec<i64>>();

    ids.truncate(config.max_length);
    mask.truncate(config.max_length);

    while ids.len() < config.max_length {
        ids.push(0);
    }
    while mask.len() < config.max_length {
        mask.push(0);
    }

    let input_ids = tract_ndarray::Array2::from_shape_vec((1, config.max_length), ids)
        .map_err(|err| format!("Input ids CLIP inválidos: {err}"))?;
    let attention_mask = tract_ndarray::Array2::from_shape_vec((1, config.max_length), mask)
        .map_err(|err| format!("Attention mask CLIP inválida: {err}"))?;

    let model = tract_onnx::onnx()
        .model_for_path(&config.text_model_path)
        .map_err(|err| format!("No se pudo abrir modelo de texto CLIP: {err}"))?
        .into_optimized()
        .map_err(|err| format!("No se pudo optimizar modelo de texto CLIP: {err}"))?
        .into_runnable()
        .map_err(|err| format!("No se pudo inicializar modelo de texto CLIP: {err}"))?;

    let outputs = model
        .run(tvec!(
            input_ids.into_tensor().into(),
            attention_mask.into_tensor().into()
        ))
        .map_err(|err| format!("Inferencia CLIP texto falló: {err}"))?;

    tensor_to_embedding(&outputs[0])
}

fn run_clip_image_embedding(config: &ClipOnnxConfig, path: &str) -> Result<Vec<f32>, String> {
    let image = image::open(path).map_err(|err| format!("No se pudo abrir imagen CLIP: {err}"))?;
    let resized = image
        .resize_exact(
            config.input_size,
            config.input_size,
            image::imageops::FilterType::CatmullRom,
        )
        .to_rgb8();

    let size = config.input_size as usize;
    let mut data = vec![0f32; 1 * 3 * size * size];
    let means = [0.48145466f32, 0.4578275, 0.40821073];
    let stds = [0.26862954f32, 0.26130258, 0.27577711];

    for y in 0..size {
        for x in 0..size {
            let pixel = resized.get_pixel(x as u32, y as u32);
            let idx = y * size + x;

            for c in 0..3 {
                let raw = pixel[c] as f32 / 255.0;
                let normalized = (raw - means[c]) / stds[c];
                data[c * size * size + idx] = normalized;
            }
        }
    }

    let input = tract_ndarray::Array4::from_shape_vec((1, 3, size, size), data)
        .map_err(|err| format!("Tensor de imagen CLIP inválido: {err}"))?;

    let model = tract_onnx::onnx()
        .model_for_path(&config.image_model_path)
        .map_err(|err| format!("No se pudo abrir modelo de imagen CLIP: {err}"))?
        .into_optimized()
        .map_err(|err| format!("No se pudo optimizar modelo de imagen CLIP: {err}"))?
        .into_runnable()
        .map_err(|err| format!("No se pudo inicializar modelo de imagen CLIP: {err}"))?;

    let outputs = model
        .run(tvec!(input.into_tensor().into()))
        .map_err(|err| format!("Inferencia CLIP imagen falló: {err}"))?;

    tensor_to_embedding(&outputs[0])
}

fn tensor_to_embedding(value: &TValue) -> Result<Vec<f32>, String> {
    let tensor = value
        .to_array_view::<f32>()
        .map_err(|err| format!("Salida tensor CLIP inválida: {err}"))?;

    let shape = tensor.shape().to_vec();
    if shape.is_empty() {
        return Err("Salida CLIP vacía".to_string());
    }

    if shape.len() == 2 {
        let dim = shape[1];
        let mut out = Vec::with_capacity(dim);
        for index in 0..dim {
            out.push(tensor[[0, index]]);
        }
        return Ok(normalize_embedding(out));
    }

    if shape.len() == 3 {
        let seq = shape[1].max(1);
        let dim = shape[2];
        let mut out = vec![0f32; dim];

        for token_index in 0..seq {
            for dim_index in 0..dim {
                out[dim_index] += tensor[[0, token_index, dim_index]];
            }
        }

        for value in &mut out {
            *value /= seq as f32;
        }

        return Ok(normalize_embedding(out));
    }

    let flat = tensor.iter().copied().collect::<Vec<f32>>();
    if flat.is_empty() {
        return Err("Salida CLIP sin dimensiones útiles".to_string());
    }

    Ok(normalize_embedding(flat))
}

fn normalize_embedding(mut values: Vec<f32>) -> Vec<f32> {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= 0.0 {
        return values;
    }

    for value in &mut values {
        *value /= norm;
    }

    values
}

fn clip_config_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    let path = app.path().app_data_dir().ok()?.join("clip-onnx-config.json");

    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    Some(path)
}

fn save_clip_config(app: &tauri::AppHandle, config: &ClipOnnxConfig) -> Result<(), String> {
    let path = clip_config_path(app).ok_or_else(|| "No se pudo resolver ruta de config CLIP".to_string())?;
    let serialized = serde_json::to_string(config)
        .map_err(|err| format!("No se pudo serializar config CLIP: {err}"))?;
    fs::write(path, serialized).map_err(|err| format!("No se pudo guardar config CLIP: {err}"))
}

fn load_clip_config(app: &tauri::AppHandle) -> Option<ClipOnnxConfig> {
    let path = clip_config_path(app)?;
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str::<ClipOnnxConfig>(&content).ok()
}

fn performance_runtime_config_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    let path = app.path().app_data_dir().ok()?.join("performance-runtime.json");

    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    Some(path)
}

fn save_performance_runtime_config(app: &tauri::AppHandle, config: &PerformanceRuntimeConfig) -> Result<(), String> {
    let path = performance_runtime_config_path(app)
        .ok_or_else(|| "No se pudo resolver ruta de runtime performance".to_string())?;
    let serialized = serde_json::to_string(config)
        .map_err(|err| format!("No se pudo serializar runtime performance: {err}"))?;
    fs::write(path, serialized)
        .map_err(|err| format!("No se pudo guardar runtime performance: {err}"))
}

fn load_performance_runtime_config(app: &tauri::AppHandle) -> Option<PerformanceRuntimeConfig> {
    let path = performance_runtime_config_path(app)?;
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str::<PerformanceRuntimeConfig>(&content).ok()
}

fn adaptive_candidate_limit(
    query: &str,
    images_only: bool,
    hardware: &HardwareProfile,
    memory_pressure_pct: f32,
) -> usize {
    let tokens = tokenize_query(query).len();
    let mut limit = if hardware.cpu_cores <= 4 || hardware.total_memory_gb < 8.0 {
        48usize
    } else if hardware.cpu_cores >= 12 && hardware.total_memory_gb >= 24.0 {
        120usize
    } else {
        80usize
    };

    if tokens <= 2 {
        limit = limit.saturating_add(16);
    } else if tokens >= 8 {
        limit = limit.saturating_sub(12);
    }

    if images_only {
        limit = limit.min(72);
    }

    let factor = compute_runtime_throttling_factor(memory_pressure_pct);
    let throttled = (limit as f32 * factor).round() as usize;
    throttled.clamp(24, 140)
}

fn adaptive_clip_inspect_limit(memory_pressure_pct: f32) -> usize {
    let factor = compute_runtime_throttling_factor(memory_pressure_pct);
    ((900f32 * factor).round() as usize).clamp(180, 900)
}

struct RuntimePressureSnapshot {
    total_memory_gb: f32,
    available_memory_gb: f32,
    memory_pressure_pct: f32,
}

fn runtime_pressure_snapshot() -> RuntimePressureSnapshot {
    let mut system = System::new_all();
    system.refresh_memory();
    let total_kib = system.total_memory() as f32;
    let available_kib = system.available_memory() as f32;

    let total_gb = (total_kib / (1024.0 * 1024.0)).max(0.0);
    let available_gb = (available_kib / (1024.0 * 1024.0)).max(0.0);
    let pressure = if total_kib <= 0.0 {
        0.0
    } else {
        ((total_kib - available_kib).max(0.0) / total_kib * 100.0).clamp(0.0, 100.0)
    };

    RuntimePressureSnapshot {
        total_memory_gb: total_gb,
        available_memory_gb: available_gb,
        memory_pressure_pct: pressure,
    }
}

fn runtime_cpu_usage_pct() -> f32 {
    let mut system = System::new_all();
    system.refresh_cpu();
    let cpus = system.cpus();
    if cpus.is_empty() {
        return 0.0;
    }

    let total = cpus.iter().map(|cpu| cpu.cpu_usage()).sum::<f32>();
    (total / cpus.len() as f32).clamp(0.0, 100.0)
}

fn runtime_pressure_level(memory_pressure_pct: f32) -> &'static str {
    if memory_pressure_pct >= 88.0 {
        "high"
    } else if memory_pressure_pct >= 74.0 {
        "medium"
    } else {
        "low"
    }
}

fn compute_runtime_throttling_factor(memory_pressure_pct: f32) -> f32 {
    if memory_pressure_pct >= 92.0 {
        0.35
    } else if memory_pressure_pct >= 88.0 {
        0.45
    } else if memory_pressure_pct >= 80.0 {
        0.62
    } else if memory_pressure_pct >= 72.0 {
        0.8
    } else {
        1.0
    }
}

fn adaptive_rag_top_k(hardware: &HardwareProfile) -> usize {
    if hardware.cpu_cores <= 4 || hardware.total_memory_gb < 8.0 {
        3
    } else if hardware.cpu_cores >= 12 && hardware.total_memory_gb >= 24.0 {
        6
    } else {
        4
    }
}

fn update_performance_decision(
    state: &AppState,
    decision: String,
    candidate_limit: Option<usize>,
    rag_top_k: Option<usize>,
) {
    if let Ok(mut runtime) = state.performance_runtime.lock() {
        runtime.last_decision = decision;
        if let Some(value) = candidate_limit {
            runtime.last_candidate_limit = value;
        }
        if let Some(value) = rag_top_k {
            runtime.last_rag_top_k = value;
        }
    }
}

fn detect_clip_config(app: &tauri::AppHandle) -> Option<ClipOnnxConfig> {
    let image_candidates = [
        "models/clip/vision_model_quantized.onnx",
        "models/clip/vision_model.onnx",
        "resources/models/clip/vision_model_quantized.onnx",
        "resources/models/clip/vision_model.onnx",
        "clip models/vision_model_quantized.onnx",
        "clip models/vision_model.onnx",
        "clip models/image_model.onnx",
    ];

    let text_candidates = [
        "models/clip/text_model_quantized.onnx",
        "models/clip/text_model.onnx",
        "resources/models/clip/text_model_quantized.onnx",
        "resources/models/clip/text_model.onnx",
        "clip models/text_model_quantized.onnx",
        "clip models/text_model.onnx",
    ];

    let tokenizer_candidates = [
        "models/clip/tokenizer.json",
        "resources/models/clip/tokenizer.json",
        "clip models/tokenizer.json",
    ];

    let image_model_path = image_candidates
        .iter()
        .find_map(|candidate| resolve_clip_candidate_path(app, candidate))?;
    let text_model_path = text_candidates
        .iter()
        .find_map(|candidate| resolve_clip_candidate_path(app, candidate))?;
    let tokenizer_path = tokenizer_candidates
        .iter()
        .find_map(|candidate| resolve_clip_candidate_path(app, candidate))?;

    Some(ClipOnnxConfig {
        enabled: true,
        image_model_path: image_model_path.to_string_lossy().to_string(),
        text_model_path: text_model_path.to_string_lossy().to_string(),
        tokenizer_path: tokenizer_path.to_string_lossy().to_string(),
        input_size: 224,
        max_length: 77,
    })
}

fn resolve_clip_candidate_path(app: &tauri::AppHandle, relative: &str) -> Option<PathBuf> {
    let resource_resolved = app
        .path()
        .resolve(relative, BaseDirectory::Resource)
        .ok()
        .filter(|path| path.exists());

    if resource_resolved.is_some() {
        return resource_resolved;
    }

    let project_resolved = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative);
    if project_resolved.exists() {
        return Some(project_resolved);
    }

    None
}

fn default_chat_base_url_value() -> String {
    "https://openrouter.ai/api/v1/chat/completions".to_string()
}

fn default_chat_model_value() -> String {
    "gpt-4o-mini".to_string()
}

fn default_ollama_embeddings_url_value() -> String {
    "http://127.0.0.1:11434/api/embed".to_string()
}

fn default_ollama_chat_url_value() -> String {
    "http://127.0.0.1:11434/api/chat".to_string()
}

fn default_ollama_model_value() -> String {
    "nomic-embed-text".to_string()
}

fn default_ollama_chat_model_value() -> String {
    "llama3.2".to_string()
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

async fn request_embeddings(state: &AppState, config: &AiProviderConfig, input: &[String]) -> Result<Vec<Vec<f32>>, String> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut output = vec![Vec::<f32>::new(); input.len()];
    let mut misses = Vec::<(usize, String, String)>::new();

    if let Ok(cache) = state.text_embedding_cache.lock() {
        for (index, text) in input.iter().enumerate() {
            let key = embedding_cache_key(config, text);
            if let Some(found) = cache.get(&key) {
                output[index] = found.clone();
                if let Ok(mut metrics) = state.runtime_metrics.lock() {
                    metrics.embedding_cache_hits = metrics.embedding_cache_hits.saturating_add(1);
                }
            } else {
                misses.push((index, key, text.clone()));
            }
        }
    } else {
        for (index, text) in input.iter().enumerate() {
            let key = embedding_cache_key(config, text);
            misses.push((index, key, text.clone()));
        }
    }

    if misses.is_empty() {
        return Ok(output);
    }

    if let Ok(mut metrics) = state.runtime_metrics.lock() {
        metrics.embedding_cache_misses = metrics.embedding_cache_misses.saturating_add(misses.len() as u64);
    }

    let missing_input = misses
        .iter()
        .map(|(_, _, text)| text.clone())
        .collect::<Vec<_>>();

    if config.provider == "ollama-local" {
        let embeddings = request_ollama_embeddings(config, &missing_input).await?;
        if embeddings.len() != misses.len() {
            return Err("Ollama devolvió una cantidad inesperada de embeddings".to_string());
        }

        if let Ok(mut cache) = state.text_embedding_cache.lock() {
            for ((index, key, _), embedding) in misses.into_iter().zip(embeddings.into_iter()) {
                bounded_cache_insert(
                    &mut cache,
                    key,
                    embedding.clone(),
                    TEXT_EMBEDDING_CACHE_MAX_ITEMS,
                );
                output[index] = embedding;
            }
        } else {
            for ((index, _, _), embedding) in misses.into_iter().zip(embeddings.into_iter()) {
                output[index] = embedding;
            }
        }

        return Ok(output);
    }

    let payload = serde_json::json!({
        "model": config.embedding_model,
        "input": missing_input,
    });

    let client = reqwest::Client::new();
    let mut parsed: Option<EmbeddingResponse> = None;
    let mut last_error = String::new();

    for attempt in 0..CLOUD_RETRY_ATTEMPTS {
        let response = client
            .post(&config.base_url)
            .header("Authorization", format!("Bearer {}", config.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await;

        match response {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    let value = resp
                        .json::<EmbeddingResponse>()
                        .await
                        .map_err(|err| {
                            format!(
                                "Respuesta de embeddings inválida: {}",
                                sanitize_remote_error(&err.to_string())
                            )
                        })?;
                    parsed = Some(value);
                    break;
                }

                let body = resp.text().await.unwrap_or_default();
                last_error = format!(
                    "Embeddings respondieron {}: {}",
                    status.as_u16(),
                    sanitize_remote_error(&body).chars().take(220).collect::<String>()
                );

                if !is_retryable_http_status(status.as_u16()) || attempt + 1 >= CLOUD_RETRY_ATTEMPTS {
                    return Err(last_error);
                }
            }
            Err(err) => {
                last_error = format!(
                    "No se pudo llamar API de embeddings: {}",
                    sanitize_remote_error(&err.to_string())
                );
                if attempt + 1 >= CLOUD_RETRY_ATTEMPTS {
                    return Err(last_error);
                }
            }
        }
    }

    let parsed = parsed.ok_or_else(|| {
        if last_error.is_empty() {
            "No se pudo obtener embeddings tras reintentos".to_string()
        } else {
            last_error
        }
    })?;

    let embeddings = parsed
        .data
        .into_iter()
        .map(|item| item.embedding)
        .collect::<Vec<_>>();

    if embeddings.len() != misses.len() {
        return Err("Embeddings devolvieron una cantidad inesperada de vectores".to_string());
    }

    if let Ok(mut cache) = state.text_embedding_cache.lock() {
        for ((index, key, _), embedding) in misses.into_iter().zip(embeddings.into_iter()) {
            bounded_cache_insert(
                &mut cache,
                key,
                embedding.clone(),
                TEXT_EMBEDDING_CACHE_MAX_ITEMS,
            );
            output[index] = embedding;
        }
    } else {
        for ((index, _, _), embedding) in misses.into_iter().zip(embeddings.into_iter()) {
            output[index] = embedding;
        }
    }

    Ok(output)
}

async fn request_chat_answer(config: &AiProviderConfig, query: &str, context: &str) -> Result<String, String> {
    if config.provider == "ollama-local" {
        return request_chat_answer_ollama(config, query, context).await;
    }

    let payload = serde_json::json!({
        "model": config.chat_model,
        "messages": [
            {
                "role": "system",
                "content": "Responde solo con base en el contexto proporcionado. Si falta evidencia, dilo explícitamente. Sé breve y preciso. Incluye referencias inline tipo [S1], [S2] cuando uses evidencia."
            },
            {
                "role": "user",
                "content": format!("Consulta: {}\n\nContexto:\n{}", query, context)
            }
        ]
    });

    let client = reqwest::Client::new();
    let mut parsed: Option<ChatCompletionResponse> = None;
    let mut last_error = String::new();

    for attempt in 0..CLOUD_RETRY_ATTEMPTS {
        let response = client
            .post(&config.chat_base_url)
            .header("Authorization", format!("Bearer {}", config.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await;

        match response {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    let value = resp
                        .json::<ChatCompletionResponse>()
                        .await
                        .map_err(|err| {
                            format!(
                                "Respuesta de chat inválida: {}",
                                sanitize_remote_error(&err.to_string())
                            )
                        })?;
                    parsed = Some(value);
                    break;
                }

                let body = resp.text().await.unwrap_or_default();
                last_error = format!(
                    "Chat respondió {}: {}",
                    status.as_u16(),
                    sanitize_remote_error(&body).chars().take(220).collect::<String>()
                );

                if !is_retryable_http_status(status.as_u16()) || attempt + 1 >= CLOUD_RETRY_ATTEMPTS {
                    return Err(last_error);
                }
            }
            Err(err) => {
                last_error = format!(
                    "No se pudo llamar API de chat: {}",
                    sanitize_remote_error(&err.to_string())
                );
                if attempt + 1 >= CLOUD_RETRY_ATTEMPTS {
                    return Err(last_error);
                }
            }
        }
    }

    let parsed = parsed.ok_or_else(|| {
        if last_error.is_empty() {
            "No se pudo obtener respuesta de chat tras reintentos".to_string()
        } else {
            last_error
        }
    })?;

    let Some(choice) = parsed.choices.into_iter().next() else {
        return Err("Chat no devolvió opciones".to_string());
    };

    extract_chat_content(&choice.message.content)
}

fn extract_chat_content(value: &serde_json::Value) -> Result<String, String> {
    if let Some(text) = value.as_str() {
        return Ok(text.to_string());
    }

    if let Some(parts) = value.as_array() {
        let joined = parts
            .iter()
            .filter_map(|item| item.get("text").and_then(|text| text.as_str()))
            .collect::<Vec<_>>()
            .join("\n");

        if !joined.trim().is_empty() {
            return Ok(joined);
        }
    }

    Err("Contenido de chat no soportado".to_string())
}

async fn request_ollama_embeddings(config: &AiProviderConfig, input: &[String]) -> Result<Vec<Vec<f32>>, String> {
    let payload = serde_json::json!({
        "model": config.embedding_model,
        "input": input,
    });

    let client = reqwest::Client::new();
    let response = client
        .post(&config.base_url)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
        .map_err(|err| format!("No se pudo llamar embeddings local (Ollama): {}", sanitize_remote_error(&err.to_string())))?;

    let status = response.status();
    let raw = response
        .text()
        .await
        .map_err(|err| format!("No se pudo leer respuesta Ollama: {}", sanitize_remote_error(&err.to_string())))?;

    if !status.is_success() {
        return Err(format!(
            "Ollama embeddings respondió {}: {}",
            status.as_u16(),
            sanitize_remote_error(&raw).chars().take(220).collect::<String>()
        ));
    }

    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|err| format!("Respuesta Ollama embeddings inválida: {}", sanitize_remote_error(&err.to_string())))?;

    if let Some(values) = value.get("embeddings").and_then(|item| item.as_array()) {
        let mut parsed = Vec::with_capacity(values.len());
        for item in values {
            let Some(row) = item.as_array() else {
                continue;
            };
            let mut vector = Vec::with_capacity(row.len());
            for value in row {
                if let Some(f) = value.as_f64() {
                    vector.push(f as f32);
                }
            }
            if !vector.is_empty() {
                parsed.push(vector);
            }
        }
        return Ok(parsed);
    }

    if let Some(single) = value.get("embedding").and_then(|item| item.as_array()) {
        let mut vector = Vec::with_capacity(single.len());
        for value in single {
            if let Some(f) = value.as_f64() {
                vector.push(f as f32);
            }
        }
        if !vector.is_empty() {
            return Ok(vec![vector]);
        }
    }

    Err("Respuesta Ollama embeddings sin campos esperados (`embeddings` o `embedding`)".to_string())
}

async fn request_chat_answer_ollama(config: &AiProviderConfig, query: &str, context: &str) -> Result<String, String> {
    let payload = serde_json::json!({
        "model": config.chat_model,
        "stream": false,
        "messages": [
            {
                "role": "system",
                "content": "Responde solo con base en el contexto proporcionado. Si falta evidencia, dilo explícitamente. Sé breve y preciso. Incluye referencias inline tipo [S1], [S2] cuando uses evidencia."
            },
            {
                "role": "user",
                "content": format!("Consulta: {}\n\nContexto:\n{}", query, context)
            }
        ]
    });

    let client = reqwest::Client::new();
    let response = client
        .post(&config.chat_base_url)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
        .map_err(|err| format!("No se pudo llamar chat local (Ollama): {}", sanitize_remote_error(&err.to_string())))?;

    let status = response.status();
    let raw = response
        .text()
        .await
        .map_err(|err| format!("No se pudo leer respuesta chat local: {}", sanitize_remote_error(&err.to_string())))?;

    if !status.is_success() {
        return Err(format!(
            "Ollama chat respondió {}: {}",
            status.as_u16(),
            sanitize_remote_error(&raw).chars().take(220).collect::<String>()
        ));
    }

    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|err| format!("Respuesta Ollama chat inválida: {}", sanitize_remote_error(&err.to_string())))?;

    if let Some(text) = value
        .get("message")
        .and_then(|msg| msg.get("content"))
        .and_then(|content| content.as_str())
    {
        return Ok(text.to_string());
    }

    if let Some(text) = value.get("response").and_then(|resp| resp.as_str()) {
        return Ok(text.to_string());
    }

    Err("Respuesta Ollama chat sin contenido esperado".to_string())
}

async fn request_image_ocr_text(config: &AiProviderConfig, path: &Path) -> Result<String, String> {
    if config.provider == "ollama-local" {
        return request_image_ocr_text_ollama(config, path).await;
    }

    request_image_ocr_text_cloud(config, path).await
}

async fn request_image_ocr_text_cloud(config: &AiProviderConfig, path: &Path) -> Result<String, String> {
    let mime = image_mime_for_path(path).ok_or_else(|| "Formato de imagen no soportado para OCR".to_string())?;
    let bytes = fs::read(path).map_err(|err| format!("No se pudo leer imagen para OCR: {err}"))?;
    let data_url = format!(
        "data:{};base64,{}",
        mime,
        base64::engine::general_purpose::STANDARD.encode(bytes)
    );

    let payload = serde_json::json!({
        "model": config.chat_model,
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": "Eres un OCR estricto. Extrae únicamente el texto visible de la imagen en orden de lectura. No inventes contenido ni agregues explicación."
            },
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Extrae todo el texto legible de esta imagen." },
                    { "type": "image_url", "image_url": { "url": data_url } }
                ]
            }
        ]
    });

    let client = reqwest::Client::new();
    let mut parsed: Option<ChatCompletionResponse> = None;
    let mut last_error = String::new();

    for attempt in 0..CLOUD_RETRY_ATTEMPTS {
        let response = client
            .post(&config.chat_base_url)
            .header("Authorization", format!("Bearer {}", config.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await;

        match response {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    let value = resp
                        .json::<ChatCompletionResponse>()
                        .await
                        .map_err(|err| format!("Respuesta OCR cloud inválida: {}", sanitize_remote_error(&err.to_string())))?;
                    parsed = Some(value);
                    break;
                }

                let body = resp.text().await.unwrap_or_default();
                last_error = format!(
                    "OCR cloud respondió {}: {}",
                    status.as_u16(),
                    sanitize_remote_error(&body).chars().take(220).collect::<String>()
                );

                if !is_retryable_http_status(status.as_u16()) || attempt + 1 >= CLOUD_RETRY_ATTEMPTS {
                    return Err(last_error);
                }
            }
            Err(err) => {
                last_error = format!(
                    "No se pudo llamar OCR cloud: {}",
                    sanitize_remote_error(&err.to_string())
                );
                if attempt + 1 >= CLOUD_RETRY_ATTEMPTS {
                    return Err(last_error);
                }
            }
        }
    }

    let parsed = parsed.ok_or_else(|| {
        if last_error.is_empty() {
            "No se pudo obtener OCR cloud tras reintentos".to_string()
        } else {
            last_error
        }
    })?;

    let Some(choice) = parsed.choices.into_iter().next() else {
        return Err("OCR cloud no devolvió opciones".to_string());
    };

    extract_chat_content(&choice.message.content)
}

async fn request_image_ocr_text_ollama(config: &AiProviderConfig, path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|err| format!("No se pudo leer imagen para OCR: {err}"))?;
    let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);

    let payload = serde_json::json!({
        "model": config.chat_model,
        "stream": false,
        "messages": [
            {
                "role": "system",
                "content": "Eres un OCR estricto. Extrae únicamente el texto visible de la imagen en orden de lectura. No inventes contenido ni agregues explicación."
            },
            {
                "role": "user",
                "content": "Extrae todo el texto legible de esta imagen.",
                "images": [encoded]
            }
        ]
    });

    let client = reqwest::Client::new();
    let response = client
        .post(&config.chat_base_url)
        .header("Content-Type", "application/json")
        .json(&payload)
        .send()
        .await
        .map_err(|err| format!("No se pudo llamar OCR local (Ollama): {}", sanitize_remote_error(&err.to_string())))?;

    let status = response.status();
    let raw = response
        .text()
        .await
        .map_err(|err| format!("No se pudo leer respuesta OCR local: {}", sanitize_remote_error(&err.to_string())))?;

    if !status.is_success() {
        return Err(format!(
            "OCR local respondió {}: {}",
            status.as_u16(),
            sanitize_remote_error(&raw).chars().take(220).collect::<String>()
        ));
    }

    let value: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|err| format!("Respuesta OCR local inválida: {}", sanitize_remote_error(&err.to_string())))?;

    if let Some(text) = value
        .get("message")
        .and_then(|msg| msg.get("content"))
        .and_then(|content| content.as_str())
    {
        return Ok(text.to_string());
    }

    if let Some(text) = value.get("response").and_then(|resp| resp.as_str()) {
        return Ok(text.to_string());
    }

    Err("Respuesta OCR local sin contenido esperado".to_string())
}

fn compute_recency_signal(modified_unix_secs: u64) -> f32 {
    if modified_unix_secs == 0 {
        return 0.0;
    }

    let now = now_timestamp_string().parse::<u64>().unwrap_or(modified_unix_secs);
    let age_secs = now.saturating_sub(modified_unix_secs);
    let day = 86_400u64;

    if age_secs <= day {
        1.0
    } else if age_secs <= day * 7 {
        0.65
    } else if age_secs <= day * 30 {
        0.35
    } else {
        0.12
    }
}

fn chat_history_path(app: &tauri::AppHandle) -> Option<PathBuf> {
    let path = app.path().app_data_dir().ok()?.join("chat-history.json");

    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    Some(path)
}

fn load_chat_history(app: &tauri::AppHandle) -> Option<Vec<ChatHistoryItem>> {
    let path = chat_history_path(app)?;
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str::<Vec<ChatHistoryItem>>(&content).ok()
}

fn append_chat_history(app: &tauri::AppHandle, item: &ChatHistoryItem) -> Result<(), String> {
    let mut history = load_chat_history(app).unwrap_or_default();
    history.push(item.clone());

    if history.len() > 300 {
        let keep_from = history.len().saturating_sub(300);
        history = history.into_iter().skip(keep_from).collect();
    }

    let path = chat_history_path(app).ok_or_else(|| "No se pudo resolver ruta de historial".to_string())?;
    let serialized = serde_json::to_string(&history)
        .map_err(|err| format!("No se pudo serializar historial: {err}"))?;
    fs::write(path, serialized).map_err(|err| format!("No se pudo guardar historial: {err}"))
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

fn compute_avg_ms(total_ms: u64, calls: u64) -> f32 {
    if calls == 0 {
        return 0.0;
    }
    total_ms as f32 / calls as f32
}

fn compute_avg_from_samples(samples: &[u64]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().copied().sum::<u64>() as f32 / samples.len() as f32
}

fn percentile_ms(samples: &[u64], percentile: f32) -> u64 {
    if samples.is_empty() {
        return 0;
    }

    let mut ordered = samples.to_vec();
    ordered.sort_unstable();
    let idx = ((ordered.len() as f32 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(ordered.len().saturating_sub(1));
    ordered[idx]
}

fn compute_hit_rate(hits: u64, misses: u64) -> f32 {
    let total = hits.saturating_add(misses);
    if total == 0 {
        return 0.0;
    }
    (hits as f32 * 100.0) / total as f32
}

fn embedding_cache_key(config: &AiProviderConfig, input: &str) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    config.provider.hash(&mut hasher);
    config.base_url.hash(&mut hasher);
    config.embedding_model.hash(&mut hasher);
    input.hash(&mut hasher);
    format!("{}:{}:{:x}", config.provider, config.embedding_model, hasher.finish())
}

fn detect_hardware_profile() -> HardwareProfile {
    let cpu_cores = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(4);

    let mut system = System::new_all();
    system.refresh_memory();
    let total_memory_gb = (system.total_memory() as f32 / (1024.0 * 1024.0)).max(0.0);
    let cpu_brand = std::env::var("PROCESSOR_IDENTIFIER")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "CPU no detectada".to_string());

    let (recommended_mode, recommended_top_k, recommended_max_file_size_mb, note) =
        if cpu_cores <= 4 || total_memory_gb < 8.0 {
            (
                "local".to_string(),
                3usize,
                64u64,
                "Equipo modesto: prioriza LOCAL, top-k bajo y archivos más pequeños para mantener fluidez.".to_string(),
            )
        } else if cpu_cores >= 12 && total_memory_gb >= 24.0 {
            (
                "auto".to_string(),
                6usize,
                256u64,
                "Equipo potente: puedes usar AUTO con top-k más alto y lotes grandes sin degradación notable.".to_string(),
            )
        } else {
            (
                "auto".to_string(),
                4usize,
                128u64,
                "Perfil equilibrado: usa AUTO, top-k medio y límites de archivo estándar.".to_string(),
            )
        };

    HardwareProfile {
        cpu_cores,
        cpu_brand,
        total_memory_gb,
        recommended_mode,
        recommended_top_k,
        recommended_max_file_size_mb,
        note,
    }
}

fn record_runtime_metric(state: &AppState, metric: &str, elapsed_ms: u64) {
    if let Ok(mut guard) = state.runtime_metrics.lock() {
        match metric {
            "semantic" => {
                guard.semantic_calls = guard.semantic_calls.saturating_add(1);
                guard.semantic_total_ms = guard.semantic_total_ms.saturating_add(elapsed_ms);
                guard.last_semantic_ms = elapsed_ms;
            }
            "rag" => {
                guard.rag_calls = guard.rag_calls.saturating_add(1);
                guard.rag_total_ms = guard.rag_total_ms.saturating_add(elapsed_ms);
                guard.last_rag_ms = elapsed_ms;
            }
            "indexing" => {
                guard.indexing_runs = guard.indexing_runs.saturating_add(1);
                guard.indexing_total_ms = guard.indexing_total_ms.saturating_add(elapsed_ms);
                guard.last_index_ms = elapsed_ms;
            }
            _ => {}
        }
    }
}

fn append_audit_log(state: &AppState, event: &str, detail: String) {
    if let Ok(mut guard) = state.audit_logs.lock() {
        guard.push(AuditLogEntry {
            timestamp: now_timestamp_string(),
            event: event.to_string(),
            detail: sanitize_for_log(&detail),
        });

        if guard.len() > 1000 {
            let drop_count = guard.len().saturating_sub(1000);
            guard.drain(0..drop_count);
        }
    }
}

fn sanitize_for_log(input: &str) -> String {
    let compact = input.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut mask_next_bearer_value = false;
    let mut parts = Vec::new();

    for raw in compact.split(' ') {
        let mut token = raw.to_string();

        if mask_next_bearer_value {
            token = "****".to_string();
            mask_next_bearer_value = false;
        } else if token.eq_ignore_ascii_case("bearer") {
            token = "Bearer".to_string();
            mask_next_bearer_value = true;
        } else if token.starts_with("sk-") {
            token = "sk-****".to_string();
        }

        parts.push(token);
    }

    parts.join(" ").chars().take(420).collect()
}

fn sanitize_remote_error(input: &str) -> String {
    sanitize_for_log(input)
}

fn is_retryable_http_status(code: u16) -> bool {
    code == 408 || code == 429 || code >= 500
}

fn bounded_cache_insert(
    cache: &mut HashMap<String, Vec<f32>>,
    key: String,
    value: Vec<f32>,
    max_items: usize,
) {
    cache.insert(key, value);

    if cache.len() <= max_items {
        return;
    }

    let remove_count = cache.len().saturating_sub(max_items);
    let keys_to_remove = cache
        .keys()
        .take(remove_count)
        .cloned()
        .collect::<Vec<_>>();

    for key in keys_to_remove {
        cache.remove(&key);
    }
}

fn now_timestamp_string() -> String {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(duration) => format!("{}", duration.as_secs()),
        Err(_) => "0".to_string(),
    }
}

fn now_millis_string() -> String {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(duration) => format!("{}", duration.as_millis()),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_text_chunks_applies_overlap() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        let chunks = split_text_chunks(text, 10, 2);

        assert!(chunks.len() >= 3);
        assert_eq!(chunks[0], "abcdefghij");
        assert!(chunks[1].starts_with("ij"));
    }

    #[test]
    fn lexical_score_boosts_full_query_match() {
        let tokens = vec!["memo".to_string(), "vaulty".to_string()];
        let lowered_query = "memo vaulty".to_string();
        let score = compute_lexical_score("memo vaulty app local", &tokens, &lowered_query);

        assert!(score > 1.0);
        assert!(score <= 1.6);
    }

    #[test]
    fn sanitize_remote_error_masks_secrets() {
        let sanitized = sanitize_remote_error("Authorization: Bearer sk-abc123456789 xyz");

        assert!(!sanitized.contains("abc123456789"));
        assert!(!sanitized.contains("Bearer sk-"));
        assert!(sanitized.contains("Bearer ****"));
    }
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

            if let Some(ai_config) = load_ai_provider_config(app.handle()) {
                if let Ok(mut guard) = app.state::<AppState>().ai_config.lock() {
                    *guard = Some(ai_config);
                }
            }

            if let Some(clip_config) = load_clip_config(app.handle()) {
                if let Ok(mut guard) = app.state::<AppState>().clip_config.lock() {
                    *guard = Some(clip_config);
                }
            } else if let Some(detected) = detect_clip_config(app.handle()) {
                if let Ok(mut guard) = app.state::<AppState>().clip_config.lock() {
                    *guard = Some(detected.clone());
                }
                let _ = save_clip_config(app.handle(), &detected);
            }

            if let Some(perf_cfg) = load_performance_runtime_config(app.handle()) {
                if let Ok(mut runtime) = app.state::<AppState>().performance_runtime.lock() {
                    runtime.config = perf_cfg;
                }
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            open_file,
            get_image_metadata,
            get_file_text_preview,
            get_file_visual_preview,
            search_stub,
            semantic_search,
            start_indexing,
            get_index_status,
            get_index_diagnostics,
            cancel_indexing,
            start_file_watcher,
            stop_file_watcher,
            clear_file_watcher_error,
            trigger_watcher_reindex,
            get_file_watcher_status,
            get_runtime_metrics,
            get_performance_telemetry,
            get_performance_runtime_status,
            configure_performance_runtime,
            get_embedding_cache_status,
            clear_embedding_cache,
            get_semantic_schema_info,
            get_embedding_model_index_status,
            sync_embedding_model_index,
            get_hardware_profile,
            apply_hardware_profile_defaults,
            run_local_semantic_benchmark,
            run_semantic_cold_hot_benchmark,
            get_clip_image_cache_status,
            clear_clip_image_cache,
            get_audit_logs,
            clear_audit_logs,
            export_audit_logs_to_file,
            configure_clip_onnx,
            get_clip_onnx_status,
            validate_clip_onnx_setup,
            clip_text_to_image_search,
            configure_ai_provider,
            get_ai_provider_status,
            clear_index_data,
            forget_index_root,
            export_app_config_to_file,
            import_app_config_from_file,
            answer_with_local_context,
            get_chat_history,
            clear_chat_history,
            export_chat_history_to_file
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
