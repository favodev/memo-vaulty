import { Fragment, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { open as openDialog, save as saveDialog } from "@tauri-apps/plugin-dialog";

type SearchResultItem = {
  title: string;
  path: string;
  snippet: string;
  match_reason?: string;
  origin?: string;
};

type IndexStatus = {
  has_index: boolean;
  indexed_files: number;
  indexed_at: string | null;
  roots: string[];
};

type IndexFeedback = {
  type: "running" | "success" | "error";
  text: string;
};

type IndexProgressEvent = {
  phase: string;
  message: string;
  scanned_files: number;
  indexed_files: number;
  done: boolean;
};

type IndexDiagnostics = {
  scanned_files: number;
  indexed_files: number;
  pdf_scanned: number;
  pdf_indexed: number;
  pdf_failed: number;
  pdf_failed_examples: string[];
  last_error: string | null;
  updated_at: string | null;
  canceled: boolean;
  lancedb_synced: boolean;
  pdf_fallback_used: number;
};

type AiProviderStatus = {
  configured: boolean;
  provider: string;
  base_url: string;
  embedding_model: string;
  chat_base_url: string;
  chat_model: string;
  api_key_hint: string | null;
};

type FileWatcherStatus = {
  running: boolean;
  roots: string[];
  pending_events: boolean;
  debounce_ms: number;
  last_event_at: string | null;
  last_reindex_at: string | null;
  last_error: string | null;
};

type ImageMetadata = {
  path: string;
  width: number | null;
  height: number | null;
  format: string | null;
  date_taken: string | null;
  orientation: string | null;
};

type FileTextPreview = {
  available: boolean;
  source: string;
  text: string;
};

type FileVisualPreview = {
  available: boolean;
  mime: string;
  data_url: string | null;
  size_bytes: number;
  reason: string | null;
};

type RagSourceItem = {
  ref_id: string;
  title: string;
  path: string;
  snippet: string;
  score: number;
};

type RagAnswerResponse = {
  answer: string;
  grounded: boolean;
  mode: string;
  sources: RagSourceItem[];
};

type ChatHistoryItem = {
  id: string;
  timestamp: string;
  query: string;
  answer: string;
  grounded: boolean;
  mode: string;
  sources: RagSourceItem[];
};

type RuntimeMetrics = {
  semantic_calls: number;
  semantic_avg_ms: number;
  rag_calls: number;
  rag_avg_ms: number;
  indexing_runs: number;
  indexing_avg_ms: number;
  last_index_ms: number;
  embedding_cache_hits: number;
  embedding_cache_misses: number;
  embedding_cache_hit_rate: number;
  embedding_cache_items: number;
};

type EmbeddingCacheStatus = {
  enabled: boolean;
  items: number;
  hits: number;
  misses: number;
  hit_rate: number;
};

type HardwareProfile = {
  cpu_cores: number;
  cpu_brand: string;
  total_memory_gb: number;
  recommended_mode: string;
  recommended_top_k: number;
  recommended_max_file_size_mb: number;
  note: string;
};

type SearchBenchmarkResult = {
  query: string;
  iterations: number;
  avg_ms: number;
  p95_ms: number;
  best_ms: number;
  worst_ms: number;
  last_result_count: number;
  candidate_limit: number;
  images_only: boolean;
  backend: string;
};

type SearchColdHotBenchmarkResult = {
  query: string;
  iterations: number;
  cold_avg_ms: number;
  cold_p95_ms: number;
  hot_avg_ms: number;
  hot_p95_ms: number;
  speedup_percent: number;
  candidate_count: number;
  backend: string;
};

type AppliedPerformanceProfile = {
  cpu_cores: number;
  total_memory_gb: number;
  recommended_mode: string;
  recommended_top_k: number;
  recommended_max_file_size_mb: number;
  applied_max_file_size_mb: number;
  note: string;
};

type ClipImageCacheStatus = {
  items: number;
};

type AuditLogEntry = {
  timestamp: string;
  event: string;
  detail: string;
};

type ClipOnnxStatus = {
  configured: boolean;
  enabled: boolean;
  image_model_path: string;
  text_model_path: string;
  tokenizer_path: string;
  input_size: number;
  max_length: number;
};

type ClipValidationStatus = {
  configured: boolean;
  tokenizer_ok: boolean;
  text_model_ok: boolean;
  image_model_ok: boolean;
  text_inference_ok: boolean;
  image_inference_ok: boolean;
  text_dim: number | null;
  image_dim: number | null;
  sample_image_path: string | null;
  message: string;
};

function App() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [searchRoots, setSearchRoots] = useState<string[]>([]);
  const [excludedExtensions, setExcludedExtensions] = useState("mkv, mp4, zip");
  const [excludedFolders, setExcludedFolders] = useState("node_modules, .git, target, AppData");
  const [maxFileSizeMb, setMaxFileSizeMb] = useState("128");
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [indexFeedback, setIndexFeedback] = useState<IndexFeedback | null>(null);
  const [indexProgress, setIndexProgress] = useState<IndexProgressEvent | null>(null);
  const [indexDiagnostics, setIndexDiagnostics] = useState<IndexDiagnostics | null>(null);
  const [aiProviderStatus, setAiProviderStatus] = useState<AiProviderStatus | null>(null);
  const [aiApiKey, setAiApiKey] = useState("");
  const [aiModel, setAiModel] = useState("text-embedding-3-small");
  const [aiBaseUrl, setAiBaseUrl] = useState("https://openrouter.ai/api/v1/embeddings");
  const [aiChatModel, setAiChatModel] = useState("gpt-4o-mini");
  const [aiChatBaseUrl, setAiChatBaseUrl] = useState("https://openrouter.ai/api/v1/chat/completions");
  const [isSavingAi, setIsSavingAi] = useState(false);
  const [watcherStatus, setWatcherStatus] = useState<FileWatcherStatus | null>(null);
  const [isWatcherLoading, setIsWatcherLoading] = useState(false);
  const [imagesOnly, setImagesOnly] = useState(false);
  const [isMaintenanceBusy, setIsMaintenanceBusy] = useState(false);
  const [quickLookPath, setQuickLookPath] = useState<string | null>(null);
  const [quickLookImageMeta, setQuickLookImageMeta] = useState<ImageMetadata | null>(null);
  const [isQuickLookOpen, setIsQuickLookOpen] = useState(false);
  const [quickLookTextPreview, setQuickLookTextPreview] = useState<FileTextPreview | null>(null);
  const [quickLookMode, setQuickLookMode] = useState<"visual" | "text">("visual");
  const [quickLookVisualLoaded, setQuickLookVisualLoaded] = useState(false);
  const [quickLookVisualFailed, setQuickLookVisualFailed] = useState(false);
  const [quickLookVisualUrl, setQuickLookVisualUrl] = useState<string | null>(null);
  const [quickLookVisualReason, setQuickLookVisualReason] = useState<string | null>(null);
  const [isQuickLookVisualLoading, setIsQuickLookVisualLoading] = useState(false);
  const [searchModeBadge, setSearchModeBadge] = useState<"LOCAL" | "CLOUD" | null>(null);
  const [isRagLoading, setIsRagLoading] = useState(false);
  const [ragResponse, setRagResponse] = useState<RagAnswerResponse | null>(null);
  const [answerMode, setAnswerMode] = useState<"auto" | "local" | "cloud">("auto");
  const [ragTopK, setRagTopK] = useState("4");
  const [strictGrounding, setStrictGrounding] = useState(true);
  const [ragMinScore, setRagMinScore] = useState("0.55");
  const [chatHistory, setChatHistory] = useState<ChatHistoryItem[]>([]);
  const [isChatHistoryLoading, setIsChatHistoryLoading] = useState(false);
  const [runtimeMetrics, setRuntimeMetrics] = useState<RuntimeMetrics | null>(null);
  const [auditLogs, setAuditLogs] = useState<AuditLogEntry[]>([]);
  const [isAuditLoading, setIsAuditLoading] = useState(false);
  const [clipStatus, setClipStatus] = useState<ClipOnnxStatus | null>(null);
  const [clipImageModelPath, setClipImageModelPath] = useState("");
  const [clipTextModelPath, setClipTextModelPath] = useState("");
  const [clipTokenizerPath, setClipTokenizerPath] = useState("");
  const [clipInputSize, setClipInputSize] = useState("224");
  const [clipMaxLength, setClipMaxLength] = useState("77");
  const [isClipSaving, setIsClipSaving] = useState(false);
  const [isClipSearching, setIsClipSearching] = useState(false);
  const [isClipValidating, setIsClipValidating] = useState(false);
  const [clipValidation, setClipValidation] = useState<ClipValidationStatus | null>(null);
  const [embeddingCacheStatus, setEmbeddingCacheStatus] = useState<EmbeddingCacheStatus | null>(null);
  const [hardwareProfile, setHardwareProfile] = useState<HardwareProfile | null>(null);
  const [benchmarkQuery, setBenchmarkQuery] = useState("nota");
  const [benchmarkIterations, setBenchmarkIterations] = useState("8");
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [benchmarkResult, setBenchmarkResult] = useState<SearchBenchmarkResult | null>(null);
  const [isApplyingPerfProfile, setIsApplyingPerfProfile] = useState(false);
  const [coldHotResult, setColdHotResult] = useState<SearchColdHotBenchmarkResult | null>(null);
  const [isColdHotBenchmarking, setIsColdHotBenchmarking] = useState(false);
  const [clipImageCacheStatus, setClipImageCacheStatus] = useState<ClipImageCacheStatus | null>(null);

  const refreshStatus = async () => {
    try {
      const status = await invoke<IndexStatus>("get_index_status");
      setIndexStatus(status);
    } catch {
      setIndexStatus(null);
    }

    try {
      const diagnostics = await invoke<IndexDiagnostics>("get_index_diagnostics");
      setIndexDiagnostics(diagnostics);
    } catch {
      setIndexDiagnostics(null);
    }

    try {
      const status = await invoke<AiProviderStatus>("get_ai_provider_status");
      setAiProviderStatus(status);
      setAiModel(status.embedding_model);
      setAiBaseUrl(status.base_url);
      setAiChatModel(status.chat_model);
      setAiChatBaseUrl(status.chat_base_url);
    } catch {
      setAiProviderStatus(null);
    }

    try {
      const watcher = await invoke<FileWatcherStatus>("get_file_watcher_status");
      setWatcherStatus(watcher);
    } catch {
      setWatcherStatus(null);
    }

    try {
      const metrics = await invoke<RuntimeMetrics>("get_runtime_metrics");
      setRuntimeMetrics(metrics);
    } catch {
      setRuntimeMetrics(null);
    }

    try {
      const cacheStatus = await invoke<EmbeddingCacheStatus>("get_embedding_cache_status");
      setEmbeddingCacheStatus(cacheStatus);
    } catch {
      setEmbeddingCacheStatus(null);
    }

    try {
      const profile = await invoke<HardwareProfile>("get_hardware_profile");
      setHardwareProfile(profile);
    } catch {
      setHardwareProfile(null);
    }

    try {
      const clipCache = await invoke<ClipImageCacheStatus>("get_clip_image_cache_status");
      setClipImageCacheStatus(clipCache);
    } catch {
      setClipImageCacheStatus(null);
    }

    try {
      const status = await invoke<ClipOnnxStatus>("get_clip_onnx_status");
      setClipStatus(status);
      if (status.configured) {
        setClipImageModelPath(status.image_model_path);
        setClipTextModelPath(status.text_model_path);
        setClipTokenizerPath(status.tokenizer_path);
        setClipInputSize(String(status.input_size));
        setClipMaxLength(String(status.max_length));
      }
    } catch {
      setClipStatus(null);
    }
  };

  const refreshAuditLogs = async () => {
    setIsAuditLoading(true);
    try {
      const logs = await invoke<AuditLogEntry[]>("get_audit_logs", { limit: 20 });
      setAuditLogs(logs);
    } catch {
      setAuditLogs([]);
    } finally {
      setIsAuditLoading(false);
    }
  };

  useEffect(() => {
    void refreshStatus();
    void refreshChatHistory();
    void refreshAuditLogs();
  }, []);

  const refreshChatHistory = async () => {
    setIsChatHistoryLoading(true);
    try {
      const history = await invoke<ChatHistoryItem[]>("get_chat_history", { limit: 12 });
      setChatHistory(history);
    } catch {
      setChatHistory([]);
    } finally {
      setIsChatHistoryLoading(false);
    }
  };

  useEffect(() => {
    if (!watcherStatus?.running) {
      return;
    }

    const interval = setInterval(() => {
      void invoke<FileWatcherStatus>("get_file_watcher_status")
        .then((status) => setWatcherStatus(status))
        .catch(() => undefined);
    }, 2500);

    return () => clearInterval(interval);
  }, [watcherStatus?.running]);

  useEffect(() => {
    if (!indexFeedback) {
      return;
    }

    if (indexFeedback.type === "running") {
      return;
    }

    const timeout = setTimeout(() => {
      setIndexFeedback(null);
    }, 2600);

    return () => clearTimeout(timeout);
  }, [indexFeedback]);

  useEffect(() => {
    let isMounted = true;

    const attach = async () => {
      const unlisten = await listen<IndexProgressEvent>("index-progress", (event) => {
        if (!isMounted) {
          return;
        }

        setIndexProgress(event.payload);

        if (event.payload.done) {
          void invoke<IndexDiagnostics>("get_index_diagnostics")
            .then((diagnostics) => setIndexDiagnostics(diagnostics))
            .catch(() => setIndexDiagnostics(null));

          setTimeout(() => {
            if (isMounted) {
              setIndexProgress((current) => (current?.done ? null : current));
            }
          }, 2400);
        }
      });

      return unlisten;
    };

    let cleanup: (() => void) | undefined;
    void attach().then((unlisten) => {
      cleanup = unlisten;
    });

    return () => {
      isMounted = false;
      cleanup?.();
    };
  }, []);

  const openFile = async (filePath: string) => {
    try {
      await invoke("open_file", { path: filePath });
    } catch {
      setErrorMessage("No se pudo abrir el archivo.");
    }
  };

  const openContainingFolder = async (filePath: string) => {
    await revealItemInDir(filePath);
  };

  const runSearch = async () => {
    const cleanQuery = query.trim();

    if (!cleanQuery) {
      setResults([]);
      setHasSearched(false);
      setErrorMessage(null);
      setSelectedIndex(-1);
      return;
    }

    setIsLoading(true);
    setHasSearched(true);
    setErrorMessage(null);

    try {
      const exclusions = excludedExtensions
        .split(",")
        .map((value) => value.trim())
        .filter((value) => value.length > 0);

      const excludedFolderRules = excludedFolders
        .split(",")
        .map((value) => value.trim())
        .filter((value) => value.length > 0);

      const parsedMaxSize = Number.parseInt(maxFileSizeMb, 10);
      const maxSizeValue = Number.isFinite(parsedMaxSize) && parsedMaxSize > 0 ? parsedMaxSize : 128;

      let response: SearchResultItem[] = [];
      let semanticCommandOk = false;

      try {
        response = await invoke<SearchResultItem[]>("semantic_search", {
          query: cleanQuery,
          limit: 30,
          roots: searchRoots,
          excludedExtensions: exclusions,
          excludedFolders: excludedFolderRules,
          maxFileSizeMb: maxSizeValue,
          imagesOnly,
        });
        semanticCommandOk = true;
      } catch {
        response = await invoke<SearchResultItem[]>("search_stub", {
          query: cleanQuery,
          roots: searchRoots,
          excludedExtensions: exclusions,
          excludedFolders: excludedFolderRules,
          maxFileSizeMb: maxSizeValue,
          imagesOnly,
        });
      }

      setResults(response);
      setSelectedIndex(response.length > 0 ? 0 : -1);
      setQuickLookPath(response.length > 0 ? response[0].path : null);

      const hasCloudSemantic = response.some((item) => item.origin === "cloud-semantic");
      const isCloud = semanticCommandOk && hasCloudSemantic && Boolean(aiProviderStatus?.configured);
      setSearchModeBadge(isCloud ? "CLOUD" : "LOCAL");
    } catch {
      setResults([]);
      setSelectedIndex(-1);
      setQuickLookPath(null);
      setSearchModeBadge(null);
      setErrorMessage("No se pudo ejecutar la búsqueda local.");
    } finally {
      setIsLoading(false);
    }
  };

  const runLocalAnswer = async () => {
    const cleanQuery = query.trim();
    if (!cleanQuery) {
      setRagResponse(null);
      return;
    }

    setIsRagLoading(true);
    setErrorMessage(null);

    try {
      const exclusions = excludedExtensions
        .split(",")
        .map((value) => value.trim())
        .filter((value) => value.length > 0);

      const excludedFolderRules = excludedFolders
        .split(",")
        .map((value) => value.trim())
        .filter((value) => value.length > 0);

      const parsedMaxSize = Number.parseInt(maxFileSizeMb, 10);
      const maxSizeValue = Number.isFinite(parsedMaxSize) && parsedMaxSize > 0 ? parsedMaxSize : 128;
      const parsedTopK = Number.parseInt(ragTopK, 10);
      const topKValue = Number.isFinite(parsedTopK) && parsedTopK > 0 ? Math.min(parsedTopK, 8) : 4;
      const parsedMinScore = Number.parseFloat(ragMinScore);
      const minScoreValue = Number.isFinite(parsedMinScore) ? Math.max(0, Math.min(parsedMinScore, 1.6)) : 0.55;

      const response = await invoke<RagAnswerResponse>("answer_with_local_context", {
        query: cleanQuery,
        roots: searchRoots,
        excludedExtensions: exclusions,
        excludedFolders: excludedFolderRules,
        maxFileSizeMb: maxSizeValue,
        topK: topKValue,
        mode: answerMode,
        strictGrounding,
        minScore: minScoreValue,
      });

      setRagResponse(response);
      await refreshChatHistory();
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo generar respuesta local" });
    } finally {
      setIsRagLoading(false);
    }
  };

  const clearChatHistory = async () => {
    setIsChatHistoryLoading(true);
    try {
      await invoke("clear_chat_history");
      setChatHistory([]);
      setIndexFeedback({ type: "success", text: "Historial de chat limpiado" });
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo limpiar historial" });
    } finally {
      setIsChatHistoryLoading(false);
    }
  };

  const clearAuditLogs = async () => {
    setIsAuditLoading(true);
    try {
      await invoke("clear_audit_logs");
      setAuditLogs([]);
      setIndexFeedback({ type: "success", text: "Auditoría limpiada" });
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo limpiar auditoría" });
    } finally {
      setIsAuditLoading(false);
    }
  };

  const exportAuditLogs = async () => {
    const target = await saveDialog({
      title: "Exportar auditoría",
      defaultPath: "memovault-audit.json",
      filters: [{ name: "JSON", extensions: ["json"] }],
    });

    if (!target) {
      return;
    }

    setIsAuditLoading(true);
    try {
      await invoke<string>("export_audit_logs_to_file", { path: target, limit: 300 });
      setIndexFeedback({ type: "success", text: "Auditoría exportada" });
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo exportar auditoría" });
    } finally {
      setIsAuditLoading(false);
    }
  };

  const saveClipConfig = async () => {
    setIsClipSaving(true);
    try {
      const parsedInputSize = Number.parseInt(clipInputSize, 10);
      const parsedMaxLength = Number.parseInt(clipMaxLength, 10);

      const status = await invoke<ClipOnnxStatus>("configure_clip_onnx", {
        imageModelPath: clipImageModelPath,
        textModelPath: clipTextModelPath,
        tokenizerPath: clipTokenizerPath,
        inputSize: Number.isFinite(parsedInputSize) ? parsedInputSize : 224,
        maxLength: Number.isFinite(parsedMaxLength) ? parsedMaxLength : 77,
        enabled: true,
      });

      setClipStatus(status);
      setIndexFeedback({ type: "success", text: "CLIP/ONNX configurado" });
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo configurar CLIP/ONNX" });
    } finally {
      setIsClipSaving(false);
    }
  };

  const runClipImageSearch = async () => {
    const cleanQuery = query.trim();
    if (!cleanQuery) {
      return;
    }

    setIsClipSearching(true);
    setErrorMessage(null);

    try {
      const exclusions = excludedExtensions
        .split(",")
        .map((value) => value.trim())
        .filter((value) => value.length > 0);

      const excludedFolderRules = excludedFolders
        .split(",")
        .map((value) => value.trim())
        .filter((value) => value.length > 0);

      const parsedMaxSize = Number.parseInt(maxFileSizeMb, 10);
      const maxSizeValue = Number.isFinite(parsedMaxSize) && parsedMaxSize > 0 ? parsedMaxSize : 128;

      const response = await invoke<SearchResultItem[]>("clip_text_to_image_search", {
        query: cleanQuery,
        limit: 30,
        roots: searchRoots,
        excludedExtensions: exclusions,
        excludedFolders: excludedFolderRules,
        maxFileSizeMb: maxSizeValue,
      });

      setResults(response);
      setSelectedIndex(response.length > 0 ? 0 : -1);
      setQuickLookPath(response.length > 0 ? response[0].path : null);
      setSearchModeBadge("LOCAL");
      setHasSearched(true);
      setIndexFeedback({ type: "success", text: `CLIP encontró ${response.length} imágenes` });
    } catch {
      setIndexFeedback({ type: "error", text: "Búsqueda CLIP/ONNX falló" });
    } finally {
      setIsClipSearching(false);
    }
  };

  const validateClipSetup = async () => {
    setIsClipValidating(true);
    try {
      const status = await invoke<ClipValidationStatus>("validate_clip_onnx_setup", {
        sampleImagePath: null,
      });
      setClipValidation(status);
      setIndexFeedback({ type: status.text_inference_ok ? "success" : "error", text: status.text_inference_ok ? "CLIP validado" : "CLIP con problemas" });
    } catch {
      setClipValidation(null);
      setIndexFeedback({ type: "error", text: "No se pudo validar CLIP" });
    } finally {
      setIsClipValidating(false);
    }
  };

  const clearEmbeddingCache = async () => {
    try {
      await invoke("clear_embedding_cache");
      setIndexFeedback({ type: "success", text: "Caché de embeddings limpiada" });
      await refreshStatus();
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo limpiar caché de embeddings" });
    }
  };

  const runLocalBenchmark = async () => {
    const cleanQuery = benchmarkQuery.trim();
    if (!cleanQuery) {
      return;
    }

    const parsedIterations = Number.parseInt(benchmarkIterations, 10);

    const exclusions = excludedExtensions
      .split(",")
      .map((value) => value.trim())
      .filter((value) => value.length > 0);

    const excludedFolderRules = excludedFolders
      .split(",")
      .map((value) => value.trim())
      .filter((value) => value.length > 0);

    const parsedMaxSize = Number.parseInt(maxFileSizeMb, 10);
    const maxSizeValue = Number.isFinite(parsedMaxSize) && parsedMaxSize > 0 ? parsedMaxSize : 128;

    setIsBenchmarking(true);
    try {
      const result = await invoke<SearchBenchmarkResult>("run_local_semantic_benchmark", {
        query: cleanQuery,
        iterations: Number.isFinite(parsedIterations) ? parsedIterations : 8,
        candidateLimit: 80,
        roots: searchRoots,
        excludedExtensions: exclusions,
        excludedFolders: excludedFolderRules,
        maxFileSizeMb: maxSizeValue,
        imagesOnly,
      });
      setBenchmarkResult(result);
      setIndexFeedback({ type: "success", text: `Benchmark OK · avg ${result.avg_ms.toFixed(1)} ms · p95 ${result.p95_ms} ms` });
      await refreshStatus();
    } catch {
      setIndexFeedback({ type: "error", text: "Benchmark local falló" });
    } finally {
      setIsBenchmarking(false);
    }
  };

  const applyPerformanceProfile = async () => {
    setIsApplyingPerfProfile(true);
    try {
      const applied = await invoke<AppliedPerformanceProfile>("apply_hardware_profile_defaults");
      setMaxFileSizeMb(String(applied.applied_max_file_size_mb));
      setRagTopK(String(applied.recommended_top_k));
      const recommendedMode = (applied.recommended_mode === "local" ? "local" : "auto") as "local" | "auto";
      setAnswerMode(recommendedMode);
      setIndexFeedback({
        type: "success",
        text: `Perfil aplicado · modo ${recommendedMode.toUpperCase()} · top-k ${applied.recommended_top_k} · max ${applied.applied_max_file_size_mb} MB`,
      });
      await refreshStatus();
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo aplicar perfil de rendimiento" });
    } finally {
      setIsApplyingPerfProfile(false);
    }
  };

  const runColdHotBenchmark = async () => {
    const cleanQuery = benchmarkQuery.trim();
    if (!cleanQuery) {
      return;
    }

    const parsedIterations = Number.parseInt(benchmarkIterations, 10);

    const exclusions = excludedExtensions
      .split(",")
      .map((value) => value.trim())
      .filter((value) => value.length > 0);

    const excludedFolderRules = excludedFolders
      .split(",")
      .map((value) => value.trim())
      .filter((value) => value.length > 0);

    const parsedMaxSize = Number.parseInt(maxFileSizeMb, 10);
    const maxSizeValue = Number.isFinite(parsedMaxSize) && parsedMaxSize > 0 ? parsedMaxSize : 128;

    setIsColdHotBenchmarking(true);
    try {
      const result = await invoke<SearchColdHotBenchmarkResult>("run_semantic_cold_hot_benchmark", {
        query: cleanQuery,
        iterations: Number.isFinite(parsedIterations) ? parsedIterations : 6,
        candidateLimit: 60,
        roots: searchRoots,
        excludedExtensions: exclusions,
        excludedFolders: excludedFolderRules,
        maxFileSizeMb: maxSizeValue,
        imagesOnly,
      });
      setColdHotResult(result);
      setIndexFeedback({
        type: "success",
        text: `Cold/Hot OK · cold ${result.cold_avg_ms.toFixed(1)} ms · hot ${result.hot_avg_ms.toFixed(1)} ms · +${result.speedup_percent.toFixed(1)}%`,
      });
      await refreshStatus();
    } catch {
      setIndexFeedback({ type: "error", text: "Benchmark cold/hot falló (requiere IA embeddings configurada)" });
    } finally {
      setIsColdHotBenchmarking(false);
    }
  };

  const clearClipImageCache = async () => {
    try {
      await invoke("clear_clip_image_cache");
      setIndexFeedback({ type: "success", text: "Caché CLIP de imágenes limpiada" });
      await refreshStatus();
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo limpiar caché CLIP" });
    }
  };

  useEffect(() => {
    if (!isConfigOpen || isClipValidating) {
      return;
    }

    if (!clipStatus?.configured || !clipStatus.enabled) {
      return;
    }

    void validateClipSetup();
  }, [isConfigOpen, clipStatus?.configured, clipStatus?.enabled]);

  const exportChatHistory = async () => {
    const target = await saveDialog({
      title: "Exportar historial de chat",
      defaultPath: "memovault-chat-history.json",
      filters: [{ name: "JSON", extensions: ["json"] }],
    });

    if (!target) {
      return;
    }

    setIsChatHistoryLoading(true);
    try {
      await invoke<string>("export_chat_history_to_file", { path: target, limit: 100 });
      setIndexFeedback({ type: "success", text: "Historial exportado" });
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo exportar historial" });
    } finally {
      setIsChatHistoryLoading(false);
    }
  };

  const pickFolders = async () => {
    const selected = await openDialog({
      directory: true,
      multiple: true,
      title: "Selecciona carpetas para buscar",
    });

    if (!selected) {
      return;
    }

    const paths = Array.isArray(selected) ? selected : [selected];
    setSearchRoots((prev) => {
      const merged = new Set([...prev, ...paths]);
      return Array.from(merged);
    });
  };

  const removeRoot = (pathToRemove: string) => {
    setSearchRoots((prev) => prev.filter((path) => path !== pathToRemove));
  };

  const forgetRootPersisted = async (rootPath: string) => {
    setIsMaintenanceBusy(true);
    setErrorMessage(null);

    try {
      const status = await invoke<IndexStatus>("forget_index_root", {
        root: rootPath,
        reindex: true,
      });
      setIndexStatus(status);
      setSearchRoots(status.roots);
      setIndexFeedback({ type: "success", text: "Carpeta olvidada y reindexada" });
      await refreshStatus();
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo olvidar carpeta" });
    } finally {
      setIsMaintenanceBusy(false);
    }
  };

  const resetIndexData = async () => {
    const accepted = window.confirm("Esto borrará snapshot e índices locales (SQLite/LanceDB). ¿Continuar?");
    if (!accepted) {
      return;
    }

    setIsMaintenanceBusy(true);
    setErrorMessage(null);

    try {
      const status = await invoke<IndexStatus>("clear_index_data");
      setIndexStatus(status);
      setSearchRoots([]);
      setResults([]);
      setHasSearched(false);
      setSelectedIndex(-1);
      setQuickLookPath(null);
      setIndexFeedback({ type: "success", text: "Índice local reiniciado" });
      await refreshStatus();
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo reiniciar índice" });
    } finally {
      setIsMaintenanceBusy(false);
    }
  };

  const exportConfigToJson = async () => {
    const target = await saveDialog({
      title: "Exportar configuración",
      defaultPath: "memovault-config.json",
      filters: [{ name: "JSON", extensions: ["json"] }],
    });

    if (!target) {
      return;
    }

    setIsMaintenanceBusy(true);
    try {
      await invoke<string>("export_app_config_to_file", {
        path: target,
        includeSecrets: false,
      });
      setIndexFeedback({ type: "success", text: "Config exportada" });
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo exportar config" });
    } finally {
      setIsMaintenanceBusy(false);
    }
  };

  const importConfigFromJson = async () => {
    const selected = await openDialog({
      multiple: false,
      directory: false,
      title: "Importar configuración",
      filters: [{ name: "JSON", extensions: ["json"] }],
    });

    if (!selected || Array.isArray(selected)) {
      return;
    }

    setIsMaintenanceBusy(true);
    try {
      const status = await invoke<IndexStatus>("import_app_config_from_file", {
        path: selected,
        reindex: true,
      });
      setIndexStatus(status);
      if (status.roots.length > 0) {
        setSearchRoots(status.roots);
      }
      setIndexFeedback({ type: "success", text: "Config importada" });
      await refreshStatus();
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo importar config" });
    } finally {
      setIsMaintenanceBusy(false);
    }
  };

  const startIndexing = async (roots?: string[]) => {
    const startedAt = Date.now();
    setIsIndexing(true);
    setIndexFeedback({ type: "running", text: "Indexando..." });
    setIndexProgress({
      phase: "start",
      message: "Preparando indexación...",
      scanned_files: 0,
      indexed_files: 0,
      done: false,
    });

    try {
      const exclusions = excludedExtensions
        .split(",")
        .map((value) => value.trim())
        .filter((value) => value.length > 0);

      const excludedFolderRules = excludedFolders
        .split(",")
        .map((value) => value.trim())
        .filter((value) => value.length > 0);

      const parsedMaxSize = Number.parseInt(maxFileSizeMb, 10);
      const maxSizeValue = Number.isFinite(parsedMaxSize) && parsedMaxSize > 0 ? parsedMaxSize : 128;

      const status = await invoke<IndexStatus>("start_indexing", {
        roots,
        excludedExtensions: exclusions,
        excludedFolders: excludedFolderRules,
        maxFileSizeMb: maxSizeValue,
      });

      setIndexStatus(status);
      try {
        const diagnostics = await invoke<IndexDiagnostics>("get_index_diagnostics");
        setIndexDiagnostics(diagnostics);
      } catch {
        setIndexDiagnostics(null);
      }

      if (status.roots.length > 0) {
        setSearchRoots(status.roots);
      }
      setIndexFeedback({ type: "success", text: "Reindexación hecha" });
    } catch (error) {
      const maybeMessage =
        typeof error === "string"
          ? error
          : error && typeof error === "object" && "message" in error && typeof error.message === "string"
          ? error.message
          : "";

      const cancelled = maybeMessage.toLowerCase().includes("cancelada");

      if (cancelled) {
        setIndexFeedback({ type: "error", text: "Indexación cancelada" });
      } else {
        setErrorMessage("Falló la indexación local. Intenta con una carpeta más pequeña primero.");
        setIndexFeedback({ type: "error", text: "Reindexación falló" });
      }

      try {
        const diagnostics = await invoke<IndexDiagnostics>("get_index_diagnostics");
        setIndexDiagnostics(diagnostics);
      } catch {
        setIndexDiagnostics(null);
      }
    } finally {
      const elapsed = Date.now() - startedAt;
      const minVisibleMs = 900;
      if (elapsed < minVisibleMs) {
        await new Promise((resolve) => setTimeout(resolve, minVisibleMs - elapsed));
      }
      setIsIndexing(false);
    }
  };

  const cancelIndexing = async () => {
    try {
      await invoke("cancel_indexing");
      setIndexFeedback({ type: "running", text: "Cancelando indexación..." });
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo cancelar" });
    }
  };

  const saveAiProvider = async () => {
    if (!aiApiKey.trim()) {
      setErrorMessage("Debes ingresar una API key para guardar la config de IA.");
      return;
    }

    setIsSavingAi(true);
    setErrorMessage(null);

    try {
      const status = await invoke<AiProviderStatus>("configure_ai_provider", {
        apiKey: aiApiKey,
        embeddingModel: aiModel,
        baseUrl: aiBaseUrl,
        chatModel: aiChatModel,
        chatBaseUrl: aiChatBaseUrl,
      });

      setAiProviderStatus(status);
      setAiApiKey("");
      setIndexFeedback({ type: "success", text: "IA configurada" });
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo guardar IA" });
    } finally {
      setIsSavingAi(false);
    }
  };

  const startWatcher = async () => {
    setIsWatcherLoading(true);

    try {
      const exclusions = excludedExtensions
        .split(",")
        .map((value) => value.trim())
        .filter((value) => value.length > 0);

      const excludedFolderRules = excludedFolders
        .split(",")
        .map((value) => value.trim())
        .filter((value) => value.length > 0);

      const parsedMaxSize = Number.parseInt(maxFileSizeMb, 10);
      const maxSizeValue = Number.isFinite(parsedMaxSize) && parsedMaxSize > 0 ? parsedMaxSize : 128;

      const status = await invoke<FileWatcherStatus>("start_file_watcher", {
        roots: searchRoots,
        excludedExtensions: exclusions,
        excludedFolders: excludedFolderRules,
        maxFileSizeMb: maxSizeValue,
        debounceMs: 1200,
      });

      setWatcherStatus(status);
      setIndexFeedback({ type: "success", text: "Watcher activo" });
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo iniciar watcher" });
    } finally {
      setIsWatcherLoading(false);
    }
  };

  const stopWatcher = async () => {
    setIsWatcherLoading(true);

    try {
      const status = await invoke<FileWatcherStatus>("stop_file_watcher");
      setWatcherStatus(status);
      setIndexFeedback({ type: "success", text: "Watcher detenido" });
    } catch {
      setIndexFeedback({ type: "error", text: "No se pudo detener watcher" });
    } finally {
      setIsWatcherLoading(false);
    }
  };

  const formatIndexedAt = (value: string | null) => {
    if (!value) {
      return null;
    }

    const unixSecs = Number.parseInt(value, 10);
    if (!Number.isFinite(unixSecs)) {
      return null;
    }

    return new Date(unixSecs * 1000).toLocaleString();
  };

  const queryTokens = useMemo(() => {
    return query
      .toLowerCase()
      .split(/[^a-z0-9áéíóúñü]+/i)
      .filter((token) => token.length > 0);
  }, [query]);

  const renderHighlighted = (text: string) => {
    if (!text || queryTokens.length === 0) {
      return text;
    }

    const escaped = queryTokens.map((token) => token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    const pattern = new RegExp(`(${escaped.join("|")})`, "gi");
    const parts = text.split(pattern);

    return parts.map((part, index) => {
      const isMatch = queryTokens.some((token) => token === part.toLowerCase());
      if (isMatch) {
        return (
          <mark key={`m-${index}`} className="rounded-sm bg-emerald-400/30 px-0.5 text-emerald-100">
            {part}
          </mark>
        );
      }

      return <Fragment key={`t-${index}`}>{part}</Fragment>;
    });
  };

  const visibleRoots = (indexStatus?.roots.length ? indexStatus.roots : searchRoots).slice(0, 4);
  const hasMoreRoots = (indexStatus?.roots.length ? indexStatus.roots.length : searchRoots.length) > visibleRoots.length;
  const quickLookItem = quickLookPath ? results.find((item) => item.path === quickLookPath) ?? null : null;

  const quickLookExt = useMemo(() => {
    if (!quickLookItem?.path) {
      return "";
    }

    const lower = quickLookItem.path.toLowerCase();
    const dot = lower.lastIndexOf(".");
    if (dot < 0) {
      return "";
    }

    return lower.slice(dot + 1);
  }, [quickLookItem?.path]);

  const isQuickLookImage = quickLookExt === "png" || quickLookExt === "jpg" || quickLookExt === "jpeg" || quickLookExt === "webp" || quickLookExt === "gif" || quickLookExt === "bmp" || quickLookExt === "tiff";
  const hasVisualPreview = isQuickLookImage || quickLookExt === "pdf";

  useEffect(() => {
    if (!quickLookItem?.path || !isQuickLookImage) {
      setQuickLookImageMeta(null);
      return;
    }

    let cancelled = false;

    void invoke<ImageMetadata>("get_image_metadata", { path: quickLookItem.path })
      .then((meta) => {
        if (!cancelled) {
          setQuickLookImageMeta(meta);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setQuickLookImageMeta(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [quickLookItem?.path, isQuickLookImage]);

  useEffect(() => {
    if (!quickLookItem?.path || !isQuickLookOpen) {
      setQuickLookTextPreview(null);
      return;
    }

    let cancelled = false;

    void invoke<FileTextPreview>("get_file_text_preview", {
      path: quickLookItem.path,
      maxChars: 14000,
    })
      .then((preview) => {
        if (!cancelled) {
          setQuickLookTextPreview(preview);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setQuickLookTextPreview(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [quickLookItem?.path, isQuickLookOpen]);

  useEffect(() => {
    if (!quickLookItem?.path || !isQuickLookOpen || quickLookMode !== "visual" || !hasVisualPreview) {
      setQuickLookVisualUrl(null);
      setQuickLookVisualReason(null);
      setIsQuickLookVisualLoading(false);
      return;
    }

    let cancelled = false;
    setIsQuickLookVisualLoading(true);
    setQuickLookVisualUrl(null);
    setQuickLookVisualReason(null);

    void invoke<FileVisualPreview>("get_file_visual_preview", {
      path: quickLookItem.path,
      maxMb: quickLookExt === "pdf" ? 24 : 16,
    })
      .then((preview) => {
        if (cancelled) {
          return;
        }

        if (preview.available && preview.data_url) {
          setQuickLookVisualUrl(preview.data_url);
        } else {
          setQuickLookVisualFailed(true);
          setQuickLookVisualReason(preview.reason ?? "Preview visual no disponible");
          if (quickLookTextPreview?.available) {
            setQuickLookMode("text");
          }
        }
      })
      .catch(() => {
        if (cancelled) {
          return;
        }

        setQuickLookVisualFailed(true);
        setQuickLookVisualReason("Preview visual falló al cargar");
        if (quickLookTextPreview?.available) {
          setQuickLookMode("text");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsQuickLookVisualLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [quickLookItem?.path, isQuickLookOpen, quickLookMode, hasVisualPreview, quickLookExt, quickLookTextPreview?.available]);

  useEffect(() => {
    if (!isQuickLookOpen || quickLookMode !== "visual" || quickLookVisualLoaded || quickLookVisualFailed) {
      return;
    }

    if (quickLookExt !== "pdf") {
      return;
    }

    const timeout = setTimeout(() => {
      if (!quickLookVisualLoaded && quickLookTextPreview?.available) {
        setQuickLookVisualFailed(true);
        setQuickLookVisualReason("El render visual tardó demasiado; cambiamos a texto");
        setQuickLookMode("text");
      }
    }, 7000);

    return () => clearTimeout(timeout);
  }, [isQuickLookOpen, quickLookMode, quickLookExt, quickLookVisualLoaded, quickLookVisualFailed, quickLookTextPreview?.available]);

  return (
    <div
      className="min-h-screen w-full bg-[radial-gradient(ellipse_at_top,var(--tw-gradient-stops))] from-[#0a0b0f] via-[#020203] to-[#010101] flex flex-col items-center justify-start px-5 pb-5 pt-8"
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="vault-scroll relative w-full max-w-5xl z-10 max-h-[calc(100vh-2rem)] overflow-y-auto pb-4"
      >
        <div className="mb-3 flex items-center justify-end">
          <button
            type="button"
            className="rounded-lg bg-white/10 p-2 text-gray-200 transition-colors hover:bg-white/20"
            title="Configuración"
            aria-label="Abrir configuración"
            onClick={() => setIsConfigOpen(true)}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 15.5A3.5 3.5 0 1 0 12 8.5a3.5 3.5 0 0 0 0 7Z"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06A1.65 1.65 0 0 0 15 19.4a1.65 1.65 0 0 0-1 .6 1.65 1.65 0 0 0-.33 1V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-.33-1A1.65 1.65 0 0 0 8 19.4a1.65 1.65 0 0 0-1-.6 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-.6-1 1.65 1.65 0 0 0-1-.33H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1-.33A1.65 1.65 0 0 0 4.6 8a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 8 4.6a1.65 1.65 0 0 0 1-.6 1.65 1.65 0 0 0 .33-1V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 .33 1A1.65 1.65 0 0 0 15 4.6a1.65 1.65 0 0 0 1 .6 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 8a1.65 1.65 0 0 0 .6 1 1.65 1.65 0 0 0 1 .33H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1 .33 1.65 1.65 0 0 0-.51 1.34Z"/></svg>
          </button>
        </div>

        <div className={`transition-opacity duration-200 ${isConfigOpen ? "opacity-35" : "opacity-100"}`}>

        {!indexStatus?.has_index && (
          <div className="mb-4 rounded-xl bg-white/4 p-4 ring-1 ring-white/10">
            <p className="text-sm font-medium text-white">Primera ejecución: crea tu índice local</p>
            <p className="mt-1 text-xs text-gray-400">
              Indexa una vez y las búsquedas por nombre/contenido (PDF, MD, TXT) serán mucho más rápidas.
            </p>
            <div className="mt-3 flex flex-wrap gap-2">
              <button
                type="button"
                className="rounded-md bg-blue-500/70 px-3 py-1.5 text-xs text-white transition-colors hover:bg-blue-500"
                disabled={isIndexing}
                onClick={() => {
                  void startIndexing(["C:\\"]);
                }}
              >
                {isIndexing ? "Indexando..." : "Indexar todo C:"}
              </button>
              <button
                type="button"
                className="rounded-md bg-white/10 px-3 py-1.5 text-xs text-gray-200 transition-colors hover:bg-white/20"
                disabled={isIndexing}
                onClick={() => {
                  setIsConfigOpen(true);
                }}
              >
                Abrir configuración
              </button>
            </div>
          </div>
        )}

        {indexStatus?.has_index && (
          <div className="mb-4 rounded-xl bg-white/4 p-4 ring-1 ring-white/10">
            <p className="text-xs text-gray-300">
              Índice activo: <span className="text-white">{indexStatus.indexed_files}</span> archivos en <span className="text-white">{indexStatus.roots.length}</span> carpetas
            </p>
            <p className="mt-1 text-[11px] text-gray-500">
              Archivos que se indexan por contenido: PDF, MD y TXT.
            </p>
            {visibleRoots.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {visibleRoots.map((root) => (
                  <span key={root} className="inline-flex max-w-full items-center gap-1 truncate rounded-md bg-emerald-500/15 px-2 py-1 text-[11px] text-emerald-300 ring-1 ring-emerald-400/30">
                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7.5A2.5 2.5 0 0 1 5.5 5h4.3a2 2 0 0 1 1.4.58l1.22 1.22a2 2 0 0 0 1.41.58H18.5A2.5 2.5 0 0 1 21 10v8.5a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 18.5Z"/></svg>
                    <span className="truncate">{root}</span>
                    <button
                      type="button"
                      className="rounded-sm bg-black/20 px-1 text-[10px] text-emerald-200 hover:bg-black/40"
                      title="Olvidar carpeta y reindexar"
                      disabled={isMaintenanceBusy || isIndexing}
                      onClick={() => {
                        void forgetRootPersisted(root);
                      }}
                    >
                      ×
                    </button>
                  </span>
                ))}
                {hasMoreRoots && (
                  <span className="rounded-md bg-emerald-500/15 px-2 py-1 text-[11px] text-emerald-300 ring-1 ring-emerald-400/30">+{(indexStatus?.roots.length ?? 0) - visibleRoots.length} más</span>
                )}
              </div>
            )}
            <div className="mt-2 flex flex-wrap gap-2">
              <button
                type="button"
                className="rounded-md bg-white/10 px-2.5 py-1 text-[11px] text-gray-200 transition-colors hover:bg-white/20"
                disabled={isIndexing}
                onClick={() => {
                  void startIndexing(indexStatus.roots.length > 0 ? indexStatus.roots : ["C:\\"]);
                }}
              >
                {isIndexing ? "Reindexando..." : "Reindexar"}
              </button>

              {isIndexing && (
                <button
                  type="button"
                  className="rounded-md bg-red-500/20 px-2.5 py-1 text-[11px] text-red-300 ring-1 ring-red-400/30 transition-colors hover:bg-red-500/30"
                  onClick={() => {
                    void cancelIndexing();
                  }}
                >
                  Cancelar
                </button>
              )}

              {indexFeedback && (
                <span
                  className={`inline-flex items-center rounded-md px-2.5 py-1 text-[11px] ${
                    indexFeedback.type === "running"
                      ? "bg-blue-500/20 text-blue-300 ring-1 ring-blue-400/30"
                      : indexFeedback.type === "success"
                      ? "bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-400/30"
                      : "bg-red-500/20 text-red-300 ring-1 ring-red-400/30"
                  }`}
                >
                  {indexFeedback.text}
                </span>
              )}
            </div>

            {indexProgress && (
              <div className="mt-2 rounded-md bg-white/5 px-2.5 py-2 ring-1 ring-white/10">
                <p className="text-[11px] text-gray-300">
                  {indexProgress.message} · escaneados {indexProgress.scanned_files.toLocaleString()} · indexados{" "}
                  {indexProgress.indexed_files.toLocaleString()}
                </p>
                <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-white/10">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${
                      indexProgress.done ? "w-full bg-emerald-400/70" : "w-1/2 animate-pulse bg-blue-400/70"
                    }`}
                  />
                </div>
              </div>
            )}

            {indexDiagnostics &&
              (indexDiagnostics.scanned_files > 0 || indexDiagnostics.pdf_scanned > 0 || Boolean(indexDiagnostics.last_error)) && (
                <div className="mt-2 rounded-md bg-white/5 px-2.5 py-2 ring-1 ring-white/10">
                  <p className="text-[11px] text-gray-300">
                    Diagnóstico: escaneados {indexDiagnostics.scanned_files.toLocaleString()} · indexados{" "}
                    {indexDiagnostics.indexed_files.toLocaleString()} · PDF {indexDiagnostics.pdf_indexed.toLocaleString()}/
                    {indexDiagnostics.pdf_scanned.toLocaleString()} ok
                  </p>
                  {indexDiagnostics.pdf_failed > 0 && (
                    <p className="mt-1 text-[11px] text-amber-300">
                      PDFs sin texto extraíble: {indexDiagnostics.pdf_failed.toLocaleString()}
                    </p>
                  )}
                  {indexDiagnostics.pdf_fallback_used > 0 && (
                    <p className="mt-1 text-[11px] text-sky-300">
                      PDFs recuperados por fallback avanzado: {indexDiagnostics.pdf_fallback_used.toLocaleString()}
                    </p>
                  )}
                  <p className="mt-1 text-[11px] text-gray-500">
                    LanceDB sync: {indexDiagnostics.lancedb_synced ? "OK" : "pendiente/fallback SQLite"}
                  </p>
                  {indexDiagnostics.last_error && (
                    <p className="mt-1 text-[11px] text-red-300">{indexDiagnostics.last_error}</p>
                  )}
                  {indexDiagnostics.pdf_failed_examples.length > 0 && (
                    <div className="mt-1 space-y-1">
                      {indexDiagnostics.pdf_failed_examples.slice(0, 3).map((example) => (
                        <p key={example} className="truncate text-[10px] text-gray-500">
                          {example}
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              )}

            {watcherStatus && (
              <div className="mt-2 rounded-md bg-white/5 px-2.5 py-2 ring-1 ring-white/10">
                <p className="text-[11px] text-gray-300">
                  Watcher: {watcherStatus.running ? "activo" : "detenido"}
                  {watcherStatus.pending_events ? " · cambios detectados" : ""}
                </p>
                {watcherStatus.last_reindex_at && (
                  <p className="mt-1 text-[11px] text-gray-500">
                    Última auto-indexación: {formatIndexedAt(watcherStatus.last_reindex_at)}
                  </p>
                )}
                {watcherStatus.last_error && (
                  <p className="mt-1 text-[11px] text-amber-300">{watcherStatus.last_error}</p>
                )}
              </div>
            )}

            {formatIndexedAt(indexStatus.indexed_at) && (
              <p className="mt-1 text-[11px] text-gray-500">
                Última indexación: {formatIndexedAt(indexStatus.indexed_at)}
              </p>
            )}
          </div>
        )}

        {runtimeMetrics && (
          <div className="mb-4 rounded-xl bg-white/4 p-3 ring-1 ring-white/10">
            <p className="text-xs text-gray-300">
              Métricas runtime · búsq semántica: <span className="text-white">{runtimeMetrics.semantic_calls}</span> ({runtimeMetrics.semantic_avg_ms.toFixed(1)} ms prom) · RAG: <span className="text-white">{runtimeMetrics.rag_calls}</span> ({runtimeMetrics.rag_avg_ms.toFixed(1)} ms prom)
            </p>
            <p className="mt-1 text-[11px] text-gray-500">
              Indexaciones: {runtimeMetrics.indexing_runs} · promedio {runtimeMetrics.indexing_avg_ms.toFixed(0)} ms · última {runtimeMetrics.last_index_ms} ms
            </p>
            <p className="mt-1 text-[11px] text-gray-500">
              Caché embeddings: {runtimeMetrics.embedding_cache_items} items · hits {runtimeMetrics.embedding_cache_hits} · misses {runtimeMetrics.embedding_cache_misses} · hit rate {runtimeMetrics.embedding_cache_hit_rate.toFixed(1)}%
            </p>
          </div>
        )}

        <div className="group flex items-center gap-3 rounded-xl bg-white/5 px-4 transition-all duration-300 focus-within:bg-white/10 ring-1 ring-white/10">
          <div className="text-gray-500 group-focus-within:text-blue-400 transition-colors pointer-events-none shrink-0">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-search"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
          </div>

          <input
            type="text"
            className="w-full appearance-none border-0 bg-transparent py-4 text-lg text-white placeholder:text-gray-400 caret-blue-400 outline-none"
            placeholder="Buscar en tu memoria..."
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedIndex(-1);
              setRagResponse(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown" && results.length > 0) {
                e.preventDefault();
                setSelectedIndex((prev) => {
                  const next = Math.min(prev + 1, results.length - 1);
                  setQuickLookPath(results[next]?.path ?? null);
                  return next;
                });
                return;
              }

              if (e.key === "ArrowUp" && results.length > 0) {
                e.preventDefault();
                setSelectedIndex((prev) => {
                  const next = Math.max(prev - 1, 0);
                  setQuickLookPath(results[next]?.path ?? null);
                  return next;
                });
                return;
              }

              if (e.key === "Enter") {
                e.preventDefault();

                if ((e.ctrlKey || e.metaKey) && selectedIndex >= 0 && selectedIndex < results.length) {
                  void openFile(results[selectedIndex].path);
                  return;
                }

                void runSearch();
              }

              if (e.key === "Escape") {
                if (isQuickLookOpen) {
                  setIsQuickLookOpen(false);
                  return;
                }

                setResults([]);
                setHasSearched(false);
                setErrorMessage(null);
                setSelectedIndex(-1);
                setQuickLookPath(null);
              }
            }}
            autoFocus
          />

          <button
            type="button"
            className={`shrink-0 rounded-md px-2 py-1 text-[10px] font-medium transition-colors ${
              imagesOnly
                ? "bg-fuchsia-500/20 text-fuchsia-300 ring-1 ring-fuchsia-400/30"
                : "bg-white/10 text-gray-300 hover:bg-white/20"
            }`}
            onClick={() => setImagesOnly((value) => !value)}
            title="Limitar búsqueda a imágenes"
          >
            Solo imágenes
          </button>

          <button
            type="button"
            className="shrink-0 rounded-md bg-white/10 px-2 py-1 text-[10px] font-medium text-gray-300 transition-colors hover:bg-white/20"
            onClick={() => {
              void runLocalAnswer();
            }}
            disabled={isRagLoading}
            title="Responder usando tus archivos indexados"
          >
            {isRagLoading ? "Respondiendo..." : "Responder"}
          </button>

          <button
            type="button"
            className="shrink-0 rounded-md bg-violet-500/20 px-2 py-1 text-[10px] font-medium text-violet-300 ring-1 ring-violet-400/30 transition-colors hover:bg-violet-500/30"
            onClick={() => {
              void runClipImageSearch();
            }}
            disabled={isClipSearching}
            title="Buscar imágenes por semántica CLIP ONNX"
          >
            {isClipSearching ? "CLIP..." : "CLIP Img"}
          </button>

          {clipStatus?.configured && (
            <span
              className={`shrink-0 rounded-md px-2 py-1 text-[10px] font-medium ring-1 ${
                clipValidation?.text_inference_ok && (!clipValidation.sample_image_path || clipValidation.image_inference_ok)
                  ? "bg-emerald-500/20 text-emerald-300 ring-emerald-400/30"
                  : "bg-amber-500/20 text-amber-300 ring-amber-400/30"
              }`}
              title={clipValidation?.message ?? "Estado CLIP pendiente de verificación"}
            >
              {clipValidation?.text_inference_ok && (!clipValidation.sample_image_path || clipValidation.image_inference_ok)
                ? "CLIP OK"
                : "CLIP CHECK"}
            </span>
          )}

          <div className="flex shrink-0 items-center gap-1">
            <button
              type="button"
              className={`rounded-md px-2 py-1 text-[10px] ${answerMode === "auto" ? "bg-cyan-500/20 text-cyan-300 ring-1 ring-cyan-400/30" : "bg-white/10 text-gray-300 hover:bg-white/20"}`}
              onClick={() => setAnswerMode("auto")}
              title="AUTO: intenta cloud y cae a local"
            >
              AUTO
            </button>
            <button
              type="button"
              className={`rounded-md px-2 py-1 text-[10px] ${answerMode === "local" ? "bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-400/30" : "bg-white/10 text-gray-300 hover:bg-white/20"}`}
              onClick={() => setAnswerMode("local")}
              title="LOCAL: solo índice local"
            >
              LOCAL
            </button>
            <button
              type="button"
              className={`rounded-md px-2 py-1 text-[10px] ${answerMode === "cloud" ? "bg-indigo-500/20 text-indigo-300 ring-1 ring-indigo-400/30" : "bg-white/10 text-gray-300 hover:bg-white/20"}`}
              onClick={() => setAnswerMode("cloud")}
              title="CLOUD: usa proveedor de chat configurado"
            >
              CLOUD
            </button>
          </div>

          <div className="flex shrink-0 items-center gap-1 rounded-md bg-white/5 px-2 py-1 ring-1 ring-white/10">
            <span className="text-[10px] text-gray-400">k</span>
            <input
              type="number"
              min={1}
              max={8}
              className="w-10 border-0 bg-transparent text-[10px] text-gray-200 outline-none"
              value={ragTopK}
              onChange={(e) => setRagTopK(e.target.value)}
              title="Número de fuentes para responder"
            />
          </div>

          <div className="flex shrink-0 items-center gap-1 rounded-md bg-white/5 px-2 py-1 ring-1 ring-white/10">
            <button
              type="button"
              className={`rounded px-1.5 py-0.5 text-[10px] ${strictGrounding ? "bg-amber-500/20 text-amber-300" : "bg-white/10 text-gray-300"}`}
              onClick={() => setStrictGrounding((value) => !value)}
              title="Bloquea respuestas cuando la evidencia no alcance el umbral"
            >
              Estricto
            </button>
            <input
              type="number"
              min={0}
              max={1.6}
              step={0.05}
              className="w-12 border-0 bg-transparent text-[10px] text-gray-200 outline-none"
              value={ragMinScore}
              onChange={(e) => setRagMinScore(e.target.value)}
              title="Umbral mínimo de score"
            />
          </div>

          {searchModeBadge && (
            <span
              className={`mr-1 rounded-md px-2 py-1 text-[10px] font-medium ${
                searchModeBadge === "CLOUD"
                  ? "bg-indigo-500/20 text-indigo-300 ring-1 ring-indigo-400/30"
                  : "bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-400/30"
              }`}
              title={searchModeBadge === "CLOUD" ? "Resultados con embeddings API" : "Resultados locales"}
            >
              {searchModeBadge}
            </span>
          )}
        </div>

        {isLoading && (
          <p className="mt-4 text-sm text-center text-gray-400">Buscando en tu bóveda...</p>
        )}

        {!isLoading && errorMessage && (
          <p className="mt-4 text-sm text-center text-red-400">{errorMessage}</p>
        )}

        {!isLoading && !errorMessage && hasSearched && results.length === 0 && (
          <p className="mt-4 text-sm text-center text-gray-500">No se encontraron resultados.</p>
        )}

        {ragResponse && (
          <div className="mt-5 rounded-xl bg-white/4 p-4 ring-1 ring-white/10">
            <div className="flex items-center justify-between gap-2">
              <p className="text-xs font-medium text-white">Respuesta con evidencia local</p>
              <div className="flex items-center gap-2">
                <span
                  className={`rounded-md px-2 py-1 text-[10px] font-medium ${
                    ragResponse.mode === "cloud"
                      ? "bg-indigo-500/20 text-indigo-300 ring-1 ring-indigo-400/30"
                      : "bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-400/30"
                  }`}
                >
                  {ragResponse.mode.toUpperCase()}
                </span>
                <span
                  className={`rounded-md px-2 py-1 text-[10px] font-medium ${
                    ragResponse.grounded
                      ? "bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-400/30"
                      : "bg-amber-500/20 text-amber-300 ring-1 ring-amber-400/30"
                  }`}
                >
                  {ragResponse.grounded ? "CON EVIDENCIA" : "SIN EVIDENCIA"}
                </span>
              </div>
            </div>

            <pre className="mt-2 whitespace-pre-wrap text-xs leading-relaxed text-gray-300">{ragResponse.answer}</pre>

            {ragResponse.sources.length > 0 && (
              <div className="mt-3 space-y-2">
                {ragResponse.sources.map((source) => (
                  <div key={`${source.path}-${source.score}`} className="rounded-md bg-white/5 p-2 ring-1 ring-white/10">
                    <div className="flex items-center gap-2">
                      <span className="rounded bg-white/10 px-1.5 py-0.5 text-[10px] text-gray-300">[{source.ref_id || "S?"}]</span>
                      <p className="truncate text-[11px] text-white">{source.title}</p>
                    </div>
                    <p className="mt-1 line-clamp-2 text-[11px] text-gray-400">{source.snippet}</p>
                    <div className="mt-1 flex items-center gap-2">
                      <p className="truncate text-[10px] text-gray-500">{source.path}</p>
                      <span className="rounded bg-white/10 px-1.5 py-0.5 text-[10px] text-gray-300">score {source.score.toFixed(2)}</span>
                    </div>
                    <div className="mt-2 flex items-center gap-2">
                      <button
                        type="button"
                        className="rounded-md bg-white/10 px-2 py-1 text-[10px] text-gray-200 transition-colors hover:bg-white/20"
                        onClick={() => {
                          void openFile(source.path);
                        }}
                      >
                        Abrir fuente
                      </button>
                      <button
                        type="button"
                        className="rounded-md bg-white/5 px-2 py-1 text-[10px] text-gray-300 transition-colors hover:bg-white/15"
                        onClick={() => {
                          void openContainingFolder(source.path);
                        }}
                      >
                        Carpeta
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        <div className="mt-4 rounded-xl bg-white/4 p-4 ring-1 ring-white/10">
          <div className="flex items-center justify-between gap-2">
            <p className="text-xs font-medium text-white">Historial local de respuestas</p>
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="rounded-md bg-white/10 px-2 py-1 text-[10px] text-gray-200 transition-colors hover:bg-white/20"
                onClick={() => {
                  void exportChatHistory();
                }}
                disabled={isChatHistoryLoading}
              >
                Exportar
              </button>
              <button
                type="button"
                className="rounded-md bg-white/10 px-2 py-1 text-[10px] text-gray-200 transition-colors hover:bg-white/20"
                onClick={() => {
                  void clearChatHistory();
                }}
                disabled={isChatHistoryLoading}
              >
                Limpiar historial
              </button>
            </div>
          </div>

          {chatHistory.length === 0 ? (
            <p className="mt-2 text-[11px] text-gray-500">Sin historial todavía.</p>
          ) : (
            <div className="mt-2 space-y-2">
              {chatHistory.map((entry) => (
                <div key={entry.id} className="rounded-md bg-white/5 p-2 ring-1 ring-white/10">
                  <div className="flex items-center justify-between gap-2">
                    <p className="truncate text-[11px] text-white">{entry.query}</p>
                    <div className="flex items-center gap-1">
                      <span className="rounded-md bg-white/10 px-1.5 py-0.5 text-[10px] text-gray-300">{entry.mode.toUpperCase()}</span>
                      <span className={`rounded-md px-1.5 py-0.5 text-[10px] ${entry.grounded ? "bg-emerald-500/20 text-emerald-300" : "bg-amber-500/20 text-amber-300"}`}>
                        {entry.grounded ? "OK" : "NO-EV"}
                      </span>
                    </div>
                  </div>
                  <p className="mt-1 line-clamp-2 text-[11px] text-gray-400">{entry.answer}</p>
                </div>
              ))}
            </div>
          )}
        </div>

        {!isLoading && results.length > 0 && (
          <div className="mt-5 max-h-[62vh] space-y-2 overflow-y-auto pr-1 pb-2">
            {results.map((item, index) => (
              <div
                key={item.path}
                className={`rounded-lg px-4 py-3 transition-colors ${
                  selectedIndex === index ? "bg-white/10" : "bg-white/4"
                }`}
                onMouseEnter={() => {
                  setSelectedIndex(index);
                }}
                onClick={() => {
                  setSelectedIndex(index);
                }}
                onDoubleClick={() => {
                  void openFile(item.path);
                }}
              >
                <p className="text-sm font-medium text-white">{renderHighlighted(item.title)}</p>
                <p className="mt-1 text-xs text-gray-400">{renderHighlighted(item.snippet)}</p>
                {item.match_reason && (
                  <div className="mt-1 flex items-center gap-2">
                    <span
                      className={`rounded-md px-1.5 py-0.5 text-[10px] font-medium ring-1 ${
                        item.origin === "cloud-semantic"
                          ? "bg-indigo-500/15 text-indigo-300 ring-indigo-400/30"
                          : "bg-emerald-500/15 text-emerald-300 ring-emerald-400/30"
                      }`}
                    >
                      {item.origin === "cloud-semantic" ? "CLOUD" : "LOCAL"}
                    </span>
                    <p className="truncate text-[11px] text-gray-500" title={item.match_reason}>
                      {renderHighlighted(item.match_reason)}
                    </p>
                  </div>
                )}
                <p className="mt-1 text-[11px] text-gray-500 truncate">{item.path}</p>
                <div className="mt-2 flex items-center gap-2">
                  <button
                    type="button"
                    className="rounded-md bg-white/10 px-2.5 py-1 text-[11px] text-gray-200 transition-colors hover:bg-white/20"
                    onClick={() => {
                      void openFile(item.path);
                    }}
                  >
                    Abrir
                  </button>
                  <button
                    type="button"
                    className="rounded-md bg-white/5 px-2.5 py-1 text-[11px] text-gray-300 transition-colors hover:bg-white/15"
                    onClick={() => {
                      void openContainingFolder(item.path);
                    }}
                  >
                    Carpeta
                  </button>
                  <button
                    type="button"
                    className="inline-flex items-center gap-1 rounded-md bg-white/5 px-2.5 py-1 text-[11px] text-gray-300 transition-colors hover:bg-white/15"
                    onClick={() => {
                      setQuickLookPath(item.path);
                      setQuickLookVisualLoaded(false);
                      setQuickLookVisualFailed(false);
                      const lower = item.path.toLowerCase();
                      const ext = lower.includes(".") ? lower.slice(lower.lastIndexOf(".") + 1) : "";
                      const visual = ["pdf", "png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"].includes(ext);
                      setQuickLookMode(visual ? "visual" : "text");
                      setIsQuickLookOpen(true);
                    }}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/><circle cx="12" cy="12" r="3"/></svg>
                    Ver
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        </div>

        {isConfigOpen && (
          <div
            className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/75 p-4 pt-8 backdrop-blur-sm"
            onClick={() => setIsConfigOpen(false)}
          >
            <div
              className="w-full max-w-3xl rounded-xl bg-[#07090d] p-4 shadow-xl ring-1 ring-white/10 max-h-[88vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-medium text-white">Configuración de indexación</h2>
                <button
                  type="button"
                  className="rounded-md bg-white/10 px-2 py-1 text-xs text-gray-200 hover:bg-white/20"
                  onClick={() => setIsConfigOpen(false)}
                >
                  Cerrar
                </button>
              </div>

              <div className="mt-4 space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-xs uppercase tracking-wide text-gray-500">Carpetas a indexar</p>
                    <button
                      type="button"
                      className="rounded-md bg-white/10 px-2.5 py-1 text-[11px] text-gray-200 transition-colors hover:bg-white/20"
                      onClick={() => {
                        void pickFolders();
                      }}
                    >
                      Añadir carpeta
                    </button>
                  </div>

                  {searchRoots.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {searchRoots.map((root) => (
                        <button
                          key={root}
                          type="button"
                          className="max-w-full truncate rounded-md bg-white/10 px-2 py-1 text-[11px] text-gray-300 hover:bg-white/15"
                          title="Quitar carpeta"
                          onClick={() => removeRoot(root)}
                        >
                          {root}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-gray-500">Sin carpetas seleccionadas: por defecto se usan Documents/Desktop/Downloads.</p>
                  )}
                </div>

                <div className="space-y-1">
                  <label className="text-[11px] text-gray-500" htmlFor="excluded-exts">
                    Excluir extensiones (coma separada)
                  </label>
                  <input
                    id="excluded-exts"
                    type="text"
                    className="w-full appearance-none rounded-md border-0 bg-white/5 px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 outline-none focus:bg-white/10"
                    placeholder="mkv, mp4, zip"
                    value={excludedExtensions}
                    onChange={(e) => setExcludedExtensions(e.target.value)}
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-[11px] text-gray-500" htmlFor="excluded-folders">
                    Excluir carpetas por nombre (coma separada)
                  </label>
                  <input
                    id="excluded-folders"
                    type="text"
                    className="w-full appearance-none rounded-md border-0 bg-white/5 px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 outline-none focus:bg-white/10"
                    placeholder="node_modules, .git, target"
                    value={excludedFolders}
                    onChange={(e) => setExcludedFolders(e.target.value)}
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-[11px] text-gray-500" htmlFor="max-size-mb">
                    Tamaño máximo por archivo (MB)
                  </label>
                  <input
                    id="max-size-mb"
                    type="number"
                    min={1}
                    className="w-full appearance-none rounded-md border-0 bg-white/5 px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 outline-none focus:bg-white/10"
                    value={maxFileSizeMb}
                    onChange={(e) => setMaxFileSizeMb(e.target.value)}
                  />
                </div>

                <div className="rounded-lg bg-white/5 p-3 ring-1 ring-white/10">
                  <p className="text-xs uppercase tracking-wide text-gray-400">Embeddings IA (API)</p>
                  <p className="mt-1 text-[11px] text-gray-500">
                    Activa ranking semántico real usando endpoint compatible OpenRouter/OpenAI.
                  </p>

                  <div className="mt-3 space-y-2">
                    <div className="space-y-1">
                      <label className="text-[11px] text-gray-500" htmlFor="ai-api-key">
                        API key
                      </label>
                      <input
                        id="ai-api-key"
                        type="password"
                        className="w-full appearance-none rounded-md border-0 bg-white/5 px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 outline-none focus:bg-white/10"
                        placeholder="sk-or-v1-..."
                        value={aiApiKey}
                        onChange={(e) => setAiApiKey(e.target.value)}
                      />
                    </div>

                    <div className="space-y-1">
                      <label className="text-[11px] text-gray-500" htmlFor="ai-model">
                        Modelo de embedding
                      </label>
                      <input
                        id="ai-model"
                        type="text"
                        className="w-full appearance-none rounded-md border-0 bg-white/5 px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 outline-none focus:bg-white/10"
                        placeholder="text-embedding-3-small"
                        value={aiModel}
                        onChange={(e) => setAiModel(e.target.value)}
                      />
                    </div>

                    <div className="space-y-1">
                      <label className="text-[11px] text-gray-500" htmlFor="ai-base-url">
                        Endpoint embeddings
                      </label>
                      <input
                        id="ai-base-url"
                        type="text"
                        className="w-full appearance-none rounded-md border-0 bg-white/5 px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 outline-none focus:bg-white/10"
                        placeholder="https://openrouter.ai/api/v1/embeddings"
                        value={aiBaseUrl}
                        onChange={(e) => setAiBaseUrl(e.target.value)}
                      />
                    </div>

                    <div className="space-y-1">
                      <label className="text-[11px] text-gray-500" htmlFor="ai-chat-model">
                        Modelo de chat
                      </label>
                      <input
                        id="ai-chat-model"
                        type="text"
                        className="w-full appearance-none rounded-md border-0 bg-white/5 px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 outline-none focus:bg-white/10"
                        placeholder="gpt-4o-mini"
                        value={aiChatModel}
                        onChange={(e) => setAiChatModel(e.target.value)}
                      />
                    </div>

                    <div className="space-y-1">
                      <label className="text-[11px] text-gray-500" htmlFor="ai-chat-base-url">
                        Endpoint chat
                      </label>
                      <input
                        id="ai-chat-base-url"
                        type="text"
                        className="w-full appearance-none rounded-md border-0 bg-white/5 px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 outline-none focus:bg-white/10"
                        placeholder="https://openrouter.ai/api/v1/chat/completions"
                        value={aiChatBaseUrl}
                        onChange={(e) => setAiChatBaseUrl(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="mt-3 flex items-center gap-2">
                    <button
                      type="button"
                      className="rounded-md bg-indigo-500/70 px-3 py-1.5 text-xs text-white transition-colors hover:bg-indigo-500"
                      disabled={isSavingAi}
                      onClick={() => {
                        void saveAiProvider();
                      }}
                    >
                      {isSavingAi ? "Guardando..." : "Guardar IA"}
                    </button>

                    {aiProviderStatus?.configured && (
                      <span className="text-[11px] text-emerald-300">
                        Configurada ({aiProviderStatus.api_key_hint ?? "key oculta"})
                      </span>
                    )}
                  </div>
                </div>

                <div className="rounded-lg bg-white/5 p-3 ring-1 ring-white/10">
                  <p className="text-xs uppercase tracking-wide text-gray-400">File Watcher (tiempo real)</p>
                  <p className="mt-1 text-[11px] text-gray-500">
                    Vigila cambios en disco y dispara reindexado automático con debounce.
                  </p>

                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      className="rounded-md bg-blue-500/70 px-3 py-1.5 text-xs text-white transition-colors hover:bg-blue-500"
                      disabled={isWatcherLoading || Boolean(watcherStatus?.running)}
                      onClick={() => {
                        void startWatcher();
                      }}
                    >
                      {isWatcherLoading && !watcherStatus?.running ? "Iniciando..." : "Iniciar watcher"}
                    </button>

                    <button
                      type="button"
                      className="rounded-md bg-white/10 px-3 py-1.5 text-xs text-gray-200 transition-colors hover:bg-white/20"
                      disabled={isWatcherLoading || !watcherStatus?.running}
                      onClick={() => {
                        void stopWatcher();
                      }}
                    >
                      {isWatcherLoading && watcherStatus?.running ? "Deteniendo..." : "Detener watcher"}
                    </button>

                    <span className="text-[11px] text-gray-400">
                      Estado: {watcherStatus?.running ? "activo" : "detenido"}
                    </span>
                  </div>
                </div>

                <div className="rounded-lg bg-white/5 p-3 ring-1 ring-white/10">
                  <p className="text-xs uppercase tracking-wide text-gray-400">Rendimiento adaptativo</p>
                  <p className="mt-1 text-[11px] text-gray-500">
                    Perfil hardware + caché de embeddings + benchmark local para ajustar rendimiento.
                  </p>

                  {hardwareProfile && (
                    <div className="mt-2 rounded-md bg-black/20 p-2 ring-1 ring-white/10">
                      <p className="text-[11px] text-gray-300">
                        CPU {hardwareProfile.cpu_cores} hilos · RAM {hardwareProfile.total_memory_gb.toFixed(1)} GB
                      </p>
                      <p className="mt-1 truncate text-[10px] text-gray-500">{hardwareProfile.cpu_brand}</p>
                      <p className="mt-1 text-[10px] text-gray-400">
                        Recomendado: modo {hardwareProfile.recommended_mode.toUpperCase()} · top-k {hardwareProfile.recommended_top_k} · max file {hardwareProfile.recommended_max_file_size_mb} MB
                      </p>
                      <p className="mt-1 text-[10px] text-gray-500">{hardwareProfile.note}</p>
                      <button
                        type="button"
                        className="mt-2 rounded-md bg-emerald-500/25 px-2.5 py-1 text-[11px] text-emerald-200 transition-colors hover:bg-emerald-500/35"
                        disabled={isApplyingPerfProfile}
                        onClick={() => {
                          void applyPerformanceProfile();
                        }}
                      >
                        {isApplyingPerfProfile ? "Aplicando..." : "Aplicar perfil recomendado"}
                      </button>
                    </div>
                  )}

                  {embeddingCacheStatus && (
                    <div className="mt-2 rounded-md bg-black/20 p-2 ring-1 ring-white/10">
                      <p className="text-[11px] text-gray-300">
                        Caché embeddings: {embeddingCacheStatus.items} items · hits {embeddingCacheStatus.hits} · misses {embeddingCacheStatus.misses}
                      </p>
                      <p className="mt-1 text-[10px] text-gray-500">Hit rate: {embeddingCacheStatus.hit_rate.toFixed(1)}%</p>
                      <button
                        type="button"
                        className="mt-2 rounded-md bg-white/10 px-2.5 py-1 text-[11px] text-gray-200 transition-colors hover:bg-white/20"
                        onClick={() => {
                          void clearEmbeddingCache();
                        }}
                      >
                        Limpiar caché embeddings
                      </button>
                    </div>
                  )}

                  <div className="mt-3 space-y-2">
                    <input
                      type="text"
                      className="w-full appearance-none rounded-md border-0 bg-white/5 px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 outline-none focus:bg-white/10"
                      placeholder="Consulta benchmark"
                      value={benchmarkQuery}
                      onChange={(e) => setBenchmarkQuery(e.target.value)}
                    />

                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        min={3}
                        max={40}
                        className="w-24 appearance-none rounded-md border-0 bg-white/5 px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 outline-none focus:bg-white/10"
                        value={benchmarkIterations}
                        onChange={(e) => setBenchmarkIterations(e.target.value)}
                      />
                      <button
                        type="button"
                        className="rounded-md bg-cyan-500/70 px-3 py-1.5 text-xs text-white transition-colors hover:bg-cyan-500"
                        disabled={isBenchmarking}
                        onClick={() => {
                          void runLocalBenchmark();
                        }}
                      >
                        {isBenchmarking ? "Benchmark..." : "Correr benchmark local"}
                      </button>
                      <button
                        type="button"
                        className="rounded-md bg-indigo-500/70 px-3 py-1.5 text-xs text-white transition-colors hover:bg-indigo-500"
                        disabled={isColdHotBenchmarking}
                        onClick={() => {
                          void runColdHotBenchmark();
                        }}
                      >
                        {isColdHotBenchmarking ? "Cold/Hot..." : "Benchmark cold/hot"}
                      </button>
                    </div>
                  </div>

                  {benchmarkResult && (
                    <div className="mt-2 rounded-md bg-black/20 p-2 ring-1 ring-white/10">
                      <p className="text-[11px] text-gray-300">
                        {benchmarkResult.backend} · {benchmarkResult.iterations} iter · avg {benchmarkResult.avg_ms.toFixed(1)} ms · p95 {benchmarkResult.p95_ms} ms
                      </p>
                      <p className="mt-1 text-[10px] text-gray-500">
                        best {benchmarkResult.best_ms} ms · worst {benchmarkResult.worst_ms} ms · resultados {benchmarkResult.last_result_count}
                      </p>
                    </div>
                  )}

                  {coldHotResult && (
                    <div className="mt-2 rounded-md bg-black/20 p-2 ring-1 ring-white/10">
                      <p className="text-[11px] text-gray-300">
                        cold avg {coldHotResult.cold_avg_ms.toFixed(1)} ms (p95 {coldHotResult.cold_p95_ms}) · hot avg {coldHotResult.hot_avg_ms.toFixed(1)} ms (p95 {coldHotResult.hot_p95_ms})
                      </p>
                      <p className="mt-1 text-[10px] text-gray-500">
                        speedup cache {coldHotResult.speedup_percent.toFixed(1)}% · candidatos {coldHotResult.candidate_count} · {coldHotResult.backend}
                      </p>
                    </div>
                  )}
                </div>

                <div className="flex flex-wrap gap-2 pt-1">
                  <button
                    type="button"
                    className="rounded-md bg-blue-500/70 px-3 py-1.5 text-xs text-white transition-colors hover:bg-blue-500"
                    disabled={isIndexing}
                    onClick={() => {
                      void startIndexing(["C:\\"]);
                    }}
                  >
                    {isIndexing ? "Indexando..." : "Indexar todo C:"}
                  </button>
                  <button
                    type="button"
                    className="rounded-md bg-white/10 px-3 py-1.5 text-xs text-gray-200 transition-colors hover:bg-white/20"
                    disabled={isIndexing || searchRoots.length === 0}
                    onClick={() => {
                      void startIndexing(searchRoots);
                    }}
                  >
                    Indexar selección
                  </button>
                </div>

                <div className="rounded-lg bg-white/5 p-3 ring-1 ring-white/10">
                  <p className="text-xs uppercase tracking-wide text-gray-400">Mantenimiento y respaldo</p>
                  <p className="mt-1 text-[11px] text-gray-500">
                    Exporta/importa configuración y gestiona limpieza total del índice local.
                  </p>

                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      className="rounded-md bg-white/10 px-3 py-1.5 text-xs text-gray-200 transition-colors hover:bg-white/20"
                      disabled={isMaintenanceBusy}
                      onClick={() => {
                        void exportConfigToJson();
                      }}
                    >
                      Exportar config
                    </button>

                    <button
                      type="button"
                      className="rounded-md bg-white/10 px-3 py-1.5 text-xs text-gray-200 transition-colors hover:bg-white/20"
                      disabled={isMaintenanceBusy}
                      onClick={() => {
                        void importConfigFromJson();
                      }}
                    >
                      Importar config
                    </button>

                    <button
                      type="button"
                      className="rounded-md bg-red-500/20 px-3 py-1.5 text-xs text-red-300 ring-1 ring-red-400/30 transition-colors hover:bg-red-500/30"
                      disabled={isMaintenanceBusy || isIndexing}
                      onClick={() => {
                        void resetIndexData();
                      }}
                    >
                      Reset índice local
                    </button>
                  </div>
                </div>

                <div className="rounded-lg bg-white/5 p-3 ring-1 ring-white/10">
                  <p className="text-xs uppercase tracking-wide text-gray-400">Privacidad y auditoría</p>
                  <p className="mt-1 text-[11px] text-gray-500">
                    Política: el contenido se indexa localmente por defecto. Solo se usa cloud cuando activas IA y eliges modo CLOUD/AUTO.
                  </p>
                  <p className="mt-1 text-[11px] text-gray-500">
                    Los logs de auditoría guardan eventos técnicos (sin exponer API keys completas) para diagnóstico.
                  </p>

                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      className="rounded-md bg-white/10 px-3 py-1.5 text-xs text-gray-200 transition-colors hover:bg-white/20"
                      disabled={isAuditLoading}
                      onClick={() => {
                        void refreshAuditLogs();
                      }}
                    >
                      Refrescar logs
                    </button>
                    <button
                      type="button"
                      className="rounded-md bg-white/10 px-3 py-1.5 text-xs text-gray-200 transition-colors hover:bg-white/20"
                      disabled={isAuditLoading}
                      onClick={() => {
                        void exportAuditLogs();
                      }}
                    >
                      Exportar logs
                    </button>
                    <button
                      type="button"
                      className="rounded-md bg-white/10 px-3 py-1.5 text-xs text-gray-200 transition-colors hover:bg-white/20"
                      disabled={isAuditLoading}
                      onClick={() => {
                        void clearAuditLogs();
                      }}
                    >
                      Limpiar logs
                    </button>
                  </div>

                  <div className="mt-2 max-h-28 space-y-1 overflow-y-auto rounded-md bg-black/20 p-2 ring-1 ring-white/10">
                    {auditLogs.length === 0 ? (
                      <p className="text-[10px] text-gray-500">Sin eventos de auditoría.</p>
                    ) : (
                      auditLogs.map((entry, index) => (
                        <p key={`${entry.timestamp}-${entry.event}-${index}`} className="text-[10px] text-gray-400">
                          [{entry.timestamp}] {entry.event} · {entry.detail}
                        </p>
                      ))
                    )}
                  </div>
                </div>

                <div className="rounded-lg bg-white/5 p-3 ring-1 ring-white/10">
                  <p className="text-xs uppercase tracking-wide text-gray-400">CLIP / ONNX (local)</p>
                  <p className="mt-1 text-[11px] text-gray-500">
                    Configura modelos ONNX de texto e imagen + tokenizer para búsqueda semántica texto→imagen local.
                  </p>

                  <div className="mt-3 space-y-2">
                    <input
                      type="text"
                      className="w-full appearance-none rounded-md border-0 bg-white/5 px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 outline-none focus:bg-white/10"
                      placeholder="Ruta modelo imagen ONNX"
                      value={clipImageModelPath}
                      onChange={(e) => setClipImageModelPath(e.target.value)}
                    />
                    <input
                      type="text"
                      className="w-full appearance-none rounded-md border-0 bg-white/5 px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 outline-none focus:bg-white/10"
                      placeholder="Ruta modelo texto ONNX"
                      value={clipTextModelPath}
                      onChange={(e) => setClipTextModelPath(e.target.value)}
                    />
                    <input
                      type="text"
                      className="w-full appearance-none rounded-md border-0 bg-white/5 px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 outline-none focus:bg-white/10"
                      placeholder="Ruta tokenizer.json"
                      value={clipTokenizerPath}
                      onChange={(e) => setClipTokenizerPath(e.target.value)}
                    />

                    <div className="grid grid-cols-2 gap-2">
                      <input
                        type="number"
                        min={128}
                        max={512}
                        className="w-full appearance-none rounded-md border-0 bg-white/5 px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 outline-none focus:bg-white/10"
                        placeholder="Input size"
                        value={clipInputSize}
                        onChange={(e) => setClipInputSize(e.target.value)}
                      />
                      <input
                        type="number"
                        min={16}
                        max={256}
                        className="w-full appearance-none rounded-md border-0 bg-white/5 px-3 py-2 text-xs text-gray-200 placeholder:text-gray-500 outline-none focus:bg-white/10"
                        placeholder="Max length"
                        value={clipMaxLength}
                        onChange={(e) => setClipMaxLength(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="mt-3 flex items-center gap-2">
                    <button
                      type="button"
                      className="rounded-md bg-violet-500/70 px-3 py-1.5 text-xs text-white transition-colors hover:bg-violet-500"
                      disabled={isClipSaving}
                      onClick={() => {
                        void saveClipConfig();
                      }}
                    >
                      {isClipSaving ? "Guardando..." : "Guardar CLIP"}
                    </button>

                    <button
                      type="button"
                      className="rounded-md bg-white/10 px-3 py-1.5 text-xs text-gray-200 transition-colors hover:bg-white/20"
                      disabled={isClipValidating}
                      onClick={() => {
                        void validateClipSetup();
                      }}
                    >
                      {isClipValidating ? "Validando..." : "Verificar CLIP"}
                    </button>

                    {clipStatus?.configured && (
                      <span className={`text-[11px] ${clipStatus.enabled ? "text-emerald-300" : "text-amber-300"}`}>
                        {clipStatus.enabled ? "CLIP activo" : "CLIP inactivo"}
                      </span>
                    )}

                    <button
                      type="button"
                      className="rounded-md bg-white/10 px-3 py-1.5 text-xs text-gray-200 transition-colors hover:bg-white/20"
                      onClick={() => {
                        void clearClipImageCache();
                      }}
                    >
                      Limpiar cache CLIP
                    </button>

                    {clipImageCacheStatus && (
                      <span className="text-[11px] text-gray-400">cache img: {clipImageCacheStatus.items}</span>
                    )}
                  </div>

                  {clipValidation && (
                    <div className="mt-3 rounded-md bg-black/20 p-2 ring-1 ring-white/10">
                      <p className="text-[11px] text-gray-300">{clipValidation.message}</p>
                      <p className="mt-1 text-[10px] text-gray-500">
                        tokenizer: {clipValidation.tokenizer_ok ? "OK" : "FAIL"} · text model: {clipValidation.text_model_ok ? "OK" : "FAIL"} · image model: {clipValidation.image_model_ok ? "OK" : "FAIL"}
                      </p>
                      <p className="mt-1 text-[10px] text-gray-500">
                        infer texto: {clipValidation.text_inference_ok ? "OK" : "FAIL"}
                        {clipValidation.text_dim ? ` (${clipValidation.text_dim} dim)` : ""}
                        {" · "}
                        infer imagen: {clipValidation.image_inference_ok ? "OK" : "PEND/FAIL"}
                        {clipValidation.image_dim ? ` (${clipValidation.image_dim} dim)` : ""}
                      </p>
                      {clipValidation.sample_image_path && (
                        <p className="mt-1 truncate text-[10px] text-gray-500">muestra: {clipValidation.sample_image_path}</p>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {isQuickLookOpen && quickLookItem && (
          <div
            className="fixed inset-0 z-60 flex items-center justify-center bg-black/80 p-4 backdrop-blur-sm"
            onClick={() => setIsQuickLookOpen(false)}
          >
            <div
              className="w-full max-w-5xl rounded-xl bg-[#07090d] p-4 shadow-xl ring-1 ring-white/10 max-h-[88vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-white">{quickLookItem.title}</p>
                  <p className="truncate text-[11px] text-gray-500">{quickLookItem.path}</p>
                  {isQuickLookImage && quickLookImageMeta && (
                    <p className="truncate text-[10px] text-gray-400">
                      {quickLookImageMeta.format ?? quickLookExt.toUpperCase()} · {quickLookImageMeta.width ?? "?"}x{quickLookImageMeta.height ?? "?"}
                      {quickLookImageMeta.date_taken ? ` · ${quickLookImageMeta.date_taken}` : ""}
                    </p>
                  )}

                  {quickLookVisualFailed && (
                    <p className="mt-1 text-[10px] text-amber-300">{quickLookVisualReason ?? "Render visual falló; cambiamos automáticamente a vista textual."}</p>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  {hasVisualPreview && (
                    <button
                      type="button"
                      className={`rounded-md px-2 py-1 text-xs transition-colors ${
                        quickLookMode === "visual" ? "bg-blue-500/70 text-white" : "bg-white/10 text-gray-200 hover:bg-white/20"
                      }`}
                      onClick={() => setQuickLookMode("visual")}
                    >
                      Visual
                    </button>
                  )}
                  <button
                    type="button"
                    className={`rounded-md px-2 py-1 text-xs transition-colors ${
                      quickLookMode === "text" ? "bg-blue-500/70 text-white" : "bg-white/10 text-gray-200 hover:bg-white/20"
                    }`}
                    onClick={() => setQuickLookMode("text")}
                  >
                    Texto
                  </button>
                  <button
                    type="button"
                    className="rounded-md bg-white/10 px-2 py-1 text-xs text-gray-200 hover:bg-white/20"
                    onClick={() => setIsQuickLookOpen(false)}
                  >
                    Cerrar
                  </button>
                </div>
              </div>

              <div className="mt-3 h-[68vh] overflow-hidden rounded-md bg-black/30 ring-1 ring-white/10">
                {quickLookMode === "visual" && isQuickLookImage && quickLookVisualUrl && (
                  <img
                    src={quickLookVisualUrl}
                    alt={quickLookItem.title}
                    className="h-full w-full object-contain"
                    loading="lazy"
                    onLoad={() => setQuickLookVisualLoaded(true)}
                    onError={() => {
                      setQuickLookVisualFailed(true);
                      if (quickLookTextPreview?.available) {
                        setQuickLookMode("text");
                      }
                    }}
                  />
                )}

                {quickLookMode === "visual" && quickLookExt === "pdf" && quickLookVisualUrl && (
                  <iframe
                    title={`quicklook-modal-${quickLookItem.path}`}
                    src={quickLookVisualUrl}
                    className="h-full w-full border-0"
                    onLoad={() => setQuickLookVisualLoaded(true)}
                  />
                )}

                {quickLookMode === "visual" && hasVisualPreview && isQuickLookVisualLoading && (
                  <div className="flex h-full items-center justify-center px-6 text-center">
                    <p className="text-xs text-gray-500">Cargando preview visual...</p>
                  </div>
                )}

                {quickLookMode === "visual" && !hasVisualPreview && (
                  <div className="flex h-full items-center justify-center px-6 text-center">
                    <p className="text-xs text-gray-500">Vista previa visual disponible para PDF e imágenes. Para este tipo de archivo, usa Abrir para verlo completo.</p>
                  </div>
                )}

                {quickLookMode === "text" && (
                  <div className="h-full overflow-y-auto p-4">
                    {quickLookTextPreview?.available ? (
                      <>
                        <p className="mb-2 text-[11px] text-gray-500">Fuente: {quickLookTextPreview.source}</p>
                        <pre className="whitespace-pre-wrap text-xs leading-relaxed text-gray-300">{quickLookTextPreview.text}</pre>
                      </>
                    ) : (
                      <p className="text-xs text-gray-500">
                        {quickLookTextPreview?.text ?? "No se pudo cargar vista textual para este archivo."}
                      </p>
                    )}
                  </div>
                )}
              </div>

              <p className="mt-2 text-[11px] text-gray-500">Snippet: {quickLookItem.snippet}</p>
              {quickLookItem.match_reason && (
                <p className="mt-1 text-[11px] text-gray-500">Por qué apareció: {quickLookItem.match_reason}</p>
              )}
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
}

export default App;
