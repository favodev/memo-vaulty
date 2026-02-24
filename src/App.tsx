import { Fragment, useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { open as openDialog } from "@tauri-apps/plugin-dialog";

type SearchResultItem = {
  title: string;
  path: string;
  snippet: string;
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
  const [isSavingAi, setIsSavingAi] = useState(false);
  const [watcherStatus, setWatcherStatus] = useState<FileWatcherStatus | null>(null);
  const [isWatcherLoading, setIsWatcherLoading] = useState(false);
  const [quickLookPath, setQuickLookPath] = useState<string | null>(null);
  const [quickLookImageMeta, setQuickLookImageMeta] = useState<ImageMetadata | null>(null);

  useEffect(() => {
    const loadStatus = async () => {
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
      } catch {
        setAiProviderStatus(null);
      }

      try {
        const watcher = await invoke<FileWatcherStatus>("get_file_watcher_status");
        setWatcherStatus(watcher);
      } catch {
        setWatcherStatus(null);
      }
    };

    void loadStatus();
  }, []);

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

      try {
        response = await invoke<SearchResultItem[]>("semantic_search", {
          query: cleanQuery,
          limit: 30,
          roots: searchRoots,
          excludedExtensions: exclusions,
          excludedFolders: excludedFolderRules,
          maxFileSizeMb: maxSizeValue,
        });
      } catch {
        response = await invoke<SearchResultItem[]>("search_stub", {
          query: cleanQuery,
          roots: searchRoots,
          excludedExtensions: exclusions,
          excludedFolders: excludedFolderRules,
          maxFileSizeMb: maxSizeValue,
        });
      }

      setResults(response);
      setSelectedIndex(response.length > 0 ? 0 : -1);
      setQuickLookPath(response.length > 0 ? response[0].path : null);
    } catch {
      setResults([]);
      setSelectedIndex(-1);
      setQuickLookPath(null);
      setErrorMessage("No se pudo ejecutar la búsqueda local.");
    } finally {
      setIsLoading(false);
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

  const quickLookUrl = useMemo(() => {
    if (!quickLookItem?.path) {
      return null;
    }

    try {
      return convertFileSrc(quickLookItem.path);
    } catch {
      return null;
    }
  }, [quickLookItem?.path]);

  const isQuickLookImage = quickLookExt === "png" || quickLookExt === "jpg" || quickLookExt === "jpeg" || quickLookExt === "webp" || quickLookExt === "gif" || quickLookExt === "bmp" || quickLookExt === "tiff";

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
                setResults([]);
                setHasSearched(false);
                setErrorMessage(null);
                setSelectedIndex(-1);
                setQuickLookPath(null);
              }
            }}
            autoFocus
          />
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

        {!isLoading && results.length > 0 && (
          <div className="mt-5 grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(280px,360px)]">
            <div className="max-h-[62vh] space-y-2 overflow-y-auto pr-1 pb-2">
              {results.map((item, index) => (
                <div
                  key={item.path}
                  className={`rounded-lg px-4 py-3 transition-colors ${
                    selectedIndex === index ? "bg-white/10" : "bg-white/4"
                  }`}
                  onMouseEnter={() => {
                    setSelectedIndex(index);
                    setQuickLookPath(item.path);
                  }}
                  onClick={() => {
                    setSelectedIndex(index);
                    setQuickLookPath(item.path);
                  }}
                  onDoubleClick={() => {
                    void openFile(item.path);
                  }}
                >
                  <p className="text-sm font-medium text-white">{renderHighlighted(item.title)}</p>
                  <p className="mt-1 text-xs text-gray-400">{renderHighlighted(item.snippet)}</p>
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
                  </div>
                </div>
              ))}
            </div>

            <div className="max-h-[62vh] overflow-hidden rounded-lg bg-white/4 p-3 ring-1 ring-white/10">
              <p className="text-[11px] uppercase tracking-wide text-gray-500">Quick Look</p>

              {!quickLookItem && (
                <p className="mt-2 text-xs text-gray-500">Selecciona un resultado para previsualizar.</p>
              )}

              {quickLookItem && (
                <>
                  <p className="mt-2 truncate text-xs text-gray-300">{quickLookItem.title}</p>
                  <p className="truncate text-[10px] text-gray-500">{quickLookItem.path}</p>

                  {isQuickLookImage && quickLookImageMeta && (
                    <p className="mt-1 truncate text-[10px] text-gray-400">
                      {quickLookImageMeta.format ?? quickLookExt.toUpperCase()} · {quickLookImageMeta.width ?? "?"}x{quickLookImageMeta.height ?? "?"}
                      {quickLookImageMeta.date_taken ? ` · ${quickLookImageMeta.date_taken}` : ""}
                    </p>
                  )}

                  <div className="mt-3 h-[46vh] overflow-hidden rounded-md bg-black/30 ring-1 ring-white/10">
                    {isQuickLookImage && quickLookUrl && (
                      <img src={quickLookUrl} alt={quickLookItem.title} className="h-full w-full object-contain" loading="lazy" />
                    )}

                    {quickLookExt === "pdf" && quickLookUrl && (
                      <iframe title={`quicklook-${quickLookItem.path}`} src={quickLookUrl} className="h-full w-full border-0" />
                    )}

                    {quickLookExt !== "pdf" && !isQuickLookImage && (
                      <div className="flex h-full items-center justify-center px-3 text-center">
                        <p className="text-xs text-gray-500">Vista previa visual disponible para PDF e imágenes. Para otros tipos, usa el snippet y el botón Abrir.</p>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        </div>

        {isConfigOpen && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4"
            onClick={() => setIsConfigOpen(false)}
          >
            <div
              className="w-full max-w-2xl rounded-xl bg-[#07090d] p-4 shadow-xl ring-1 ring-white/10"
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
              </div>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
}

export default App;
