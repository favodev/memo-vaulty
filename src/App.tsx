import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open as openDialog, save as saveDialog } from "@tauri-apps/plugin-dialog";
import { revealItemInDir } from "@tauri-apps/plugin-opener";

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
  chunk_db_synced: boolean;
  pdf_fallback_used: number;
};

type IndexProgressEvent = {
  phase: string;
  message: string;
  scanned_files: number;
  indexed_files: number;
  done: boolean;
};

type FileWatcherStatus = {
  running: boolean;
  roots: string[];
  pending_events: boolean;
  debounce_ms: number;
  last_event_at: string | null;
  last_event_kind: string | null;
  pending_event_count: number;
  total_event_count: number;
  last_batch_event_count: number;
  last_batch_reason: string | null;
  last_reindex_at: string | null;
  last_error: string | null;
};

type FileTextPreview = {
  available: boolean;
  source: string;
  text: string;
};

function fromCsv(value: string): string[] {
  return value
    .split(",")
    .map((v) => v.trim())
    .filter((v) => v.length > 0);
}

function unixToDate(value: string | null): string {
  if (!value) return "—";
  const numeric = Number.parseInt(value, 10);
  if (!Number.isFinite(numeric) || numeric <= 0) return "—";
  return new Date(numeric * 1000).toLocaleString();
}

function getPathTail(path: string): string {
  const normalized = path.replace(/\\/g, "/");
  const parts = normalized.split("/").filter(Boolean);
  return parts[parts.length - 1] ?? path;
}

function App() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(-1);

  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [searchRoots, setSearchRoots] = useState<string[]>([]);
  const [excludedFolders, setExcludedFolders] = useState("node_modules, .git, target, AppData");
  const [maxFileSizeMb, setMaxFileSizeMb] = useState("128");

  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [indexDiagnostics, setIndexDiagnostics] = useState<IndexDiagnostics | null>(null);
  const [indexProgress, setIndexProgress] = useState<IndexProgressEvent | null>(null);
  const [isIndexing, setIsIndexing] = useState(false);

  const [watcherStatus, setWatcherStatus] = useState<FileWatcherStatus | null>(null);
  const [isWatcherLoading, setIsWatcherLoading] = useState(false);
  const [watcherDebounceMs, setWatcherDebounceMs] = useState("1200");

  const [quickLookPath, setQuickLookPath] = useState<string | null>(null);
  const [quickLookPreview, setQuickLookPreview] = useState<FileTextPreview | null>(null);
  const [isQuickLookOpen, setIsQuickLookOpen] = useState(false);
  const [isQuickLookLoading, setIsQuickLookLoading] = useState(false);
  const [isLiveSearchEnabled] = useState(true);

  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const searchRequestIdRef = useRef(0);

  const selectedItem = useMemo(() => {
    if (selectedIndex < 0 || selectedIndex >= results.length) return null;
    return results[selectedIndex] ?? null;
  }, [results, selectedIndex]);

  const refreshStatus = async () => {
    try {
      const [status, diagnostics, watcher] = await Promise.all([
        invoke<IndexStatus>("get_index_status"),
        invoke<IndexDiagnostics>("get_index_diagnostics"),
        invoke<FileWatcherStatus>("get_file_watcher_status"),
      ]);
      setIndexStatus(status);
      setIndexDiagnostics(diagnostics);
      setWatcherStatus(watcher);
      if (watcher.debounce_ms > 0) {
        setWatcherDebounceMs(String(watcher.debounce_ms));
      }
      if (status.roots.length > 0 && searchRoots.length === 0) {
        setSearchRoots(status.roots);
      }
    } catch {
      setIndexStatus(null);
      setIndexDiagnostics(null);
      setWatcherStatus(null);
    }
  };

  const executeSearch = async (rawQuery: string) => {
    const clean = rawQuery.trim();
    if (!clean) {
      setResults([]);
      setSelectedIndex(-1);
      setHasSearched(false);
      setErrorMessage(null);
      return;
    }

    const requestId = ++searchRequestIdRef.current;
    setIsLoading(true);
    setHasSearched(true);
    setErrorMessage(null);

    const excludedFolderRules = fromCsv(excludedFolders);
    const parsedMaxSize = Number.parseInt(maxFileSizeMb, 10);
    const maxSizeValue = Number.isFinite(parsedMaxSize) && parsedMaxSize > 0 ? parsedMaxSize : 128;

    try {
      let response: SearchResultItem[] = [];
      try {
        response = await invoke<SearchResultItem[]>("semantic_search", {
          query: clean,
          limit: 35,
          roots: searchRoots,
          excludedExtensions: [],
          excludedFolders: excludedFolderRules,
          maxFileSizeMb: maxSizeValue,
        });
      } catch {
        response = await invoke<SearchResultItem[]>("search_stub", {
          query: clean,
          roots: searchRoots,
          excludedExtensions: [],
          excludedFolders: excludedFolderRules,
          maxFileSizeMb: maxSizeValue,
        });
      }

      if (requestId === searchRequestIdRef.current) {
        setResults(response);
        setSelectedIndex(response.length > 0 ? 0 : -1);
        if (response.length > 0) {
          setQuickLookPath(response[0].path);
        }
      }
    } catch {
      if (requestId === searchRequestIdRef.current) {
        setResults([]);
        setSelectedIndex(-1);
        setErrorMessage("No se pudo ejecutar la búsqueda local.");
      }
    } finally {
      if (requestId === searchRequestIdRef.current) {
        setIsLoading(false);
      }
    }
  };

  const runSearch = async () => {
    await executeSearch(query);
  };

  const startIndexing = async () => {
    setIsIndexing(true);
    setErrorMessage(null);
    try {
      const excludedFolderRules = fromCsv(excludedFolders);
      const parsedMaxSize = Number.parseInt(maxFileSizeMb, 10);
      const maxSizeValue = Number.isFinite(parsedMaxSize) && parsedMaxSize > 0 ? parsedMaxSize : 128;

      const status = await invoke<IndexStatus>("start_indexing", {
        roots: searchRoots,
        excludedExtensions: [],
        excludedFolders: excludedFolderRules,
        maxFileSizeMb: maxSizeValue,
      });
      setIndexStatus(status);
      await refreshStatus();
    } catch {
      setErrorMessage("No se pudo iniciar la indexación.");
    } finally {
      setIsIndexing(false);
    }
  };

  const cancelIndexing = async () => {
    try {
      await invoke("cancel_indexing");
    } catch {
      setErrorMessage("No se pudo cancelar la indexación.");
    }
  };

  const handleIndexPrimaryAction = async () => {
    if (isIndexing) {
      await cancelIndexing();
      return;
    }
    await startIndexing();
  };

  const toggleWatcher = async () => {
    setIsWatcherLoading(true);
    try {
      const excludedFolderRules = fromCsv(excludedFolders);
      const parsedMaxSize = Number.parseInt(maxFileSizeMb, 10);
      const maxSizeValue = Number.isFinite(parsedMaxSize) && parsedMaxSize > 0 ? parsedMaxSize : 128;
      const debounceValue = Number.parseInt(watcherDebounceMs, 10);
      if (watcherStatus?.running) {
        const status = await invoke<FileWatcherStatus>("stop_file_watcher");
        setWatcherStatus(status);
      } else {
        const status = await invoke<FileWatcherStatus>("start_file_watcher", {
          roots: searchRoots,
          excludedExtensions: [],
          excludedFolders: excludedFolderRules,
          maxFileSizeMb: maxSizeValue,
          debounceMs: Number.isFinite(debounceValue) ? debounceValue : 1200,
        });
        setWatcherStatus(status);
      }
    } catch {
      setErrorMessage("No se pudo actualizar el watcher.");
    } finally {
      setIsWatcherLoading(false);
    }
  };

  const pickFolders = async () => {
    const selected = await openDialog({
      directory: true,
      multiple: true,
      title: "Seleccionar carpetas para indexar",
    });

    if (!selected) return;

    const selectedPaths = Array.isArray(selected) ? selected : [selected];
    const unique = Array.from(new Set([...searchRoots, ...selectedPaths]));
    setSearchRoots(unique);
  };

  const removeRoot = (root: string) => {
    setSearchRoots((prev) => prev.filter((item) => item !== root));
  };

  const removeRootPersisted = async (root: string) => {
    try {
      const status = await invoke<IndexStatus>("forget_index_root", { root, reindex: true });
      setIndexStatus(status);
      setSearchRoots(status.roots);
      await refreshStatus();
    } catch {
      setErrorMessage("No se pudo olvidar la carpeta en el índice.");
    }
  };

  const exportConfig = async () => {
    const target = await saveDialog({
      title: "Exportar configuración",
      defaultPath: "memovault-config.json",
      filters: [{ name: "JSON", extensions: ["json"] }],
    });
    if (!target) return;

    try {
      await invoke("export_app_config_to_file", { path: target, includeSecrets: false });
    } catch {
      setErrorMessage("No se pudo exportar la configuración.");
    }
  };

  const importConfig = async () => {
    const source = await openDialog({
      title: "Importar configuración",
      multiple: false,
      filters: [{ name: "JSON", extensions: ["json"] }],
    });
    if (!source || Array.isArray(source)) return;

    try {
      const status = await invoke<IndexStatus>("import_app_config_from_file", { path: source, reindex: false });
      setIndexStatus(status);
      setSearchRoots(status.roots);
      await refreshStatus();
    } catch {
      setErrorMessage("No se pudo importar la configuración.");
    }
  };

  const resetIndexData = async () => {
    try {
      const status = await invoke<IndexStatus>("clear_index_data");
      setIndexStatus(status);
      setSearchRoots([]);
      setResults([]);
      setSelectedIndex(-1);
      setQuickLookPath(null);
      await refreshStatus();
    } catch {
      setErrorMessage("No se pudo resetear el índice.");
    }
  };

  const loadQuickLook = async (path: string) => {
    setIsQuickLookOpen(true);
    setQuickLookPath(path);
    setQuickLookPreview(null);
    setIsQuickLookLoading(true);
    try {
      const preview = await invoke<FileTextPreview>("get_file_text_preview", { path, maxChars: 18000 });
      setQuickLookPreview(preview);
    } catch {
      setQuickLookPreview({ available: false, source: "error", text: "No se pudo cargar vista previa." });
    } finally {
      setIsQuickLookLoading(false);
    }
  };

  useEffect(() => {
    void refreshStatus();
  }, []);

  useEffect(() => {
    if (!isLiveSearchEnabled) return;

    const clean = query.trim();
    if (!clean) {
      setResults([]);
      setSelectedIndex(-1);
      setHasSearched(false);
      setErrorMessage(null);
      return;
    }

    if (clean.length < 2) {
      setResults([]);
      setSelectedIndex(-1);
      setHasSearched(false);
      return;
    }

    const debounceMs = isIndexing ? 420 : 280;
    const timer = window.setTimeout(() => {
      void executeSearch(clean);
    }, debounceMs);

    return () => window.clearTimeout(timer);
  }, [query, isIndexing, isLiveSearchEnabled]);

  useEffect(() => {
    if (!watcherStatus?.running) return;
    const timer = setInterval(() => {
      void invoke<FileWatcherStatus>("get_file_watcher_status")
        .then((status) => setWatcherStatus(status))
        .catch(() => undefined);
    }, 2200);
    return () => clearInterval(timer);
  }, [watcherStatus?.running]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing =
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.isContentEditable;

      if (!typing && event.key === "/") {
        event.preventDefault();
        searchInputRef.current?.focus();
      }

      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        searchInputRef.current?.focus();
      }

      if (event.key === "Enter" && !typing) {
        event.preventDefault();
        void runSearch();
      }

      if (event.key === "ArrowDown" && !typing && results.length > 0) {
        event.preventDefault();
        setSelectedIndex((current) => Math.min(current + 1, results.length - 1));
      }

      if (event.key === "ArrowUp" && !typing && results.length > 0) {
        event.preventDefault();
        setSelectedIndex((current) => Math.max(current - 1, 0));
      }

      if (event.key === "Escape") {
        setIsQuickLookOpen(false);
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [results.length]);

  useEffect(() => {
    let mounted = true;
    let cleanup: (() => void) | undefined;

    const bind = async () => {
      const unlisten = await listen<IndexProgressEvent>("index-progress", (event) => {
        if (!mounted) return;
        setIndexProgress(event.payload);
        if (event.payload.done) {
          void refreshStatus();
        }
      });
      cleanup = unlisten;
    };

    void bind();
    return () => {
      mounted = false;
      cleanup?.();
    };
  }, []);

  useEffect(() => {
    if (!selectedItem) return;
    setQuickLookPath(selectedItem.path);
  }, [selectedItem]);

  return (
    <div className="relative min-h-screen overflow-hidden bg-black text-zinc-100">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -left-28 top-8 h-72 w-72 rounded-full bg-emerald-500/8 blur-3xl" />
        <div className="absolute right-0 top-0 h-80 w-80 rounded-full bg-cyan-500/6 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-64 w-64 rounded-full bg-violet-500/6 blur-3xl" />
      </div>
      <div className="mx-auto grid min-h-screen w-full max-w-400 grid-cols-1 gap-4 p-4 lg:grid-cols-[340px_1fr]">
        <motion.aside
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.25, ease: "easeOut" }}
          className="rounded-3xl border border-zinc-800/80 bg-linear-to-b from-black via-zinc-950/80 to-black p-4 shadow-2xl shadow-black/40 backdrop-blur"
        >
          <div className="mb-4 flex items-center justify-between">
            <h1 className="text-lg font-semibold tracking-tight">MemoVault</h1>
            <span className="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-2 py-1 text-xs text-emerald-300">
              Texto
            </span>
          </div>

          <div className="space-y-3">
            <motion.button
              type="button"
              onClick={pickFolders}
              whileHover={{ y: -1 }}
              whileTap={{ scale: 0.99 }}
              className="w-full rounded-2xl border border-zinc-700/80 bg-zinc-900/80 px-3 py-2.5 text-sm font-medium transition hover:border-zinc-500 hover:bg-zinc-800"
            >
              + Añadir carpetas
            </motion.button>

            <motion.button
              type="button"
              onClick={() => void handleIndexPrimaryAction()}
              whileHover={{ y: -1 }}
              whileTap={{ scale: 0.99 }}
              className={`w-full rounded-xl px-3 py-2.5 text-sm font-semibold transition ${
                isIndexing
                  ? "border border-amber-400/30 bg-amber-500/20 text-amber-200 hover:bg-amber-500/30"
                  : "bg-emerald-500 text-zinc-950 shadow-lg shadow-emerald-500/20 hover:bg-emerald-400"
              }`}
            >
              {isIndexing ? "Pausar / cancelar indexación" : "Indexar ahora"}
            </motion.button>

            <motion.button
              type="button"
              onClick={() => setIsConfigOpen(true)}
              whileHover={{ y: -1 }}
              whileTap={{ scale: 0.99 }}
              className="w-full rounded-2xl border border-zinc-700/80 px-3 py-2.5 text-sm transition hover:border-zinc-500"
            >
              Configuración
            </motion.button>
          </div>

          <div className="mt-5 rounded-2xl border border-zinc-800/80 bg-zinc-900/40 p-3">
            <div className="mb-2 flex items-center justify-between">
              <p className="text-xs font-medium uppercase tracking-wide text-zinc-400">Carpetas activas</p>
              <span className="rounded-md border border-zinc-700 px-2 py-0.5 text-[10px] text-zinc-300">
                {searchRoots.length}
              </span>
            </div>

            <div className="grid grid-cols-1 gap-2">
              {searchRoots.length === 0 ? (
                <p className="text-xs text-zinc-500">Aún no agregas carpetas para indexar.</p>
              ) : (
                searchRoots.map((root) => (
                  <motion.div
                    key={root}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    whileHover={{ y: -1 }}
                    className="rounded-xl border border-zinc-800/90 bg-zinc-900/80 px-2.5 py-2 text-xs"
                  >
                    <div className="flex items-start gap-2">
                      <div className="mt-0.5 grid h-7 w-7 place-items-center rounded-lg border border-zinc-700 bg-zinc-950 text-zinc-300">
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 7.5A2.5 2.5 0 0 1 5.5 5h4.3a2 2 0 0 1 1.4.58l1.22 1.22a2 2 0 0 0 1.41.58H18.5A2.5 2.5 0 0 1 21 10v8.5a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 18.5Z"/></svg>
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-[11px] font-medium text-zinc-300" title={getPathTail(root)}>{getPathTail(root)}</p>
                        <p className="truncate text-[10px] text-zinc-500" title={root}>{root}</p>
                      </div>
                    </div>
                    <div className="mt-2 flex gap-2">
                      <button
                        type="button"
                        onClick={() => removeRoot(root)}
                        className="rounded-md border border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-300 hover:border-zinc-500"
                      >
                        Quitar
                      </button>
                      <button
                        type="button"
                        onClick={() => void removeRootPersisted(root)}
                        className="rounded-md border border-zinc-700 px-2 py-0.5 text-[11px] text-zinc-300 hover:border-zinc-500"
                      >
                        Olvidar
                      </button>
                    </div>
                  </motion.div>
                ))
              )}
            </div>
          </div>

          <div className="mt-4 space-y-2 rounded-xl border border-zinc-800/80 bg-zinc-900/40 p-3 text-xs text-zinc-300">
            <div className="flex items-center justify-between">
              <span>Watcher</span>
              <button
                type="button"
                onClick={toggleWatcher}
                disabled={isWatcherLoading}
                className={`relative h-6 w-11 rounded-full border transition ${
                  watcherStatus?.running
                    ? "border-emerald-400/60 bg-emerald-500/60"
                    : "border-zinc-600 bg-zinc-800"
                }`}
                aria-label="Alternar watcher"
              >
                <span
                  className={`absolute top-0.5 h-4.5 w-4.5 rounded-full bg-white shadow transition ${
                    watcherStatus?.running ? "left-5.5" : "left-0.5"
                  }`}
                />
              </button>
            </div>
            <div>Debounce: {watcherStatus?.debounce_ms ?? 1200} ms</div>
            <div>Eventos: {watcherStatus?.total_event_count ?? 0}</div>
            {watcherStatus?.last_error ? (
              <div className="text-rose-300">Error: {watcherStatus.last_error}</div>
            ) : null}
          </div>

          <div className="mt-4 space-y-2 text-xs text-zinc-400">
            <div>Archivos indexados: {indexStatus?.indexed_files ?? 0}</div>
            <div>Última indexación: {unixToDate(indexStatus?.indexed_at ?? null)}</div>
            {indexProgress ? (
              <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-2 text-zinc-300">
                {indexProgress.message} · {indexProgress.indexed_files} archivos
              </div>
            ) : null}
          </div>
        </motion.aside>

        <motion.main
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.25, ease: "easeOut", delay: 0.06 }}
          className="rounded-3xl border border-zinc-800/80 bg-linear-to-b from-zinc-950/80 via-black to-black p-4 shadow-2xl shadow-black/40 backdrop-blur"
        >
          <div className="mb-1 flex flex-col gap-3 md:flex-row">
            <input
              ref={searchInputRef}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  void runSearch();
                }
              }}
              placeholder="Buscar en texto, markdown, código, PDFs, DOCX, ODT, RTF..."
              className="h-12 w-full rounded-2xl border border-zinc-800 bg-black/80 px-4 text-sm outline-none transition focus:border-emerald-400"
            />
            <motion.button
              type="button"
              onClick={runSearch}
              disabled={isLoading}
              whileTap={{ scale: 0.98 }}
              className="h-12 rounded-2xl bg-zinc-100 px-5 text-sm font-semibold text-zinc-900 transition hover:bg-white disabled:opacity-60"
            >
              {isLoading ? "Buscando..." : "Buscar"}
            </motion.button>
          </div>

          <div className="mb-4 flex items-center gap-2 px-1 text-[11px] text-zinc-500">
            <span className={`h-1.5 w-1.5 rounded-full ${isLoading ? "bg-emerald-400 animate-pulse" : "bg-zinc-600"}`} />
            <span>
              {isLiveSearchEnabled
                ? "Búsqueda en tiempo real activa (debounce inteligente)"
                : "Búsqueda manual"}
            </span>
          </div>

          {errorMessage ? (
            <div className="mb-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-200">
              {errorMessage}
            </div>
          ) : null}

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1fr_380px]">
            <section className="min-h-130 rounded-2xl border border-zinc-800/90 bg-black/60 p-3">
              {hasSearched && results.length === 0 && !isLoading ? (
                <div className="mt-12 text-center text-sm text-zinc-400">No hay resultados con esa consulta.</div>
              ) : null}

              <div className="vault-scroll max-h-[70vh] space-y-2 overflow-auto pr-1">
                {results.map((item, index) => {
                  const active = index === selectedIndex;
                  return (
                    <motion.article
                      key={`${item.path}-${index}`}
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.16, delay: index * 0.01 }}
                      whileHover={{ y: -2, scale: 1.002 }}
                      onClick={() => setSelectedIndex(index)}
                      className={`cursor-pointer rounded-lg border p-3 transition ${
                        active
                          ? "border-emerald-400/60 bg-emerald-500/10 shadow-lg shadow-emerald-500/10"
                          : "border-zinc-800 bg-zinc-900/40 hover:border-zinc-700"
                      }`}
                    >
                      <div className="mb-1 flex items-center justify-between gap-3">
                        <h3 className="truncate text-sm font-medium text-zinc-100">{item.title}</h3>
                        <span className="rounded border border-zinc-700 px-2 py-0.5 text-[10px] uppercase tracking-wide text-zinc-400">
                          {item.origin ?? "local"}
                        </span>
                      </div>
                      <p className="mb-2 line-clamp-2 text-xs text-zinc-400">{item.path}</p>
                      <p className="line-clamp-3 text-xs text-zinc-300">{item.snippet}</p>
                      {item.match_reason ? (
                        <p className="mt-2 line-clamp-2 text-[11px] text-zinc-500">{item.match_reason}</p>
                      ) : null}
                      <div className="mt-3 flex gap-2">
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            void invoke("open_file", { path: item.path });
                          }}
                          className="rounded-md border border-zinc-700 px-2 py-1 text-[11px] hover:border-zinc-500"
                        >
                          Abrir
                        </button>
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            void revealItemInDir(item.path);
                          }}
                          className="rounded-md border border-zinc-700 px-2 py-1 text-[11px] hover:border-zinc-500"
                        >
                          Carpeta
                        </button>
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            void loadQuickLook(item.path);
                          }}
                          className="rounded-md border border-zinc-700 px-2 py-1 text-[11px] hover:border-zinc-500"
                        >
                          Vista previa
                        </button>
                      </div>
                    </motion.article>
                  );
                })}
              </div>
            </section>

            <section className="rounded-2xl border border-zinc-800/90 bg-black/60 p-3">
              <h2 className="mb-2 text-sm font-medium">Diagnóstico</h2>
              <div className="space-y-2 text-xs text-zinc-300">
                <div>Escaneados: {indexDiagnostics?.scanned_files ?? 0}</div>
                <div>Indexados: {indexDiagnostics?.indexed_files ?? 0}</div>
                <div>PDF extraídos: {indexDiagnostics?.pdf_indexed ?? 0}</div>
                <div>PDF fallidos: {indexDiagnostics?.pdf_failed ?? 0}</div>
                <div>Fallback PDF: {indexDiagnostics?.pdf_fallback_used ?? 0}</div>
                {indexDiagnostics?.last_error ? (
                  <div className="rounded-md border border-rose-500/30 bg-rose-500/10 p-2 text-rose-200">
                    {indexDiagnostics.last_error}
                  </div>
                ) : null}
              </div>
            </section>
          </div>
        </motion.main>
      </div>

      {isConfigOpen ? (
        <div className="fixed inset-0 z-40 grid place-items-center bg-black/60 p-4">
          <div className="w-full max-w-xl rounded-2xl border border-zinc-700 bg-zinc-900 p-4">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-base font-semibold">Configuración</h3>
              <button
                type="button"
                onClick={() => setIsConfigOpen(false)}
                className="rounded-md border border-zinc-700 px-2 py-1 text-xs hover:border-zinc-500"
              >
                Cerrar
              </button>
            </div>

            <div className="space-y-3">
              <div>
                <label className="mb-1 block text-xs text-zinc-400">Carpetas excluidas</label>
                <input
                  value={excludedFolders}
                  onChange={(event) => setExcludedFolders(event.target.value)}
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm outline-none focus:border-emerald-400"
                />
              </div>

              <div>
                <label className="mb-1 block text-xs text-zinc-400">Tamaño máximo por archivo (MB)</label>
                <input
                  value={maxFileSizeMb}
                  onChange={(event) => setMaxFileSizeMb(event.target.value)}
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm outline-none focus:border-emerald-400"
                />
              </div>

              <div>
                <label className="mb-1 block text-xs text-zinc-400">Watcher debounce (ms)</label>
                <input
                  value={watcherDebounceMs}
                  onChange={(event) => setWatcherDebounceMs(event.target.value)}
                  className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm outline-none focus:border-emerald-400"
                />
              </div>

              <div className="flex flex-wrap gap-2 pt-1">
                <button
                  type="button"
                  onClick={exportConfig}
                  className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs hover:border-zinc-500"
                >
                  Exportar config
                </button>
                <button
                  type="button"
                  onClick={importConfig}
                  className="rounded-md border border-zinc-700 px-3 py-1.5 text-xs hover:border-zinc-500"
                >
                  Importar config
                </button>
                <button
                  type="button"
                  onClick={() => void resetIndexData()}
                  className="rounded-md border border-rose-500/40 px-3 py-1.5 text-xs text-rose-200 hover:border-rose-400"
                >
                  Reset índice
                </button>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {isQuickLookOpen ? (
        <div className="fixed inset-0 z-50 grid place-items-center bg-black/70 p-4" onClick={() => setIsQuickLookOpen(false)}>
          <div
            className="w-full max-w-4xl rounded-2xl border border-zinc-700 bg-zinc-900 p-4"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="mb-3 flex items-center justify-between gap-3">
              <h3 className="truncate text-sm font-medium text-zinc-100">{quickLookPath}</h3>
              <button
                type="button"
                onClick={() => setIsQuickLookOpen(false)}
                className="rounded-md border border-zinc-700 px-2 py-1 text-xs hover:border-zinc-500"
              >
                Cerrar
              </button>
            </div>

            <div className="vault-scroll max-h-[68vh] overflow-auto rounded-lg border border-zinc-800 bg-zinc-950/80 p-3">
              {isQuickLookLoading ? (
                <p className="text-sm text-zinc-400">Cargando vista previa...</p>
              ) : null}

              {!isQuickLookLoading && quickLookPreview?.available ? (
                <pre className="whitespace-pre-wrap wrap-break-word text-xs leading-6 text-zinc-200">{quickLookPreview.text}</pre>
              ) : null}

              {!isQuickLookLoading && !quickLookPreview?.available ? (
                <p className="text-sm text-zinc-400">{quickLookPreview?.text ?? "Vista previa no disponible."}</p>
              ) : null}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default App;
