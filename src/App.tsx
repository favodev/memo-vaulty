import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { invoke } from "@tauri-apps/api/core";
import { openPath, revealItemInDir } from "@tauri-apps/plugin-opener";
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

function App() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const [searchRoots, setSearchRoots] = useState<string[]>([]);
  const [excludedExtensions, setExcludedExtensions] = useState("mkv, mp4, zip");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [indexMessage, setIndexMessage] = useState<string | null>(null);

  useEffect(() => {
    const loadStatus = async () => {
      try {
        const status = await invoke<IndexStatus>("get_index_status");
        setIndexStatus(status);
      } catch {
        setIndexStatus(null);
      }
    };

    void loadStatus();
  }, []);

  const openFile = async (filePath: string) => {
    await openPath(filePath);
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

      const response = await invoke<SearchResultItem[]>("search_stub", {
        query: cleanQuery,
        roots: searchRoots,
        excludedExtensions: exclusions,
      });
      setResults(response);
      setSelectedIndex(response.length > 0 ? 0 : -1);
    } catch {
      setResults([]);
      setSelectedIndex(-1);
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
    setIsIndexing(true);
    setIndexMessage("Indexando archivos locales... esto puede tardar varios minutos.");

    try {
      const exclusions = excludedExtensions
        .split(",")
        .map((value) => value.trim())
        .filter((value) => value.length > 0);

      const status = await invoke<IndexStatus>("start_indexing", {
        roots,
        excludedExtensions: exclusions,
      });

      setIndexStatus(status);
      if (status.roots.length > 0) {
        setSearchRoots(status.roots);
      }
      setIndexMessage(`Índice listo: ${status.indexed_files} archivos.`);
    } catch {
      setIndexMessage("Falló la indexación local. Intenta con una carpeta más pequeña primero.");
    } finally {
      setIsIndexing(false);
    }
  };

  const shouldAnchorTop =
    showAdvanced ||
    isLoading ||
    Boolean(errorMessage) ||
    hasSearched ||
    results.length > 0;

  return (
    <div
      className={`min-h-screen w-full bg-[radial-gradient(ellipse_at_top,var(--tw-gradient-stops))] from-gray-900 via-black to-black flex flex-col items-center p-4 ${
        shouldAnchorTop ? "justify-start" : "justify-center"
      }`}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="vault-scroll w-full max-w-lg z-10 max-h-[calc(100vh-2rem)] overflow-y-auto pb-4"
      >
        {!indexStatus?.has_index && (
          <div className="mb-4 rounded-xl bg-white/4 p-4">
            <p className="text-sm font-medium text-white">Primera ejecución: crea tu índice local</p>
            <p className="mt-1 text-xs text-gray-400">
              Indexa una vez y las búsquedas por nombre/contenido (TXT/MD) serán mucho más rápidas.
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
                  void pickFolders();
                }}
              >
                Elegir carpetas
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
        )}

        {indexStatus?.has_index && (
          <div className="mb-4 rounded-xl bg-white/4 p-3">
            <p className="text-xs text-gray-300">
              Índice activo: <span className="text-white">{indexStatus.indexed_files}</span> archivos
            </p>
            <p className="mt-1 text-[11px] text-gray-500">
              Incluye nombre de archivo y contenido para TXT/MD (extracto inicial).
            </p>
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
            </div>
          </div>
        )}

        {indexMessage && (
          <p className="mb-3 text-xs text-center text-gray-400">{indexMessage}</p>
        )}

        <div className="group flex items-center gap-3 rounded-xl bg-white/5 px-4 transition-all duration-300 focus-within:bg-white/10">
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
                setSelectedIndex((prev) => Math.min(prev + 1, results.length - 1));
                return;
              }

              if (e.key === "ArrowUp" && results.length > 0) {
                e.preventDefault();
                setSelectedIndex((prev) => Math.max(prev - 1, 0));
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
              }
            }}
            autoFocus
          />
        </div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="mt-4 text-gray-500 text-sm text-center"
        >
          Presiona <kbd className="bg-white/10 px-1.5 py-0.5 rounded text-xs">Enter</kbd> para buscar.
        </motion.p>

        <div className="mt-4 flex items-center justify-center">
          <button
            type="button"
            className="text-xs text-gray-500 transition-colors hover:text-gray-300"
            onClick={() => setShowAdvanced((prev) => !prev)}
          >
            {showAdvanced ? "Ocultar opciones avanzadas" : "Mostrar opciones avanzadas"}
          </button>
        </div>

        {showAdvanced && (
          <div className="mt-3 space-y-3 rounded-lg bg-white/3 p-3">
            <div className="flex items-center justify-between gap-3">
              <p className="text-xs uppercase tracking-wide text-gray-500">Fuentes de búsqueda</p>
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
              <p className="text-xs text-gray-500">Sin carpetas seleccionadas: se usan Documents/Desktop/Downloads.</p>
            )}

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
          </div>
        )}

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
          <div className="mt-5 max-h-[56vh] space-y-2 overflow-y-auto pr-1 pb-2">
            {results.map((item, index) => (
              <div
                key={item.path}
                className={`rounded-lg px-4 py-3 transition-colors ${
                  selectedIndex === index ? "bg-white/10" : "bg-white/4"
                }`}
                onMouseEnter={() => setSelectedIndex(index)}
                onDoubleClick={() => {
                  void openFile(item.path);
                }}
              >
                <p className="text-sm font-medium text-white">{item.title}</p>
                <p className="mt-1 text-xs text-gray-400">{item.snippet}</p>
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
        )}
      </motion.div>
    </div>
  );
}

export default App;
