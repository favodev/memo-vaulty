import { useState } from "react";
import { motion } from "framer-motion";
import { invoke } from "@tauri-apps/api/core";
import { openPath, revealItemInDir } from "@tauri-apps/plugin-opener";

type SearchResultItem = {
  title: string;
  path: string;
  snippet: string;
};

function App() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(-1);

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
      const response = await invoke<SearchResultItem[]>("search_stub", { query: cleanQuery });
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

  return (
    <div className="min-h-screen w-full bg-[radial-gradient(ellipse_at_top,var(--tw-gradient-stops))] from-gray-900 via-black to-black flex flex-col items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="w-full max-w-lg z-10"
      >
        <div className="group flex items-center gap-3 rounded-xl bg-white/5 px-4 transition-all duration-300 focus-within:bg-white/10">
          <div className="text-gray-500 group-focus-within:text-blue-400 transition-colors pointer-events-none shrink-0">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-search"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
          </div>

          <input
            type="text"
            className="w-full appearance-none border-0 bg-transparent py-4 text-lg text-white placeholder:text-gray-400 caret-blue-400 outline-none"
            placeholder="Buscar en tu memoria..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
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

                if (selectedIndex >= 0 && selectedIndex < results.length) {
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
          <div className="vault-scroll mt-5 max-h-[52vh] space-y-2 overflow-y-auto pr-1">
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
