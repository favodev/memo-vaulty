import { useState } from "react";
import { motion } from "framer-motion";

function App() {
  const [query, setQuery] = useState("");

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
      </motion.div>
    </div>
  );
}

export default App;
