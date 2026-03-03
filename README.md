# MEMOVAULT

MemoVault es un indexador y buscador local de texto, enfocado en velocidad, privacidad y simplicidad.

## Alcance actual

- Indexación local de archivos de texto y documentos.
- Búsqueda por contenido y metadatos locales.
- Vista previa de texto dentro de la app.
- Reindexado manual y automático con watcher.
- Exportación e importación de configuración.

## Formatos soportados

- Texto plano y markup: TXT, MD, CSV, LOG, JSON, YAML, TOML, INI.
- Código y configuración: TS/JS, RS, PY, TF, HCL, SQL, entre otros.
- Documentos: PDF, DOCX, ODT, RTF, PPTX, XLSX.

## Arquitectura

- Frontend: React + TypeScript + Tailwind + Framer Motion.
- Backend: Tauri + Rust.
- Índice local persistente para búsqueda rápida.
