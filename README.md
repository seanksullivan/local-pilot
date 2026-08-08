# Local Pilot

Focused **Probate Chat** app with optional browser-side RAG over one or more sqlite-vector databases.

## Prerequisites

1. **Node.js** with **npm** or **pnpm**
2. **Ollama** running locally (`ollama serve` on port `11434`)
3. At least one chat model, e.g.:

```bash
ollama pull gemma4:e4b
```

RAG retrieval is **lexical** (keyword overlap on chunk text). No separate embedding model is required at query time.

## Setup

```bash
npm install
npm run dev
```

Or with pnpm: `pnpm install` / `pnpm dev`.

Open http://localhost:5173 — Vite proxies `/ollama` → `http://127.0.0.1:11434`.

## Features

- Streaming chat via Ollama `/api/chat`
- Model picker (embedding models filtered out)
- Generation controls: max tokens, temperature, top-K
- Optional RAG over sqlite DBs loaded in the browser (`@sqlite.org/sqlite-wasm`)
- Lexical retrieval against `label` / `content` (no `nomic-embed-text` for queries)
- **RAG source selector** — switch among registered corpora (ships with RCW Title 11)

## Adding a RAG database

1. Place a single-file SQLite DB in [`public/data/`](public/data/) with table:

   `embeddings (id, label, content, metadata, embedding)`

   (`embedding` may be present from index builds; query-time RAG does not use it.)

2. Register it in [`src/lib/rag-sources.ts`](src/lib/rag-sources.ts).

3. Restart the app and select the source under **Retrieval**.

See [`public/data/README.md`](public/data/README.md) for schema and journal-mode notes.

## Scripts

| Script | npm | pnpm |
| --- | --- | --- |
| Dev server | `npm run dev` | `pnpm dev` |
| Typecheck + build | `npm run build` | `pnpm build` |
| Preview build | `npm run preview` | `pnpm preview` |
