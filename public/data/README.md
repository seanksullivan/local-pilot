# RAG databases

SQLite stores used by Local Pilot Probate Chat.

## Shipped corpus

```text
public/data/rcw-title-11-web.db
```

| Property | Value |
| --- | --- |
| Corpus | Washington RCW Title 11 (Probate and Trust Law) |
| Table | `embeddings` (`id`, `label`, `content`, `metadata`, `embedding`) |
| Catalog id | `rcw-title-11` |
| Query retrieval | Lexical overlap on `label` / `content` (no embedding model at query time) |

The `embedding` column may still be present from how the DB was built; Local Pilot does not call Ollama embeddings when retrieving.

## Runtime requirements

1. Keep `.db` files in this folder (served as `/data/...`).
2. Run **Ollama** with a chat model for generation (`ollama serve`). No `nomic-embed-text` pull is required for RAG queries.

## Adding another source

1. Place a single-file SQLite DB here (journal mode `DELETE`, not open WAL).
2. Register it in `src/lib/rag-sources.ts` with the same table schema.
3. Restart the app — the Retrieval source selector will list it.

If you rebuild the DB, run `PRAGMA wal_checkpoint(TRUNCATE); PRAGMA journal_mode=DELETE;` before serving.
