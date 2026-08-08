/**
 * Copyright 2026 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Generalized from the RCW Title 11 RAG engine for any sqlite-vector corpus
 * with the same embeddings schema. Query retrieval is lexical (no Ollama
 * embedding model required at query time).
 */

import sqlite3InitModule, { type Database, type Sqlite3Static } from '@sqlite.org/sqlite-wasm';

export interface RagHit {
  id: number;
  label: string;
  content: string;
  score: number;
  sectionCite?: string;
  sectionTitle?: string;
  chapterTitle?: string;
  sourceUrl?: string;
}

export interface SqliteVectorRagInitOptions {
  /** Absolute URL to the sqlite vector database file. */
  dbUrl: string;
  /** Prompt prepended when building augmented user messages. */
  systemPrompt: string;
  /** Corpus label used in the augmented prompt header (e.g. "RCW Title 11"). */
  corpusLabel?: string;
}

interface IndexedChunk {
  id: number;
  label: string;
  content: string;
  metadata: Record<string, unknown>;
  /** Pre-tokenized searchable text (label + content). */
  searchText: string;
}

const STOPWORDS = new Set([
  'a',
  'an',
  'and',
  'are',
  'as',
  'at',
  'be',
  'by',
  'for',
  'from',
  'has',
  'he',
  'in',
  'is',
  'it',
  'its',
  'of',
  'on',
  'or',
  'that',
  'the',
  'to',
  'was',
  'were',
  'will',
  'with',
]);

/**
 * Browser-side RAG over a sqlite store (`embeddings` table).
 * Document rows may include precomputed vectors, but retrieval scores queries
 * with lexical overlap so no embedding model is needed at query time.
 */
export class SqliteVectorRagEngine {
  private sqlite3: Sqlite3Static | null = null;
  private db: Database | null = null;
  private chunks: IndexedChunk[] = [];
  private _systemPrompt = '';
  private corpusLabel = 'corpus';
  private ready = false;

  get isReady(): boolean {
    return this.ready;
  }

  get chunkCount(): number {
    return this.chunks.length;
  }

  get systemPrompt(): string {
    return this._systemPrompt;
  }

  async init(options: SqliteVectorRagInitOptions): Promise<void> {
    this.dispose();
    this._systemPrompt = options.systemPrompt;
    this.corpusLabel = options.corpusLabel ?? 'corpus';

    this.sqlite3 = await sqlite3InitModule();

    const response = await fetch(options.dbUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch RAG database (${response.status}): ${options.dbUrl}`);
    }
    const bytes = new Uint8Array(await response.arrayBuffer());
    this.db = openDatabaseFromBytes(this.sqlite3, bytes);
    this.chunks = loadChunks(this.db);
    this.ready = true;
  }

  async retrieve(query: string, topK = 5): Promise<RagHit[]> {
    if (!this.ready || this.chunks.length === 0) {
      throw new Error('RAG engine is not initialized');
    }

    const queryTerms = tokenize(query);
    if (queryTerms.length === 0) {
      return this.chunks.slice(0, Math.max(1, topK)).map((chunk) => toHit(chunk, 0));
    }

    const scored = this.chunks.map((chunk) => ({
      chunk,
      score: lexicalScore(queryTerms, chunk),
    }));

    scored.sort((a, b) => b.score - a.score);
    const hits = scored.filter((s) => s.score > 0).slice(0, Math.max(1, topK));
    if (hits.length === 0) {
      // No lexical overlap — return top chunks so the model still gets context.
      return scored.slice(0, Math.max(1, topK)).map(({ chunk, score }) => toHit(chunk, score));
    }
    return hits.map(({ chunk, score }) => toHit(chunk, score));
  }

  buildAugmentedUserMessage(question: string, hits: RagHit[]): string {
    const excerpts = hits
      .map((hit, i) => {
        const header = [
          `[${i + 1}] ${hit.label}`,
          hit.sectionTitle ? `— ${hit.sectionTitle}` : '',
          `(score ${hit.score.toFixed(3)})`,
        ]
          .filter(Boolean)
          .join(' ');
        return `${header}\n${hit.content.trim()}`;
      })
      .join('\n\n---\n\n');

    return `${this._systemPrompt}

Retrieved ${this.corpusLabel} excerpts:
${excerpts}

User question:
${question}`;
  }

  dispose(): void {
    try {
      this.db?.close();
    } catch {
      // ignore
    }
    this.db = null;
    this.sqlite3 = null;
    this.chunks = [];
    this.ready = false;
  }
}

function toHit(chunk: IndexedChunk, score: number): RagHit {
  return {
    id: chunk.id,
    label: chunk.label,
    content: chunk.content,
    score,
    sectionCite: typeof chunk.metadata.sectionCite === 'string' ? chunk.metadata.sectionCite : undefined,
    sectionTitle: typeof chunk.metadata.sectionTitle === 'string' ? chunk.metadata.sectionTitle : undefined,
    chapterTitle: typeof chunk.metadata.chapterTitle === 'string' ? chunk.metadata.chapterTitle : undefined,
    sourceUrl:
      typeof chunk.metadata.sectionSourceUrl === 'string' ? chunk.metadata.sectionSourceUrl : undefined,
  };
}

/** Tokenize into lowercase alphanumerics, keeping dotted statute refs (e.g. 11.62). */
function tokenize(text: string): string[] {
  const matches = text.toLowerCase().match(/[a-z0-9]+(?:\.[a-z0-9]+)*/g) ?? [];
  return matches.filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

/**
 * Simple lexical score: sum of query-term hits, with label matches weighted higher.
 * Normalized by query length so scores stay roughly in 0–1+.
 */
function lexicalScore(queryTerms: string[], chunk: IndexedChunk): number {
  if (queryTerms.length === 0) return 0;
  const label = chunk.label.toLowerCase();
  const body = chunk.searchText;
  let score = 0;
  for (const term of queryTerms) {
    const inLabel = label.includes(term);
    const inBody = body.includes(term);
    if (inLabel) score += 2;
    else if (inBody) score += 1;
  }
  return score / queryTerms.length;
}

function openDatabaseFromBytes(sqlite3: Sqlite3Static, bytes: Uint8Array): Database {
  const db = new sqlite3.oo1.DB(':memory:');
  if (db.pointer === undefined) {
    db.close();
    throw new Error('Failed to open in-memory SQLite database');
  }
  const ptr = sqlite3.wasm.allocFromTypedArray(bytes);
  const capi = sqlite3.capi as typeof sqlite3.capi & {
    SQLITE_DESERIALIZE_FREEONCLOSE: number;
    SQLITE_DESERIALIZE_RESIZEABLE: number;
  };
  const flags = capi.SQLITE_DESERIALIZE_FREEONCLOSE | capi.SQLITE_DESERIALIZE_RESIZEABLE;
  const rc = sqlite3.capi.sqlite3_deserialize(
    db.pointer,
    'main',
    ptr,
    bytes.byteLength,
    bytes.byteLength,
    flags
  );
  if (rc !== sqlite3.capi.SQLITE_OK) {
    db.close();
    throw new Error(`sqlite3_deserialize failed with code ${rc}`);
  }
  return db;
}

function loadChunks(db: Database): IndexedChunk[] {
  const chunks: IndexedChunk[] = [];
  db.exec({
    sql: 'SELECT id, label, content, metadata FROM embeddings',
    rowMode: 'object',
    callback: (row: Record<string, unknown>) => {
      const label = String(row.label ?? '');
      const content = String(row.content ?? '');
      if (!content.trim() && !label.trim()) return;

      let metadata: Record<string, unknown> = {};
      if (typeof row.metadata === 'string' && row.metadata) {
        try {
          metadata = JSON.parse(row.metadata) as Record<string, unknown>;
        } catch {
          metadata = {};
        }
      }
      chunks.push({
        id: Number(row.id),
        label,
        content,
        metadata,
        searchText: `${label}\n${content}`.toLowerCase(),
      });
    },
  });
  return chunks;
}
