import {
  RAG_SOURCES,
  defaultRagSourceId,
  getRagSource,
  type RagSourceConfig,
} from './rag-sources';
import { SqliteVectorRagEngine, type RagHit } from './sqlite-vector-rag';

export interface RagSourceStatus {
  id: string;
  ready: boolean;
  chunkCount: number;
  error?: string;
}

/**
 * Lazy-loads sqlite-vector engines per catalog entry and retrieves from the
 * currently selected source only (no cross-DB merge).
 */
export class RagSourceManager {
  private engines = new Map<string, SqliteVectorRagEngine>();
  private statuses = new Map<string, RagSourceStatus>();
  private activeId = defaultRagSourceId();
  private ollamaBaseUrl = '/ollama';
  private resolveDbUrl: (dbPath: string) => string;

  constructor(options: {
    ollamaBaseUrl?: string;
    resolveDbUrl: (dbPath: string) => string;
  }) {
    this.ollamaBaseUrl = (options.ollamaBaseUrl ?? '/ollama').replace(/\/$/, '');
    this.resolveDbUrl = options.resolveDbUrl;
    for (const source of RAG_SOURCES) {
      this.statuses.set(source.id, { id: source.id, ready: false, chunkCount: 0 });
    }
  }

  get sources(): RagSourceConfig[] {
    return RAG_SOURCES;
  }

  get activeSourceId(): string {
    return this.activeId;
  }

  getActiveConfig(): RagSourceConfig | undefined {
    return getRagSource(this.activeId);
  }

  getActiveEngine(): SqliteVectorRagEngine | null {
    return this.engines.get(this.activeId) ?? null;
  }

  getStatus(id: string): RagSourceStatus | undefined {
    return this.statuses.get(id);
  }

  get activeReady(): boolean {
    return this.statuses.get(this.activeId)?.ready === true;
  }

  get activeChunkCount(): number {
    return this.statuses.get(this.activeId)?.chunkCount ?? 0;
  }

  async setActiveSource(id: string): Promise<void> {
    if (!getRagSource(id)) {
      throw new Error(`Unknown RAG source: ${id}`);
    }
    this.activeId = id;
    await this.ensureLoaded(id);
  }

  /** Load the active source (and optionally warm the default on startup). */
  async ensureLoaded(id = this.activeId): Promise<SqliteVectorRagEngine> {
    const config = getRagSource(id);
    if (!config) {
      throw new Error(`Unknown RAG source: ${id}`);
    }

    const existing = this.engines.get(id);
    if (existing?.isReady) {
      return existing;
    }

    const engine = existing ?? new SqliteVectorRagEngine();
    this.engines.set(id, engine);

    try {
      await engine.init({
        dbUrl: this.resolveDbUrl(config.dbPath),
        systemPrompt: config.systemPrompt,
        corpusLabel: config.label,
      });
      this.statuses.set(id, {
        id,
        ready: true,
        chunkCount: engine.chunkCount,
      });
      return engine;
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      this.statuses.set(id, {
        id,
        ready: false,
        chunkCount: 0,
        error: message,
      });
      throw error;
    }
  }

  async retrieve(query: string, topK: number): Promise<RagHit[]> {
    const engine = await this.ensureLoaded(this.activeId);
    return engine.retrieve(query, topK);
  }

  buildAugmentedUserMessage(question: string, hits: RagHit[]): string {
    const engine = this.getActiveEngine();
    if (!engine?.isReady) {
      throw new Error('Active RAG source is not ready');
    }
    return engine.buildAugmentedUserMessage(question, hits);
  }

  get systemPrompt(): string {
    return this.getActiveConfig()?.systemPrompt ?? '';
  }

  async pingOllama(): Promise<boolean> {
    try {
      const res = await fetch(`${this.ollamaBaseUrl}/api/tags`);
      return res.ok;
    } catch {
      return false;
    }
  }

  dispose(): void {
    for (const engine of this.engines.values()) {
      engine.dispose();
    }
    this.engines.clear();
    for (const source of RAG_SOURCES) {
      this.statuses.set(source.id, { id: source.id, ready: false, chunkCount: 0 });
    }
  }
}
