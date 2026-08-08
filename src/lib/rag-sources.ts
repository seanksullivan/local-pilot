/** Registered sqlite-vector RAG corpora available to the chat UI. */
export interface RagSourceConfig {
  id: string;
  label: string;
  description: string;
  /** Path under public/, e.g. data/rcw-title-11-web.db */
  dbPath: string;
  /** Used when building augmented prompts for this corpus. */
  systemPrompt: string;
  /** Short label for system-prompt preset UI. */
  systemPresetLabel?: string;
  /** Compact system prompt for the composer preset button. */
  systemPresetText?: string;
}

export const RCW_TITLE_11_SYSTEM_PROMPT = `You are a knowledgeable professional specializing in Washington State probate and trust law (RCW Title 11).

Answer the user's question using ONLY the provided RCW excerpts when possible. Cite section numbers (e.g., RCW 11.12.010) in your answer. If the excerpts are insufficient, say what is missing and avoid inventing statutes. Be clear, precise, and practical. This is educational information, not formal legal advice.`;

export const RCW_TITLE_11_PRESET = `You are a professional specializing in Washington State probate and trust law (RCW Title 11). Cite statutes when relevant. This is educational information, not formal legal advice.`;

/**
 * Catalog of RAG sources. Add another entry (same embeddings schema) to expose
 * a second database in the Retrieval source selector.
 */
export const RAG_SOURCES: RagSourceConfig[] = [
  {
    id: 'rcw-title-11',
    label: 'RCW Title 11',
    description: 'Ground answers in Washington probate statutes',
    dbPath: 'data/rcw-title-11-web.db',
    systemPrompt: RCW_TITLE_11_SYSTEM_PROMPT,
    systemPresetLabel: 'Probate',
    systemPresetText: RCW_TITLE_11_PRESET,
  },
];

export function getRagSource(id: string): RagSourceConfig | undefined {
  return RAG_SOURCES.find((s) => s.id === id);
}

export function defaultRagSourceId(): string {
  return RAG_SOURCES[0]?.id ?? '';
}
