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
 */

export interface OllamaModelInfo {
  name: string;
  capabilities?: string[];
}

export interface OllamaChatStats {
  /** Wall time from request start to final chunk (client). */
  wallMs: number;
  /** Ollama total_duration (ns → ms when present). */
  totalMs?: number;
  /** Ollama eval_duration — token generation only. */
  evalMs?: number;
  /** Ollama prompt_eval_duration — prompt processing. */
  promptEvalMs?: number;
  /** Output tokens (eval_count). */
  evalCount?: number;
  /** Input / prompt tokens (prompt_eval_count). */
  promptEvalCount?: number;
}

export interface OllamaChatOptions {
  baseUrl?: string;
  model: string;
  systemPrompt?: string;
  userMessage: string;
  temperature?: number;
  topK?: number;
  numPredict?: number;
  onPartial?: (text: string, done: boolean, stats?: OllamaChatStats) => void;
  signal?: AbortSignal;
}

export class OllamaChatEngine {
  private baseUrl: string;
  private abortController: AbortController | null = null;
  private generating = false;

  constructor(baseUrl = '/ollama') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  get isGenerating(): boolean {
    return this.generating;
  }

  async listModels(): Promise<OllamaModelInfo[]> {
    const res = await fetch(`${this.baseUrl}/api/tags`);
    if (!res.ok) {
      throw new Error(`Ollama unreachable (${res.status}). Is \`ollama serve\` running?`);
    }
    const data = (await res.json()) as {
      models?: Array<{ name: string; details?: unknown; model?: string }>;
    };
    return (data.models ?? []).map((m) => ({ name: m.name || m.model || '' })).filter((m) => m.name);
  }

  async ping(): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/api/tags`);
      return res.ok;
    } catch {
      return false;
    }
  }

  cancel(): void {
    this.abortController?.abort();
    this.abortController = null;
  }

  async chat(options: OllamaChatOptions): Promise<string> {
    if (this.generating) {
      throw new Error('Generation already in progress');
    }

    this.abortController = new AbortController();
    if (options.signal) {
      options.signal.addEventListener('abort', () => this.abortController?.abort(), { once: true });
    }

    this.generating = true;
    let full = '';
    const wallStart = performance.now();
    let lastStats: OllamaChatStats | undefined;
    let finished = false;

    const nsToMs = (ns?: number) => (typeof ns === 'number' && ns > 0 ? ns / 1e6 : undefined);

    const emitDone = () => {
      if (finished) return;
      finished = true;
      const stats: OllamaChatStats = {
        wallMs: performance.now() - wallStart,
        totalMs: lastStats?.totalMs,
        evalMs: lastStats?.evalMs,
        promptEvalMs: lastStats?.promptEvalMs,
        evalCount: lastStats?.evalCount,
        promptEvalCount: lastStats?.promptEvalCount,
      };
      options.onPartial?.('', true, stats);
    };

    try {
      const messages: Array<{ role: string; content: string }> = [];
      const system = options.systemPrompt?.trim();
      if (system) {
        messages.push({ role: 'system', content: system });
      }
      messages.push({ role: 'user', content: options.userMessage });

      const res = await fetch(`${this.baseUrl}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: this.abortController.signal,
        body: JSON.stringify({
          model: options.model,
          messages,
          stream: true,
          options: {
            temperature: options.temperature ?? 0.8,
            top_k: options.topK ?? 40,
            num_predict: options.numPredict ?? 1024,
          },
        }),
      });

      if (!res.ok) {
        const body = await res.text().catch(() => '');
        throw new Error(`Ollama chat failed (${res.status}): ${body || res.statusText}`);
      }
      if (!res.body) {
        throw new Error('Ollama response has no body to stream');
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      const handleChunk = (parsed: {
        message?: { content?: string };
        response?: string;
        done?: boolean;
        total_duration?: number;
        eval_duration?: number;
        prompt_eval_duration?: number;
        eval_count?: number;
        prompt_eval_count?: number;
      }) => {
        const piece = parsed.message?.content ?? parsed.response ?? '';
        if (piece) {
          full += piece;
          options.onPartial?.(piece, false, {
            wallMs: performance.now() - wallStart,
          });
        }
        if (parsed.done) {
          lastStats = {
            wallMs: performance.now() - wallStart,
            totalMs: nsToMs(parsed.total_duration),
            evalMs: nsToMs(parsed.eval_duration),
            promptEvalMs: nsToMs(parsed.prompt_eval_duration),
            evalCount: parsed.eval_count,
            promptEvalCount: parsed.prompt_eval_count,
          };
          emitDone();
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;
          try {
            handleChunk(JSON.parse(trimmed));
          } catch {
            continue;
          }
        }
      }

      if (buffer.trim()) {
        try {
          handleChunk(JSON.parse(buffer.trim()));
        } catch {
          // ignore trailing partial JSON
        }
      }

      emitDone();
      return full;
    } catch (error: unknown) {
      if (error instanceof Error && error.name === 'AbortError') {
        emitDone();
        return full;
      }
      throw error;
    } finally {
      this.generating = false;
      this.abortController = null;
    }
  }
}
