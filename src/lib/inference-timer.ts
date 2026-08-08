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

export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(2)} s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = ((ms % 60_000) / 1000).toFixed(1);
  return `${minutes}m ${seconds}s`;
}

export function formatTokenCount(n: number | undefined | null): string {
  if (n == null || !Number.isFinite(n) || n < 0) return '—';
  return Math.round(n).toLocaleString();
}

export interface InferenceFinishStats {
  /** Client wall time for generation (excludes RAG when timed separately). */
  wallMs?: number;
  /** Server-side generation time (e.g. Ollama eval_duration). */
  evalMs?: number;
  /** Server-side prompt eval time. */
  promptEvalMs?: number;
  evalCount?: number;
  ragMs?: number;
  inputTokens?: number;
  outputTokens?: number;
}

export interface LiveInferenceClockLabels {
  setTime: (text: string) => void;
  setTokens?: (text: string) => void;
}

/** Live-updating inference time + token labels during streaming generation. */
export class LiveInferenceClock {
  private timer: number | null = null;
  private start = 0;
  private ragMs = 0;
  private inputTokens: number | undefined;
  private outputTokens: number | undefined;
  private tokensExact = false;

  constructor(private labels: LiveInferenceClockLabels) {}

  /** Start live refresh (every 100ms). Pass RAG duration if already measured. */
  begin(ragMs = 0, inputTokens?: number) {
    this.stop();
    this.ragMs = ragMs;
    this.inputTokens = inputTokens;
    this.outputTokens = undefined;
    this.tokensExact = false;
    this.start = performance.now();
    this.paintLive(0);
    this.paintTokens();
    this.timer = window.setInterval(() => {
      this.paintLive(performance.now() - this.start);
    }, 100);
  }

  /** Update token counts during / after streaming. */
  setTokens(input?: number, output?: number, exact = false) {
    if (input != null) this.inputTokens = input;
    if (output != null) this.outputTokens = output;
    if (exact) this.tokensExact = true;
    this.paintTokens();
  }

  /** Elapsed generation ms since begin(). */
  elapsed(): number {
    return this.start ? performance.now() - this.start : 0;
  }

  finish(stats: InferenceFinishStats = {}) {
    this.stop();
    const ragMs = stats.ragMs ?? this.ragMs;
    const parts: string[] = [];

    if (stats.evalMs != null && stats.evalMs > 0) {
      parts.push(`${formatDuration(stats.evalMs)} eval`);
      if (stats.promptEvalMs != null && stats.promptEvalMs > 0) {
        parts.push(`${formatDuration(stats.promptEvalMs)} prompt`);
      }
      if (stats.evalCount != null && stats.evalCount > 0 && stats.evalMs > 0) {
        const tps = (stats.evalCount / (stats.evalMs / 1000)).toFixed(1);
        parts.push(`${tps} tok/s`);
      }
    } else {
      const wall = stats.wallMs ?? this.elapsed();
      parts.push(formatDuration(wall));
    }

    if (ragMs > 0) {
      parts.push(`RAG ${formatDuration(ragMs)}`);
    }

    this.labels.setTime(`Inference Time: ${parts.join(' · ')}`);

    if (stats.inputTokens != null) this.inputTokens = stats.inputTokens;
    if (stats.outputTokens != null) this.outputTokens = stats.outputTokens;
    this.tokensExact = true;
    this.paintTokens();
  }

  stop() {
    if (this.timer != null) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  resetLabels() {
    this.stop();
    this.inputTokens = undefined;
    this.outputTokens = undefined;
    this.tokensExact = false;
    this.labels.setTime('Inference Time: —');
    this.labels.setTokens?.('Tokens: in — · out —');
  }

  private paintLive(genMs: number) {
    const gen = formatDuration(genMs);
    if (this.ragMs > 0) {
      this.labels.setTime(`Inference Time: ${gen} · RAG ${formatDuration(this.ragMs)}`);
    } else {
      this.labels.setTime(`Inference Time: ${gen}`);
    }
  }

  private paintTokens() {
    if (!this.labels.setTokens) return;
    const inn = formatTokenCount(this.inputTokens);
    const out = formatTokenCount(this.outputTokens);
    const suffix = this.tokensExact || (this.inputTokens == null && this.outputTokens == null) ? '' : ' (est.)';
    this.labels.setTokens(`Tokens: in ${inn} · out ${out}${suffix}`);
  }
}
