import template from './ollama-chat.html?raw';
import { OllamaChatEngine } from '../lib/ollama-chat-engine';
import { LiveInferenceClock } from '../lib/inference-timer';
import { setMarkdownContent } from '../lib/markdown-render';
import { RagSourceManager } from '../lib/rag-source-manager';
import { defaultRagSourceId, type RagSourceConfig } from '../lib/rag-sources';
import type { RagHit } from '../lib/sqlite-vector-rag';

const PREFERRED_MODELS = ['gemma4:e4b', 'gemma4:e2b', 'gemma3:4b', 'llama3.2', 'qwen3.5:9b'];

class OllamaChatApp {
  private engine = new OllamaChatEngine('/ollama');
  private rag!: RagSourceManager;
  private isReady = false;
  private inferenceClock = new LiveInferenceClock({
    setTime: (text) => {
      const el = document.getElementById('inference-time');
      if (el) el.textContent = text;
    },
    setTokens: (text) => {
      const el = document.getElementById('token-counts');
      if (el) el.textContent = text;
    },
  });

  private modelName = '';
  private maxTokens = 3072;
  private temperature = 0.4;
  private topK = 1;
  private ragEnabled = true;
  private ragTopK = 10;

  private modelSelect!: HTMLSelectElement;
  private ragSourceSelect!: HTMLSelectElement;
  private systemInput!: HTMLTextAreaElement;
  private textInput!: HTMLTextAreaElement;
  private generateBtn!: HTMLButtonElement;
  private cancelBtn!: HTMLButtonElement;
  private outputEl!: HTMLElement;
  private ollamaStatusEl!: HTMLElement;
  private ragSourcesEl!: HTMLElement;
  private ragStatusEl!: HTMLElement;
  private ragToggleSubEl!: HTMLElement;
  private chatSubtitleEl!: HTMLElement;
  private systemHintEl!: HTMLElement;
  private systemPresetsEl!: HTMLElement;

  constructor(private container: HTMLElement) {
    this.rag = new RagSourceManager({
      ollamaBaseUrl: '/ollama',
      resolveDbUrl: (dbPath) => this.assetUrl(dbPath),
    });
  }

  async initialize() {
    this.container.innerHTML = template;
    this.bindUi();
    this.populateRagSourceSelect();
    this.applySourceUi(this.rag.getActiveConfig());
    await Promise.all([this.refreshModels(), this.initializeActiveRag()]);
  }

  cleanup() {
    this.engine.cancel();
    this.inferenceClock.stop();
    this.rag.dispose();
    this.isReady = false;
  }

  private assetUrl(path: string): string {
    const base = import.meta.env.BASE_URL || '/';
    return `${base.replace(/\/?$/, '/')}${path.replace(/^\//, '')}`;
  }

  private bindUi() {
    this.modelSelect = document.getElementById('ollama-model-select') as HTMLSelectElement;
    this.ragSourceSelect = document.getElementById('rag-source-select') as HTMLSelectElement;
    this.systemInput = document.getElementById('system-input') as HTMLTextAreaElement;
    this.textInput = document.getElementById('text-input') as HTMLTextAreaElement;
    this.generateBtn = document.getElementById('generate-btn') as HTMLButtonElement;
    this.cancelBtn = document.getElementById('cancel-btn') as HTMLButtonElement;
    this.outputEl = document.getElementById('llm-output') as HTMLElement;
    this.ollamaStatusEl = document.getElementById('ollama-status') as HTMLElement;
    this.ragSourcesEl = document.getElementById('rag-sources') as HTMLElement;
    this.ragStatusEl = document.getElementById('rag-status') as HTMLElement;
    this.ragToggleSubEl = document.getElementById('rag-toggle-sub') as HTMLElement;
    this.chatSubtitleEl = document.getElementById('chat-subtitle') as HTMLElement;
    this.systemHintEl = document.getElementById('system-hint') as HTMLElement;
    this.systemPresetsEl = document.getElementById('system-presets') as HTMLElement;

    this.generateBtn.addEventListener('click', () => void this.runGenerate());
    this.cancelBtn.addEventListener('click', () => {
      this.engine.cancel();
      this.inferenceClock.finish({ wallMs: this.inferenceClock.elapsed() });
      this.updateStatus('Cancelled');
      this.setBusy(false);
    });

    this.modelSelect.addEventListener('change', () => {
      this.modelName = this.modelSelect.value;
      this.updateReadyState();
    });

    this.ragSourceSelect.addEventListener('change', () => {
      void this.onRagSourceChange(this.ragSourceSelect.value);
    });

    this.textInput.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !this.generateBtn.disabled) {
        void this.runGenerate();
      }
    });

    this.container.querySelectorAll('.sample-btn').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const text = (e.currentTarget as HTMLElement).dataset.text;
        if (text) {
          this.textInput.value = text;
          void this.runGenerate();
        }
      });
    });

    const maxTokensInput = document.getElementById('max-tokens') as HTMLInputElement;
    const maxTokensValue = document.getElementById('max-tokens-value')!;
    maxTokensInput.addEventListener('input', () => {
      this.maxTokens = parseInt(maxTokensInput.value, 10);
      maxTokensValue.innerText = String(this.maxTokens);
    });

    const temperatureInput = document.getElementById('temperature') as HTMLInputElement;
    const temperatureValue = document.getElementById('temperature-value')!;
    temperatureInput.addEventListener('input', () => {
      this.temperature = parseInt(temperatureInput.value, 10) / 100;
      temperatureValue.innerText = this.temperature.toFixed(2);
    });

    const topKInput = document.getElementById('top-k') as HTMLInputElement;
    const topKValue = document.getElementById('top-k-value')!;
    topKInput.addEventListener('input', () => {
      this.topK = parseInt(topKInput.value, 10);
      topKValue.innerText = String(this.topK);
    });

    const ragToggle = document.getElementById('rag-enabled') as HTMLInputElement;
    ragToggle.checked = this.ragEnabled;
    ragToggle.addEventListener('change', () => {
      this.ragEnabled = ragToggle.checked;
      this.updateRagStatus();
    });

    const ragTopKInput = document.getElementById('rag-top-k') as HTMLInputElement;
    const ragTopKValue = document.getElementById('rag-top-k-value')!;
    ragTopKInput.addEventListener('input', () => {
      this.ragTopK = parseInt(ragTopKInput.value, 10);
      ragTopKValue.innerText = String(this.ragTopK);
      this.updateRagStatus();
    });
  }

  private populateRagSourceSelect() {
    const sources = this.rag.sources;
    this.ragSourceSelect.innerHTML = sources
      .map((s) => `<option value="${escapeHtml(s.id)}">${escapeHtml(s.label)}</option>`)
      .join('');
    const initial = this.rag.activeSourceId || defaultRagSourceId();
    this.ragSourceSelect.value = initial;
  }

  private applySourceUi(config: RagSourceConfig | undefined) {
    if (!config) return;

    this.ragToggleSubEl.textContent = config.description;
    this.chatSubtitleEl.textContent = `${config.label} · local Ollama`;
    const presets = config.systemPresets ?? [];
    this.systemHintEl.textContent = presets[0]?.label ?? config.label;
    this.textInput.placeholder = `Ask about ${config.label}…`;

    const presetText = presets[0]?.text ?? config.systemPrompt;
    this.systemInput.value = presetText;

    this.systemPresetsEl.innerHTML = '';
    for (const preset of presets) {
      const btn = document.createElement('button');
      btn.className = 'sample-system-btn';
      btn.type = 'button';
      btn.textContent = preset.label;
      btn.addEventListener('click', () => {
        this.systemInput.value = preset.text;
        this.systemHintEl.textContent = preset.label;
      });
      this.systemPresetsEl.appendChild(btn);
    }
  }

  private async onRagSourceChange(id: string) {
    this.ragStatusEl.textContent = 'Loading RAG source…';
    try {
      await this.rag.setActiveSource(id);
      this.applySourceUi(this.rag.getActiveConfig());
      const ollamaOk = await this.rag.pingOllama();
      const label = this.rag.getActiveConfig()?.label ?? id;
      this.ragStatusEl.textContent = ollamaOk
        ? `Loaded ${this.rag.activeChunkCount} chunks · ${label}`
        : `DB loaded (${this.rag.activeChunkCount} chunks) · Ollama unreachable at /ollama`;
      this.updateRagStatus();
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      this.ragStatusEl.textContent = `RAG error: ${message}`;
      console.error(error);
    }
  }

  private async initializeActiveRag() {
    const config = this.rag.getActiveConfig();
    this.ragStatusEl.textContent = `Loading ${config?.label ?? 'RAG'} vector DB…`;
    try {
      await this.rag.ensureLoaded();
      const ollamaOk = await this.rag.pingOllama();
      const label = this.rag.getActiveConfig()?.label ?? 'RAG';
      this.ragStatusEl.textContent = ollamaOk
        ? `Loaded ${this.rag.activeChunkCount} chunks · ${label}`
        : `DB loaded (${this.rag.activeChunkCount} chunks) · Ollama unreachable at /ollama`;
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      this.ragStatusEl.textContent = `RAG error: ${message}`;
      console.error(error);
    }
  }

  private updateRagStatus() {
    if (!this.ragStatusEl) return;
    if (!this.ragEnabled) {
      this.ragStatusEl.textContent = 'RAG disabled — plain chat';
      return;
    }
    if (this.rag.activeReady) {
      const label = this.rag.getActiveConfig()?.label ?? 'RAG';
      this.ragStatusEl.textContent = `Loaded ${this.rag.activeChunkCount} chunks · ${label} · top ${this.ragTopK}`;
    }
  }

  private async refreshModels() {
    this.updateStatus('Connecting to Ollama…');
    this.ollamaStatusEl.textContent = 'Connecting to Ollama via /ollama…';
    this.modelSelect.innerHTML = `<option value="">Loading models…</option>`;
    this.isReady = false;
    this.setBusy(false);
    this.generateBtn.disabled = true;

    try {
      const models = await this.engine.listModels();
      const chatModels = models
        .map((m) => m.name)
        .filter((name) => !/embed/i.test(name))
        .sort((a, b) => a.localeCompare(b));

      if (chatModels.length === 0) {
        this.modelSelect.innerHTML = `<option value="">No chat models found</option>`;
        this.ollamaStatusEl.textContent = 'Ollama is up, but no chat models are installed';
        this.updateStatus('No models');
        return;
      }

      const preferred = PREFERRED_MODELS.find((p) => chatModels.includes(p)) ?? chatModels[0];
      this.modelSelect.innerHTML = chatModels
        .map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`)
        .join('');
      this.modelSelect.value = preferred;
      this.modelName = preferred;
      this.ollamaStatusEl.textContent = `${chatModels.length} model(s) available · proxy /ollama → :11434`;
      this.updateStatus('Ready');
      this.updateReadyState();
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      this.modelSelect.innerHTML = `<option value="">Unavailable</option>`;
      this.ollamaStatusEl.textContent = message;
      this.updateStatus(`Error: ${message}`);
      console.error(error);
    }
  }

  private updateReadyState() {
    this.isReady = !!this.modelName;
    this.generateBtn.disabled = !this.isReady || this.engine.isGenerating;
  }

  private renderRagSources(hits: RagHit[]) {
    if (!hits.length) {
      this.ragSourcesEl.innerHTML = '';
      return;
    }
    this.ragSourcesEl.innerHTML = `
      <div class="rag-sources-card">
        <div class="rag-sources-title">Retrieved sources</div>
        <div class="rag-sources-body">
          <ol class="rag-sources-list">
            ${hits
              .map((hit) => {
                const title = hit.sectionTitle ? ` — ${escapeHtml(hit.sectionTitle)}` : '';
                const link = hit.sourceUrl
                  ? `<a href="${escapeHtml(hit.sourceUrl)}" target="_blank" rel="noopener">${escapeHtml(hit.label)}</a>`
                  : escapeHtml(hit.label);
                return `<li>${link}${title} <span class="rag-score">${hit.score.toFixed(3)}</span></li>`;
              })
              .join('')}
          </ol>
        </div>
      </div>`;
  }

  private async runGenerate() {
    const userMessage = this.textInput.value.trim();
    if (!userMessage || !this.isReady || this.engine.isGenerating) return;

    this.setBusy(true);
    this.updateStatus('Generating…');
    this.outputEl.innerHTML = `<div class="llm-response" id="llm-response-text"></div>`;
    const responseEl = document.getElementById('llm-response-text')!;
    this.ragSourcesEl.innerHTML = '';
    this.inferenceClock.resetLabels();

    try {
      let message = userMessage;
      let systemPrompt = this.systemInput.value;
      let ragMs = 0;

      if (this.ragEnabled) {
        if (!this.rag.activeReady) {
          throw new Error('RAG database is not ready');
        }
        const label = this.rag.getActiveConfig()?.label ?? 'corpus';
        this.updateStatus(`Retrieving ${label} passages…`);
        const ragStart = performance.now();
        const hits = await this.rag.retrieve(userMessage, this.ragTopK);
        ragMs = performance.now() - ragStart;
        this.renderRagSources(hits);
        message = this.rag.buildAugmentedUserMessage(userMessage, hits);
        if (!systemPrompt.trim()) {
          systemPrompt = this.rag.systemPrompt;
        }
      }

      this.updateStatus('Generating…');
      this.inferenceClock.begin(ragMs);
      let streamed = '';

      await this.engine.chat({
        model: this.modelName,
        systemPrompt,
        userMessage: message,
        temperature: this.temperature,
        topK: this.topK,
        numPredict: this.maxTokens,
        onPartial: (piece, done, stats) => {
          if (piece) {
            streamed += piece;
            setMarkdownContent(responseEl, streamed);
            this.scrollChatToBottom();
            this.inferenceClock.setTokens(undefined, estimateTokens(streamed), false);
          }
          if (done) {
            if (streamed) setMarkdownContent(responseEl, streamed);
            this.scrollChatToBottom();
            this.inferenceClock.finish({
              wallMs: stats?.wallMs,
              evalMs: stats?.evalMs,
              promptEvalMs: stats?.promptEvalMs,
              evalCount: stats?.evalCount,
              ragMs,
              inputTokens: stats?.promptEvalCount,
              outputTokens: stats?.evalCount,
            });
            this.updateStatus('Done');
            this.setBusy(false);
          }
        },
      });
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : String(error);
      this.inferenceClock.stop();
      this.updateStatus(`Error: ${message}`);
      console.error(error);
      this.setBusy(false);
    }
  }

  private setBusy(busy: boolean) {
    this.generateBtn.disabled = busy || !this.isReady;
    this.cancelBtn.disabled = !busy;
    this.generateBtn.innerHTML = busy
      ? `<span class="material-icons" style="font-size: 18px; margin-right: 6px">hourglass_top</span> Working…`
      : `<span class="material-icons" style="font-size: 18px; margin-right: 6px">send</span> Generate`;
  }

  private scrollChatToBottom() {
    const scroll = this.container.querySelector('.chat-scroll') as HTMLElement | null;
    if (scroll) scroll.scrollTop = scroll.scrollHeight;
  }

  private updateStatus(message: string) {
    const el = document.getElementById('status-message');
    if (el) el.textContent = message;
  }
}

function escapeHtml(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/** Cheap live estimate until the server reports exact token counts. */
function estimateTokens(text: string): number {
  if (!text) return 0;
  return Math.max(1, Math.round(text.length / 4));
}

let activeApp: OllamaChatApp | null = null;

export async function setupOllamaChat(container: HTMLElement) {
  activeApp = new OllamaChatApp(container);
  await activeApp.initialize();
}

export function cleanupOllamaChat() {
  if (activeApp) {
    activeApp.cleanup();
    activeApp = null;
  }
}
