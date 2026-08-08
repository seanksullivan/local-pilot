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

import { marked } from 'marked';
import DOMPurify from 'dompurify';

marked.setOptions({
  gfm: true,
  breaks: true,
});

/** Parse Markdown to sanitized HTML for safe DOM insertion. */
export function renderMarkdown(markdown: string): string {
  const html = marked.parse(markdown || '', { async: false }) as string;
  return DOMPurify.sanitize(html, {
    USE_PROFILES: { html: true },
    ADD_ATTR: ['target', 'rel'],
  });
}

/** Set an element's contents from Markdown (streaming-safe: re-parse full text). */
export function setMarkdownContent(el: HTMLElement, markdown: string): void {
  el.innerHTML = renderMarkdown(markdown);
  el.querySelectorAll('a[href]').forEach((anchor) => {
    const a = anchor as HTMLAnchorElement;
    if (/^https?:/i.test(a.getAttribute('href') || '')) {
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
    }
  });
}
