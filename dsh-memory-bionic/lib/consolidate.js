/**
 * dsh-memory-bionic — memory consolidation (记忆巩固).
 *
 * Bio-mapping: 睡眠/巩固期 — 将短期记忆 (近期会话跨度) 提炼为长期记忆条目。
 * 在每轮结束 (turn/end) 或压缩 (compaction) 发生时, 若新内容超过阈值,
 * 后台调用 LLM 做结构化提炼 (facts/decisions/preferences/errors...),
 * 失败时回退到原文分块 (raw), 保证内容不丢。捕获进度按会话 seq 记录,
 * 增量推进, 压缩遮蔽的历史永远不会被遗漏。
 */
import { estimateTokens, messageText, isHumanUserMessage } from "./util.js";

const EXTRACTION_SYSTEM = `You are a memory-consolidation engine for an AI coding assistant.
The text below is a recent conversation span. Extract the durable, reusable memories
worth remembering long-term, and output ONLY a JSON array, nothing else.

Each element: { "kind": string, "text": string, "importance": number, "tags": [string] }
- kind ∈ { "fact", "decision", "preference", "error-fix", "task", "file-change", "summary" }
  - fact:       established facts about the project, environment, or domain
  - decision:   decisions made and their rationale (architecture, approach, choices)
  - preference: the user's stated preferences, conventions, and corrections
  - error-fix:  a concrete error and how it was resolved (include exact strings/paths)
  - task:       explicitly requested work, in-flight or pending
  - file-change: significant file-level changes (exact paths, what/why)
  - summary:    only if the span is too broad for the kinds above
- text: concise, self-contained, precise. Keep exact file paths, commands, error
  strings, identifiers, and numeric values. 1–3 sentences. Do not quote the user.
- importance: 0..1 (0.9+ critical decisions/corrections, 0.5 typical, 0.2 trivia)
- tags: 0..6 short lowercase keywords (e.g. ["auth", "refactor"])

Rules:
- Skip transient chatter, pleasantries, and anything already covered by another element.
- If nothing is worth remembering, output [].
- Output ONLY the JSON array — no markdown fences, no commentary.`;

const SPAN_HEADER = "Recent conversation span (older first):\n\n";

export class Consolidator {
  constructor(ctx, config) {
    this.ctx = ctx;
    this.config = config;
    this.agents = new Map(); // sessionId -> agent (for provider/model routing)
    this.inflight = new Set(); // sessionIds currently consolidating
  }

  registerAgent(sessionId, agent) {
    this.agents.set(sessionId, agent);
  }

  unregisterAgent(sessionId) {
    this.agents.delete(sessionId);
  }

  /** Collect a text span of surface events after `sinceSeq` (capped). */
  collectSpan(session, sinceSeq) {
    const log = session.log;
    const events = [];
    for (let seq = sinceSeq; seq < log.length; seq++) {
      const event = log[seq];
      if (event === undefined) continue;
      if (event.type === "user/message") {
        if (!isHumanUserMessage(event.data)) continue;
        events.push({ seq, role: "user", text: messageText(event.data) });
      } else if (event.type === "assistant/message") {
        const text = messageText(event.data.message);
        if (text.length > 0) events.push({ seq, role: "assistant", text });
      } else if (event.type === "tool/result") {
        const text = messageText(event.data.message);
        if (text.length > 0) events.push({ seq, role: "tool", text });
      }
    }
    if (events.length === 0) return null;
    // keep the tail (most recent) within maxSpanChars
    let chars = 0;
    const kept = [];
    for (let i = events.length - 1; i >= 0; i--) {
      chars += events[i].text.length + 8;
      kept.unshift(events[i]);
      if (chars > this.config.maxSpanChars) {
        kept.shift();
        break;
      }
    }
    const text = kept
      .map((e) => `[${e.role} @seq ${e.seq}] ${e.text}`)
      .join("\n\n");
    return {
      text,
      firstSeq: kept[0].seq,
      lastSeq: kept[kept.length - 1].seq,
      approxTokens: estimateTokens(text)
    };
  }

  /** Run consolidation for one session when the new span clears the threshold. */
  maybeConsolidate(session, force = false) {
    const store = this.config.storeFor(session);
    if (!store) return;
    if (this.inflight.has(session.id)) return;
    const capture = store.getCapture(session.id);
    const sinceSeq = capture ? capture.lastCapturedSeq + 1 : 0;
    let span;
    try {
      span = this.collectSpan(session, sinceSeq);
    } catch (error) {
      this.ctx.logger.warn(`memory: span collection failed: ${error instanceof Error ? error.message : error}`);
      return;
    }
    if (!span) return;
    if (!force && span.approxTokens < this.config.captureTokenThreshold) return;
    this.inflight.add(session.id);
    const sessionId = session.id;
    // Background task — never block the agent loop.
    const run = async () => {
      try {
        await this.consolidate(session, store, span);
        store.setCapture(sessionId, span.lastSeq);
        const stats = store.stats();
        this.ctx.logger.info(
          `memory: consolidated ${span.approxTokens} tokens (seqs ${span.firstSeq}-${span.lastSeq}) → ${stats.items} items total`
        );
      } catch (error) {
        this.ctx.logger.warn(`memory: consolidation failed: ${error instanceof Error ? error.message : error}`);
      } finally {
        this.inflight.delete(sessionId);
      }
    };
    // Fire-and-forget with a microtask delay so the turn can commit first.
    void Promise.resolve().then(run);
  }

  async consolidate(session, store, span) {
    const items =
      this.config.captureMode === "raw"
        ? this.rawChunks(span)
        : await this.extractWithLlm(session, span);
    const sourceRange = [span.firstSeq, span.lastSeq];
    let added = 0;
    for (const candidate of items) {
      const item = store.add({
        kind: candidate.kind,
        text: candidate.text,
        tags: candidate.tags,
        sourceSeqs: sourceRange,
        importance: candidate.importance
      });
      if (item !== null) added++;
    }
    if (items.length > 0 && added === 0) {
      this.ctx.logger.info(`memory: ${items.length} candidates were already remembered (deduped)`);
    }
  }

  /** Raw fallback: chunk the span into overlapping pieces so nothing is lost. */
  rawChunks(span) {
    const size = 700;
    const overlap = 120;
    const text = span.text;
    const chunks = [];
    for (let i = 0; i < text.length; i += size - overlap) {
      const chunk = text.slice(i, i + size).trim();
      if (chunk.length >= 32) {
        chunks.push({ kind: "raw", text: chunk, importance: 0.5, tags: [] });
      }
      if (i + size >= text.length) break;
    }
    return chunks;
  }

  /** LLM extraction of structured memories from the span. */
  async extractWithLlm(session, span) {
    const agent = this.agents.get(session.id);
    const header = session.requestHeader?.()?.config;
    const target =
      header && header.provider && header.model
        ? { provider: header.provider, model: header.model }
        : agent?.options?.provider && agent?.options?.model
          ? { provider: agent.options.provider, model: agent.options.model }
          : undefined;
    if (target === undefined) {
      this.ctx.logger.info("memory: no routed provider/model yet — using raw capture");
      return this.rawChunks(span);
    }
    const messages = [
      {
        role: "user",
        content: [{ type: "text", text: SPAN_HEADER + span.text }]
      }
    ];
    let text = "";
    try {
      const stream = this.ctx.llm.stream({
        provider: target.provider,
        model: target.model,
        messages,
        system: EXTRACTION_SYSTEM,
        maxTokens: this.config.extractMaxTokens,
        sessionId: session.id,
        purpose: "memory-consolidation"
      });
      const perBlock = new Map();
      for await (const chunk of stream) {
        if (chunk.type === "text-delta") {
          perBlock.set(chunk.index, (perBlock.get(chunk.index) ?? "") + chunk.text);
        } else if (chunk.type === "finish" && chunk.reason === "length") {
          this.ctx.logger.warn("memory: extraction hit maxTokens; output may be truncated");
        }
      }
      text = [...perBlock.values()].join("");
    } catch (error) {
      this.ctx.logger.warn(
        `memory: LLM extraction failed (${error instanceof Error ? error.message : String(error)}); falling back to raw capture`
      );
      return this.rawChunks(span);
    }
    const parsed = this.parseExtraction(text);
    if (parsed === null) {
      this.ctx.logger.warn("memory: extraction returned unparseable output; falling back to raw capture");
      return this.rawChunks(span);
    }
    return parsed;
  }

  /** Parse and validate the JSON array the model returns. */
  parseExtraction(text) {
    let cleaned = text.trim();
    cleaned = cleaned.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "");
    let raw;
    try {
      raw = JSON.parse(cleaned);
    } catch {
      const start = cleaned.indexOf("[");
      const end = cleaned.lastIndexOf("]");
      if (start < 0 || end <= start) return null;
      try {
        raw = JSON.parse(cleaned.slice(start, end + 1));
      } catch {
        return null;
      }
    }
    if (!Array.isArray(raw)) return null;
    const out = [];
    for (const entry of raw.slice(0, 60)) {
      if (!entry || typeof entry !== "object") continue;
      const text = String(entry.text ?? "").trim();
      if (text.length < 8) continue;
      const importance = Number(entry.importance);
      out.push({
        kind: String(entry.kind ?? "fact"),
        text,
        importance: Number.isFinite(importance) ? importance : 0.5,
        tags: Array.isArray(entry.tags) ? entry.tags.map(String) : []
      });
    }
    return out;
  }
}
