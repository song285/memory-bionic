/**
 * dsh-memory-bionic — the long-term memory store (仿生记忆的长期记忆皮层).
 *
 * Bio-mapping:
 *  - 记忆条目 (memory items)     → 皮质表征 (cortical traces)
 *  - 向量索引 (vector index)     → 海马索引 (hippocampal index): 快速检索指针
 *  - 遗忘曲线 (Ebbinghaus decay) → 可访问性衰减: 不被复习(rehearsal)的记忆
 *                                  随时间在召回排序中下沉, 但从不被删除
 *  - 复习/重激活 (rehearsal)     → 每次成功召回(touch)强化记忆强度 (间隔重复)
 *
 * Persistence (all under one per-workspace directory):
 *  - items.jsonl  append-only 操作日志 (add / touch)
 *  - index.bin    float32 向量行 (与 index.json 的 ids 对齐)
 *  - index.json   { v, dim, ids }
 *  - meta.json    每会话捕获进度 (lastCapturedSeq)
 */
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { appendJsonl, readJson, readJsonl, atomicWrite, nowMs, clamp } from "./util.js";
import { VectorIndex } from "./vector-store.js";

const KIND_SET = new Set(["fact", "decision", "preference", "error-fix", "task", "file-change", "summary", "raw"]);
const LOG_VERSION = 1;

export class MemoryStore {
  dir;
  embedder;
  config;
  items = new Map(); // id -> item
  index;
  loaded = false;
  _saveTimer = null;
  meta = { v: 1, sessions: {} };

  constructor(dir, embedder, config) {
    this.dir = dir;
    this.embedder = embedder;
    this.config = config;
    this.index = new VectorIndex(config.embedDim);
  }

  get ready() {
    return this.loaded;
  }

  /** Load persisted state; rebuilds the vector index from text when missing. */
  async load() {
    // items.jsonl replay
    const rows = readJsonl(join(this.dir, "items.jsonl"));
    for (const row of rows) {
      if (row.v !== LOG_VERSION) continue;
      if (row.op === "add" && row.item) {
        this.items.set(row.item.id, row.item);
      } else if (row.op === "touch" && row.id) {
        const item = this.items.get(row.id);
        if (item) {
          item.lastAccessAt = row.at;
          item.accessCount = row.count;
        }
      } else if (row.op === "remove" && row.id) {
        this.items.delete(row.id);
      }
    }
    // vector index — load it, but rebuild from text whenever it is missing,
    // stale, or out of sync with the item log (e.g. an empty index written by
    // an earlier boot before any items existed).
    if (!this.index.load(this.dir) || this.index.size() !== this.items.size) {
      this.index = new VectorIndex(this.config.embedDim);
      for (const item of this.items.values()) {
        this.index.upsert(item.id, this.embedder.embed(item.text));
      }
      this.saveIndexNow();
    }
    this.meta = readJson(join(this.dir, "meta.json"), { v: 1, sessions: {} }) ?? { v: 1, sessions: {} };
    this.loaded = true;
    return this;
  }

  /** Insert a new memory item (dedupes against existing vectors). */
  add({ kind = "fact", text, tags = [], sourceSeqs = [], importance = 0.5 }) {
    if (!this.loaded) return null;
    const cleaned = String(text ?? "").trim();
    if (cleaned.length < 8) return null;
    const vector = this.embedder.embed(cleaned);
    const best = this.index.maxSimilarity(vector);
    if (best >= this.config.dedupeSimilarity) return null; // already remembered
    const item = {
      id: randomUUID(),
      kind: KIND_SET.has(kind) ? kind : "fact",
      text: cleaned,
      tags: Array.isArray(tags) ? tags.map(String).slice(0, 12) : [],
      sourceSeqs: Array.isArray(sourceSeqs) ? sourceSeqs.map(Number) : [],
      importance: clamp(Number(importance) || 0.5, 0, 1),
      createdAt: nowMs(),
      lastAccessAt: nowMs(),
      accessCount: 0
    };
    if (this.items.size >= this.config.maxItems) this.evictLeastAccessible();
    this.items.set(item.id, item);
    this.index.upsert(item.id, vector);
    appendJsonl(join(this.dir, "items.jsonl"), { v: LOG_VERSION, op: "add", item });
    this.scheduleSave();
    return item;
  }

  /**
   * Bounded-store safety valve: when the item cap is reached, evict the trace
   * with the lowest current accessibility (the one least likely to be needed).
   * The eviction is recorded as a durable op so the append-only log stays
   * replayable. The cap exists only to bound disk; raise it for unbounded
   * "海量记忆" (the default is already generous).
   */
  evictLeastAccessible() {
    let worst = null;
    let worstScore = Infinity;
    const now = nowMs();
    for (const item of this.items.values()) {
      // Accessibility without a fresh query: fixed nominal similarity term.
      const score = this.accessibility(item, 0.3, now);
      if (score < worstScore) {
        worstScore = score;
        worst = item;
      }
    }
    if (!worst) return;
    this.items.delete(worst.id);
    this.index.remove(worst.id);
    appendJsonl(join(this.dir, "items.jsonl"), { v: LOG_VERSION, op: "remove", id: worst.id });
  }

  /** Rehearsal: strengthen one item's accessibility (间隔重复). */
  touch(id) {
    const item = this.items.get(id);
    if (!item) return;
    item.accessCount++;
    item.lastAccessAt = nowMs();
    appendJsonl(join(this.dir, "items.jsonl"), {
      v: LOG_VERSION,
      op: "touch",
      id,
      at: item.lastAccessAt,
      count: item.accessCount
    });
    this.scheduleSave();
  }

  /**
   * Recall: embed the query, fetch candidate vectors, then rank by
   * *accessibility* — similarity × recency decay × importance × strength —
   * the forgetting-curve term that makes unrehearsed memories sink (but never
   * vanish). Returns `[{ item, similarity, accessibility }]`.
   */
  search(query, topK, minSimilarity) {
    if (!this.loaded || this.index.size() === 0) return [];
    const vector = this.embedder.embed(String(query ?? ""));
    const candidates = this.index.search(vector, Math.max(topK * 4, 32));
    const now = nowMs();
    const results = [];
    for (const { id, similarity } of candidates) {
      if (similarity < minSimilarity) continue;
      const item = this.items.get(id);
      if (!item) continue;
      results.push({
        item,
        similarity,
        accessibility: this.accessibility(item, similarity, now)
      });
    }
    results.sort((a, b) => b.accessibility - a.accessibility);
    return results.slice(0, topK);
  }

  /** Ebbinghaus-style accessibility: how reachable is this trace right now? */
  accessibility(item, similarity, at) {
    const halfLifeMs = this.config.halfLifeHours * 3600 * 1000;
    const ageMs = Math.max(0, at - item.createdAt);
    const recency = Math.exp(-ageMs / halfLifeMs); // 遗忘曲线衰减项
    const strength = 1 - Math.exp(-item.accessCount / 3); // 复习强化项
    const base = 0.55 * similarity + 0.25 * recency + 0.2 * item.importance;
    return base * (0.5 + 0.5 * strength);
  }

  stats() {
    const byKind = {};
    for (const item of this.items.values()) byKind[item.kind] = (byKind[item.kind] ?? 0) + 1;
    return { items: this.items.size, vectors: this.index.size(), byKind, ready: this.loaded };
  }

  getCapture(sessionId) {
    return this.meta.sessions[sessionId] ?? null;
  }

  setCapture(sessionId, seq, at = nowMs()) {
    const prev = this.meta.sessions[sessionId];
    if (prev && prev.lastCapturedSeq >= seq) return;
    this.meta.sessions[sessionId] = { lastCapturedSeq: seq, at };
    this.scheduleSave();
  }

  saveIndexNow() {
    this.index.save(this.dir);
  }

  /** Debounced persistence of index + meta (items.jsonl is already append-only). */
  scheduleSave() {
    if (this._saveTimer !== null) return;
    this._saveTimer = setTimeout(() => {
      this._saveTimer = null;
      try {
        this.index.save(this.dir);
        atomicWrite(join(this.dir, "meta.json"), JSON.stringify(this.meta));
      } catch (error) {
        /* persistence is best-effort; log happens at the plugin layer */
      }
    }, 400);
  }

  /** Synchronous flush for shutdown. */
  flush() {
    if (this._saveTimer !== null) {
      clearTimeout(this._saveTimer);
      this._saveTimer = null;
    }
    if (this.loaded) {
      this.index.save(this.dir);
      atomicWrite(join(this.dir, "meta.json"), JSON.stringify(this.meta));
    }
  }
}
