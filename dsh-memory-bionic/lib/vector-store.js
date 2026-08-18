/**
 * dsh-memory-bionic — flat cosine vector index with binary persistence.
 *
 * Exact top-k cosine search over L2-normalized dense vectors, kept entirely in
 * memory and persisted as a raw float32 rows file (`index.bin`) plus an id
 * manifest (`index.json`). Exact search over a few tens of thousands of
 * vectors is a few milliseconds — fast enough for per-request recall, and the
 * format is trivially replaceable by an ANN index (HNSW etc.) later.
 */
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { dot, normalize } from "./util.js";
import { atomicWrite } from "./util.js";

export class VectorIndex {
  dim;
  ids = [];
  map = new Map(); // id -> row index
  vectors; // Float32Array(dim * capacity)
  capacity = 0;
  length = 0;

  constructor(dim = 512) {
    this.dim = dim;
    this.vectors = new Float32Array(0);
  }

  size() {
    return this.length;
  }

  _grow() {
    const next = this.capacity === 0 ? 256 : this.capacity * 2;
    const grown = new Float32Array(next * this.dim);
    grown.set(this.vectors.subarray(0, this.length * this.dim));
    this.vectors = grown;
    this.capacity = next;
  }

  /** Insert or replace one vector (copied; must be L2-normalized). */
  upsert(id, vector) {
    const row = this.map.get(id);
    const dim = this.dim;
    if (row === undefined) {
      if (this.length === this.capacity) this._grow();
      const idx = this.length;
      this.vectors.set(vector, idx * dim);
      this.ids.push(id);
      this.map.set(id, idx);
      this.length++;
    } else {
      this.vectors.set(vector, row * dim);
    }
  }

  remove(id) {
    const row = this.map.get(id);
    if (row === undefined) return false;
    const last = this.length - 1;
    const dim = this.dim;
    if (row !== last) {
      // move the last row into the hole
      this.vectors.copyWithin(row * dim, last * dim, last * dim + dim);
      const movedId = this.ids[last];
      this.ids[row] = movedId;
      this.map.set(movedId, row);
    }
    this.ids.pop();
    this.map.delete(id);
    this.length--;
    return true;
  }

  /**
   * Exact top-k by cosine similarity. `query` must be normalized; returns
   * `[{ id, similarity }]` sorted descending, capped at `topK`.
   */
  search(query, topK) {
    const dim = this.dim;
    const scores = new Float64Array(this.length);
    for (let r = 0; r < this.length; r++) {
      let s = 0;
      const base = r * dim;
      for (let i = 0; i < dim; i++) s += this.vectors[base + i] * query[i];
      scores[r] = s;
    }
    const order = [];
    for (let i = 0; i < this.length; i++) order.push(i);
    order.sort((a, b) => scores[b] - scores[a]);
    const out = [];
    const limit = Math.min(topK, order.length);
    for (let i = 0; i < limit; i++) {
      const r = order[i];
      out.push({ id: this.ids[r], similarity: scores[r] });
    }
    return out;
  }

  /** Max cosine similarity against every stored vector (used for dedupe). */
  maxSimilarity(query) {
    const dim = this.dim;
    let best = -1;
    for (let r = 0; r < this.length; r++) {
      let s = 0;
      const base = r * dim;
      for (let i = 0; i < dim; i++) s += this.vectors[base + i] * query[i];
      if (s > best) best = s;
    }
    return best;
  }

  /** Persist rows + id manifest atomically. */
  save(dir) {
    mkdirSync(dir, { recursive: true });
    const bin = Buffer.alloc(this.length * this.dim * 4);
    for (let r = 0; r < this.length; r++) {
      bin.writeFloatLE(this.vectors[r * this.dim], r * this.dim * 4);
      for (let i = 1; i < this.dim; i++) {
        bin.writeFloatLE(this.vectors[r * this.dim + i], (r * this.dim + i) * 4);
      }
    }
    atomicWrite(join(dir, "index.bin"), bin);
    atomicWrite(join(dir, "index.json"), JSON.stringify({ v: 1, dim: this.dim, ids: [...this.ids] }));
  }

  /** Load persisted rows; returns false when the files are missing/corrupt. */
  load(dir) {
    const metaPath = join(dir, "index.json");
    const binPath = join(dir, "index.bin");
    if (!existsSync(metaPath) || !existsSync(binPath)) return false;
    try {
      const meta = JSON.parse(readFileSync(metaPath, "utf8"));
      if (meta.v !== 1 || meta.dim !== this.dim || !Array.isArray(meta.ids)) return false;
      const bin = readFileSync(binPath);
      const rows = meta.ids.length;
      if (bin.length !== rows * this.dim * 4) return false;
      this.ids = [...meta.ids];
      this.vectors = new Float32Array(bin.buffer, bin.byteOffset, rows * this.dim);
      this.length = rows;
      this.capacity = Math.max(rows, 256);
      this.vectors = new Float32Array(this.vectors); // detach from the Buffer
      this.map = new Map();
      for (let r = 0; r < rows; r++) this.map.set(this.ids[r], r);
      return true;
    } catch {
      return false;
    }
  }
}
