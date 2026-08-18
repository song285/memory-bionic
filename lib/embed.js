/**
 * dsh-memory-bionic — local deterministic embeddings (hashed character n-grams).
 *
 * Zero-dependency "hashing trick" embedder: tokenizes text into character
 * n-grams (2/3/4-grams — strong for CJK and code) plus latin word unigrams,
 * hashes each feature into a fixed-dimension dense vector with signed weights
 * and TF-style magnitude, then L2-normalizes. Cosine similarity over these
 * vectors is a solid lexical/semantic proxy for retrieval and needs no
 * network, model weights, or native modules — everything stays local and
 * deterministic (same text → same vector on every machine).
 *
 * The embedder is a swappable seam: any object exposing
 * `embed(text) -> Float32Array(dim)` and a `dim` property can replace this
 * one (e.g. a neural embedding API) without touching the store or recall code.
 */
import { fnv1a, hashSign, normalize } from "./util.js";

export class HashedNGramEmbedder {
  dim;
  constructor(dim = 512) {
    this.dim = dim;
  }

  /** Normalize text for stable feature extraction. */
  normalize(text) {
    return String(text).normalize("NFKC").toLowerCase();
  }

  /** Extract weighted character n-gram + word features from normalized text. */
  features(text) {
    const counts = new Map();
    const add = (gram, weight = 1) => counts.set(gram, (counts.get(gram) ?? 0) + weight);
    const chars = [...text];
    const n = chars.length;
    for (let i = 0; i < n; i++) {
      add(chars[i]);
      for (const len of [2, 3, 4]) {
        if (i + len <= n) add(chars.slice(i, i + len).join(""));
      }
    }
    // Latin word unigrams + bigrams add topic-level signal.
    const words = text.match(/[a-z0-9_]{3,}/g);
    if (words) {
      for (const word of words) add("w:" + word, 2);
      for (let i = 0; i + 1 < words.length; i++) {
        add("wb:" + words[i] + " " + words[i + 1], 3);
      }
    }
    // CJK bigram/trigram boundaries are already covered by char n-grams above.
    return counts;
  }

  /** Embed text into an L2-normalized dense vector. */
  embed(text) {
    const dim = this.dim;
    const vec = new Float32Array(dim);
    const counts = this.features(this.normalize(text));
    for (const [gram, count] of counts) {
      const idx = fnv1a(gram) % dim;
      const sign = hashSign(gram);
      vec[idx] += sign * (1 + 0.5 * Math.log1p(count));
    }
    return normalize(vec);
  }

  /** Batch-embed (helper for index rebuilds). */
  embedAll(texts) {
    return texts.map((t) => this.embed(t));
  }
}
