/**
 * dsh-memory-bionic — shared low-level helpers (zero external dependencies).
 */
import { mkdirSync, renameSync, writeFileSync, readFileSync, existsSync, appendFileSync } from "node:fs";
import { dirname, join } from "node:path";

/** Stable FNV-1a 32-bit hash (unsigned). */
export function fnv1a(str, seed = 0x811c9dc5) {
  let h = seed >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** Signed hash in {-1, 1} — sign for hashing-trick embeddings. */
export function hashSign(str, salt = "") {
  return (fnv1a(str + "|" + salt) & 1) === 0 ? 1 : -1;
}

/** Dot product of two same-length Float32Arrays (vectors are L2-normalized). */
export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** L2-normalize a Float32Array in place; returns the same array. */
export function normalize(vec) {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < vec.length; i++) vec[i] /= norm;
  return vec;
}

/**
 * Workspace slug, mirroring the harness sessions-dir convention
 * (`--Users-macbookpro-Documents-Project--`) so memory scopes are stable
 * across sessions of the same workspace.
 */
export function slugifyWorkspace(cwd) {
  const cleaned = cwd.replace(/[^a-zA-Z0-9._-]/g, "-").replace(/-+/g, "-");
  return "--" + cleaned.replace(/^-|-$/g, "") + "--";
}

/** Atomic file write: temp file in the same dir, then rename. */
export function atomicWrite(file, data) {
  const dir = dirname(file);
  mkdirSync(dir, { recursive: true });
  const tmp = join(dir, "." + basenameSafe(file) + "." + process.pid + "." + Date.now() + ".tmp");
  writeFileSync(tmp, data);
  renameSync(tmp, file);
}

function basenameSafe(file) {
  const idx = file.lastIndexOf("/");
  return idx >= 0 ? file.slice(idx + 1) : file;
}

export function readJson(file, fallback = undefined) {
  if (!existsSync(file)) return fallback;
  try {
    return JSON.parse(readFileSync(file, "utf8"));
  } catch {
    return fallback;
  }
}

/** Append one line to a JSONL file (creates the file and dirs). */
export function appendJsonl(file, record) {
  mkdirSync(dirname(file), { recursive: true });
  appendFileSync(file, JSON.stringify(record) + "\n", "utf8");
}

/** Read a JSONL file into an array of records (missing file → []). */
export function readJsonl(file) {
  if (!existsSync(file)) return [];
  const out = [];
  for (const line of readFileSync(file, "utf8").split("\n")) {
    const t = line.trim();
    if (t.length === 0) continue;
    try {
      out.push(JSON.parse(t));
    } catch {
      /* skip corrupt line */
    }
  }
  return out;
}

/** Rough token estimate (heuristic; only used for trigger thresholds). */
export function estimateTokens(text) {
  let cjk = 0;
  let other = 0;
  for (const ch of text) {
    const code = ch.codePointAt(0);
    if (code > 0x2e80 && code < 0x9fff) cjk++;
    else if (!/\s/.test(ch)) other++;
  }
  return Math.ceil(cjk * 0.6 + other / 4);
}

/** Extract plain text from a message's content blocks. */
export function messageText(message) {
  if (!message) return "";
  const blocks = message.content;
  if (!Array.isArray(blocks)) return "";
  const parts = [];
  for (const block of blocks) {
    if (block && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join(" ");
}

/** Whether a user message was authored by the human (not a plugin/runtime snapshot). */
export function isHumanUserMessage(message) {
  if (!message || message.role !== "user") return false;
  const source = message.source;
  if (!source) return true;
  if (source.kind === "plugin") return false; // steering, runtime snapshots, checkpoints
  return true;
}

export function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

export function nowMs() {
  return Date.now();
}
