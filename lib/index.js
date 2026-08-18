/**
 * dsh-memory-bionic — 仿生记忆 + RAG 向量检索 (DeepSeek Harness profile plugin).
 *
 * 把智能体的"记忆"从"上下文窗口内保留 + 压缩摘要丢弃"升级为:
 *
 *   短期记忆 (工作记忆)   → 会话表层原文保留 (沿用 harness 自身机制)
 *   巩固 (consolidation)  → 每轮结束, 增量地把新会话跨度提炼为长期记忆条目
 *                          (LLM 结构化提炼; 失败回退原文分块, 内容不丢)
 *   长期记忆 (LTM 皮层)   → 按 workspace 持久化的记忆条目库 (items.jsonl)
 *   海马索引 (向量索引)   → 本地哈希 n-gram 嵌入 + 扁平余弦索引, 全内存检索
 *   回忆 (recall)         → 每步请求前, 用用户最新消息做 RAG 检索, 命中结果
 *                          作为运行时上下文快照注入 (systemPrompt.context)
 *   遗忘曲线 (Ebbinghaus) → 可访问性 = 相似度×衰减×重要性×复习强度;
 *                          不被复习的记忆在召回排序中下沉, 但永不删除 —
 *                          RAG 保证"海量记忆 + 快速回忆 + 不遗忘"
 *
 * 零外部依赖 (仅 Node 内置模块), 纯 ESM, 安装于 profile 的 node_modules。
 */
import { join } from "node:path";
import { homedir } from "node:os";
import { HashedNGramEmbedder } from "./embed.js";
import { MemoryStore } from "./memory-store.js";
import { Consolidator } from "./consolidate.js";
import { RecallProvider } from "./recall.js";
import { slugifyWorkspace } from "./util.js";

export const name = "dsh-memory-bionic";
export const inject = ["llm", "systemPrompt"];

const DEFAULTS = {
  enabled: true,
  storageRoot: "", // '' → $DSH_HOME/memory/<workspace-slug>
  topK: 6,
  minSimilarity: 0.24,
  captureMode: "llm", // 'llm' | 'raw'
  captureTokenThreshold: 6000,
  maxSpanChars: 24000,
  maxItems: 100000,
  dedupeSimilarity: 0.87,
  halfLifeHours: 168, // 遗忘曲线半衰期 (7 天)
  embedDim: 512,
  extractMaxTokens: 2048,
  recallOrder: 140
};

export function apply(ctx, rawConfig) {
  const config = { ...DEFAULTS, ...(rawConfig ?? {}) };
  if (config.enabled === false) return;

  const embedder = new HashedNGramEmbedder(config.embedDim);
  const stores = new Map(); // workspace cwd -> MemoryStore
  const home = () =>
    process.env.DSH_HOME && process.env.DSH_HOME.length > 0
      ? process.env.DSH_HOME
      : join(homedir(), ".dsh");

  /** Resolve (and lazily create) the shared store for one session's workspace. */
  const storeFor = (session) => {
    if (!session) return undefined;
    const cwd =
      session.header?.cwd ?? session.identity?.cwd ?? process.cwd();
    let store = stores.get(cwd);
    if (store === undefined) {
      const root =
        config.storageRoot && config.storageRoot.length > 0
          ? config.storageRoot
          : join(home(), "memory", slugifyWorkspace(cwd));
      store = new MemoryStore(root, embedder, config);
      stores.set(cwd, store);
      void store
        .load()
        .then(() => {
          const stats = store.stats();
          ctx.logger.info(
            `memory: long-term store ready (${root}) — ${stats.items} items, ${stats.vectors} vectors`
          );
        })
        .catch((error) => {
          ctx.logger.warn(
            `memory: store load failed: ${error instanceof Error ? error.message : String(error)}`
          );
        });
    }
    return store;
  };
  config.storeFor = storeFor;

  const consolidator = new Consolidator(ctx, config);
  const recall = new RecallProvider(config);

  // Capture + query tracking from the session event firehose.
  ctx.on("session/event", (session, event) => {
    try {
      switch (event.type) {
        case "agent/inbox/spliced":
          recall.handleInboxSplice(session, event);
          break;
        case "user/message":
          storeFor(session);
          recall.handleUserMessage(session, event);
          break;
        case "assistant/message":
        case "tool/result":
          storeFor(session);
          break;
        case "turn/end":
          storeFor(session);
          consolidator.maybeConsolidate(session);
          break;
        default:
          if (typeof event.type === "string" && event.type.startsWith("compaction/")) {
            storeFor(session);
            consolidator.maybeConsolidate(session, true);
          }
          break;
      }
    } catch (error) {
      ctx.logger.warn(
        `memory: session/event handler failed: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  });

  // Route consolidation LLM calls through the owning agent when no header exists yet.
  ctx.on("agent/created", ({ agent }) => {
    try {
      if (agent?.session?.id) consolidator.registerAgent(agent.session.id, agent);
    } catch {
      /* best effort */
    }
  });
  ctx.on("agent/disposed", ({ agent }) => {
    try {
      if (agent?.session?.id) consolidator.unregisterAgent(agent.session.id);
    } catch {
      /* best effort */
    }
  });

  // Recall injection: a dynamic runtime-context contribution evaluated per
  // request assembly. Renders the top-k accessible memories for the current
  // query; empty text contributes nothing to the snapshot.
  ctx.systemPrompt.context({
    name: "memory:bionic-recall",
    order: config.recallOrder,
    text: (context) => recall.render(context)
  });

  // Eagerly preload the store for the process workspace so the first request
  // already has recall available.
  ctx.on("ready", () => {
    try {
      storeFor({ header: { cwd: process.cwd() }, id: "preload" });
    } catch {
      /* best effort */
    }
  });

  ctx.on("dispose", () => {
    for (const store of stores.values()) {
      try {
        store.flush();
      } catch {
        /* best effort */
      }
    }
  });

  ctx.logger.info(
    `memory: dsh-memory-bionic active (mode=${config.captureMode}, topK=${config.topK}, threshold=${config.captureTokenThreshold} tokens, halfLife=${config.halfLifeHours}h)`
  );
}
