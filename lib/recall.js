/**
 * dsh-memory-bionic — recall (回忆): RAG 向量检索 → 工作记忆注入.
 *
 * Bio-mapping: 回忆即海马体按线索(query)检索皮质痕迹, 再把命中结果带进
 * 工作记忆。这里的"工作记忆"是每步请求前的运行时上下文快照
 * (systemPrompt.context 贡献), 随话题变化自动更新、旧快照被新快照取代。
 *
 * 查询线索来自用户最新消息 (通过 agent/inbox/spliced 事件提前捕获,
 * 以及 user/message 兜底), 检索按 accessibility (相似度 × 遗忘曲线 ×
 * 重要性 × 复习强度) 排序, 命中即复习 (touch), 实现间隔重复。
 */
import { messageText, isHumanUserMessage } from "./util.js";

const QUERY_MAX_CHARS = 1200;

export class RecallProvider {
  constructor(config) {
    this.config = config;
    this.pendingQueries = new Map(); // sessionId -> { text, at }
    this.lastQueryKeys = new Map(); // sessionId -> key (rehearsal dedup)
  }

  /** New user input entered the inbox (fires before the loop's pre-step). */
  handleInboxSplice(session, event) {
    const data = event.data;
    if (!data || data.target !== "next-turn") return;
    const inserted = Array.isArray(data.inserted) ? data.inserted : [];
    for (const message of inserted) {
      if (!message || message.role !== "user") continue;
      const text = messageText(message).trim();
      if (text.length >= 2) {
        this.pendingQueries.set(session.id, { text, at: Date.now() });
      }
    }
  }

  /** Fallback: a user-authored message was appended to the session. */
  handleUserMessage(session, event) {
    if (!isHumanUserMessage(event.data)) return;
    const text = messageText(event.data).trim();
    if (text.length >= 2) {
      this.pendingQueries.set(session.id, { text, at: Date.now() });
    }
  }

  /** Render the recalled-memories context contribution (synchronous). */
  render(context) {
    const agent = context?.agent;
    if (!agent) return "";
    const store = this.config.storeFor(agent.session);
    if (!store || !store.ready) return "";
    const sessionId = agent.session.id;
    const pending = this.pendingQueries.get(sessionId);
    const query = (pending?.text ?? this.lastUserText(agent.session)).slice(0, QUERY_MAX_CHARS);
    if (query.length < 2) return "";
    const results = store.search(query, this.config.topK, this.config.minSimilarity);
    if (results.length === 0) return "";
    const now = Date.now();
    const lines = [
      "Recalled long-term memories (retrieved by semantic relevance; use them when they bear on the current task, ignore otherwise):"
    ];
    for (const { item, similarity, accessibility } of results) {
      const ageDays = Math.max(0, (now - item.createdAt) / 86400000).toFixed(1);
      const rehearsals = item.accessCount;
      lines.push(
        `- [${item.kind} | rel ${similarity.toFixed(2)} | importance ${item.importance.toFixed(1)} | ${ageDays}d old | rehearsed ${rehearsals}×] ${item.text}`
      );
    }
    // Rehearsal: only when the query actually changed (avoid inflating on repeats).
    const key = query + "|" + results.map((r) => r.item.id).join(",");
    if (this.lastQueryKeys.get(sessionId) !== key) {
      this.lastQueryKeys.set(sessionId, key);
      for (const { item } of results) store.touch(item.id);
    }
    return lines.join("\n");
  }

  /** Last substantive user text already in the session (fallback query). */
  lastUserText(session) {
    try {
      const messages = session.deriveMessages();
      for (let i = messages.length - 1; i >= 0; i--) {
        const message = messages[i];
        if (message && message.role === "user" && isHumanUserMessage(message)) {
          const text = messageText(message).trim();
          if (text.length >= 2) return text;
        }
      }
    } catch {
      /* session may be mid-rewrite; fall through */
    }
    return "";
  }
}
