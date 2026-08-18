# dsh-memory-bionic — 仿生记忆 + RAG 向量检索

DeepSeek Harness 的长期记忆插件: 把智能体记忆从"上下文窗口内保留 + 压缩摘要丢失"
升级为 **仿生记忆 (短期→巩固→长期→回忆) + RAG 向量检索 (海量存储 + 语义召回)**。

## 仿生映射

| 人脑机制 | 本插件实现 |
|---|---|
| 感觉/工作记忆 | 会话表层原文 (沿用 harness 自身保留机制) |
| 短期记忆 (STM) | 未巩固的新会话跨度 (consolidation 缓冲) |
| 睡眠巩固 | 每轮结束 (turn/end) 增量把新跨度提炼为长期记忆条目: LLM 结构化提炼 (facts / decisions / preferences / error-fix / task / file-change / summary), 失败自动回退原文分块, **内容不丢** |
| 海马索引 | 本地哈希 n-gram 向量嵌入 + 扁平余弦索引 (全内存, 毫秒级检索) |
| 长期记忆皮层 | 按 workspace 持久化记忆条目库 (`items.jsonl` + 二进制向量 `index.bin`) |
| 回忆 (recall) | 每步请求前用用户最新消息做 RAG 检索, 按可访问性排序, 命中注入运行时上下文快照 (systemPrompt.context, 随话题变化自动更新) |
| 遗忘曲线 (Ebbinghaus) | 可访问性 = 0.55×相似度 + 0.25×时间衰减 + 0.20×重要性, 再乘复习强度 (0.5+0.5×strength); 不被复习的记忆在排序中下沉, **永不删除** |
| 间隔重复 (复习) | 每次成功召回即 touch 强化 (accessCount++, 时间戳刷新), 陈旧记忆可被复习"救活" |

## 为什么"海量 + 快速 + 不遗忘"

- **海量**: 记忆条目存在磁盘 (默认上限 100,000 条), 不受 1M token 上下文窗口约束;
  召回时只取 top-K (默认 6 条) 注入, 上下文成本固定。
- **快速**: 全部本地计算 — 无网络嵌入请求、无原生依赖; 5 万条向量精确余弦检索约几毫秒。
- **不遗忘**: 压缩/遗忘只影响"表层可见性", 记忆条目与向量永不因压缩丢失; 提炼失败有 raw
  回退; 捕获进度按会话 seq 增量推进, 压缩遮蔽的历史在被遮蔽前必已入册。

## 工作原理

1. **捕获**: 订阅 `session/event` 火鹤流, 增量追踪每个会话的未巩固跨度。
2. **巩固**: `turn/end` 时若新跨度 ≥ `captureTokenThreshold` (默认 6000 tokens),
   后台调用 `ctx.llm.stream()` 做一次结构化提炼 (直连, 不经过 agent loop),
   输出 JSON 数组 → 逐条嵌入、去重、入库; 失败回退 raw 分块。
3. **检索**: `agent/pre-step` 组装请求前, 用用户最新消息 (经 `agent/inbox/spliced`
   提前捕获) 做向量检索, 按可访问性排序取 top-K。
4. **注入**: 作为 `systemPrompt.context` 贡献 ("memory:bionic-recall", order 140)
   进入"Current runtime context"快照 — 不污染会话日志 (快照机制自带新旧取代语义),
   也不影响压缩/计费之外的开销 (topK 受控)。
5. **复习**: 查询变化时对命中条目 touch — 间隔重复。

## 存储布局

默认 `$DSH_HOME/memory/<workspace-slug>/` (可用 `storageRoot` 覆盖):

- `items.jsonl` — 追加式操作日志 (add / touch / remove), 可重放
- `index.bin` + `index.json` — 二进制 float32 向量 + id 清单 (索引缺失/失配时自动从文本重建)
- `meta.json` — 每会话捕获进度

同一 workspace 的多个会话共享一个长期记忆库 — 跨会话"记得"。

## 配置 (cordis.patch.yml)

```yaml
- insert:
    - id: memory-bionic
      name: 'dsh-memory-bionic'
      config:
        enabled: true          # 总开关
        storageRoot: ''        # 默认 $DSH_HOME/memory/<workspace-slug>
        topK: 6                # 每次回忆注入的条目数
        minSimilarity: 0.24    # 召回最低余弦相似度
        captureMode: llm       # 'llm' 结构化提炼 | 'raw' 仅分块
        captureTokenThreshold: 6000  # 触发巩固的新跨度 token 阈值
        maxSpanChars: 24000    # 单次提炼跨度字符上限
        maxItems: 100000       # 记忆条目上限 (到达后淘汰最低可访问性条目)
        dedupeSimilarity: 0.87 # 新条目与已有记忆的余弦去重阈值
        halfLifeHours: 168     # 遗忘曲线半衰期 (7 天)
        embedDim: 512          # 向量维度
        extractMaxTokens: 2048 # 提炼 LLM 调用输出上限
```

## 验证

```sh
# 组合配置里应出现 memory-bionic 行
dsh --profile web --dump-config | grep memory-bionic
# 重启后日志出现:
#   memory: dsh-memory-bionic active (mode=llm, topK=6, ...)
#   memory: long-term store ready ($DSH_HOME/memory/...) — N items, N vectors
#   memory: consolidated X tokens (seqs a-b) → N items total
```

## 扩展点

- **嵌入器可替换**: `lib/embed.js` 的 `HashedNGramEmbedder` 实现
  `embed(text) → Float32Array(dim)` + `dim` 接口; 换用神经网络嵌入 (API 或本地模型)
  无需改动存储与召回代码。
- **索引可替换**: `lib/vector-store.js` 的扁平余弦索引可换 HNSW/IVF 等 ANN。
- **提炼可替换**: `lib/consolidate.js` 的 `extractWithLlm` 是唯一提炼钩子。
