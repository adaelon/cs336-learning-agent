# CS336 Lecture 01 Phase 2: Educator 认知脚手架

## 0. 全局知识树

**本讲的知识树只有一个根：如何在有限资源下，从原始文本构建一个可训练、可扩展、可使用的 language model。**

```text
CS336 Lecture 01
├── 为什么学 from scratch
│   ├── API 抽象提高生产力
│   ├── 但 LM 抽象会泄漏
│   └── fundamental research 需要拆开 stack
├── 效率总原则
│   ├── accuracy = efficiency x resources
│   ├── resources = data + hardware
│   └── hardware = compute + memory + communication bandwidth
├── LM 历史地图
│   ├── pre-neural: entropy / n-gram
│   ├── neural ingredients: LSTM / seq2seq / Adam / attention / Transformer / MoE / parallelism
│   ├── foundation models: ELMo / BERT / T5
│   ├── scaling: GPT-2 / GPT-3 / PaLM / Chinchilla
│   ├── open models: Llama / Mistral / DeepSeek / Qwen / OLMo / Nemotron / Marin
│   └── interface shift: fine-tune -> prompt -> talk -> act
├── 课程五个工程面
│   ├── basics: tokenization / architecture / training
│   ├── systems: kernels / parallelism / inference
│   ├── scaling laws: small runs predict large runs
│   ├── data: evaluation / curation / processing / mixing
│   └── alignment: weak supervision / preferences / RL systems
└── 本讲展开的第一个技术单元
    ├── Tokenizer interface: encode / decode
    ├── bad baselines: character / byte / word
    └── BPE: byte start + frequent pair merges
```

---

## 1. 课程为什么强调 From Scratch

### 1.1 溯源与关联拓扑

**From scratch 不是怀旧，而是为了重新暴露 API 背后被隐藏的 data、model、training、systems 决策。**

这个概念挂在“leaky abstraction”上。普通软件抽象通常能把底层藏起来，用户不必知道 CPU cache 也能写业务代码；但 LM 抽象会把训练数据偏差、context length、inference latency、alignment failure 等底层因素泄漏到最终行为中。

### 1.2 直观类比与极简案例

**把 API model 当作黑盒，就像只会开车但要研究发动机燃烧效率：能用，不等于能解释性能极限。**

极简案例：

```text
任务：解释一个 model 为什么在长 prompt 后变慢
只会 prompt：只能说“它慢了”
懂系统：会追到 sequence length、attention cost、KV cache、decode memory bandwidth
```

### 1.3 差异鉴别

| 概念 | 处理对象 | 目标 | 风险 |
|---|---|---|---|
| 使用 API | prompt 和 response | 快速得到结果 | 底层失败不可解释 |
| Fine-tuning | pretrained weights + task data | 改善某个任务 | 仍依赖已有架构和训练 recipe |
| From scratch | tokenizer 到 alignment 全栈 | 建立机制理解 | 成本高、工作量大 |

### 1.4 认知陷阱

**最容易误解的是：from scratch 不是要复现 frontier model，而是复现能迁移的 mechanics 和 mindset。**

讲义明确区分三类知识：

- Mechanics：Transformer、model parallelism 等如何工作。
- Mindset：压榨 hardware、认真对待 scaling。
- Intuitions：哪些 data/modeling decision 带来 accuracy，只有部分可迁移。

---

## 2. 效率总原则

### 2.1 溯源与关联拓扑

**效率是所有模块的共同父概念；tokenizer、architecture、systems、data、scaling、alignment 都是在不同位置减少浪费。**

公式 `accuracy = efficiency x resources` 给出课程主线。resources 包括 data 与 hardware；hardware 又拆成 compute、memory、communication bandwidth。后续每个 assignment 都是在问同一个问题：固定预算下，哪个决策最不浪费？

### 2.2 直观类比与极简案例

**把训练预算想成一张只能花一次的账本：每个低质量 token、每次多余 HBM 读写、每个过长 sequence 都在扣钱。**

极简案例：

```text
同样 45 分钟 B200 训练预算
方案 A：byte tokenizer，sequence 很长，attention FLOPs 高
方案 B：BPE tokenizer，sequence 更短，但 vocabulary 更大
课程关心：哪个方案在固定时间内得到更低 perplexity
```

### 2.3 差异鉴别

| 概念 | 看起来像 | 实际差异 |
|---|---|---|
| 资源更多 | 买更多 GPU / 更多 data | 不一定能补救低效算法 |
| 效率更高 | 同样资源下更少浪费 | frontier scale 下尤其关键 |
| 直觉更好 | 经验判断某 recipe 好 | 可能不跨 scale 迁移 |

### 2.4 认知陷阱

**不要把 bitter lesson 理解成“algorithm 不重要”；讲义给出的正确读法是“能 scale 的 algorithm 才重要”。**

如果一个技术在小规模漂亮，但在长 context、分布式训练或 inference decode 阶段浪费严重，它就不符合课程的效率主线。

---

## 3. LM 历史地图

### 3.1 溯源与关联拓扑

**历史部分不是论文清单，而是在告诉你今天的 LM 栈由哪些压力逐步塑形。**

从 n-gram 到 Transformer，主要压力是表达能力；从 GPT-2 到 Chinchilla，主要压力是 scale predictability；从 OPT/BLOOM/Llama 到 OLMo/Marin，主要压力是 openness；从 BERT 到 agents，主要压力是交互接口变化。

### 3.2 直观类比与极简案例

**可以把 LM 历史看成同一台机器不断增加三个旋钮：参数、数据、系统并行。旋钮变大后，旧的调参方式会失效。**

极简时间线：

```text
2018 BERT: model 是拿来 fine-tune 的东西
2020 GPT-3: model 是拿来 prompt 的东西
2022 ChatGPT: model 是拿来对话的东西
2026 agents: model 是能自主行动的系统部件
```

### 3.3 差异鉴别

| 模型/阶段 | 核心信号 | 对课程的意义 |
|---|---|---|
| GPT-3 | in-context learning, 175B | scale 改变使用方式 |
| PaLM | 540B, MFU, large TPU system | architecture 与 systems 绑定 |
| Chinchilla | 70B + 更多 data | compute-optimal allocation |
| Llama/Mistral/DeepSeek | open-weight recipes | 可研究、可教学 |
| OLMo/Marin | 更完整 open artifacts | 支撑透明科学 |

### 3.4 认知陷阱

**不要把 open model 只理解成“免费权重”；讲义强调真正有教学价值的是 recipe、data、code、logs、checkpoints 等构建证据链。**

只释放 weights 能使用模型，但很难回答“为什么这样训练、哪里失败、如何复现”。

---

## 4. 五个 Assignment 的系统地图

### 4.1 溯源与关联拓扑

**五个 assignment 是完整 LM 生命周期的切面，不是五个孤立作业。**

```text
Assignment 1 basics: 造出能训练的小 LM
Assignment 2 systems: 让同一计算更高效地跑
Assignment 3 scaling: 用小实验预测大实验
Assignment 4 data: 让 token stream 更有价值
Assignment 5 alignment: 用 weak supervision 改善行为
```

### 4.2 直观类比与极简案例

**训练 LM 像建工厂：basics 是产品设计，systems 是生产线，scaling laws 是产能规划，data 是原材料，alignment 是出厂调校。**

极简依赖：

```text
没有 tokenizer -> 没有 token IDs
没有 training loop -> 没有 base model
没有 systems -> 预算内跑不动
没有 scaling -> 大规模决策靠猜
没有 data work -> compute 花在垃圾 token 上
没有 alignment -> 模型不一定按用户意图行动
```

### 4.3 差异鉴别

| 模块 | 主要问题 | 典型失败 |
|---|---|---|
| Basics | 模型是否能学 | loss 不降、表达不足、序列太长 |
| Systems | 计算是否跑满硬件 | memory-bound、通信慢、kernel overhead |
| Scaling laws | 大 run 前能否预测 | full-scale 调参浪费巨大 |
| Data | token 是否有用 | contamination、重复、低质量、harmful |
| Alignment | 行为是否符合偏好 | RL 不稳、on-policy 与效率冲突 |

### 4.4 认知陷阱

**不要把 systems 当成“训练好以后再优化速度”的附属模块；在 CS336 中，systems 从一开始就决定哪些模型设计可行。**

例如 tokenization 决定 sequence length，sequence length 直接影响 attention cost；这不是后处理优化能完全补救的。

---

## 5. Tokenizer Interface

### 5.1 溯源与关联拓扑

**Tokenizer 是 raw text 与 model tensor world 的入口适配器。**

**Tokenizer (分词器)**: 一个在 raw string 与 integer token sequence 之间双向转换的接口；它把人类可读文本变成模型能索引 embedding 的整数序列。

父概念是 sequence modeling：model 不处理“字符串意义”，只处理 token IDs。Tokenizer 的质量直接影响 sequence length、vocabulary size、rare pattern 的表示方式。

### 5.2 直观类比与极简案例

**Tokenizer 像把连续文本切成积木编号；模型只看到积木编号，不直接看到原始文字。**

极简案例：

```text
string = "Hello!"
encode(string) -> [token_id_1, token_id_2]
model 只处理 [token_id_1, token_id_2]
decode([token_id_1, token_id_2]) -> "Hello!"
```

关键检查是 roundtrip：

```text
decode(encode(s)) == s
```

### 5.3 差异鉴别

| 概念 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `encode` | string | `list[int]` | 进入模型前 |
| `decode` | `list[int]` | string | 模型输出后 |
| `compression_ratio` | string + token IDs | bytes/token | 衡量 sequence 压缩 |

### 5.4 认知陷阱

**Roundtrip 正确只说明 tokenizer 不丢信息，不说明它适合 Transformer。**

CharacterTokenizer、ByteTokenizer 都能 roundtrip，但一个 vocabulary 大且 rare character 低效，另一个 sequence 长且 attention 成本高。

---

## 6. Character / Byte / Word 三个 Baseline

### 6.1 溯源与关联拓扑

**这三个 baseline 是为了把 tokenizer 的 trade-off 暴露出来：没有哪个朴素切分同时满足小词表、短序列、开放词表。**

它们都挂在 Tokenizer interface 下，区别只是 chunk 的粒度：

```text
character: Unicode character
byte: UTF-8 byte
word: regex word/chunk
```

### 6.2 直观类比与极简案例

**Character 是按字母装箱，Byte 是按最小零件装箱，Word 是按完整单词装箱；BPE 后面会学会常见零件组合。**

极简案例：

```text
"🌍"
CharacterTokenizer: 一个 code point，但 Unicode 词表约 150K
ByteTokenizer: 多个 UTF-8 bytes，但词表只有 256
WordTokenizer: 如果从未在训练中出现，可能变成 UNK
```

### 6.3 差异鉴别

| Tokenizer | 优点 | 缺点 | 讲义结论 |
|---|---|---|---|
| Character | 直观，`ord/chr` 简单 | vocabulary 很大，rare characters 低效，compression 低 | worst of both worlds |
| Byte | vocabulary 固定为 256，开放 | compression ratio = 1，sequence 太长 | context/attention 成本差 |
| Word | token 有人类语义，compression 好 | vocabulary 巨大，OOV/UNK，固定词表困难 | classical 但不适合目标 |

### 6.4 认知陷阱

**不要把“token 有语义”当作唯一目标；LM 训练还要考虑固定 vocabulary、rare words、perplexity 与 attention 成本。**

WordTokenizer 看起来最像人类阅读，但对模型训练来说，OOV 和巨大 vocabulary 会制造工程问题。

---

## 7. BPE

### 7.1 溯源与关联拓扑

**BPE 是 ByteTokenizer 与 WordTokenizer 之间的折中：从 byte 的开放性出发，学习常见 subword 的压缩。**

**Byte Pair Encoding (BPE, 字节对编码)**: 从 byte token 开始，反复把训练语料里最常见的相邻 token pair 合并成新 token，从而得到 data-driven vocabulary。

拓扑关系：

```text
ByteTokenizer 提供安全起点
WordTokenizer 提供“常见 chunk 应该更短”的直觉
BPE = byte 起点 + 训练得到的 merge rules
```

### 7.2 直观类比与极简案例

**BPE 像给常见短语发快捷键：常见组合一个键输入，罕见组合仍然能逐字节拼出来。**

极简训练例子来自讲义：

```text
string = "the cat in the hat"
start: UTF-8 byte IDs
repeat 3 times:
  count adjacent pairs
  merge most common pair into token 256+i
result: vocab + merges
```

极简使用：

```text
params = train_bpe("the cat in the hat", num_merges=3)
tokenizer = BPETokenizer(params)
encode("the quick brown fox") -> token IDs
decode(token IDs) -> original string
```

### 7.3 差异鉴别

| 对比项 | BPE | WordTokenizer |
|---|---|---|
| 起点 | bytes | words/chunks |
| 未见词 | 拆成 byte/subword | 可能 UNK |
| vocabulary size | 由 `256 + num_merges` 控制 | 由训练数据 distinct chunks 决定 |
| 压缩来源 | 数据中 frequent adjacent pairs | 人类定义的 word boundary |

| 对比项 | BPE | ByteTokenizer |
|---|---|---|
| vocabulary | 更大 | 256 |
| sequence length | 更短 | 等于 byte length |
| 训练步骤 | 需要 train merges | 不需要 |
| 稀有字符串 | 可回退为多个 byte/subword | 全部 byte |

### 7.4 认知陷阱

**BPE 不是语义理解算法；它是一个频率驱动的压缩 heuristic。**

最反直觉的一点是：BPE 可能把空格和后面的 word 合成一个 token，或者让句首 word 与句中 word token 不同。这不是 bug，而是因为 tokenizer 学到的是 byte sequence 的统计频率，不是人类词典。

---

## 8. BPE 代码心智模型

### 8.1 溯源与关联拓扑

**BPE 的代码可以拆成两个阶段：训练 params，使用 params。**

训练阶段输出：

```python
vocab: dict[int, bytes]              # index -> bytes
merges: dict[tuple[int, int], int]   # adjacent pair -> new index
```

使用阶段只依赖这两个对象。

### 8.2 直观类比与极简案例

**`vocab` 是字典，`merges` 是压缩规则表；encode 负责查规则压缩，decode 负责查字典展开。**

toy flow：

```text
初始 vocab:
  116 -> b"t"
  104 -> b"h"
  101 -> b"e"

某次 merge:
  pair = (116, 104)
  new_index = 256
  merges[(116, 104)] = 256
  vocab[256] = b"t" + b"h" = b"th"
```

之后 sequence 中相邻的 `[116, 104]` 会被替换成 `[256]`。

### 8.3 差异鉴别

| 函数 | 做什么 | 容易混淆点 |
|---|---|---|
| `count_adjacent_pairs` | 统计当前 sequence 的相邻 pair 次数 | 统计的是 token pair，不一定是原始 byte pair |
| `merge` | left-to-right 替换指定 pair | 每轮只替换当前选中的 pair |
| `train_bpe` | 反复 count -> max -> merge | merges 的顺序会影响 encode |
| `BPETokenizer.encode` | 对新 string 应用 learned merges | 讲义版本很慢，loop over all merges |
| `BPETokenizer.decode` | token IDs -> bytes -> string | decode 依赖 vocab，不重新运行 merges |

### 8.4 认知陷阱

**不要以为 BPE encode 会重新统计新文本的 pair；统计只发生在 tokenizer training。**

使用 tokenizer 时，merge rules 已固定。新文本只是按已有规则被压缩，否则同一个模型的 token vocabulary 会随输入变化，embedding 接口就不稳定。

---

## 9. Tokenization 与 Efficiency 的连接

### 9.1 溯源与关联拓扑

**Tokenization 是 lecture 中第一个具体展示“效率不是系统课才有”的模块。**

它挂在 efficiency 主线下，因为 sequence length 直接决定 Transformer attention 成本。讲义明确说，更高 compression ratio 会让 sequence 更短，而 attention 对 sequence length 是 quadratic。

### 9.2 直观类比与极简案例

**如果 attention 是两两比较，那么 token 越多，比较次数增长越快。**

极简数值：

```text
1000 bytes 直接进模型 -> 约 1000 个位置
BPE 压成约 250 tokens -> 约 250 个位置
两两 attention 的位置对数量从 1000^2 变为 250^2
```

这只是讲义中的效率直觉，不代表所有成本都只由 attention 决定；但它解释了为什么 raw byte 虽优雅却 compute-inefficient。

### 9.3 差异鉴别

| 目标 | 增大 vocabulary 的效果 | 风险 |
|---|---|---|
| 更短 sequence | 通常提高 compression ratio | embedding/head 更大，token 更稀疏 |
| 更开放表示 | byte 起点保证可表示任意 UTF-8 | sequence 可能变长 |
| 更语义 chunk | word/subword 更像人类单位 | OOV 或统计偏差 |

### 9.4 认知陷阱

**不要把 tokenizer-free 当成已经替代 BPE 的事实；讲义只说它是 dream，相关方法 promising，但尚未 scaled up to frontier。**

ByT5、MEGABYTE、BLT、T-FREE、H-Net 都是在探索 raw bytes 或动态 chunks，但本课程 assignment 仍要求实现 BPE。

---

## 10. Scaling Laws 的学习位置

### 10.1 溯源与关联拓扑

**Scaling laws 是把“训练大模型靠直觉”改成“先在小预算上建立预测关系”。**

它依赖 basics 和 systems，因为没有稳定 training API 与可控 resource accounting，就无法比较不同 FLOPs budgets 下的 loss。

### 10.2 直观类比与极简案例

**Scaling recipe 像风洞实验：不能每次都造全尺寸飞机，所以先用小模型测规律，再外推大模型。**

极简例子：

```text
目标预算: 1e25 FLOPs
直接调参: 太贵
课程方法:
  在 1e24 FLOPs 以下跑多个设置
  fit scaling law
  predict 1e25 FLOPs 下的 loss
```

### 10.3 差异鉴别

| 概念 | 目标 |
|---|---|
| Hyperparameter tuning | 找某个 scale 的好设置 |
| Hyperparameter transfer | 小 scale 的设置能迁移到大 scale |
| Scaling recipe | FLOPs -> hyperparameters 的函数 |
| Scaling law | 预测 recipe 在目标 scale 的 loss |

### 10.4 认知陷阱

**不要把 Chinchilla 的 `D ≈ 20N` 当成无条件定律；讲义提醒它没有计入 inference cost。**

如果 serving 成本很重要，更小模型训练更多 tokens 可能更有吸引力；课程会把 training 与 inference 放在同一资源视角下权衡。

---

## 11. Data 与 Alignment 的学习位置

### 11.1 溯源与关联拓扑

**Data 决定模型学什么，alignment 决定已有模型如何偏向更可用的行为。**

Data 在 pretraining 前后都影响 token stream；alignment 在 base model 已经 reasonable 后，用 weak supervision 进一步改造行为。

### 11.2 直观类比与极简案例

**Data 像原材料筛选，alignment 像成品调校。原材料差会让训练预算浪费，调校差会让可用性下降。**

极简流程：

```text
Raw web HTML -> text extraction -> filtering -> deduplication -> mixture -> pretraining tokens
Base model -> generate responses -> score responses -> update toward preferred responses
```

### 11.3 差异鉴别

| 模块 | 信号来源 | 主要风险 |
|---|---|---|
| Pretraining data | raw documents | 低质量、重复、污染、版权风险 |
| Evaluation | private docs / benchmark tasks | contamination 或不符合真实 use case |
| Alignment | human/verifier/LM judge scores | RL 不稳、rollout infrastructure 复杂 |
| DPO/GRPO | preference 或 group-relative signal | 本讲只定位，不展开公式 |

### 11.4 认知陷阱

**不要认为 alignment 可以弥补所有 pretraining/data 问题；讲义的顺序是先有 reasonable model，再用 weak supervision 改进。**

如果 base model 对语言、知识、代码或长上下文没有基本能力，alignment 阶段只能在有限范围内调整偏好。

---

## 12. 本讲应留下的核心心智模型

**Lecture 01 最重要的记忆不是“列出很多模型名”，而是建立一个从资源到接口的因果链。**

```text
资源有限
-> 必须减少浪费
-> tokenizer 减少 sequence length
-> architecture 平衡 expressivity/stability/efficiency
-> systems 减少 data movement
-> scaling laws 减少 full-scale 试错
-> data pipeline 提高 token utility
-> alignment 用更弱但更便宜的 critique signal 改善行为
```

最终落点：

- 能 roundtrip 的 tokenizer 不一定好。
- BPE 是 frequency-driven compression heuristic，不是语义词典。
- Open models 的价值在于暴露 recipe，使 mechanics 和 mindset 可学习。
- CS336 的所有模块都围绕同一个问题：给定固定 resources，如何最大化 useful model quality。

全局护栏：本产物只基于 `lecture/lecture_01.md` 和本地 deconstruction SOP，没有使用外部搜索或外部教程。
