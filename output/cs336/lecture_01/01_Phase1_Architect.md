# CS336 Lecture 01 Phase 1: Architect 技术结构拆解

## 0. 全局定位

**本讲不是从某个模型公式开始，而是在建立 CS336 的总约束：在固定 data 与 hardware 资源下，训练出尽可能好的 language model。**

Lecture 01 的结构是三层递进：

1. 课程为什么存在：研究者正在远离底层技术，但 frontier model 的抽象仍然是 leaky abstraction。
2. 现代 LM 版图如何形成：从 entropy / n-gram，到 neural LM / Transformer，到 scaling laws、open models、agents。
3. 本课程如何拆解 LM：basics、systems、scaling laws、data、alignment 五个 assignment 对应完整训练栈；本讲最后实际进入第一个技术单元 Tokenization。

历史状态：当前为 `lecture_01`，无前序 lecture 产物输入。

---

## 1. 课程核心目标：Understanding via Building

### 1.1 技术路线与演进逻辑

**课程的起点是“抽象层上移造成理解断层”，因此教学路线选择从 scratch 重建 LM 栈。**

| 阶段 | 研究者默认工作方式 | 抽象层变化 | 讲义指出的风险 |
|---|---|---|---|
| 2016 | 自己实现并训练模型 | 直接接触模型与训练 | 底层细节可见 |
| 2018 | 下载 BERT 等模型并 fine-tune | 模型变成可复用基座 | 训练细节开始隐藏 |
| Today | prompt API models | 模型变成远程接口 | 研究者可能只接触行为，不接触机制 |

前置基线是 API / pretrained model 的高层使用方式。核心崩溃点是这些抽象不像编程语言或 OS 那样稳定封装，模型行为、训练数据、系统瓶颈、alignment 机制都可能泄漏到研究问题本身。破局机制是用 assignment 重新实现关键部件，而不是只背诵论文结论。

### 1.2 系统设计与资源权衡链

**课程的统一资源模型是 `accuracy = efficiency x resources`：资源昂贵且不透明，因此效率不是优化细节，而是核心目标。**

讲义给出两个约束：

- Frontier models 不可直接复现：GPT-4 训练成本被描述为约 **$100M**，xAI 训练集群被描述为 **230K GPUs**。
- Small models 可训练，但不一定代表 large models：attention 与 MLP 的 FLOPs 占比会随规模变化，能力也可能随规模出现 emergence。

资源置换：

| 资源维度 | 讲义中的体现 | 设计后果 |
|---|---|---|
| Compute | 训练 frontier models 极贵；assignment 设 leaderboard 与时间预算 | 所有设计都要问是否浪费 FLOPs |
| Memory | 后续 systems 关注参数、activation、gradient、optimizer state | 模型结构和并行策略要配合显存 |
| Bandwidth | GPU HBM、GPU 间通信会成为瓶颈 | kernel fusion、parallelism、inference batching 成为课程核心 |
| Data | 当前先 compute-constrained，后续会转向 data-constrained | scaling laws 与 data curation 都服务于预算分配 |

### 1.3 数学原则与推导链

**本模块属于课程哲学与资源框架，讲义未给出可推导的模型公式；唯一显式公式是效率框架。**

公式：

$$
\text{accuracy} = \text{efficiency} \times \text{resources}
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $\text{accuracy}$ | 给定任务或评估上的模型质量 |
| $\text{efficiency}$ | 同样资源下更好地利用算法、架构、数据和系统的能力 |
| $\text{resources}$ | data 与 hardware，包括 compute、memory、communication bandwidth |

该公式不是精确物理定律，而是课程的设计准则：当 resources 无法无限扩张时，必须把 tokenizer、architecture、training、systems、data、alignment 都看成效率问题。

### 1.4 系统增量与接口绑定

**Lecture 01 建立的是后续所有模块的接口契约：每个组件都要说明自己如何消耗资源、改变序列或张量、影响训练质量。**

本讲引出的系统输入输出链：

```text
raw data -> tokenizer -> token IDs -> model architecture -> loss/training -> trained model
          -> systems acceleration / parallelism -> scaling prediction -> data iteration -> alignment
```

---

## 2. 语言模型历史版图

### 2.1 技术路线与演进逻辑

**LM 的历史路线在本讲中被组织为“表示能力、规模、开放性和交互形态”的连续扩张。**

| 时期 | 代表内容 | 关键转折 |
|---|---|---|
| Pre-neural | Shannon entropy, n-gram LM, 5-gram on 2T tokens | 用统计方式预测语言 |
| Neural ingredients | LSTM, first neural LM, seq2seq, Adam, attention, Transformer, MoE, model parallelism | 学习表示与可扩展训练成为中心 |
| Early foundation models | ELMo, BERT, T5 | pretraining + fine-tuning 成为范式 |
| Embracing scaling | GPT-2, scaling laws, GPT-3, PaLM, Chinchilla | scale 与 compute-optimal allocation 成为可预测工程问题 |
| Open models | The Pile/GPT-J, OPT, BLOOM, Llama, Mistral, DeepSeek, Qwen, Kimi, GLM, OLMo, Nemotron, Marin | 开放权重、论文、代码、数据让课程可教学 |
| Interaction shift | BERT -> GPT-3 -> ChatGPT -> agents | LM 从 fine-tune 对象变成 prompt/talk/act 的系统 |

前置基线是 n-gram 和早期 supervised NLP。核心崩溃点是固定特征与小规模任务系统无法吸收 web-scale data，也无法支撑通用能力。破局机制依次是 neural representations、attention/Transformer、scaling laws、systems parallelism、open recipes。

### 2.2 系统设计与资源权衡链

**历史版图中的每次进展都对应一种资源重新分配：更多参数、更多 tokens、更强硬件、更高系统复杂度，换取更通用的能力。**

讲义中的关键资源事实：

- Brants 5-gram 使用 **2T tokens**。
- GPT-2 为 **1.5B parameters**。
- GPT-3 为 **175B parameters**、**300B tokens**。
- PaLM 为 **540B parameters**、**6144 TPU v4**、**46.2% MFU**。
- Chinchilla 为 **70B parameters**、MassiveText **1.5T tokens**，强调 model 与 data 同速扩张。
- Llama 1: **7B-65B**，Llama 2: **2T tokens / 70B**，Llama 3: **15T tokens / 405B**。
- OLMo 提供 weights、training data、code、recipes、logs、checkpoints 等更完整开放链路。

### 2.3 数学原则与推导链

**历史模块中的主要数学线索是 compute-optimal scaling：在固定训练 FLOPs 下决定 parameters 与 tokens 的分配。**

讲义显式给出训练 compute 近似：

$$
C = 6ND
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $C$ | 训练 FLOPs budget |
| $N$ | model size，通常指非 embedding 或总 parameter 数量，讲义在 scaling context 中用作模型规模 |
| $D$ | training tokens 数量 |
| $6$ | 讲义采用的 Transformer 训练 FLOPs 粗略系数 |

讲义中的 scaling-law 目标不是只拟合 loss，而是为固定 $C$ 找到更好的 $N, D$。Chinchilla 结论在讲义中被压缩为：

$$
D \approx 20N
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $D$ | 训练 token 数 |
| $N$ | 模型参数数 |
| $20$ | 讲义给出的粗略 compute-optimal token/parameter 比例 |

推导链按讲义描述为：在多个小 FLOPs budget 上找 IsoFLOP optimal $N$，再外推到大 budget；或者拟合

$$
L(N,D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $L(N,D)$ | 训练后 loss |
| $E$ | irreducible loss 下限项 |
| $A, B$ | 模型规模项与数据规模项的系数 |
| $N$ | parameter 数 |
| $D$ | token 数 |
| $\alpha, \beta$ | loss 随 model/data 扩展下降的幂律指数 |

### 2.4 系统增量与接口绑定

**历史版图为课程提供真实系统样本：每个 open model 的 recipe 都会在后续模块中被拆成 tokenizer、architecture、training、systems、data、alignment。**

例如 Llama recipe 中出现的 pre-norm、SwiGLU、RoPE 会进入 model architecture；Mistral 的 GQA 与 sliding window attention 会进入 inference efficiency；DeepSeek 的 MLA、MoE、aux-free load balancing、multi-token prediction 会进入 architecture/training/alignment 交界。

---

## 3. 可执行讲义与课程组织

### 3.1 技术路线与演进逻辑

**本课程把 lecture 本身写成 executable lecture，让代码、层级结构和教学内容共享同一个源文件。**

本模块属于课程载体设计，讲义未涉及特定历史演进痛点 beyond “everything is code”。默认方案是普通 slides 或 notes；当前方案改为程序执行生成 lecture trace，使学生能查看代码、运行代码、看到 lecture 的函数层级。

### 3.2 系统设计与资源权衡链

**Executable lecture 是上层教学抽象，不涉及底层硬件、显存带宽的显著资源置换。**

它的主要工程接口是将 lecture sections 表达为 Python functions，例如 `welcome()`、`current_lm_landscape()`、`tokenization()`；技术收益是结构可导航，代码片段可直接对齐 assignment。

### 3.3 数学原则与推导链

**本模块无数学推导；讲义只给出一个最小代码执行例子。**

代码片段：

```python
total = 0
for x in [1, 2, 3]:
    total += x
```

该代码只服务于说明 lecture 是可执行程序，不承担 LM 算法含义。

### 3.4 系统增量与接口绑定

**课程目录被显式绑定到五个 assignment，形成完整 LM 构建栈。**

| Assignment | 课程模块 | 核心构件 |
|---|---|---|
| 1 | basics | tokenization, model architecture, training |
| 2 | systems | kernels, parallelism, inference |
| 3 | scaling laws | FLOPs budget -> hyperparameters -> loss prediction |
| 4 | data | evaluation, curation, processing, mixing |
| 5 | alignment | weak supervision, PPO, DPO, GRPO |

---

## 4. Basics 模块：Tokenization / Architecture / Training

### 4.1 技术路线与演进逻辑

**Basics 的目标是训练一个 basic language model，三件事缺一不可：输入如何离散化、模型如何表达依赖、参数如何被优化。**

痛点链：

1. 原始输入是 bytes / strings，Transformer 不能直接吃任意字符串。
2. tokenization 将 raw input 转成 integer token sequence。
3. architecture 决定这些 token 如何交互。
4. training 决定如何把 loss 转成 parameter update。

### 4.2 系统设计与资源权衡链

**Basics 的统一原则是 expressivity、stability、efficiency 三者平衡。**

| 目标 | 讲义定义 | 对应设计项 |
|---|---|---|
| Expressivity | 能表达复杂数据依赖 | Transformer shape, heads, depth, MoE |
| Stability | parameter 与 gradient norms 保持在合适区间 | normalization, initialization, optimizer, LR schedule |
| Efficiency | training 与 inference 都快 | BPE, GQA, MLA, sparse/local attention, multi-token prediction |

### 4.3 数学原则与推导链

**Basics 中本讲只列出局部公式和目标函数方向，详细推导将在后续 lecture 展开。**

SwiGLU 在讲义中给出：

$$
\mathrm{FFN\text{-}SwiGLU}(x) = \mathrm{Swish}(xW_1) * xV W_2
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $x$ | 输入 hidden state |
| $W_1$ | 第一条投影矩阵，用于产生经过 Swish 的门控分支 |
| $V$ | 第二条投影矩阵，用于产生被门控的值分支 |
| $W_2$ | 输出投影矩阵 |
| $*$ | elementwise product |
| $\mathrm{Swish}$ | activation function，讲义把它列为 ReLU/GeLU/Swish 对比之一 |

设计调整：因为从 2 个矩阵变成 3 个矩阵，讲义说明 hidden dimension 设为原 2-matrix 版本的 **2/3**，以控制参数/FLOPs。

### 4.4 系统增量与接口绑定

**Basics 输出的是一个能训练的最小 LM 系统，并把 token IDs 接到 Transformer、cross-entropy loss、AdamW、training loop。**

Assignment 1 明确要求：

- implement BPE tokenizer
- implement Transformer, cross-entropy loss, AdamW optimizer, training loop
- do resource accounting
- train on TinyStories and OpenWebText
- leaderboard: **45 minutes on a B200** 内最小化 OpenWebText perplexity

---

## 5. Systems 模块：Kernels / Parallelism / Inference

### 5.1 技术路线与演进逻辑

**Systems 的核心痛点是模型质量不能只靠数学定义，必须让计算真实跑满 GPU/TPU。**

前置基线是直接用 PyTorch primitive operations。核心崩溃点是每个 primitive launch 标准 kernel，频繁读写 HBM，数据移动成本压过计算。破局机制是 resource accounting、custom kernels、parallelism、inference-specific optimization。

### 5.2 系统设计与资源权衡链

**Systems 的底层原则是 minimize data movement。**

讲义给出 B200 示例：

- **2.25 PFLOP/sec bf16**
- **8 TB/sec memory bandwidth**

naive 执行流：

```text
read HBM -> compute A -> write HBM -> read HBM -> compute B -> write HBM
```

fused 执行流：

```text
read HBM -> compute A and B -> write HBM
```

资源置换：

| 技术 | 牺牲/增加 | 换取 |
|---|---|---|
| operator fusion | 更复杂 kernel 实现 | 更少 HBM round trips |
| tiling / FlashAttention | 更复杂内存调度 | attention 更少中间写回 |
| distributed parallelism | 通信与同步复杂度 | 更大模型/更大 batch |
| sharding states | collective operations | 降低单卡 memory pressure |
| continuous batching | serving scheduler 复杂度 | 更高 inference throughput |

### 5.3 数学原则与推导链

**Systems 本讲只给出 FLOPs accounting，不推导具体 kernel 数学。**

讲义示例：

$$
\text{total\_flops} = 6 \times 70\mathrm{e}9 \times 1\mathrm{e}12 = 4.2\mathrm{e}23
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $6$ | 训练 FLOPs 粗略系数 |
| $70\mathrm{e}9$ | 70B parameter model |
| $1\mathrm{e}12$ | 1T training tokens |
| $4.2\mathrm{e}23$ | 总训练 FLOPs |

### 5.4 系统增量与接口绑定

**Systems 接在 basic model 后面，改变的不是模型语义，而是同一语义在硬件上的执行计划。**

Assignment 2 明确要求：

- fused RMSNorm kernel in Triton
- distributed data parallel training
- optimizer state sharding
- benchmark and profile implementations

Inference 接口被拆成两阶段：

| 阶段 | 输入输出形态 | 资源瓶颈 |
|---|---|---|
| prefill | prompt tokens 已知，可并行处理 | compute-bound |
| decode | 一次生成一个 token | memory-bound |

---

## 6. Scaling Laws 模块

### 6.1 技术路线与演进逻辑

**Scaling laws 解决的不是“如何训练一个模型”，而是“不能在目标规模试错时如何选择 hyperparameters”。**

前置基线是 full-scale hyperparameter tuning。核心崩溃点是如果目标预算是 **1e25 FLOPs**，直接调参过于昂贵。破局机制是 scaling recipe：用较小规模实验拟合 loss，再预测目标规模。

### 6.2 系统设计与资源权衡链

**Scaling laws 用小 compute 换大 compute 上的决策确定性。**

流程：

1. 定义 scaling recipe：FLOPs -> hyperparameters。
2. 在较小 scales 上运行实验，例如到 **1e24 FLOPs**。
3. 拟合 scaling law。
4. 预测目标 scale，例如 **1e25 FLOPs**。
5. 用 predictability 指导 full run。

讲义强调：predictability 至少和 optimality 一样重要。

### 6.3 数学原则与推导链

**核心公式是 $C=6ND$ 与 Chinchilla 风格的 loss decomposition。**

同第 2.3 节：

$$
C = 6ND
$$

$$
L(N,D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

关键工程结论：

$$
D \approx 20N
$$

讲义 caveat：该结论不考虑 inference costs；如果部署成本重要，可能偏向 smaller model。

### 6.4 系统增量与接口绑定

**Scaling laws 输入是 training API 的 hyperparameters -> loss，输出是 extrapolated hyperparameters 与 loss predictions。**

Assignment 3 明确要求：

- submit training jobs under FLOPs budget
- gather data points
- fit scaling laws
- submit extrapolated hyperparameters and loss predictions
- leaderboard: minimize loss given FLOPs budget

---

## 7. Data 模块

### 7.1 技术路线与演进逻辑

**Data 模块的痛点是模型能力来自数据分布，而 raw web data 不是可直接训练的干净文本。**

前置基线是假设 data 自动存在。核心崩溃点是来源包括 webpages、books、arXiv、GitHub 等，原始形态可能是 HTML、PDF、directories，且质量、版权、重复、污染、harmfulness 都会影响训练。破局机制是 evaluation、curation、processing、mixing、rewriting/synthetic data。

### 7.2 系统设计与资源权衡链

**Data design 是用前处理 compute 与筛选复杂度，换取训练 compute 不被低质量 token 浪费。**

| 子模块 | 工程目标 | 资源逻辑 |
|---|---|---|
| Evaluation | 指导开发与衡量真实 use case | 私有 perplexity 避免 contamination；高级任务看生态有效性 |
| Transformation | HTML/PDF -> text | 前处理成本换训练可用性 |
| Filtering | 保留高质量、移除 harmful content | classifier 成本换更高 token utility |
| Deduplication | 去重 | Bloom filter / MinHash 成本换少 memorization 与少浪费 compute |
| Data mixing | 调整 source ratios | 小模型/回归实验成本换大模型 mixture |
| Synthetic rewriting | 用 LM 改写 web data | 额外生成成本换更贴近 downstream 的 style/quality |

### 7.3 数学原则与推导链

**本讲未给出 data 模块公式；只给出 RegMix 的方法论：把 mixture selection 形式化为 regression。**

留白原则：讲义没有具体回归目标函数或 MinHash 推导，因此不补写外部公式。

### 7.4 系统增量与接口绑定

**Data 模块向 training loop 提供 token stream，并通过 evaluation 反向改写 data pipeline。**

Assignment 4 明确要求：

- convert Common Crawl HTML to text
- train classifiers to filter quality and harmful content
- deduplication using MinHash
- leaderboard: minimize perplexity given token budget

---

## 8. Alignment 模块

### 8.1 技术路线与演进逻辑

**Alignment 从 full supervision 的 next-token prediction 转向 weak supervision：当 critique 比 generate 更容易时，用评分信号继续改进模型。**

前置基线是 pretraining：预测下一个 token。核心崩溃点是大模型并不会自然遵循用户意图，也可能不 truthful、不 helpful。破局机制是生成 responses、用 human/verifier/LM judge 打分、更新模型偏好更好 responses。

### 8.2 系统设计与资源权衡链

**Alignment 的系统瓶颈来自 on-policy generation 与训练更新之间的耦合。**

讲义列出的挑战：

- RL algorithms unstable and hard to tune
- at scale requires new infrastructure: inference with async rollouts
- constantly trading off systems efficiency and on-policyness

### 8.3 数学原则与推导链

**本讲只列出 PPO、DPO、GRPO 的算法位置，没有给出 objective 公式。**

留白原则：不补写 PPO/DPO/GRPO 的外部数学推导。讲义中的可提取结构是：

```text
generate responses -> score responses -> update model to prefer better responses
```

### 8.4 系统增量与接口绑定

**Alignment 接在已有 reasonable model 后面，新增 preference/reward 信号与 rollout infrastructure。**

Assignment 5 明确要求：

- implement Direct Preference Optimization (DPO)
- implement Group Relative Preference Optimization (GRPO)

---

## 9. Tokenization 总接口

### 9.1 技术路线与演进逻辑

**Tokenization 的核心问题是把 string 变成 integer token sequence，同时控制 vocabulary size 与 sequence length。**

**Tokenizer (分词器)**: 一个在 raw string 与 integer token sequence 之间双向转换的接口；模型不直接处理字符串，而处理这些整数索引。

抽象接口：

```python
class Tokenizer(ABC):
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError
```

数据流：

```text
string -> encode -> list[int] -> model
model output token IDs -> decode -> string
```

### 9.2 系统设计与资源权衡链

**Tokenization 是 sequence length、vocabulary size、sparsity、model compute 之间的资源置换。**

讲义给出关键效率观点：

- 更大的 compression ratio 意味着更短 sequence。
- 更短 sequence 对 Transformer 有利，因为 attention 对 sequence length 是 quadratic。
- 通过增大 vocabulary size 可以提高 compression ratio，但会带来 sparsity。
- 理想方向是 tokenizer-free，但当前 frontier scale 尚未证明完全替代。

### 9.3 数学原则与推导链

**Tokenization 的核心度量是 UTF-8 bytes per token。**

公式：

$$
\mathrm{compression\_ratio}(s, z) = \frac{\mathrm{num\_bytes}(s)}{\mathrm{num\_tokens}(z)}
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $s$ | 输入 string |
| $z$ | tokenizer 输出的 token index list |
| $\mathrm{num\_bytes}(s)$ | `len(bytes(s, encoding="utf-8"))` |
| $\mathrm{num\_tokens}(z)$ | `len(z)` |

代码映射：

```python
def get_compression_ratio(string: str, indices: list[int]) -> float:
    num_bytes = len(bytes(string, encoding="utf-8"))
    num_tokens = len(indices)
    return num_bytes / num_tokens
```

### 9.4 系统增量与接口绑定

**Tokenizer 输出的 `list[int]` 是后续 embedding lookup 与 sequence model 的直接输入。**

Lecture 01 中 tokenizer 单元的后续接口要求：

1. Model should operate on chunks / abstractions of sequence。
2. Chunks should be variable，给 interesting chunks 分配更多 model capacity。

---

## 10. CharacterTokenizer

### 10.1 技术路线与演进逻辑

**CharacterTokenizer 直接把 Unicode character 映射成 code point，是最直观但资源效率很差的方案。**

本模块属于基础定义/前置概念，讲义未涉及特定历史演进痛点。它用于建立 tokenization design space 的一端：直接按字符切分。

### 10.2 系统设计与资源权衡链

**CharacterTokenizer 同时承受 large vocabulary 与 low compression ratio 两个问题。**

讲义事实：

- Unicode characters 约 **150K**。
- rare characters 例如 `🌍` 会占用 vocabulary capacity。
- `vocabulary_size = max(indices) + 1` 在示例里只是 lower bound。

### 10.3 数学原则与推导链

**CharacterTokenizer 的 encode/decode 是 `ord` 与 `chr` 的互逆。**

公式化写法：

$$
z_i = \mathrm{ord}(c_i)
$$

$$
\hat{s} = \mathrm{concat}_{i=1}^{T}\mathrm{chr}(z_i)
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $c_i$ | string 中第 $i$ 个 Unicode character |
| $z_i$ | 第 $i$ 个 token index / Unicode code point |
| $T$ | character 数 |
| $\hat{s}$ | decode 后重构出的 string |

代码映射：

```python
def encode(self, string: str) -> list[int]:
    return list(map(ord, string))

def decode(self, indices: list[int]) -> str:
    return "".join(map(chr, indices))
```

### 10.4 系统增量与接口绑定

**CharacterTokenizer 满足 roundtrip，但它不是后续 assignment 的目标实现。**

它输出 code point IDs，可进入模型，但 sequence/chunk 质量差；讲义用它证明“能 roundtrip”不足以成为好 tokenizer。

---

## 11. ByteTokenizer

### 11.1 技术路线与演进逻辑

**ByteTokenizer 把 string 先转为 UTF-8 bytes，再把每个 byte 当作 token，解决 vocabulary size 但牺牲 sequence length。**

前置基线是 CharacterTokenizer。核心崩溃点是 Unicode vocabulary 太大且 rare character 低效。破局机制是 byte-level representation：每个 byte 只有 0 到 255。

### 11.2 系统设计与资源权衡链

**ByteTokenizer 用固定小 vocabulary 换来固定糟糕的 compression ratio。**

讲义事实：

- byte vocabulary size = **256**。
- `compression_ratio == 1`。
- sequence 过长导致 Transformer context length 与 quadratic attention 成本变差。

### 11.3 数学原则与推导链

**ByteTokenizer 的 encode/decode 是 UTF-8 bytes 与整数列表的互逆。**

公式化写法：

$$
b = \mathrm{UTF8Encode}(s)
$$

$$
z_i = \mathrm{int}(b_i), \quad z_i \in \{0,\dots,255\}
$$

$$
\hat{s} = \mathrm{UTF8Decode}(\mathrm{bytes}(z))
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $s$ | 输入 string |
| $b_i$ | UTF-8 byte sequence 中第 $i$ 个 byte |
| $z_i$ | 第 $i$ 个 token index |
| $\hat{s}$ | decode 后重构 string |

代码映射：

```python
string_bytes = string.encode("utf-8")
indices = list(map(int, string_bytes))
string = bytes(indices).decode("utf-8")
```

### 11.4 系统增量与接口绑定

**ByteTokenizer 是 BPE 的起点：BPE 从 256 个 byte token 开始，通过 merge 构造更大的 data-driven vocabulary。**

---

## 12. WordTokenizer

### 12.1 技术路线与演进逻辑

**WordTokenizer 接近 classical NLP：token 更符合人类语义，但 vocabulary 不固定且 OOV 问题严重。**

前置基线是 character/byte 切分过细。核心崩溃点是过长 sequence 或低语义 chunk。破局尝试是按 word/chunk 切分，例如：

```python
chunks = regex.findall(r"\w+|.", string)
```

### 12.2 系统设计与资源权衡链

**WordTokenizer 用较高 compression ratio 和语义 chunk，换来巨大 vocabulary、rare words、UNK token 与 perplexity 计算问题。**

讲义列出问题：

- many words are rare
- fixed vocabulary size 不明显
- training 未见过的新词需要 UNK
- UNK 会影响 perplexity calculations

### 12.3 数学原则与推导链

**本模块偏向上层切分逻辑，讲义未给出底层数学推导。**

可抽取的接口是：

```text
string -> regex chunks -> chunk-to-id mapping -> list[int]
```

`vocabulary_size = number of distinct chunks in the training data`。

### 12.4 系统增量与接口绑定

**WordTokenizer 说明“人类有意义”不是 tokenizer 的充分条件；模型需要开放词表、固定 vocabulary 和可控 sequence length。**

---

## 13. Byte Pair Encoding (BPE)

### 13.1 技术路线与演进逻辑

**BPE 是本讲 tokenization 的目标方案：从 bytes 出发，训练 data-driven merges，让 common byte sequences 变成单 token。**

**Byte Pair Encoding (BPE, 字节对编码)**: 从 byte-level token 开始，反复合并训练语料中最常见的相邻 token pair，以构造固定 vocabulary 的 subword tokenizer。

技术演进：

1. Philip Gage 1994: BPE for data compression。
2. Sennrich et al. 2015: adapted to NLP for neural machine translation rare words。
3. GPT-2 使用 BPE。

痛点链：

| 旧方案 | 崩溃点 | BPE 的处理 |
|---|---|---|
| Character | vocabulary 大，compression 差 | 从 bytes 起步，不直接用 Unicode code point vocabulary |
| Byte | vocabulary 小但 sequence 长 | 合并常见 byte pairs，提高 compression |
| Word | vocabulary 巨大，OOV | rare sequences 可拆回多个 byte/subword token |

### 13.2 系统设计与资源权衡链

**BPE 的资源置换是用 tokenizer training 与更大 vocabulary，换取更短 sequence 与开放词表。**

讲义对 BPE 的直觉：

- common sequences of bytes -> single token
- rare sequences -> many tokens
- start with each byte as a token
- successively merge most common adjacent token pair

Assignment 1 要求进一步优化：

- `encode()` currently loops over all merges; only loop over merges that matter
- detect and preserve special tokens such as `<|endoftext|>`
- use pre-tokenization such as GPT-2 regex
- make implementation fast

### 13.3 数学原则与推导链

**BPE training 的每一步都在最大化当前序列中最常见 adjacent pair 的局部压缩收益。**

初始化：

$$
z = [\mathrm{int}(b_1), \dots, \mathrm{int}(b_T)]
$$

$$
V_0 = \{i \mapsto \mathrm{bytes}([i]) \mid i \in \{0,\dots,255\}\}
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $z$ | 当前 token index sequence |
| $b_i$ | string 的第 $i$ 个 UTF-8 byte |
| $T$ | byte sequence length |
| $V_0$ | 初始 vocabulary，index -> bytes |

相邻 pair 计数：

$$
\mathrm{count}(a,b)=\sum_{i=1}^{|z|-1}\mathbf{1}[z_i=a \land z_{i+1}=b]
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $(a,b)$ | adjacent token pair |
| $z_i$ | 当前 sequence 第 $i$ 个 token |
| $\mathbf{1}[\cdot]$ | 条件成立为 1，否则为 0 |
| $\mathrm{count}(a,b)$ | pair 在当前 sequence 中出现次数 |

选择 merge pair：

$$
(a^\*, b^\*) = \arg\max_{(a,b)} \mathrm{count}(a,b)
$$

新 token：

$$
k = 256 + i
$$

$$
\mathrm{vocab}[k] = \mathrm{vocab}[a^\*] \Vert \mathrm{vocab}[b^\*]
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $i$ | 第 $i$ 次 merge，从 0 开始 |
| $k$ | 新 token index |
| $\Vert$ | bytes 拼接 |
| `merges[(a*, b*)] = k` | 记录 pair -> new_index |

代码映射：

```python
for i in range(num_merges):
    counts = count_adjacent_pairs(indices)
    pair = max(counts, key=counts.get)
    new_index = 256 + i
    merges[pair] = new_index
    vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]
    indices = merge(indices, pair, new_index)
```

`merge` 函数语义：

```text
scan left-to-right; whenever pair appears, replace both tokens by new_index; otherwise copy current token
```

### 13.4 系统增量与接口绑定

**BPE 输出 `BPETokenizerParams(vocab, merges)`，`BPETokenizer` 用同一套 params 对新 string 编码并可 decode 回原 string。**

数据结构：

```python
@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: dict[tuple[int, int], int]
```

encode:

```text
string -> UTF-8 byte IDs -> apply learned merges in order -> token IDs
```

decode:

```text
token IDs -> vocab lookup bytes -> concatenate bytes -> UTF-8 decode -> string
```

---

## 14. Lecture 01 总结接口

**Lecture 01 的技术产物不是一个完整 LM，而是一个完整学习栈的蓝图加上第一个可实现部件 BPE tokenizer。**

关键接口检查：

| 位置 | 输入 | 输出 | 后续消费方 |
|---|---|---|---|
| Tokenizer | raw string / UTF-8 bytes | `list[int]` token IDs | embedding / Transformer |
| BPE training | raw training text, `num_merges` | `vocab`, `merges` | BPE encode/decode |
| Basic LM | token IDs | logits/loss/model params | systems, scaling, data, alignment |
| Systems | model computation graph | optimized execution | faster training/inference |
| Scaling laws | runs under FLOPs budgets | predicted loss/hyperparameters | full-scale training decision |
| Data pipeline | raw web/books/code/etc. | filtered/mixed token stream | pretraining |
| Alignment | reasonable pretrained model + scores | preference-aligned model | chat/agent use |

全局护栏：以上所有内容严格来自 `lecture/lecture_01.md` 与本地 SOP 文件；未使用外部搜索或外部教程补全。
