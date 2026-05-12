# Phase 1 - Agent A: Architect Structural & Technical Extraction

目标参数：`lecture_01`

输入文件：`E:\allwork\cs336\lecture\lecture_01.md`

输出文件：`output/cs336/lecture_01/01_Phase1_Architect.md`

数据源约束：本产物只基于本地讲义 `lecture/lecture_01.md`，未引入外部资料。

## 0. 全局工程蓝图

**本讲的主线是：在 frontier models 昂贵且不透明的现实下，CS336 用“从零构建”训练学生掌握可迁移的 mechanics 与 efficiency mindset。**

讲义先说明课程存在的原因：研究者正在远离底层技术。2016 年研究者常常自己实现和训练模型；2018 年开始下载 BERT 这类模型并 fine-tune；今天很多研究者直接 prompt GPT、Claude、Gemini 这类 API。抽象层提高生产力，但语言模型的抽象是 leaky abstraction；若要做 fundamental research，就必须能拆开模型栈。

本讲的总公式是：

$$
\text{accuracy} = \text{efficiency} \times \text{resources}
$$

这不是说规模本身无关，而是说 **能随规模继续工作的算法和系统实现才真正重要**。讲义把 resources 落到 data、hardware compute、memory、communication bandwidth，并反复追问：给定固定资源，如何训练出最好的模型。

课程被组织为五个工程子系统：

| 单元 | 工程目标 | 主要约束 |
| --- | --- | --- |
| Basics | 训练一个 basic language model | 表达能力、稳定性、训练/推理效率 |
| Systems | 从 GPU/TPU 中榨出有效吞吐 | FLOPs、HBM、kernel launch、跨 GPU 通信 |
| Scaling laws | 用小规模实验预测大规模训练 | full-scale 调参过贵 |
| Data | 让 token budget 用在有价值数据上 | 数据质量、污染、重复、混合比例 |
| Alignment | 用 weak supervision 改善行为 | RL 稳定性、异步 rollout、on-policyness |

本讲最后进入第一个可实现组件：**tokenization**。它把 raw Unicode string 转成模型实际处理的 integer token sequence：

```text
encode: str -> list[int]
decode: list[int] -> str
```

最低正确性不变量是：

```text
decode(encode(s)) == s
```

## 1. 课程动机与可迁移知识

**本模块的技术路线是从“直接使用封闭模型”退回到“理解模型栈”，因为封闭 frontier models 不能提供可验证的研究底层。**

### 技术路线综述

讲义指出 frontier models 有两类不可控性。第一，训练成本极高，例如 GPT-4 被描述为训练成本约 `$100M`，xAI 为 Grok 训练建设 `230K GPUs` 级别集群。第二，公开技术报告缺少数据和模型架构细节。小模型可以训练，但不一定代表大模型，因为 FLOPs 在 attention 与 MLP 间的比例会随 scale 改变，行为也可能随 scale 涌现。

因此课程只承诺稳定传授两类知识，并谨慎处理第三类：

| 知识类型 | 迁移性 | 讲义判断 |
| --- | --- | --- |
| Mechanics | 高 | Transformer、model parallelism、tokenizer 等工作机制 |
| Mindset | 高 | squeezing the most out of hardware、taking scaling seriously |
| Intuitions | 部分 | 许多数据/建模决策来自实验，不一定跨 scale 迁移 |

讲义用 SwiGLU 作为 intuition 的例子：ReLU、GeLU、Swish 与 GLU 变体通过实验比较，SwiGLU 使用三组矩阵，因此把 hidden dimension 调为两矩阵版本的 `2/3`。这里的重点不是推导出某个唯一公式，而是承认现代 LM recipe 中有实验成分。

### 系统设计

本模块的系统设计是课程边界本身：不把封闭 API 当作事实来源，不把 small LM 的行为直接外推到 frontier LM，而是学习可跨规模复用的底层机制、效率思维和资源核算方法。课程的核心判断是：在大规模下浪费会被放大，因此 efficiency 在更大 scale 上更重要。

### 数学原则

核心公式是：

$$
\text{accuracy} = \text{efficiency} \times \text{resources}
$$

当 resources 固定时，改进只能来自 efficiency；当 resources 增大时，低效率设计会被规模放大成巨大浪费。讲义引用的 algorithmic efficiency 例子给出一个经验事实：2012 到 2019 年，达到 AlexNet-level ImageNet performance 所需 FLOPs 降低了 `44x`。

## 2. 语言模型技术谱系

**本模块的技术路线显示：LLM 的进步不是单纯“变大”，而是可扩展算法、数据、优化和系统工程连续叠加。**

### 技术路线综述

讲义按时间组织语言模型技术谱系：

| 阶段 | 代表内容 | 工程意义 |
| --- | --- | --- |
| Pre-neural | Shannon entropy、5-gram LM | 用统计方法刻画语言序列 |
| Neural ingredients | LSTM、Bengio neural LM、seq2seq、Adam、attention、Transformer、MoE、model parallelism | 引入可学习表示、序列建模、优化器、注意力和可扩展训练 |
| Early foundation models | ELMo、BERT、T5 | pretraining + fine-tuning；任务统一成 text-to-text |
| Embracing scaling | GPT-2、scaling laws、GPT-3、PaLM、Chinchilla | fluency、zero-shot、in-context learning、compute-optimal training |
| Open models | The Pile/GPT-J、OPT、BLOOM、Llama、Mistral/Mixtral、DeepSeek、Qwen、Kimi、GLM、OLMo、Nemotron、Marin | 开放权重、论文、代码、数据推动可研究性 |

讲义最后给出语言模型定义的演化：2018 年是 something you fine-tune；2020 年是 something you prompt；2022 年是 something you talk to；2026 年是 something that acts autonomously。虽然 specs 变成 longer context 和更重的 inference efficiency，但 fundamentals 仍然是 attention、kernels、optimization。

### 系统设计

技术谱系中的系统瓶颈逐步变化：

| 技术变化 | 解决的痛点 | 新引入的系统问题 |
| --- | --- | --- |
| attention / Transformer | fixed-vector bottleneck、RNN 串行性 | attention 的 sequence cost |
| MoE | 增加参数容量但不等比例增加 compute | routing、load balancing、通信 |
| model parallelism / ZeRO / Megatron | 单卡 memory 不足 | sharding、collectives、pipeline/tensor parallel |
| scaling laws | full-scale 调参不可承受 | 小规模实验设计与外推可靠性 |
| open models | closed frontier 不可研究 | 开放程度、训练数据和 recipe 可复现性 |

### 数学原则

讲义在本模块没有展开完整 LM 公式，但明确语言模型处理的是 token sequence 的概率分布。后续 tokenizer 把 raw text 映射为 token indices：

$$
x_1, x_2, \ldots, x_T
$$

其中 `T` 是 tokenizer 产生的 sequence length。`T` 直接影响后续 attention、训练 FLOPs 和 context 使用效率。

## 3. Basics：从文本到可训练 LM

**Basics 单元的工程目标是训练一个 basic language model，核心组件是 tokenization、model architecture 和 training。**

### 技术路线综述

Basics 从三个问题切入：

| 组件 | 讲义问题 | 方向 |
| --- | --- | --- |
| Tokenization | 模型操作的 atoms 是什么？ | raw inputs/bytes 到 integer tokens，主流为 BPE |
| Model architecture | token 如何相互作用？ | 从 Transformer 出发，加入 SwiGLU、RoPE、norm、attention 变体、MoE 等 |
| Training | 参数如何被设定？ | loss、optimizer、initialization、LR schedule、regularization、batch size、MoE load balancing |

Assignment 1 的工程任务是实现 BPE tokenizer、Transformer、cross-entropy loss、AdamW optimizer、training loop，做 resource accounting，并在 TinyStories 与 OpenWebText 上训练。

### 系统设计

Basics 的高层设计原则是平衡三件事：

| 目标 | 含义 |
| --- | --- |
| Expressivity | 能表示复杂数据依赖 |
| Stability | 参数和梯度范数保持在合适范围 |
| Efficiency | 训练和推理都能在硬件上高效运行 |

tokenization 已经是系统设计问题：byte-level 模型优雅且 tokenizer-free，但在当前 Transformer 架构中 compute-inefficient，因为 byte sequence 更长。讲义提到 tokenizer-free 方向有前景，但尚未 scale 到 frontier。

### 数学原则

本单元与本讲最直接相关的量是 compression ratio：

$$
\text{compression\_ratio}(s, z) =
\frac{|\mathrm{UTF8}(s)|}{|z|}
$$

其中 `s` 是原始字符串，`z = encode(s)` 是 token id 序列。compression ratio 越高，token sequence 越短；讲义明确指出这很好，因为 attention 对 sequence length 是 quadratic。

## 4. Model Architecture：从原始 Transformer 到现代配方

**模型架构模块的技术路线是围绕 expressivity、stability 和 efficiency 对 Transformer 做局部替换。**

### 技术路线综述

讲义把 original Transformer 作为起点，然后列出现代 LM 中常见的 refinements：

| 子模块 | 讲义列出的技术 | 解决的问题 |
| --- | --- | --- |
| Activation | ReLU、SwiGLU | 改善 FFN 表达能力；SwiGLU 用 gated unit 思路 |
| Positional encoding | sinusoidal、RoPE | 给 attention 注入位置信息；RoPE 用 rotation matrix 并包含 relative position dependency |
| Normalization | LayerNorm、RMSNorm、QK norm、pre-norm vs post-norm | 稳定训练；RMSNorm 更简单；pre-norm 改善初始化梯度行为 |
| Attention | full、sparse/local、GQA、MLA | 处理 quadratic cost、KV cache、inference speed |
| Recurrence / SSM / linear attention | linear attention、Mamba、Gated DeltaNet、Mamba-3 | 尝试降低 long sequence 和 decode 成本 |
| MLP | dense、MoE | 用 sparse activation 增加容量 |
| Shape | hidden dimension、depth、heads、experts | 决定容量、成本与并行方式 |

### 系统设计

架构不是纯数学审美，而是资源分配。SwiGLU 增加矩阵数后用较小 hidden dimension 抵消成本；GQA/MQA 减少 key-value heads 来加速 decoder inference；MLA 通过压缩 KV cache 改善推理；MoE 让每个 token 只激活部分 experts，用通信和 routing 复杂度换取更大参数容量。

### 数学原则

讲义给出的核心架构公式包括 SwiGLU 与 RoPE 的简写：

$$
\mathrm{FFN\text{-}SwiGLU}(x) = \mathrm{Swish}(xW_1) * xV W_2
$$

讲义同时指出 RoPE 的 key 是：

$$
R W x
$$

其中 `R` 是由 `d/2` 个 rotation matrices 组成的 block-diagonal 结构。这里的数学作用是把位置信息以旋转方式注入表示，并在 self-attention 中包含 relative position dependency。

attention 相关的系统边界来自 sequence length：full attention 的时间和内存随长度呈 quadratic 增长；linear attention 讲义中被描述为利用矩阵乘法结合律把复杂度从 $O(N^2)$ 降到 $O(N)$，其中 `N` 是 sequence length。

## 5. Training：参数更新系统

**训练模块的工程目标是用稳定且高效的更新规则把模型参数移动到低 loss 区域。**

### 技术路线综述

讲义列出的 training knobs 包括：

| 训练旋钮 | 讲义例子 | 工程作用 |
| --- | --- | --- |
| Loss | next-token prediction、multi-token prediction | 定义模型预测目标；multi-token prediction 可提高 sample efficiency |
| Optimizer | Adam、AdamW、SOAP、Muon | 控制梯度如何转化为参数更新 |
| Initialization scale | Xavier init、muP | 控制不同规模下的激活/梯度尺度 |
| LR schedule | cosine、WSD | 控制训练动态和持续训练方式 |
| Regularization | dropout、weight decay | 控制泛化和参数规模 |
| Batch size | critical batch size | 平衡 compute-efficiency 与 time-efficiency |
| MoE load balancing | aux-free | 避免 expert load 不均衡和 routing collapse |

### 系统设计

训练系统同时受优化稳定性和硬件效率约束。AdamW 将 weight decay 与 Adam 的 loss gradient update 解耦；muP 支持在小模型上调超参再迁移到大模型；critical batch size 用 gradient noise scale 判断最大有用 batch；MoE load balancing 要避免 expert 过载，同时避免 auxiliary loss 给主任务引入干扰梯度。

### 数学原则

本讲没有完整推导 optimizer 公式，但给出训练成本与模型/数据规模的核心连接：

$$
C = 6ND
$$

其中 `N` 为参数量，`D` 为训练 token 数。这个公式把训练 knobs 与 scaling laws、systems、tokenization 连接起来：tokenizer 影响 `D` 的计数和 sequence length；architecture 决定 `N` 与每 token 的实际计算形态；systems 决定这些 FLOPs 是否高效落到硬件。

## 6. Systems：硬件效率与推理路径

**Systems 单元的工程目标是减少数据搬运，让有限硬件真正用于计算。**

### 技术路线综述

系统单元包括 resource accounting、kernels、parallelism、inference。讲义给出训练成本例子：

```python
total_flops = 6 * 70e9 * 1e12
```

对应训练 `70B` 参数、`1T` tokens，约为：

$$
C = 6ND = 4.2 \times 10^{23}\ \text{FLOPs}
$$

### 系统设计

系统瓶颈围绕 data movement 展开：

| 模块 | 工程判断 |
| --- | --- |
| Resource accounting | 模型参数必须从 HBM 移到 SM；B200 例子为 `2.25 PFLOP/sec bf16` 与 `8TB/sec` memory bandwidth |
| Kernels | PyTorch primitive 会发标准 kernel；custom kernel 通过 fusion、tiling 减少 HBM 往返 |
| Parallelism | 多 GPU 中 data movement 更慢，需要 gather、reduce、all-reduce，并 shard parameters、activations、gradients、optimizer states |
| Inference | prefill 类似训练，可并行处理给定 tokens，偏 compute-bound；decode 逐 token 生成，偏 memory-bound |

讲义列出的 parallelism 维度包括 data、tensor、pipeline、sequence、expert parallelism。推理加速路径包括 cheaper model、speculative decoding、fused kernels、continuous batching。

### 数学原则

系统单元的核心公式仍是：

$$
C = 6ND
$$

但是系统层面关心的不只是总 FLOPs，还关心 FLOPs 是否被 memory bandwidth、communication bandwidth 或 kernel launch overhead 卡住。roofline analysis 的作用就是判断一个 workload 是 compute-bound 还是 memory-bound。

## 7. Scaling Laws：从小实验外推大训练

**Scaling laws 单元的工程目标是用小规模实验选择大规模训练 recipe，因为 full-scale hyperparameter tuning 太贵。**

### 技术路线综述

讲义提出 conceptual shift：不要只考虑单个 scale，而要考虑 scaling recipe：

```text
FLOPs -> hyperparameters
```

流程是：在较小 FLOPs 下运行实验，拟合 scaling law，预测目标 scale 的 loss，并优化面向大 scale 的 recipe。讲义强调 predictability 至少和 optimality 一样重要。

### 系统设计

scaling laws 解决预算配置问题：给定 FLOPs budget，应该用更大的模型 `N`，还是训练更多 tokens `D`？讲义比较 Kaplan 与 Chinchilla，指出 Chinchilla 的关键结论是 model size 和 training tokens 应该大致同速 scaling，并给出实用近似：

$$
D \approx 20N
$$

讲义也给出 caveat：这没有考虑 inference costs，因此实际部署中可能希望模型更小。

### 数学原则

讲义给出 Chinchilla 式参数化 loss：

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

其系统含义是：模型参数和训练 tokens 都有边际收益递减；固定 compute 下需要在 `N` 与 `D` 之间找最优分配。结合训练成本公式：

$$
C = 6ND
$$

`N` 和 `D` 的选择就是 compute allocation 问题。

## 8. Data：从 raw corpus 到有效 token budget

**Data 单元的工程目标是让模型学到目标能力，而不是把 compute 浪费在坏数据、重复数据或错误混合上。**

### 技术路线综述

数据单元从能力目标出发：模型是否需要 multilingual、conversation、agentic coding capabilities？随后拆成 evaluation、curation、processing：

| 子模块 | 讲义内容 |
| --- | --- |
| Evaluation | internal evaluation 指导开发，external evaluation 衡量真实 use case；perplexity 应避开 Internet contamination |
| Curation | 数据来自 webpages、books、arXiv、GitHub 等，也涉及 copyright/fair use 和 licensing |
| Processing | HTML/PDF 转 text、filtering、deduplication、data mixing、synthetic rewriting |
| Data types | pretraining data、mid-training data、post-training data |

### 系统设计

数据不是自然掉下来的干净文本，而是大量非结构化 raw sources。transformation、filtering、deduplication、mixing 都是 compute efficiency 的组成部分：低质量数据会浪费梯度更新，重复数据会浪费 token budget 并增加 memorization 风险，错误 mixture 会把能力推向错误方向。

### 数学原则

本讲没有给数据处理的具体推导公式，但它与训练成本直接相连。若训练成本近似为：

$$
C = 6ND
$$

那么数据模块主要控制 `D` 的内容质量，而 scaling laws 主要控制 `D` 的数量配置。同样数量的 tokens，不同数据 mixture 会产生不同能力。

## 9. Alignment：从 full supervision 到 weak supervision

**Alignment 单元的技术路线是在 next-token pretraining 之后，用更容易获得的 critique/preference 信号继续改善模型行为。**

### 技术路线综述

讲义先给出前提：到这里模型已经通过 full supervision，即 next-token prediction，训练成 reasonable model。之后可以用 weak supervision 改善，因为很多任务中 critique 比 generation 更容易。

基本模板是：

1. Generate responses from the model.
2. Score responses with a `{human, verifier, LM judge}`.
3. Update the model to prefer better responses.

讲义列出的算法包括 PPO、DPO、GRPO。DPO 被定位为 preference data 上更简单的方法；GRPO 被定位为去掉 value function 的方法。

### 系统设计

alignment 的系统问题不只是优化目标。讲义列出的挑战包括：RL algorithms unstable and hard to tune；at scale requires new infrastructure，例如 inference with async rollouts；需要 constantly trading off systems efficiency and on-policyness。

### 数学原则

本讲没有展开 PPO/DPO/GRPO 的数学推导，因此 Phase 1 不补充外部公式。这里保留讲义层面的技术事实：alignment 的核心数据流是 model generation -> scoring -> preference update；系统瓶颈来自大规模推理和 RL 稳定性。

## 10. Tokenization：本讲展开的第一个实现单元

**Tokenization 的工程目标是把 raw string 映射为可训练的离散 token ids，同时在 vocabulary size、sequence length 和可逆性之间取平衡。**

### 技术路线综述

讲义先定义抽象接口：

```python
class Tokenizer(ABC):
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError
```

随后比较四种 tokenizer：

| 方案 | 编码单位 | 优点 | 主要问题 |
| --- | --- | --- | --- |
| CharacterTokenizer | Unicode code point | 可逆，直观 | 约 `150K` Unicode characters；稀有字符浪费词表；compression ratio 低 |
| ByteTokenizer | UTF-8 byte | 词表固定且小，`256` values | `compression_ratio == 1`，序列太长 |
| Word tokenizer | regex 切 words/chunks | token 语义强，compression ratio 好 | vocabulary huge；rare words 学不好；未见词需要 `UNK` |
| BPE tokenizer | 从 bytes 出发，迭代合并高频相邻 pair | data-driven、固定词表、兼顾可逆与压缩 | naive encode 会遍历所有 merges；需处理 special tokens、pre-tokenization 和速度 |

### 系统设计

tokenizer 输出影响整个模型栈：

| tokenizer 设计量 | 下游影响 |
| --- | --- |
| sequence length `T` | attention/context 成本；byte-level 让 `T` 变大 |
| vocabulary size `V` | embedding/output head 的规模；过大导致稀疏和参数浪费 |
| reversibility | decode 是否能还原原始字符串 |
| special token handling | 训练数据边界和控制 token 是否被破坏 |
| pre-tokenization | BPE merge 的统计和 encode 速度 |

讲义明确说：较大的 compression ratio 意味着较短序列，因为 attention 对 sequence length 是 quadratic。也指出可以通过增大 vocabulary size 提高 compression ratio，但会导致 sparsity。

### 数学原则

核心函数：

$$
\mathrm{encode}(s) = z,\quad \mathrm{decode}(z) = s
$$

压缩率：

$$
\text{compression\_ratio}(s, z) =
\frac{|\mathrm{UTF8}(s)|}{|z|}
$$

BPE 训练从 bytes 开始：

```python
indices = list(map(int, string.encode("utf-8")))
merges: dict[tuple[int, int], int] = {}
vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
```

每轮统计相邻 pair：

```python
counts[(index1, index2)] += 1
```

选择最高频 pair，分配新 token id：

```python
new_index = 256 + i
merges[pair] = new_index
vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]
indices = merge(indices, pair, new_index)
```

`merge` 的不变量是：在 `indices` 中从左到右扫描，把所有与目标 `pair` 相同的相邻 token 替换成 `new_index`，其他 token 原样保留。BPE 的可逆性来自 `vocab: index -> bytes`，decode 时拼接 bytes 再 UTF-8 decode。

## 11. Phase 1 结构化结论

**本讲的技术骨架可以压缩为一条资源约束链：封闭 frontier models 迫使课程回到底层机制；固定资源迫使所有设计服从 efficiency；tokenization 是第一个影响全栈成本的具体接口。**

最短链条：

```text
frontier models expensive + opaque
  -> learn mechanics and efficiency mindset
  -> course split into basics/systems/scaling/data/alignment
  -> all modules optimize accuracy under resource constraints
  -> tokenization maps raw strings to token ids
  -> token ids determine sequence length and vocabulary size
  -> sequence length drives attention/context cost
  -> BPE balances byte-level coverage and word-level compression
```

后续进入 Phase 2 时，必须以本 Phase 1 的技术事实为准：不补充讲义外的算法证明，不把 modern LM 外部知识写成讲义事实，不直接给 assignment 完整实现。

