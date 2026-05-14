# CS336 Lecture 03 Phase 2: Educator 认知脚手架

## 0. 全局知识树

**Lecture 03 的知识树只有一个主干：现代 LLM 架构不是神秘拼装，而是在 Transformer 基线上的稳定性、资源和表达能力三方折中。**

```text
CS336 历史主线
├── Lecture 01: 目标
│   ├── accuracy = efficiency x resources
│   └── tokenizer 把 raw text 接到 token IDs
├── Lecture 02: 账本
│   ├── tensor shape -> memory
│   ├── operations -> FLOPs
│   ├── arithmetic intensity -> compute-bound / memory-bound
│   └── training/inference 都要先算资源账
└── Lecture 03: 架构选择
    ├── Transformer block 现代化
    │   ├── pre-norm / RMSNorm
    │   ├── no bias
    │   ├── SwiGLU / GeGLU
    │   └── serial vs parallel layers
    ├── Position handling
    │   ├── absolute additive embeddings
    │   ├── relative attention
    │   └── RoPE on Q/K
    ├── Hyperparameters
    │   ├── FFN multiplier
    │   ├── head dim ratio
    │   ├── depth-width aspect ratio
    │   ├── vocab size
    │   └── dropout / weight decay
    ├── Stability tricks
    │   ├── z-loss
    │   ├── QK norm
    │   └── logit soft-capping
    └── Attention efficiency
        ├── KV Cache bottleneck
        ├── MQA / GQA
        └── sparse / sliding / interleaved attention
```

最重要的认知重排：

| 如果只看名字 | 容易误解成 | 正确挂载点 |
|---|---|---|
| RMSNorm | 一个小 normalization 变体 | Lecture 02 memory movement 问题的架构回应 |
| RoPE | 一种 position embedding | attention score 如何依赖 relative position 的接口改造 |
| GQA/MQA | 换 head 数量 | KV Cache bandwidth 优化 |
| z-loss/QK norm | 训练小技巧 | softmax 数值尺度控制 |
| hyperparameters | 调参经验 | 参数量、FLOPs、latency、vocab 覆盖范围的资源分配模板 |

---

## 1. 从 Original Transformer 到现代 LLM Block

### 1.1 溯源与关联拓扑

**本讲的第一个认知动作是把“Transformer”拆成可替换部件，而不是把它当成一个不可改动的整体。**

Original Transformer 给出的是基线：

```text
sinusoidal position embeddings
post-norm LayerNorm
ReLU FFN with bias
```

Modern variant 给出的是当前默认：

```text
RoPE
pre-norm
SwiGLU
no bias
```

它继承 Lecture 01/02 的方式是：Lecture 01 关心整体效率，Lecture 02 教你数 FLOPs/memory，Lecture 03 告诉你哪些 block 选择会改变这些账本和训练稳定性。

### 1.2 极简例子

**把一个 Transformer block 想成一条处理 `hidden state` 的生产线，本讲讨论的是每个工位换成哪种工具。**

极简流：

```text
输入 token IDs: [17, 42, 9]
embedding 后: X shape = [1, 3, d_model]
block 内:
  norm(X)
  attention(norm(X))
  MLP(...)
输出:
  logits shape = [1, 3, vocab_size]
```

本讲所有 architecture changes 都基本不改变 `[batch, seq, d_model]` 这个外部接口，而是改变 block 内部如何计算。

### 1.3 差异鉴别

| 概念 | 处理对象 | 作用目标 |
|---|---|---|
| architecture variation | block 内部计算图 | 训练稳定、表达能力、推理效率 |
| hyperparameter | dimension / depth / vocab 等配置 | 参数量、FLOPs、latency 分配 |
| stability trick | softmax/logit 附近 | 防止训练数值失控 |

### 1.4 认知陷阱

**不要把“现代 LLM 架构”理解成完全不同于 Transformer；讲义反复展示的是大多数变化都很小，但小变化在大规模训练里会被放大。**

真正的难点不是记住每个模型用哪个选项，而是知道每个选项挂在哪条约束链上：稳定性、memory movement、inference KV Cache、relative position、或参数/FLOPs 比例。

---

## 2. Pre-Norm、RMSNorm 与 No Bias：稳定性和带宽意识

### 2.1 溯源与关联拓扑

**pre-norm 解决的是深层训练的 gradient 行为，RMSNorm/no bias 解决的是 normalization 与小操作在 runtime 中被低估的问题。**

关系拓扑：

```text
Normalization
├── 放在哪里
│   ├── post-norm: original/BERT 基线
│   └── pre-norm: modern LMs 主流
└── 怎么 normalize
    ├── LayerNorm: mean + variance + gamma + beta
    └── RMSNorm: RMS scale + gamma

Bias terms
└── 从 FFN/linear/norm 中删除，减少参数和额外加法路径
```

Lecture 02 的关键挂载点是：stat normalization 和 elementwise FLOPs 占比小，但 runtime 占比可以很高。因此“只看 FLOPs”会误判 RMSNorm 的价值。

### 2.2 直观类比与极简例子

**LayerNorm 像先把一组数整体平移到均值 0，再调整尺度；RMSNorm 像只调整尺度，不做平移。**

toy hidden vector：

```text
x = [2, 4]
LayerNorm 关心:
  mean = 3
  variance = 1
  x - mean = [-1, 1]
RMSNorm 关心:
  norm-like scale from [2, 4]
  不执行 x - mean 这一步
```

这个例子只说明操作差异，不代表讲义给出完整数值推导。关键是：RMSNorm 少做统计和平移，执行路径更简单。

### 2.3 差异鉴别

| 易混概念 | 核心差异 |
|---|---|
| pre-norm vs RMSNorm | pre-norm 是 norm 的位置；RMSNorm 是 norm 的公式 |
| RMSNorm vs no bias | RMSNorm 删除部分 normalization 机制；no bias 删除 linear/norm 的平移参数 |
| FLOPs 少 vs runtime 少 | FLOPs 少不必然 runtime 少；Lecture 02 已说明 bandwidth/kernel 才可能是瓶颈 |

### 2.4 认知陷阱

**最容易犯的错是说“RMSNorm 更快，因为 FLOPs 少”；讲义真正强调的是 FLOPs 与 runtime 不等价。**

RMSNorm 的价值要放在 memory movement 里理解：normalization 需要读写 tensor、做归约、做 elementwise 操作，这些都可能让 GPU 计算单元等数据，而不是等乘法。

---

## 3. Activations 与 Gating：FFN 的表达能力开关

### 3.1 溯源与关联拓扑

**FFN activation 的知识树从“单通道非线性”长到“带 gate 的双通道控制”。**

```text
FFN activation
├── ReLU
│   └── max(0, xW1)
├── GeLU
│   └── GELU(xW1)
└── GLU family
    ├── ReGLU
    ├── GeGLU
    └── SwiGLU
        └── activation branch ⊗ gate branch
```

挂载点：FFN 通常是 Transformer block 中参数和计算的重要部分，所以 activation 的变化会影响质量，也会影响 $d_{ff}$ multiplier 的默认值。

### 3.2 直观类比与极简例子

**普通 FFN 像“先加工再输出”，GLU 像“加工结果还要经过一个逐维阀门”。**

极简向量：

```text
candidate = [3, -1, 2]
gate      = [0.2, 0.9, 0.0]
gated     = candidate ⊗ gate
          = [0.6, -0.9, 0.0]
```

在模型中，`candidate` 来自 $xW_1$ 的 activation branch，`gate` 来自 $xV$。这不是把整个 token 开关掉，而是每个 hidden dimension 独立调节。

### 3.3 差异鉴别

| 概念 | 是否有 gate branch | 常见 FF multiplier |
|---|---:|---:|
| ReLU FFN | 否 | 约 $4d_{model}$ |
| GeLU FFN | 否 | 约 $4d_{model}$ |
| SwiGLU/GeGLU | 是 | 约 $\frac{8}{3}d_{model}$ |

### 3.4 认知陷阱

**不要只记“GLU 更好”；要同时记住它通常缩小 $d_{ff}$，否则参数和计算量比较不公平。**

讲义的重点不是 gated units 必然统治一切，GPT-3 这类非 GLU 模型也能工作。更准确的结论是：GLU variants 在现代经验中非常常见，并形成了与 $d_{ff}$ multiplier 配套的默认配置。

---

## 4. RoPE：Position 信息从输入加法变成 Q/K 旋转

### 4.1 溯源与关联拓扑

**RoPE 的母概念不是“embedding table”，而是“attention score 应该如何感知相对距离”。**

拓扑：

```text
Position information
├── absolute additive
│   ├── sine: fixed PE_pos
│   └── learned: learned u_i
├── relative attention
│   └── 在 attention score 里加入 relative term
└── RoPE
    ├── 对 Q/K 做 position-dependent rotation
    └── dot product 变成 relative-position-aware
```

RoPE 的目标性质：

```text
两个位置 i,j 的 attention score
不应该只记“i 是第几个、j 是第几个”
而应该自然看见 “i-j 相距多远”
```

### 4.2 直观类比与极简例子

**RoPE 像给每个位置的 query/key 指针旋转一个角度；比较两个指针时，真正留下的是两者角度差。**

二维玩具例子：

```text
原始向量: [1, 0]
位置 m=0: 旋转 0 度  -> [1, 0]
位置 m=1: 旋转 θ 度  -> [cos θ, sin θ]
位置 m=2: 旋转 2θ 度 -> [cos 2θ, sin 2θ]
```

当 query 和 key 都按各自位置旋转后，它们的 dot product 会携带相对角度信息。讲义中用 block diagonal rotation matrix 把这个二维过程扩展到多维 hidden vector 的每一对维度。

### 4.3 差异鉴别

| 方法 | position 加在哪里 | 是否 additive | attention 是否天然看到相对位置 |
|---|---|---:|---:|
| sinusoidal PE | embedding 输入端 | 是 | 不如 RoPE 直接 |
| learned absolute | embedding 输入端 | 是 | 不如 RoPE 直接 |
| relative attention | attention 内部 | 否 | 是 |
| RoPE | Q/K projection 后 | 否 | 是 |

### 4.4 认知陷阱

**实现 RoPE 时最容易错的位置是把它当成普通 embedding 加到 token vector 上；讲义明确说它在每次 attention operation 中作用于 query/key。**

因此正确心智模型不是：

```text
token embedding + RoPE embedding
```

而是：

```text
x -> q,k -> rotate(q,k, position) -> attention(q,k,v)
```

这也解释了为什么 RoPE 属于 attention 内部接口，而不是 tokenizer 或 embedding lookup 的接口。

---

## 5. Hyperparameters：不是玄学，而是形状账本

### 5.1 溯源与关联拓扑

**Lecture 03 的 hyperparameters 不是“随便调”，而是每个 dimension 都会落到 Lecture 02 的 memory/FLOPs/latency 账本。**

```text
Hyperparameters
├── d_model
│   ├── hidden state width
│   ├── projection matrix size
│   └── residual stream width
├── d_ff
│   └── MLP compute/params
├── num_heads × head_dim
│   └── attention projection layout
├── depth
│   ├── serial latency
│   └── parallelization difficulty
└── vocab size
    ├── embedding table
    └── output softmax logits
```

讲义的反直觉结论是：许多大模型在这些比例上非常保守。

### 5.2 极简例子

**如果 $d_{model}=1024$，普通 FFN 默认会把中间层设到约 $4096$；GLU 则常设到约 $2730$ 左右。**

toy config：

```text
d_model = 1024
non-gated FFN:
  d_ff = 4 * 1024 = 4096
GLU FFN:
  d_ff ≈ 8/3 * 1024 ≈ 2731
```

这解释了为什么不能直接比较“有没有 gate”，还要看中间维度是否按规则缩放。

### 5.3 差异鉴别

| 问题 | 不是在问 | 实际在问 |
|---|---|---|
| FFN 多宽 | activation 喜好 | MLP 参数/FLOPs 分配多少 |
| head dim 和 head 数 | head 越多越好 | attention projection width 如何对齐 $d_{model}$ |
| deep or wide | 层数审美 | latency、parallelization、表达能力折中 |
| vocab size | 词表越大越智能 | 语言覆盖、embedding/logit 成本、tokenization 粒度 |
| dropout | 是否一定防过拟合 | 大规模单 pass pretraining 中是否值得引入随机丢弃 |

### 5.4 认知陷阱

**不要把 T5 的 64 倍 FFN 当成新默认；讲义反而用 T5 v1.1 回到 2.5 倍说明极端选择可能可行但未必最优。**

本讲给出的默认值更像“安全工作区”：

- non-GLU FFN：约 4
- GLU FFN：约 $8/3$
- heads × head_dim / model_dim：多数约 1
- 单语 vocab：30-50k
- 多语/生产 vocab：100-250k
- 新模型 pretraining：多数不用 dropout，更多依赖 weight decay

---

## 6. Stability Tricks：Softmax 是数值风险集中区

### 6.1 溯源与关联拓扑

**z-loss、QK norm、soft-capping 都挂在同一个母问题上：softmax 前后的 logit 尺度不能失控。**

```text
Softmax risk
├── output softmax
│   └── z-loss controls log Z(x)
├── attention softmax
│   └── QK norm controls attention logits
└── logits too large
    └── tanh soft-capping limits magnitude
```

Lecture 02 让我们关注 dtype 和数值范围；Lecture 03 把这个问题落在 softmax 的 exponentials 和 division 上。

### 6.2 直观类比与极简例子

**softmax 像把分数先指数放大再归一化；一旦分数尺度太大，小差异会被指数变成极端差异。**

toy logits：

```text
logits A = [1, 2, 3]
logits B = [10, 20, 30]
```

两组 logits 的排序一样，但第二组进入 exponential 后会更极端。稳定性技巧关心的不是“哪个 token 最大”，而是 logits 和 normalizer 的尺度是否让训练进入危险区域。

### 6.3 差异鉴别

| 技巧 | 控制对象 | 不要混淆成 |
|---|---|---|
| z-loss | output softmax 的 $\log Z(x)$ | 普通 weight decay |
| QK norm | attention softmax 前的 Q/K 尺度 | FFN normalization |
| soft-capping | raw logits 的上限 | hard clipping；讲义说是通过 Tanh soft cap |

### 6.4 认知陷阱

**z-loss 不是在惩罚参数大小，而是在约束 softmax normalizer 的 log 值。**

另一个陷阱是把讲义里的 $L=\sum[\log(P)-\alpha\log^2Z]$ 直接改写成常见 negative log-likelihood 形式。Phase 1 已保留讲义写法；理解时只需抓住机制：额外项把 $\log Z(x)$ 压向 0，帮助 output softmax 稳定。

---

## 7. MQA/GQA 与 Sliding Attention：推理时真正贵的是搬 KV

### 7.1 溯源与关联拓扑

**MQA/GQA 是 Lecture 02 arithmetic intensity 思想在 autoregressive decoding 上的直接应用：生成不能完全并行，所以 KV Cache 读写变成瓶颈。**

```text
Attention efficiency
├── training/full attention
│   └── arithmetic intensity 较高，GPU 容易跑满
├── incremental generation
│   ├── step-by-step decoding
│   ├── KV Cache 反复读取
│   └── arithmetic intensity 变差
├── MQA/GQA
│   └── 减少 K/V heads，降低 KV Cache movement
└── sparse/sliding attention
    └── 限制 attention pattern，降低长上下文 quadratic 成本
```

### 7.2 直观类比与极简例子

**MHA 像每个 query head 都有自己的历史档案柜；MQA 像所有 query heads 共用一个档案柜；GQA 则是几个 heads 共用一组档案柜。**

toy layout：

```text
num query heads = 8

MHA:
  KV heads = 8
MQA:
  KV heads = 1
GQA:
  KV heads = 2 or 4
```

输出 shape 仍回到 `[batch, seq, d_model]`，但 KV Cache 中要存和读的 K/V 数量不同。这就是 MQA/GQA 的推理收益来源。

### 7.3 差异鉴别

| 概念 | 改什么 | 主要收益 | 可能代价 |
|---|---|---|---|
| MHA | 每个 head 独立 K/V | 表达最完整 | KV Cache 最大 |
| MQA | 所有 query heads 共享 K/V | 推理 memory movement 低 | 可能有小 PPL hit |
| GQA | 分组共享 K/V | 表达与效率折中 | 需要选 group ratio |
| sliding attention | attention mask 局部化 | 长上下文成本低 | 长距离信息通路变弱 |
| interleaved full/local | 层间混合 full 与 local | 保留部分长程通路 | 架构和实现更复杂 |

### 7.4 认知陷阱

**不要把 MQA/GQA 理解成“减少 query heads”；它真正减少的是 key/value heads，query heads 仍可以很多。**

另一个陷阱是把 sparse/sliding attention 当成免费午餐。它降低 quadratic 成本，但会限制 token 能直接看到的上下文。因此当前常见折中是 interleaving：例如讲义里的 Command A 每 4 层一个 full attention，其余使用 SWA。

---

## 8. 全讲综合：一张现代 LLM 架构判断表

**读完 Lecture 03 后，判断一个 LLM architecture 配置时应该先问“它在改哪条约束链”，而不是先问“这个名字新不新”。**

| 看到的设计 | 第一反应应该挂到哪里 | 应追问的问题 |
|---|---|---|
| pre-norm | training stability | 是否还加 double norm / non-residual postnorm？ |
| RMSNorm | runtime + normalization simplicity | 是否删除 beta/bias？ |
| no bias | 参数/内存/稳定性小优化 | 是所有 linear 都删，还是局部删？ |
| SwiGLU | FFN expressiveness | $d_{ff}$ 是否缩到 GLU 默认区间？ |
| RoPE | relative position in attention | 是作用在每层 Q/K，还是错误地加到 embedding？ |
| large vocab | tokenizer/output logits | 是多语/生产需求，还是无谓扩大？ |
| z-loss | output softmax stability | $\alpha$ 多大，是否只在训练用？ |
| QK norm | attention softmax stability | 对 Q/K 哪个位置 normalize？ |
| GQA/MQA | decoding KV Cache | KV groups 与 query heads 比例是多少？ |
| sliding attention | long context cost | full/local 层如何 interleave？ |

最终心智模型：

```text
现代 LLM = Transformer 主干
       + 更稳的 norm placement
       + 更轻的 normalization/bias 选择
       + 更强的 FFN gate
       + 更合理的 relative position interface
       + 更保守的 dimension ratios
       + softmax 稳定性保护
       + decoding KV Cache 优化
```

**这就是 Lecture 03 对前两讲的推进：Lecture 01 告诉你目标是效率，Lecture 02 教你数资源，Lecture 03 告诉你现代 LLM 架构里的每个“默认选择”如何服务于这些目标。**
