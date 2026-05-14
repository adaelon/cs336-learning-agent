# CS336 Lecture 03 Phase 1: Architect 技术结构拆解

## 0. 全局定位

**Lecture 03 把 Lecture 02 的“tensor 资源账本”落地到一组**具体可选**的架构与超参数旋钮：在已知 FLOPs/memory/MFU 约束下，工业界 LLM 实际收敛到了哪几条 architecture lane。**

历史状态继承（来自 `output/cs336/lecture_02/01_Phase1_Architect.md`）：

- Lecture 02 给出 training step 的 compute 公式 $F_{\text{step}}=6BN$、memory 公式 `params + grads + optimizer states + activations`，并以 roofline 区分 compute-bound（大 matmul）与 memory-bound（elementwise、matvec）。
- Lecture 02 的 `MFU = F_actual / F_promised` 和 arithmetic intensity 概念是本讲所有 stability/efficiency 讨论的隐含 baseline：每一个 architecture knob 的胜负判定，最终都要回到「FLOPs 省没省 / bandwidth 省没省 / MFU 升没升」。
- Lecture 02 的 tensor shape 约定 `[B, S, H, D]`（其中 $H$=num heads, $D$=head dim）以及 mixed precision policy（params/activations/grads bf16；optimizer states fp32）在本讲被进一步操作：head dim 与 model dim 关系、$d_{ff}/d_{model}$ 比例、QK-norm 注入点都是对这套 tensor flow 的细节调整。

本讲目标：

1. 用 LLaMA-like 现代架构作为 anchor，逐项对比和 *原始 transformer* 的差异（pre-norm, RMSNorm, RoPE, SwiGLU, no-bias）。
2. 给出 hyperparameter 行业共识：$d_{ff}=4d_{model}$（GLU 用 $8/3$）、head_dim·num_heads ≈ $d_{model}$、aspect ratio $d_{model}/n_{layer} \approx 100$–$200$。
3. 列出训练稳定性三件套：z-loss、QK-norm、logit soft-capping。
4. 列出推理友好的 attention 变体：MQA / GQA / sliding window / interleaved full+SWA。

---

## 1. 模块零：现代 LLaMA-like 起点 vs 原始 Transformer

### 1.1 技术路线演进逻辑

**Lecture 03 的开场不是从头讲 Transformer，而是用「现代变体」对原始结构做四点定向 diff，把所有差异归到“为什么不再这么写”。**

【前置基线】：原始 Transformer（Vaswani et al. 2017，讲义 Page 3）的默认组合：

- Position embedding: **sines and cosines (正弦余弦位置嵌入)** — 把位置 $i$ 编码成不同频率的 sin/cos，加到 token embedding 上。
- FFN activation: **ReLU (修正线性单元)** — $\max(0, x)$，无 gating。
- Norm placement: **post-norm (后置归一化)** — LayerNorm 在 residual 加法之后。
- Linear/Norm bias: 全部带 bias term。

【核心崩溃点】：在 large-scale pretraining 体制下，这四点都在 stability 或 efficiency 上撞墙：post-norm 需要 warmup 才能稳；ReLU 与 sine PE 在 long-context 与 GLU 时代被超越；bias term 既吃 memory 又是 optimization 不稳定来源。

【破局机制】：讲义 Page 4 提出现代 simple variant：

- LayerNorm 移到 block 输入侧（**pre-norm**）。
- Position embedding 改为 **RoPE (Rotary Position Embedding，旋转位置编码)**: 在 query/key 上施加 2D 平面旋转，使 attention inner product 只依赖相对位置 $i-j$。
- FFN 改为 **SwiGLU (Swish-Gated Linear Unit)**: gated activation，含额外参数矩阵 $V$。
- 所有 linear 与 norm 层 **去 bias**。

### 1.2 系统设计与资源权衡链

**这四个改动在 Lecture 02 的资源账本上各自命中一项：bandwidth、memory、stability。**

| 改动 | 在 Lecture 02 账本上的影响 |
|---|---|
| pre-norm | 训练稳定性提升，可去掉 warmup、容纳更大 LR；改变的是梯度路径而非 FLOPs |
| RMSNorm（与 pre-norm 经常打包）| 少一遍 mean 扫描 + 少 bias，减少 memory move，对 normalization 这类 memory-bound op 有可测加速 |
| RoPE | 不增加可训练参数，只在 QK 上做旋转；FLOPs 极少，但保证 long-context 外推与相对位置不变性 |
| SwiGLU | 引入额外 $V$ 矩阵，参数翻倍，因此 $d_{ff}$ 要按 $2/3$ 缩回，保证 FLOPs 等量 |
| no-bias | 直接减少 parameter 数与 gradient/optimizer state 占用 |

【硬件对齐】：所有这些改动都符合 Ivanov et al. 2023 的结论——matrix multiply 已经吃掉绝大多数 FLOPs/memory，省下 normalization/bias 的 memory traffic 才是 wallclock 收益所在。

### 1.3 数学原理与推导链

**本模块属于 anchor 介绍，无新公式；新公式集中在下游 RMSNorm / GLU / RoPE 节展开。**

### 1.4 系统演进与接口对接

**本讲所有后续模块都是对这套现代 anchor 的“替换组件”——它把 Lecture 02 中抽象的 transformer block 具体化为可配置的 architecture knobs。**

接口继承：

```text
Lecture 02 tensor flow: [B, S, H·D] -> linear -> norm -> attention -> linear -> norm -> ffn
Lecture 03 替换: norm = pre-RMSNorm; PE = RoPE on Q/K; FFN = SwiGLU; bias = None
```

---

## 2. Pre-norm vs Post-norm

### 2.1 技术路线演进逻辑

**讲义 Page 10–12：现代 LMs 几乎全员 pre-norm，BERT 是历史性的 post-norm 例外，OPT350M 是“奇怪的”post-norm 异类。**

【前置基线】：post-norm（原始 Transformer / BERT）— LayerNorm 在 residual addition 之后：

$$
y = \mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))
$$

【核心崩溃点】：

- **Gradient attenuation** [Xiong 2020]：post-norm 把每层信号都重新尺度归一化，反向回传时梯度被层数 $L$ 衰减得很厉害，导致需要 warmup 才能让 LR 起步。
- **Gradient spikes** [Salazar and Nguyen 2019]：在大模型/大 LR 下 post-norm 容易出现梯度尖峰，训练曲线“炸毛”。

【破局机制】：pre-norm 把 LayerNorm 放到 sublayer 内部，**让主 residual stream 始终不经过 norm**：

$$
y = x + \mathrm{Sublayer}(\mathrm{LayerNorm}(x))
$$

讲义原话：「set up LayerNorm so that it doesn't affect the main residual signal path」。

### 2.2 系统设计与资源权衡链

**架构变更：normalization 从“收尾保险”变成“开头预处理”，主 residual 路径在 layer 间是干净的 identity path。**

权衡链：

- 牺牲：post-norm 对 train-time loss 曲线略好的“理论 well-behavedness”。
- 换得：可以**去 warmup**、用**更大 LR**、获得**更稳的大模型训练曲线**——这是 Lecture 03 后续 stability tricks 的 baseline 防线。

【新变体】：讲义 Page 13 的 **double-norm / non-residual post-norm**（Grok、Gemma 2、Olmo 2）— 在 residual stream 之外再加一次 post-norm，相当于“pre-norm 主路径 + 残差外辅助 post-norm”，进一步压制 outlier activations。Olmo 2 只用 non-residual post-norm。

### 2.3 数学原理与推导链

**Pre-norm 的核心数学诉求是：让 residual 路径在层数 $L$ 增加时仍保持 identity-like 的 gradient flow。**

设第 $\ell$ 层的输出：

- Post-norm: $x_{\ell+1} = \mathrm{Norm}(x_\ell + F_\ell(x_\ell))$
- Pre-norm: $x_{\ell+1} = x_\ell + F_\ell(\mathrm{Norm}(x_\ell))$

符号字典：

| 符号 | 含义 / 工程映射 |
|---|---|
| $x_\ell$ | 第 $\ell$ 层 residual stream 上的 activations，shape `[B, S, d_model]` |
| $F_\ell$ | 第 $\ell$ 层的 sublayer（attention 或 FFN） |
| $\mathrm{Norm}$ | LayerNorm 或 RMSNorm |

post-norm 下 $\partial x_{\ell+1}/\partial x_\ell$ 必经过 norm 的 Jacobian，多层连乘后衰减；pre-norm 下因为有 identity 旁路，反传梯度天然为 $I + J_F$ 形式，避免衰减。讲义没有展开严格证明，引用 Xiong 2020 即可。

### 2.4 系统演进与接口对接

**对接 Lecture 02：pre-norm 改的是 backward gradient flow 而非 forward FLOPs；Lecture 02 中的 $6ND$ 训练 FLOPs 公式不受影响。**

Tensor shape 不变：norm 仍是 `[..., d_model] -> [..., d_model]` 的 elementwise op，roofline 上仍为 memory-bound（与 Lecture 02 给出的 GeLU/ReLU intensity 同一象限）。

---

## 3. LayerNorm → RMSNorm

### 3.1 技术路线演进逻辑

**讲义 Page 14–17：原始 Transformer/GPT 家族用 LayerNorm；LLaMA、PaLM、Chinchilla、T5 等现代家族切换到 RMSNorm。RMSNorm 的胜出**不是因为它做了更多事，而是因为它干掉了**不必要的均值计算和 bias 参数（补充自论文 RMSNorm, Zhang & Sennrich 2019）*。

【前置基线】：LayerNorm（Ba et al. 2016）— 对每个 token 在 $d_{model}$ 维度做 mean/variance 归一化：

$$
\bar{a}_i = \frac{a_i - \mu}{\sigma} g_i + b_i,\quad \mu=\frac{1}{n}\sum a_i,\quad \sigma=\sqrt{\frac{1}{n}\sum (a_i-\mu)^2}
$$

历史时代背景*（补充自论文 RMSNorm）*：2019 年单卡 TITAN X/V100 上，RNN 引入 LayerNorm 后每步时间增加约 67%，mean reduction 的串行 kernel 成为可见瓶颈。

【核心崩溃点】：LayerNorm 在大规模 LM 中并非精度瓶颈，而是 **bandwidth/参数瓶颈**——讲义 Page 15 引用 Ivanov et al. 2023：矩阵乘已经吃掉绝大多数 FLOPs/memory，剩下的 normalization 越精简越好。

【破局机制】：RMSNorm 直接抛弃「mean-centering」假设——*（补充自论文 RMSNorm）* 作者押注 LayerNorm 的成功来自 **re-scaling invariance（重缩放不变性）**，而非 **re-centering invariance（重平移不变性）**。这是反直觉点：人们通常认为 zero-mean 很重要，但实验表明删除均值并不损害最终性能。

### 3.2 系统设计与资源权衡链

**RMSNorm 用一个数学假设的弱化（删除 mean/centering）换来两个工程收益：少一遍 reduction，少一组 bias 参数。**

【架构变更】：normalization kernel 从「两趟扫描（先求 $\mu$，再用 $\mu$ 求 $\sigma^2$）」收缩为「一趟扫描（直接求 RMS）」。

【资源置换】*（补充自论文 RMSNorm 的实测数据）*：

| 架构 | normalization 加速 |
|---|---|
| RNN | 25%–64% |
| Transformer | 7%–9% |
| pRMSNorm（仅前 6.25% 神经元估计 RMS） | 理论更快，但因 PyTorch/Theano tensor slicing kernel 未优化，实测反而慢于 RMSNorm |

讲义 Page 17 引用 Narang et al. 2020：RMSNorm 在 perf 上不输（甚至偶有微涨）。

【硬件对齐】：normalization 本身是 memory-bound op（Lecture 02 已论证 elementwise/小 reduction 为 memory-bound）；少一次 reduction 直接减少 memory traffic，且没有 bias tensor 也减少了 parameter 与 gradient/optimizer state 在 HBM 与 SM 间的搬运。

### 3.3 数学原理与推导链

**RMSNorm 的本质：用 RMS 代替 standard deviation，完全删除均值与 bias 项。**

【动机导入】：上一段 LayerNorm 公式中的 $\mu$ 与 $b_i$ 是这次要被移除的对象。

【绝对数学内核】*（补充自论文 RMSNorm 的核心定义）*：

$$
\mathrm{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} a_i^2},\qquad \bar{a}_i = \frac{a_i}{\mathrm{RMS}(\mathbf{a})}\, g_i
$$

讲义 Page 14 给出的简化形式（带 $\varepsilon$ 与逐元素 $\gamma$）等价：

$$
y = \frac{x}{\sqrt{\tfrac{1}{n}\sum x_i^2 + \varepsilon}} \cdot \gamma
$$

【符号字典】：

| 符号 | 工程含义 / 代码映射 |
|---|---|
| $\mathbf{a}\in\mathbb{R}^n$ | 单个 token 在 $d_{model}$ 维度上的 pre-norm activations |
| $n$ | $d_{model}$ |
| $\mathrm{RMS}(\mathbf{a})$ | 该向量的均方根，将其投影到半径 $\sqrt{n}$ 的球面上 |
| $g_i / \gamma_i$ | 可学习 gain，shape `[d_model]`，初始化为 1，对应 `RMSNorm.weight` |
| $\varepsilon$ | 数值稳定项，典型 `1e-5`/`1e-6`，防止 RMS 趋零时溢出 |
| $b_i$（缺席） | RMSNorm 中无 bias，比 LayerNorm 少一组 `[d_model]` 参数 |

【项的拆解】：

- 去掉 $\mu$ → 去掉「re-centering」自由度；该自由度的工程价值是允许激活分布有非零中心，但本质上和下游 affine 层冗余。
- 去掉 $b_i$ → 与讲义 Page 18 的全局「dropping bias terms」呼应：bias 在 GLU/SwiGLU 等 gated 体系下既不必要又拖累 stability。

【隐式学习率自适应】*（补充自论文 RMSNorm 的梯度分析）*：

$$
\mathbf{R}' = \tfrac{1}{\delta}\mathbf{R}\;\Rightarrow\;\frac{\partial \mathcal{L}}{\partial \mathbf{W}} \propto \frac{1}{\|\mathbf{W}\|}
$$

权重 $\mathbf{W}$ 被任意常数 $\delta$ 放大时，输入到 RMSNorm 的 $\mathbf{a}$ 也被同步放大，但 normalize 之后输出不变，因此梯度对 $\mathbf{W}$ 的依赖被 $\|\mathbf{W}\|$ 自然反比，等价于免费的 gradient clipping。

### 3.4 系统演进与接口对接

**与 Lecture 02 的资源账本对接：RMSNorm 同时减少 normalization 时间（bandwidth）与训练 memory（少一组 bias 参数及其 grad/optimizer state）。**

接口对比：

| 项 | LayerNorm | RMSNorm |
|---|---|---|
| 可训练参数 shape | `weight [d_model]`、`bias [d_model]` | 仅 `weight [d_model]` |
| 前向 reductions | mean + variance | 单个 mean of squares |
| 前向 FLOPs/byte | 偏低（memory-bound） | 更低 memory traffic，但仍 memory-bound |
| pre-norm 适配 | 可，但更常配合 LayerNorm 出现于历史模型 | LLaMA 家族的标准搭配（pre-RMSNorm） |

【现代 Pre-RMSNorm 实现】*（补充自论文 RMSNorm 的代码指引）*：

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight
```

下游模块（Pre-Norm Transformer block）：

```text
x -> RMSNorm -> attention(QK with RoPE) -> + residual
  -> RMSNorm -> SwiGLU FFN -> + residual
```

---

## 4. Dropping Bias Terms 与 Norm 总结

### 4.1 技术路线演进逻辑

**讲义 Page 18–19：现代 LM 普遍丢掉 linear/norm 的 bias。**

【前置基线】：原始 Transformer 的 FFN 是

$$
\mathrm{FFN}(x) = \sigma(xW_1 + b_1)W_2 + b_2
$$

【核心崩溃点】：bias 在 large-scale pretraining 中的边际价值很低，但每个 bias 都要参与 grad/optimizer state 计算，并在大模型中数值不稳定。

【破局机制】：移除所有 bias，FFN 退化为

$$
\mathrm{FFN}(x) = \sigma(xW_1)W_2
$$

讲义点名两个动机：「memory（同 RMSNorm 一样的搬运成本）」与「optimization stability」。

### 4.2 系统设计与资源权衡链

**这是一个零成本的“减肥”：模型容量损失可忽略，但参数 / gradient / optimizer state 都减少同等量。**

留白原则：本模块不涉及新的硬件特性对齐，是上层参数表的剪裁。

### 4.3 数学原理与推导链

本模块属于参数集合层面的剪裁，讲义未提供新的推导。

### 4.4 系统演进与接口对接

**所有现代 weight shape 都假定 `linear.bias = None`、`norm.bias = None`；在 Lecture 02 训练 memory accounting 中，每个 linear 节省 $d_{out}$ 个 fp32-equivalent optimizer state slot。**

---

## 5. Activations: ReLU → GeLU → *GLU

### 5.1 技术路线演进逻辑

**讲义 Page 20–26：activation function 在十年里从 ReLU 走到 *GLU 家族；现代主流是 SwiGLU。**

【前置基线】：

- **ReLU**: $\max(0, x)$ — 原始 Transformer、T5、Gopher、Chinchilla、OPT。
- **GeLU (Gaussian Error Linear Unit)**: $x\Phi(x)$ — GPT1/2/3、GPTJ、GPT-NeoX、BLOOM。

【核心崩溃点】：纯非线性激活无 gating，FFN 的非线性表达受限于「逐元素函数」。

【破局机制】：**GLU (Gated Linear Unit)** —— 在 FFN 第一段乘上一支可学的线性 gate，逐元素相乘后再过非线性。讲义 Page 22 写出从 FFN 到 ReGLU 的演进：

$$
\mathrm{FF}(x) = \max(0, xW_1)W_2 \quad\Longrightarrow\quad \mathrm{FF}_{\text{ReGLU}}(x) = \big(\max(0, xW_1)\otimes xV\big) W_2
$$

不同非线性给出不同 GLU 变体：

| 名称 | 非线性 $\sigma(\cdot)$ | 代表模型 |
|---|---|---|
| ReGLU | ReLU | — |
| GeGLU | GeLU | T5 v1.1, mT5, LaMDA, Phi3, Gemma 2/3/4 |
| **SwiGLU** | Swish $= x\cdot\sigma_{\text{sig}}(x)$ | LLaMA 1/2/3, PaLM, Mistral, OLMo, 2023 后绝大多数 |

【效果证据】：讲义 Page 24–25 引 Shazeer 2020 / Narang et al. 2020 — GLU 家族对 perplexity 有一致小幅提升。Nemotron 340B 选 Squared ReLU 是少数 outlier。

### 5.2 系统设计与资源权衡链

**GLU 引入额外参数矩阵 $V$，使第一段线性层参数翻倍；通过把 $d_{ff}$ 缩到 $2/3$ 维度，FLOPs 与原始 FFN 持平。**

【架构变更】：FFN 第一段从「一个 linear」变成「两个并行 linear + 逐元素乘」，第二段不变。

【资源置换】：

- 不缩 $d_{ff}$：参数 +50%、FLOPs +50%、收益有限。
- 缩 $d_{ff} \leftarrow (2/3) d_{ff}$：参数与 FLOPs 与 ReLU FFN 持平，但获得 gating 表达力。

【硬件对齐】：两支线性矩阵可并行执行；与 attention 的 QKV 投影一样属于「fused matmul」友好结构。

### 5.3 数学原理与推导链

**【动机导入】**：FFN 想保留容量、加 gating，但不希望 FLOPs 翻倍。

**【公式】SwiGLU 完整形式**：

$$
\mathrm{SwiGLU}(x) = \big(\mathrm{Swish}(xW_1)\otimes (xV)\big) W_2,\quad \mathrm{Swish}(z) = z\cdot \sigma_{\text{sig}}(z)
$$

【符号字典】：

| 符号 | 工程含义 |
|---|---|
| $x \in \mathbb{R}^{B\times S\times d_{model}}$ | block 输入 activations |
| $W_1 \in \mathbb{R}^{d_{model}\times d_{ff}}$ | gate 之前的非线性支线性层 |
| $V \in \mathbb{R}^{d_{model}\times d_{ff}}$ | gate 支线性层 |
| $W_2 \in \mathbb{R}^{d_{ff}\times d_{model}}$ | 输出投影 |
| $\sigma_{\text{sig}}$ | sigmoid，逐元素 |
| $\otimes$ | Hadamard（逐元素）积 |
| $d_{ff}$ | 在 GLU 下取 $(2/3)\cdot 4 d_{model} = (8/3) d_{model}$（见 §7） |

【项的拆解】：

- $\mathrm{Swish}(xW_1)$：可学非线性。
- $xV$：信号 gate，逐元素相乘起到「软开关」作用，压制噪声 channel。
- $W_2$：把 gated representation 投回 $d_{model}$。

### 5.4 系统演进与接口对接

**对接 Lecture 02 的 `2BDK` matmul 账本：SwiGLU 把单层 FFN 从「`[B,S,d_model] @ W_1[d_model,d_ff]`」拆成两条同 shape 的并行 matmul，输出仍是 `[B,S,d_model]`。**

对 Lecture 02 的训练 FLOPs $6ND$：参数 $N$ 在 GLU 下因 $W_1$ 与 $V$ 而增加，但 $d_{ff}$ 缩到 $2/3$ 后总参数等同 ReLU FFN，公式直接复用。

---

## 6. Serial vs Parallel Transformer Block

### 6.1 技术路线演进逻辑

**讲义 Page 27–28：默认的 Transformer block 是 serial（attention → MLP），少数模型（GPT-J、PaLM、GPT-NeoX、Cohere Command A、Falcon 2 11B、Command R+）改为 parallel。现代主流仍是 serial。**

【前置基线】：serial block

$$
y = x + \mathrm{Attn}(\mathrm{Norm}(x));\quad z = y + \mathrm{MLP}(\mathrm{Norm}(y))
$$

【核心崩溃点】：attention 与 MLP 之间存在数据依赖，两者的 norm + matmul 不能并行，深层 transformer 的 wallclock 受限。

【破局机制】：parallel block

$$
z = x + \mathrm{Attn}(\mathrm{Norm}(x)) + \mathrm{MLP}(\mathrm{Norm}(x))
$$

如果 attention 与 MLP 共享同一个 norm，norm 只算一次；attention/MLP 的 matmul 可 fuse 成一个更大的 GEMM。

### 6.2 系统设计与资源权衡链

**架构变更**：dependency graph 从「attn → mlp」变成「attn ∥ mlp，最后求和」。
**资源置换**：用「数学上稍弱的 expressiveness（attn 与 mlp 看到同一份输入，不能像 serial 那样让 mlp 加工 attn 的结果）」换「同一层 norm 与 matmul 的可融合性」。
**硬件对齐**：在 large model 上 fused QKV/MLP matmul 是高 arithmetic intensity 的 compute-bound op，更接近 H100 peak。

### 6.3 数学原理与推导链

讲义未给出 parallel block 的严格表达力对比，只提工程效果，本节按留白原则不补外部公式。

### 6.4 系统演进与接口对接

**对 Lecture 02 的 tensor flow 而言，parallel block 不改 shape，只改 dependency graph；FLOPs 总量相近，但 kernel fusion 潜力更大。Serial 仍是 default。**

---

## 7. Position Embeddings 与 RoPE

### 7.1 技术路线演进逻辑

**讲义 Page 30–35：position encoding 从 sine → absolute learnable → relative bias → RoPE，是一条「让 attention 只看相对位置」的逐步逼近史。**

【前置基线】（讲义 Page 30）：

| 类型 | 公式 | 代表模型 |
|---|---|---|
| Sine | $\mathrm{Embed}(x, i) = v_x + \mathrm{PE}_{pos}$ | 原始 Transformer |
| Absolute learnable | $\mathrm{Embed}(x, i) = v_x + u_i$ | GPT 1/2/3, OPT |
| Relative attention bias | 在 attention logits 上加 $b_{i-j}$ | T5, Gopher, Chinchilla |
| **RoPE** | 见下式 | GPT-J, PaLM, LLaMA, 大多数 2024+ 模型 |

【核心崩溃点】：sine 与 absolute 在 attention inner product 中会展开出 *非相对位置* 的 cross terms；relative bias 不是 inner-product 形式，不易在 attention kernel 里高效实现。

【破局机制】：RoPE 把位置编码改写成 query/key 上的 **2D 平面旋转**：在 $i$ 这个位置把每一对 $(q_{2k}, q_{2k+1})$ 在 2D 平面旋转角度 $\theta_k \cdot i$，让 inner product 自动只剩 $i-j$ 项：

$$
\langle f(q, i), f(k, j)\rangle = g(q, k, i-j)
$$

### 7.2 系统设计与资源权衡链

**架构变更**：position information 从「加性 embedding（一次性注入）」变成「乘性旋转（每层 attention 都施加一次）」。
**资源置换**：不增加 trainable 参数；每层 attention 多两次 elementwise 乘法 + sin/cos table lookup，FLOPs 极少。
**硬件对齐**：rotary 矩阵稀疏（每个 2D 块独立），可用 fused kernel 在 attention forward 中合并。
**模型变体**：讲义 Page 33 提到 Gemma 4 的折中——「只对前两个维度对做 rotation」，进一步省 FLOPs。

### 7.3 数学原理与推导链

**【动机导入】**：希望存在函数 $f$ 使得 attention 只依赖相对位置 $i-j$，且能写成 inner product 形式。

**【关键性质】**：inner product 对正交变换不变；2D 旋转矩阵是正交且参数化简单的家族。

**【RoPE 公式】**：将 query $q\in\mathbb{R}^{d}$ 按相邻两维分组 $(q_{2k}, q_{2k+1})$，每组按位置 $i$ 旋转角 $i\theta_k$：

$$
\begin{pmatrix} q'_{2k} \\ q'_{2k+1} \end{pmatrix} = \begin{pmatrix} \cos(i\theta_k) & -\sin(i\theta_k) \\ \sin(i\theta_k) & \cos(i\theta_k) \end{pmatrix} \begin{pmatrix} q_{2k} \\ q_{2k+1} \end{pmatrix}
$$

同样的旋转作用到 key 上。Inner product 写为：

$$
\langle q'_i, k'_j\rangle = \sum_k \big[ q_{2k} k_{2k} + q_{2k+1} k_{2k+1} \big] \cos((i-j)\theta_k) + \big[\dots\big]\sin((i-j)\theta_k)
$$

只依赖 $i-j$，与 absolute position 无关。

【符号字典】：

| 符号 | 工程含义 |
|---|---|
| $q, k$ | attention 的 query/key 向量，shape `[head_dim]` |
| $d$ | head_dim，必须为偶数以便配对 |
| $\theta_k$ | 第 $k$ 个 2D 平面的旋转基频，常取 $\theta_k = 10000^{-2k/d}$ |
| $i, j$ | query/key 在 sequence 中的绝对位置 |

【与 sine PE 的差异】：sine PE 是**加性**注入到 embedding，再一路传到 attention，inner product 会展开出 $\langle v_x, \mathrm{PE}_i\rangle$ 这类 cross terms；RoPE 是**乘性**注入到 Q/K，且只在 attention 内部出现，inner product 没有跨项。

### 7.4 系统演进与接口对接

**对接 Lecture 02 的 tensor flow**：

```text
Q, K = [B, H, S, head_dim]
RoPE: 对 [B, H, S, head_dim] 中相邻两维 (k, k+1) 施加按位置 i 的 2D 旋转
attention = softmax(Q' K'^T / sqrt(head_dim)) V
```

实现要点（讲义 Page 35）：rotary 矩阵 cos/sin 表预计算；每次 attention 前对 Q/K 各做一次 elementwise 乘；其余 attention 计算与标准 multi-head self-attention 完全一致。

---

## 8. 超参数 1：$d_{ff} / d_{model}$ 比

### 8.1 技术路线演进逻辑

**讲义 Page 37–41：行业共识 $d_{ff} = 4 d_{model}$；GLU 模型缩到 $(8/3) d_{model}$；T5 11B 是 64× 的极端反例。**

【前置基线】：原始 Transformer 设定 $d_{ff} = 4 d_{model}$。

【核心崩溃点】：每代新模型都要重新选择 FFN 宽度；选错既浪费参数又损害性能。

【破局机制】：基于多年实验形成的「near-optimal basin」共识。讲义 Page 40 引用 Kaplan+ 2020：在 $1$–$10$ 倍区间内 FFN 比例几乎平坦。

### 8.2 系统设计与资源权衡链

**资源置换**：FFN 矩阵参数占 $2 \cdot d_{model} \cdot d_{ff}$；比值过大会浪费 memory/optimizer state，过小会损失非线性容量。

【经验对照表】（讲义 Page 38）：

| 模型 | $d_{ff} / d_{model}$ |
|---|---:|
| PaLM | 4 |
| Mistral 7B | 3.5 |
| LLaMA-2 70B | 3.5 |
| LLaMA 70B | 2.68 |
| Qwen 14B | 2.67 |
| DeepSeek 67B | 2.68 |
| Yi 34B | 2.85 |
| T5 v1.1 | 2.5 |
| **T5 (11B)** | **64** |
| Gemma 2 | 8 |
| SmolLM / Gemma 3 / Gemma 4 | 4 (GLU) |

【硬件对齐】：$d_{ff}$ 是 matmul 的 reduction 维之一，决定 FFN GEMM 的 arithmetic intensity（Lecture 02 中 $I_{\text{matmul}} \approx n/3$）；选 8/3 与 4 在主流 GPU 上都能落在 compute-bound 区。

### 8.3 数学原理与推导链

**【动机导入】**：希望同 GLU FFN 与 ReLU FFN 在「参数量 / FLOPs」上保持等价。

**【推导】**：

- ReLU FFN 第一段参数：$d_{model} \cdot d_{ff}$。
- GLU FFN 第一段参数：$2 \cdot d_{model} \cdot d_{ff}^{\text{GLU}}$（多一个 $V$）。
- 等参数条件 $\Rightarrow$ $d_{ff}^{\text{GLU}} = (1/2) d_{ff}^{\text{ReLU}}$，再考虑实际 GLU 实现常用稍大比例，工业上落到 $d_{ff}^{\text{GLU}} = (2/3)\cdot 4 d_{model} = (8/3) d_{model}$。

讲义 Page 41 对 T5 的 64× 评价：「跑通了，但 T5 v1.1 的 follow-up 已经回归 2.5×，说明 64× 大概率次优」。

### 8.4 系统演进与接口对接

**对接 Lecture 02 的 training memory 公式：FFN 占了大多数 weight memory（两条 $d_{model}\times d_{ff}$ 矩阵），选 $d_{ff}/d_{model}$ 直接决定 $N_{\text{params}}$，进而经过 $6ND$ 决定训练时间。**

---

## 9. 超参数 2：head_dim · num_heads vs $d_{model}$

### 9.1 技术路线演进逻辑

**讲义 Page 42–43：行业共识 head_dim · num_heads = $d_{model}$（比例 1），少数 Google 模型偏离。**

【前置基线】：CS224n 教学版本默认 $d_{model} = h \cdot k$，即 head 切片刚好拼回 model dim。

【核心崩溃点】：从代数上 nothing forces it — 可以让 head_dim · num_heads > $d_{model}$（更冗余的 attention 空间）。

【破局机制】：让该比值约为 1 是大多数模型的「不必思考的默认」。讲义实证：

| 模型 | num_heads | head_dim | $d_{model}$ | ratio |
|---|---:|---:|---:|---:|
| GPT-3 | 96 | 128 | 12288 | 1 |
| T5 | 128 | 128 | 1024 | 16 |
| T5 v1.1 | 64 | 64 | 4096 | 1 |
| LaMDA | 128 | 128 | 8192 | 2 |
| PaLM | 48 | 258 | 18432 | 1.48 |
| LLaMA 2 | 64 | 128 | 8192 | 1 |
| Qwen 3.5 (27B) | 24 | 256 | 5120 | 1.2 |

### 9.2 系统设计与资源权衡链

**架构变更**：head_dim · num_heads 决定 QKV 投影矩阵 shape `[d_model, head_dim · num_heads]`；选 ratio = 1 等价于「不增大 QKV 参数」。
**资源置换**：ratio > 1 增加 attention 表达力但 QKV 矩阵参数和 KV cache 同步膨胀。
**硬件对齐**：选 head_dim 为 64/128/256 是为了对齐 GPU tensor core 的 K 维度。

### 9.3 数学原理与推导链

本模块属于参数表设定，讲义未给出推导。

### 9.4 系统演进与接口对接

**对接 Lecture 02 的 attention shape**：QKV projection 是 `[..., d_model] @ [d_model, h·k] -> [..., h·k] -> reshape [..., h, k]`。当 ratio = 1 时，KV cache 与 $d_{model}$ 同量级；ratio > 1 时 KV cache 同比膨胀，对推理 memory-bound 阶段（Lecture 02 §7 的 matvec 情景）直接放大压力。

---

## 10. 超参数 3：Aspect Ratio $d_{model} / n_{layer}$

### 10.1 技术路线演进逻辑

**讲义 Page 44–46：大部分 LLM 的 aspect ratio 落在 60–200 之间；超出此范围（如 T5 11B 的 33）则是异类。**

【前置基线】：早期没有共识，T5 (11B) 选 33（很深）。

【核心崩溃点】：极深模型 (大 $n_{layer}$) 的 pipeline parallelism 和 latency 不友好；极宽模型 (大 $d_{model}$) 又损 sample efficiency。

【破局机制】：经验上 100–200 之间是 sweet spot。

| 模型 | $d_{model}/n_{layer}$ |
|---|---:|
| BLOOM | 205 |
| T5 v1.1 | 171 |
| PaLM (540B) | 156 |
| GPT-3 / OPT / Mistral / Qwen / OLMo 3 | 128 |
| LLaMA / LLaMA 2 | 102 |
| Gemma 3 | 87 |
| Gemma 4 | 61 |
| T5 (11B) | 33 |

### 10.2 系统设计与资源权衡链

**架构变更**：固定总参数预算 $N \approx 12 \cdot n_{layer}\cdot d_{model}^2$ 下，aspect ratio 直接决定「深」还是「宽」。
**资源置换**（讲义 Page 45–46 引 Tay et al. 2021 / Kaplan et al. 2020）：

- 太深：pipeline depth 高，inference latency 大，pipeline-parallel bubble 多。
- 太宽：activation memory 与 attention 维度增大，长序列下 attention compute 受影响。

【硬件对齐】：现代 LLM 倾向 $\approx 128$，原因是匹配主流 GPU 的 tensor parallel + pipeline parallel 切分粒度。

### 10.3 数学原理与推导链

讲义未给出闭式 scaling law，引用 Kaplan/Tay 的经验曲线，本节按留白原则不补外部推导。

### 10.4 系统演进与接口对接

**对接 Lecture 02 的 training memory：$d_{model}^2$ 决定单层参数与 activation footprint，$n_{layer}$ 是 activation checkpointing 的 $L$（决定 $O(\sqrt{L})$ memory tradeoff）。aspect ratio 是「系统层」决定，而非纯模型质量决定。**

---

## 11. 超参数 4：Vocabulary Size

### 11.1 技术路线演进逻辑

**讲义 Page 47：monolingual 模型 30–50k，multilingual / 生产模型 100–250k。**

| 类型 | 范围 | 代表模型 |
|---|---|---|
| Monolingual | 32k–64k | LLaMA 32000, GPT-3 50257, T5 32128, Yi 64000 |
| Multilingual / production | 100k–262k | GPT-4 100276, DeepSeek 100000, Qwen 15B 152064, mT5 250000, PaLM 256000, Gemma 4 262144 |

### 11.2 系统设计与资源权衡链

**资源置换**：vocab size $V$ 直接决定 embedding/output projection 矩阵 `[V, d_model]` 的参数与 FLOPs；multilingual 必须吃这个 cost。

### 11.3 数学原理与推导链

讲义未展开 BPE/SentencePiece 等具体 tokenization 数学，按留白原则不补。

### 11.4 系统演进与接口对接

**对接 Lecture 02 的 tensor shape：embedding lookup `[V, d_model]` 与 final softmax `[d_model, V]` 在 vocab 越大时越重，且 softmax 是 Lecture 02 中典型的 memory-bound op，会被本讲第 13 节的 z-loss 进一步“关心”。**

---

## 12. 正则化：Dropout 与 Weight Decay

### 12.1 技术路线演进逻辑

**讲义 Page 48–50：大规模 pretraining 不需要传统正则；但 weight decay 仍被普遍保留，且不是为了防过拟合。**

【前置基线】：CV/NLP 时代 dropout 0.1 + weight decay 0.0/0.1 是标配。

【核心崩溃点】：LLM 数据量远大于参数量、单 epoch SGD，「过拟合 corpus」不是主要矛盾。

【破局机制】：

- 老模型（Original Transformer、GPT-2、GPT-3、T5、OPT、Qwen 14B）仍用 dropout 0.1。
- 新模型（T5 v1.1、PaLM、LLaMA）几乎不用 dropout，但保留 weight decay 0.1。

| 模型 | Dropout | Weight Decay |
|---|---:|---:|
| Original Transformer | 0.1 | 0 |
| GPT-2 | 0.1 | 0.1 |
| GPT-3 | 0.1 | 0.1 |
| T5 v1.1 | 0 | 0 |
| PaLM | 0 | variable |
| OPT | 0.1 | 0.1 |
| LLaMA | 0 | 0.1 |
| Qwen 14B | 0.1 | 0.1 |

### 12.2 系统设计与资源权衡链

**Weight decay 在 LLM 里的角色已经变了**：讲义 Page 50 引 Andriushchenko et al. 2023——weight decay 不再是「控制 overfitting」，而是「与 cosine LR schedule 交互，改变 optimization dynamics」（影响有效学习率的衰减节奏）。

### 12.3 数学原理与推导链

讲义未展开 weight decay 与 cosine schedule 的耦合公式，本节按留白原则不补外部推导。

### 12.4 系统演进与接口对接

**对接 Lecture 02 的 optimizer accounting：weight decay 在 AdamW 中是 update 时的解耦项 $\theta \leftarrow \theta - \eta(\hat{m}/(\sqrt{\hat{v}}+\varepsilon) + \lambda \theta)$，本身不增加 optimizer state，但通过改变 effective step size 影响训练曲线。**

---

## 13. Stability Tricks：Softmax 三件套

### 13.1 技术路线演进逻辑

**讲义 Page 52–56：现代 LM 的训练曲线如果出现「blue curve」（loss spike），通常元凶是 softmax 端的数值不稳定。三件套是 z-loss、QK-norm、logit soft-capping。**

【前置基线】：标准 softmax + cross entropy；标准 attention $\mathrm{softmax}(QK^\top/\sqrt{d_k})$。

【核心崩溃点】：softmax 的指数与归一化对极端 logits 高度敏感——某个 logit 跳到 $+\infty$ 就会让 partition function 爆炸或一个概率独占。这在 fp16/bf16 训练中更脆弱。

【破局机制】：在 softmax 的入口或归一化项上加约束。

#### 13a. Output Softmax — Z-loss

**【动机导入】**：softmax 的 partition $Z = \sum_j e^{x_j}$ 越大越容易溢出；希望让 $\log Z$ 保持靠近 0。

**【公式】**：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha \cdot (\log Z)^2,\qquad Z = \sum_j \exp(x_j)
$$

讲义 Page 54 注明 z-loss 由 PaLM 使用，并被 Baichuan 2 (2023)、DCLM (2024)、OLMo 2/3 (2025) 采纳；source 是 Devlin 2014（早期 NMT）。

**【符号字典】**：

| 符号 | 工程含义 |
|---|---|
| $\mathcal{L}_{\text{CE}}$ | 标准 cross-entropy loss |
| $x_j$ | output logits（vocab 上的 raw score） |
| $Z$ | softmax partition function |
| $\alpha$ | z-loss 权重，典型 $10^{-4}$ 数量级 |

**【项的拆解】**：$(\log Z)^2$ 是一个对 partition 量级的「锚定项」，把 $\log Z$ 推近 0，从而压住 logits 的整体偏移。它**不限制每个 logit 的相对差异**，因此不直接损害学习能力。

#### 13b. Attention Softmax — QK-norm

**【动机导入】**：在 long-context 或 mixed precision 下，$QK^\top$ 容易出现极大值导致 attention softmax 退化为 one-hot 或下溢。

**【公式】**：在 attention 前对 Q、K 各加一次 RMSNorm / LayerNorm：

$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{\mathrm{Norm}(Q) \mathrm{Norm}(K)^\top}{\sqrt{d_k}}\right) V
$$

讲义 Page 55 列出使用模型：DCLM、OLMo 2、Gemma 2、Qwen 3、OLMo 3、Gemma 4；起源是 vision/multimodal 模型（Dehghani 2023、Idefics、Chameleon）。

**【符号字典】**：

| 符号 | 工程含义 |
|---|---|
| $Q, K$ | attention 的 query/key tensor，shape `[B, H, S, head_dim]` |
| $\mathrm{Norm}$ | RMSNorm 或 LayerNorm，作用于最后一维 head_dim |
| $d_k$ | head_dim |

**【项的拆解】**：QK-norm 把 Q、K 的 magnitude 锚到固定 RMS，logits = $QK^\top/\sqrt{d_k}$ 的范围被有效压缩；这等价于给 attention 加了一层「pre-softmax 温度控制」。

#### 13c. Logit Soft-capping

**【动机导入】**：兜底 stability——即使 z-loss/QK-norm 都加了，仍可能出现 outlier logits。

**【公式】**（讲义 Page 56）：

$$
\hat{x} = c \cdot \tanh(x / c)
$$

把每个 logit $x$ 经过 $\tanh$，映射到区间 $(-c, c)$。

【符号字典】：

| 符号 | 工程含义 |
|---|---|
| $x$ | softmax 之前的 logit |
| $c$ | soft cap 上限，典型 $c=30$ 或 $50$ |
| $\hat{x}$ | 截断后的 logit，喂进 softmax |

【项的拆解】：$\tanh$ 是平滑的 saturating 函数，比 hard clip 更易反向传播；但讲义提示其有「perf issues?」——可能影响最终精度，尚未达成共识。

### 13.2 系统设计与资源权衡链

| 技巧 | 注入位置 | 增加 FLOPs | 增加显存 | 主要风险 |
|---|---|---|---|---|
| z-loss | output softmax loss | 极少（一个标量项） | 无 | 调 $\alpha$ 不当反而拖累训练 |
| QK-norm | attention 内部 | 每层 attention +2 次 RMSNorm | 多两组 gain 参数 | 与 RoPE 顺序需明确 |
| Soft-capping | output / attention logits | $\tanh$ 调用 | 无 | 可能损 perf |

### 13.3 系统演进与接口对接

**对接 Lecture 02 的 mixed precision policy**：这三件套都是为「activations/grads 在 bf16 下不爆」服务的；本质上是把 softmax 端的数值动态范围**主动收窄**，让 bf16 与 fp32 在 loss/attention 上的差距不放大。

---

## 14. Attention Heads：MQA / GQA / Sliding Window / Interleaved

### 14.1 技术路线演进逻辑

**讲义 Page 57–66：训练阶段 attention 是 compute-bound 的大 matmul（Lecture 02 §7），但推理 decode 阶段是 memory-bound 的 matrix-vector（KV cache 主导），主要在 attention head 维度做手术。**

【前置基线】：标准 Multi-Head Attention (MHA) — $h$ 个 head 各自有独立的 K、V projection。

【核心崩溃点】：自回归 decode 阶段每次只生成一个 token，KV cache 是 `[B, H, S, head_dim]`；arithmetic intensity 为

$$
I_{\text{decode}} \sim \left(\frac{n}{d}+\frac{1}{b}\right)^{-1}
$$

n/d 项难以减小，导致 GPU 长期 memory-bound（讲义 Page 60）。

【破局机制】：

- **MQA (Multi-Query Attention)** [Shazeer 2019]：所有 query head 共享同一组 K、V（即 K、V 退化为 `[B, 1, S, head_dim]`）。
- **GQA (Grouped-Query Attention)** [Ainslie 2023]：MHA 与 MQA 之间的中点，$g$ 个 query head 共享一组 K、V（KV cache shape `[B, h/g, S, head_dim]`）。
- **MLA (Multihead Latent Attention)** [DeepSeek v2]：把 K、V 压缩进 latent 空间，进一步缩 KV cache。
- **Sparse / Sliding-Window Attention (SWA)** [Child et al. 2019, Mistral, GPT-4, GPT-OSS, Gemma 4]：每个 token 只看局部窗口 $w$，attention complexity 从 $O(S^2)$ 降到 $O(Sw)$。
- **Interleaved full + SWA**（Cohere Command A、LLaMA 4、Gemma 3/4、OLMo 3）：每 $k$ 层插一层 full attention 保留长程能力，其余 SWA。

### 14.2 系统设计与资源权衡链

**【架构变更】**：attention 从「所有 head 等价」变为「query head ≠ key/value head 数」。
**【资源置换】**：

| 变体 | KV cache 维度 | 推理 arithmetic intensity | 训练精度损失 |
|---|---|---|---|
| MHA | `[B, H, S, k]` | 低（讲义 $I \sim n/d$） | baseline |
| MQA | `[B, 1, S, k]` | 显著提高（多了 $1/(d\cdot h)$ 项） | 轻微 perplexity 损失（讲义 Page 63 Shazeer 2019） |
| GQA | `[B, H/g, S, k]` | 居中可调 | 几乎无损（讲义 Page 63 Ainslie 2023） |
| SWA | `[B, H, w, k]` 窗口截断 | 视窗口决定，长程能力下降 | 与 full attention 互补 |

**【硬件对齐】**：KV cache 是 decode 阶段 HBM 带宽的最大消费者；缩小 KV cache 直接减少每步 memory traffic。

**【讲义计算细节，Page 58–61】**：

- MHA decode total ops: $\Theta(bnd^2)$；total memory access: $\Theta(bn^2 d + nd^2)$；intensity $\sim (n/d + 1/b)^{-1}$。
- MQA decode：memory access 中 KV term 从 $bn^2 d$ 降到 $bn^2 k$（其中 $k = d/h$），intensity 提升为 $(1/d + n/(dh) + 1/b)^{-1}$。

### 14.3 数学原理与推导链

**【动机导入】**：从 forward 单步 prefill 切换到 autoregressive decode 后，序列长度由 $n$ 个并行 query 收缩为 1 个 query，arithmetic intensity 暴跌。

**【训练阶段 attention intensity】**：

- ops: $\Theta(bnd^2)$；memory: $\Theta(bnd + bhn^2 + d^2)$；
- intensity: $\Theta((1/k + 1/(bn))^{-1})$，与 $k = d/h$、batch、序列长有关。

**【decode 阶段 attention intensity】**：

- ops: $\Theta(bnd^2)$；memory: $\Theta(bn^2 d + nd^2)$；
- intensity: $\Theta((n/d + 1/b)^{-1})$。当 $n$ 远小于 $d$ 时还能维持，当 $n$ 接近或超过 $d$ 时退化严重。

**【MQA 修正】**：

- memory：$\Theta(bnd + bn^2 k + nd^2)$；
- intensity: $\Theta((1/d + n/(dh) + 1/b)^{-1})$。

**【符号字典】**：

| 符号 | 工程含义 |
|---|---|
| $b$ | batch size |
| $n$ | 当前 sequence length |
| $d$ | $d_{model}$ |
| $h$ | number of query heads |
| $k$ | head_dim $= d/h$ |
| $g$ | GQA group size（query head 数 / kv head 数） |
| $w$ | sliding window 宽度 |

【项的拆解】：

- intensity 公式中 $n/d$ 项来自「KV cache scan 与 hidden dim matmul 的比值」；这是 MHA decode 最难压的项。
- MQA 把该项变成 $n/(d h)$，立刻获得 $h$ 倍的 intensity 改进。
- GQA 在该项上变成 $n/(d \cdot h/g)$，可调 $g$ 平衡 expressiveness 与 cache。

### 14.4 系统演进与接口对接

**对接 Lecture 02 §7 roofline：MHA decode 是「matrix-vector 类 memory-bound」的代表；MQA/GQA 用更小 KV cache 直接降低 memory traffic，把 decode 从 strict memory-bound 推回更平衡的位置。**

**对接 Lecture 02 §11 gradient accumulation / §12 activation checkpointing**：SWA 的训练 activation memory 与窗口 $w$ 线性相关，而非 sequence length $S$，等价于在 attention 维度做了「天然的 checkpointing」。Interleaved full+SWA 则把这种 checkpoint 模式按 layer 间隔分布，让长程信息靠少数 full layer 与 NoPE 维持，而每层默认是 RoPE+SWA（讲义 Page 65）。

后续接口（讲义 Page 57 末尾）：「Exotic SSM stuff (Jamba, Falcon 3, Qwen 3.5)」会在 **Lecture 04** 接入，那是下一讲对 attention 进一步替换的预告。

---

## 15. Lecture 03 总结接口

**Lecture 03 给 Lecture 02 的资源账本配上了一张「现代 LLM 默认参数表」：当你要训一个新模型时，下表的每一行都是「不思考就用 / 思考之后再偏离」的 baseline。**

| 维度 | 现代默认 | 主要替代 | 选择依据 |
|---|---|---|---|
| Norm 位置 | pre-norm | non-residual post-norm (Olmo 2) | stability + 大 LR |
| Norm 类型 | RMSNorm（无 bias） | LayerNorm | bandwidth + parameter |
| FFN 激活 | SwiGLU | GeGLU / ReLU / Squared ReLU | 一致小幅 perplexity 收益 |
| Linear bias | 无 | 有 | memory + stability |
| Position encoding | RoPE | NoPE（与 SWA 配合） | inner-product 形式的相对位置 |
| Block 拓扑 | serial | parallel (GPT-J, PaLM) | expressiveness vs fusion |
| $d_{ff}/d_{model}$ | 4（无 GLU）/ 8/3（GLU） | T5 11B 的 64× | Kaplan 平台 + 等参数 GLU |
| head_dim · num_heads | ≈ $d_{model}$ | LaMDA 2×、T5 16× | tensor core 对齐 |
| $d_{model}/n_{layer}$ | 100–200 | 极深 T5 11B 33 | parallelism + latency |
| Vocab | mono 30–50k / multi 100–250k | — | language coverage |
| Regularization | weight decay 0.1，无 dropout | 老式 dropout 0.1 | optimization dynamics |
| Output 稳定 | z-loss | 无 | softmax 数值边界 |
| Attention 稳定 | QK-norm | 无 | mixed precision |
| Logits 兜底 | soft-cap tanh（部分模型） | 无 | outlier 压制 |
| Attention head | GQA | MQA / MHA / MLA | 推理 KV cache |
| Attention pattern | full / interleaved full+SWA | sparse 等 | 长程 vs 短程 |

全局护栏：以上内容严格来自 `lecture/lecture_03.md`、历史本地产物 `output/cs336/lecture_02/`、以及 `paperAfterC/RootMeanSquareLayerNormalization_ConceptCard.md`；未使用外部搜索或外部教程补全。所有 *(补充自论文 RMSNorm)* 段落均显式标注，主线仍以 Stanford 讲义教学意图为准。
