# CS336 Lecture 03 Phase 1: Architect 技术结构拆解

## 0. 全局护栏与历史继承

**Lecture 03 的核心位置是：在 Lecture 01 的效率目标和 Lecture 02 的资源账本之上，解释现代 LLM architecture 与 hyperparameters 为什么收敛到一组相当保守的工程默认值。**

全局护栏：

- 本文只使用本地材料：`lecture/lecture_03/lecture_03.md`、`output/cs336/lecture_01/`、`output/cs336/lecture_02/`。
- 对 OCR 破损或讲义未展开的公式，不补外部证明；只保留讲义可读部分并标注不确定处。
- Phase 1 只做结构、技术路线、资源权衡和接口绑定；认知类比留给 Phase 2。

历史系统状态：

| 前序讲义 | 已建立接口 | Lecture 03 的增量 |
|---|---|---|
| Lecture 01 | `raw text -> tokenizer -> token IDs`，目标是 `accuracy = efficiency x resources` | 讨论 tokenizer 之后的模型本体设计：position embeddings、activation、normalization、attention heads |
| Lecture 02 | tensor memory、FLOPs、arithmetic intensity、training/inference resource accounting | 用这些账本解释 RMSNorm、no bias、GQA/MQA、SWA 等设计为何与 runtime/memory movement 绑定 |

本讲目标可以压缩成一句话：**现代 dense Transformer 的主干非常相似，真正仍在显著分化的地方主要是 position embeddings、activations、tokenization，以及为稳定性和推理成本加入的局部技巧。**

---

## 1. 从 Original Transformer 到现代 Assignment Variant

### 1.1 技术路线与演进逻辑

**本讲先把原始 Transformer 作为基线，再逐项说明现代 LLM 如何替换掉其中的高成本或不稳定默认值。**

前置基线是原始 Transformer：

- sinusoidal position embeddings
- ReLU FFN
- post-norm LayerNorm
- linear / LayerNorm 中保留 bias terms

现代简单变体，也就是 assignment 所实现的主线：

- LayerNorm 放在 block 前面，即 pre-norm
- RoPE 替代 additive sinusoidal / learned absolute embeddings
- SwiGLU 替代 ReLU
- linear / LayerNorm 不使用 bias terms

核心崩溃点不是原始 Transformer 不能训练，而是当模型规模、上下文长度和训练步数扩大后，训练稳定性、内存移动、推理 KV Cache 成本会变成主导约束。破局机制不是发明全新架构，而是在原始 Transformer 的接口上做小幅替换。

### 1.2 系统设计与资源权衡链

**现代 variant 的每一项替换都不是孤立美学选择，而是在稳定性、内存带宽、推理 latency 和表达能力之间做工程置换。**

| 替换项 | 架构变化 | 主要收益 | 主要代价或保留风险 |
|---|---|---|---|
| post-norm -> pre-norm | norm 移到 residual branch 前 | 更稳定，允许更大的 learning rate | 可能需要额外 norm 变体处理输出尺度 |
| sinusoidal / absolute -> RoPE | position 不再加到 embedding，而是旋转 query/key | attention score 天然绑定相对位置 | 实现时必须在每次 attention 的 Q/K 上施加 |
| ReLU -> SwiGLU | FFN 增加 gate branch | 表达能力更强，实证效果更好 | FF dimension 通常缩到约 $8/3 d_{model}$ 控制参数量 |
| LayerNorm -> RMSNorm | 去掉 mean-centering 和 beta | 减少统计归约与内存访问 | 不是所有收益都能由 FLOPs 解释 |
| keep bias -> no bias | linear/norm 删除 bias | 少量参数与 memory movement 下降，可能更稳 | 讲义强调经验实践，非严格定理 |

### 1.3 数学原理与推导链

**原始 Transformer 的三个可替换数学接口分别是 position encoding、FFN 和 normalization。**

原始 sinusoidal position embedding：

$$
PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $PE$ | position embedding table 中的数值 |
| $pos$ | sequence 中的 token 位置 |
| $i$ | embedding dimension 的 pair index |
| $2i,2i+1$ | 一对偶数/奇数维度，分别放 sine 与 cosine |
| $d_{model}$ | hidden/model dimension |
| $10000$ | 讲义给出的固定频率基数 |

原始 ReLU FFN：

$$
\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $x$ | 当前 token 的 hidden state |
| $W_1$ | 从 $d_{model}$ 投到 $d_{ff}$ 的第一层权重 |
| $b_1$ | 第一层 bias |
| $\max(0,\cdot)$ | ReLU activation |
| $W_2$ | 从 $d_{ff}$ 投回 $d_{model}$ 的第二层权重 |
| $b_2$ | 第二层 bias |

### 1.4 系统演进与接口绑定

**Lecture 03 接入 Lecture 02 的 tensor 账本后，Transformer block 不再只是数学图，而是一个会产生 memory movement、KV Cache 和 arithmetic intensity 差异的执行图。**

典型数据流：

```text
token IDs -> embedding -> hidden states [batch, seq, d_model]
hidden states -> norm -> attention(Q,K,V) -> residual
hidden states -> norm -> FFN/SwiGLU -> residual
hidden states -> output logits [batch, seq, vocab]
```

Lecture 01 的 tokenizer 决定 sequence length 和 vocab size；Lecture 02 的 resource accounting 决定这些 shape 在 memory、FLOPs、bandwidth 上的后果；Lecture 03 开始选择具体 block 内部机制。

---

## 2. Pre-Norm、Post-Norm 与 Non-Residual Postnorm

### 2.1 技术路线与演进逻辑

**现代 LLM 基本收敛到 pre-norm，因为它在大规模训练时比 original post-norm 更稳定。**

前置基线：original Transformer 与 BERT 使用 post-norm。现代 dense LMs 大多使用 pre-norm；讲义还列出 Grok、Gemma 2、OLMo 2 等引入 double norm 或 non-residual postnorm 的新变体。

讲义给出的解释包括 gradient attenuation 与 gradient spikes。核心崩溃点是 residual path 与 normalization 的位置会改变 gradient 在深层网络中的传播形态。破局机制是把 norm 放到 attention/MLP 子层之前，让 residual path 更直接。

### 2.2 系统设计与资源权衡链

**norm 位置几乎不改变大矩阵乘的 FLOPs，却显著影响训练稳定性和可用 learning rate。**

资源置换：

- 算力：额外 FLOPs 很小，主要不是 compute trade-off。
- 内存/带宽：norm 本身是统计归约和 elementwise 操作，Lecture 02 的 roofline 视角说明它可能比 FLOPs 占比看起来更贵。
- 稳定性：pre-norm 用结构位置换取更平滑的训练曲线。
- 延迟：norm 增减和位置变化对每层执行顺序有影响，但本讲重点是稳定性。

### 2.3 数学原理与推导链

**讲义未给出 pre-norm/post-norm 的完整公式推导，因此这里只绑定执行顺序，不补外部梯度证明。**

serial block 的讲义公式：

$$
y=x+\mathrm{MLP}(\mathrm{LayerNorm}(x+\mathrm{Attention}(\mathrm{LayerNorm}(x))))
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $x$ | block 输入 hidden states |
| $y$ | block 输出 hidden states |
| $\mathrm{LayerNorm}$ | 对 hidden dimension 做 normalization 的模块 |
| $\mathrm{Attention}$ | multi-head self-attention 子层 |
| $\mathrm{MLP}$ | feedforward 子层 |
| $+$ | residual connection |

这条公式表达的是：attention 前有 norm，attention 输出进入 residual 后，再经 norm 进入 MLP，最后再 residual。

### 2.4 系统演进与接口绑定

**接口形状不变，语义发生变化：`[batch, seq, d_model]` 仍进仍出，但 residual 与 norm 的相对位置改变了训练动力学。**

从 Lecture 02 的角度看，norm 是 memory-bandwidth 敏感操作；从 Lecture 03 的角度看，norm 位置是稳定性开关。两者合并后，实际架构会同时考虑稳定训练和 kernel/runtime 代价。

---

## 3. LayerNorm、RMSNorm 与 Dropping Bias Terms

### 3.1 技术路线与演进逻辑

**RMSNorm 和 no bias 的共同趋势是：去掉对大模型收益不明显、但会制造额外参数或 memory movement 的细节。**

前置基线是 LayerNorm 和带 bias 的 linear layers。现代模型更常见 RMSNorm，并且在线性层和 normalization 中移除 bias terms。

核心崩溃点：LayerNorm 与 elementwise/stat normalization 的 FLOPs 占比很低，但 runtime 占比可能很高。讲义表格显示：

| 操作类别 | FLOPs 占比 | Runtime 占比 |
|---|---:|---:|
| tensor contraction | 99.80% | 61.0% |
| stat normalization | 0.17% | 25.5% |
| elementwise | 0.03% | 13.5% |

这直接对接 Lecture 02：**FLOPs 不是 runtime，memory movement 和 kernel 形态同样关键。**

### 3.2 系统设计与资源权衡链

**RMSNorm 牺牲 mean-centering 与 beta 的表达自由度，换取更少统计操作、更少参数和更简单的数据移动。**

| 设计 | 架构变化 | 资源影响 | 讲义态度 |
|---|---|---|---|
| LayerNorm | 减均值、除方差、乘 gamma、加 beta | 统计归约更多 | 原始/常见基线 |
| RMSNorm | 不减均值，只按 RMS 缩放并乘 gamma | 更少操作和参数 | 现代 LMs 常见 |
| no bias | 删除 $b_1,b_2$ 等 bias | 参数和访问略减 | 与 RMSNorm 类似，也有稳定性动机 |

### 3.3 数学原理与推导链

**LayerNorm 的目标是把 hidden vector 重新定标到稳定范围；RMSNorm 只保留尺度归一化，不处理均值平移。**

LayerNorm：

$$
y=\frac{x-\mathbb{E}[x]}{\sqrt{\mathrm{Var}[x]+\epsilon}}\gamma+\beta
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $x$ | 单个 token 的 hidden vector |
| $y$ | normalization 后的 hidden vector |
| $\mathbb{E}[x]$ | 在 hidden dimension 上的均值 |
| $\mathrm{Var}[x]$ | 在 hidden dimension 上的方差 |
| $\epsilon$ | 防止除零的数值稳定项 |
| $\gamma$ | 可学习缩放参数 |
| $\beta$ | 可学习平移参数 |

RMSNorm：

$$
y=\frac{x}{\sqrt{\lVert x\rVert_2^2+\epsilon}}\gamma
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $x$ | 单个 token 的 hidden vector |
| $y$ | normalization 后的 hidden vector |
| $\lVert x\rVert_2^2$ | hidden vector 的平方范数；讲义公式未显式写平均因子，按讲义保留 |
| $\epsilon$ | 防止除零的数值稳定项 |
| $\gamma$ | 可学习缩放参数 |

现代无 bias FFN 形式：

$$
\mathrm{FFN}(x)=\sigma(xW_1)W_2
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $\sigma$ | activation function |
| $W_1,W_2$ | FFN 两层权重矩阵 |
| $x$ | hidden state |

### 3.4 系统演进与接口绑定

**RMSNorm/no bias 不改变 block 的输入输出 shape，但改变 parameter set、kernel 组成和 memory access pattern。**

接口仍是：

```text
[batch, seq, d_model] -> norm/linear -> [batch, seq, d_model or d_ff]
```

增量在内部：LayerNorm 的 `mean/variance/beta` 路径减少，linear 的 bias 加法消失，整体更贴近 Lecture 02 中“大矩阵乘昂贵但高效，小 elementwise/normalization 便宜但可能拖 runtime”的系统观察。

---

## 4. Activations 与 Gated Linear Units

### 4.1 技术路线与演进逻辑

**FFN activation 的主线从 ReLU/GeLU 走向 SwiGLU/GeGLU，是用 gate 分支换取更强表达能力，再用较小 $d_{ff}$ 控制参数量。**

前置基线：

- ReLU：original Transformer
- GeLU：GPT-2、GPT-3 等
- SwiGLU / GeGLU：现代模型常见

核心崩溃点：单一路径 activation 只对 $xW_1$ 做非线性变换。Gated variants 增加一条 $xV$ 分支，让模型能用逐元素乘法控制信息流。破局机制是引入 gate，但把 hidden dimension 缩小到约 $2/3$，使整体参数量接近传统 FFN。

### 4.2 系统设计与资源权衡链

**GLU 用额外投影和逐元素乘法换表达能力，但通过 $d_{ff}$ 缩放把参数和 FLOPs 控制在现代默认范围。**

| activation | 结构 | 资源变化 | 讲义结论 |
|---|---|---|---|
| ReLU | 一条 FFN activation path | 最简单 | 可工作，但现代不主流 |
| GeLU | 一条平滑 activation path | 类似 ReLU | GPT 系列常见 |
| SwiGLU/GeGLU | activation branch 与 gate branch 相乘 | 多一条 projection；通常缩小 $d_{ff}$ | 实证效果好，现代常用 |

### 4.3 数学原理与推导链

**GLU 的数学增量是把单个非线性通道变成“候选值 × gate”的逐元素控制。**

ReLU FF：

$$
\mathrm{FF}(x)=\max(0,xW_1)W_2
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $x$ | hidden state |
| $W_1$ | input-to-FF 权重 |
| $W_2$ | FF-to-output 权重 |
| $\max(0,\cdot)$ | ReLU |

GeLU FF：

$$
\mathrm{FF}(x)=\mathrm{GELU}(xW_1)W_2,\quad \mathrm{GELU}(x):=x\Phi(x)
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $\mathrm{GELU}$ | Gaussian Error Linear Unit |
| $\Phi(x)$ | 标准高斯 CDF，讲义用来定义 GeLU |
| $xW_1$ | FFN 第一层 pre-activation |

ReGLU：

$$
\mathrm{FF}_{\mathrm{ReGLU}}(x)=(\max(0,xW_1)\otimes xV)W_2
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $V$ | gate branch 的投影矩阵 |
| $\otimes$ | elementwise multiplication |
| $\max(0,xW_1)$ | 候选非线性分支 |
| $xV$ | gate/value 分支 |

### 4.4 系统演进与接口绑定

**GLU 在接口上仍输出 `[batch, seq, d_model]`，但内部从两次大投影变成三次投影加逐元素乘法。**

典型流向：

```text
x [B,S,D] -> xW1 [B,S,Dff]
x [B,S,D] -> xV  [B,S,Dff]
elementwise gate -> [B,S,Dff]
project W2 -> [B,S,D]
```

这与 Lecture 02 的账本直接相关：更多 projection 增加 compute，但可通过缩小 $Dff$ 控制总量；是否值得取决于 validation performance。

---

## 5. Serial vs Parallel Layers

### 5.1 技术路线与演进逻辑

**parallel layers 曾被 GPT-J、PaLM、GPT-NeoX 使用，但现代模型多数仍回到 serial，因为收益依赖实现细节且不是主流共识。**

前置基线是 serial：attention 后接 MLP。parallel 把 attention 和 MLP 从同一个 normalized input 并行计算后相加。

核心崩溃点：serial 的执行链更长；parallel 试图减少顺序依赖并融合部分计算。破局机制是共享 LayerNorm 并融合 matmuls。但讲义总结多数模型现在仍使用 serial。

### 5.2 系统设计与资源权衡链

**parallel layers 的潜在收益来自 kernel fusion 和较短依赖链，而不是改变模型输入输出接口。**

| 形式 | 执行流 | 潜在收益 | 讲义状态 |
|---|---|---|---|
| serial | attention -> residual -> MLP -> residual | 简单、主流 | 现代多数使用 |
| parallel | attention 与 MLP 并行从同一 norm 输入计算 | 可共享 LN、融合 matmul | 少数模型使用 |

### 5.3 数学原理与推导链

**serial 与 parallel 的差异就是 MLP 是否读取 attention 更新后的 hidden state。**

serial：

$$
y=x+\mathrm{MLP}(\mathrm{LayerNorm}(x+\mathrm{Attention}(\mathrm{LayerNorm}(x))))
$$

parallel：

$$
y=x+\mathrm{MLP}(\mathrm{LayerNorm}(x))+\mathrm{Attention}(\mathrm{LayerNorm}(x))
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $x$ | block 输入 |
| $y$ | block 输出 |
| $\mathrm{MLP}$ | feedforward branch |
| $\mathrm{Attention}$ | self-attention branch |
| $\mathrm{LayerNorm}(x)$ | 两个 branch 的 normalized input |

### 5.4 系统演进与接口绑定

**parallel 仍保持 `[B,S,D] -> [B,S,D]`，但内部依赖图从串行变成分支相加。**

如果实现正确，parallel 可以共享 LN 并融合矩阵乘；如果实现不当，只会增加复杂度而不产生 runtime 收益。本讲将其归为 architecture variation，而非现代默认。

---

## 6. Position Embeddings 与 RoPE

### 6.1 技术路线与演进逻辑

**Position embedding 的演进目标是让 attention score 依赖相对位置，而不是把绝对位置当作一个额外 token feature 粗暴相加。**

前置路线：

- sine embeddings：固定函数，加到 token embedding 上。
- learned absolute embeddings：学习一个位置向量，也加到 token embedding 上。
- relative embeddings：在 attention 中显式加入相对位置项。
- RoPE：旋转 query/key，使 dot product 自然成为相对位置函数。

核心崩溃点：additive absolute position 会把 token identity 和 position 混在 hidden vector 中，attention score 不天然只依赖相对距离。RoPE 的破局机制是在 Q/K 空间执行位置相关旋转，并要求：

$$
\langle f(x,i), f(y,j)\rangle=g(x,y,i-j)
$$

### 6.2 系统设计与资源权衡链

**RoPE 把 position handling 从 embedding 输入端移动到每次 attention 的 Q/K 端，因此表达更贴合 attention，但实现位置更深。**

| 方法 | 位置注入点 | 资源/接口影响 | 关键性质 |
|---|---|---|---|
| sine absolute | embedding 加法 | 简单，输入端一次完成 | 固定绝对位置 |
| learned absolute | embedding 加法 | 增加位置参数 | 学习绝对位置 |
| relative attention | attention score 计算 | attention 内部更复杂 | 直接建模相对位置 |
| RoPE | Q/K 旋转 | 每个 attention 操作都要应用 | dot product 依赖 $i-j$ |

### 6.3 数学原理与推导链

**RoPE 的数学核心是二维旋转：两个向量分别按位置旋转后，其内积只与相对位移有关。**

目标性质：

$$
\langle f(x,i), f(y,j)\rangle=g(x,y,i-j)
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $x,y$ | 两个 token 的 hidden/input vector |
| $i,j$ | 两个 token 的位置 |
| $f(\cdot,\cdot)$ | 位置编码后的 query/key 映射 |
| $\langle \cdot,\cdot\rangle$ | dot product，用于 attention score |
| $g$ | 只依赖内容与相对位置差的函数 |
| $i-j$ | relative position |

实际 RoPE：

$$
f_{\{q,k\}}(\boldsymbol{x}_m,m)=\boldsymbol{R}_{\Theta,m}^{d}\boldsymbol{W}_{\{q,k\}}\boldsymbol{x}_m
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $f_{\{q,k\}}$ | query 或 key 的 RoPE 后表示 |
| $\boldsymbol{x}_m$ | 第 $m$ 个 token 的输入向量 |
| $m$ | token position |
| $\boldsymbol{W}_{\{q,k\}}$ | query 或 key projection matrix |
| $\boldsymbol{R}_{\Theta,m}^{d}$ | 维度为 $d$、位置为 $m$ 的 block diagonal rotation matrix |
| $\Theta$ | 一组旋转频率参数 |
| $d$ | query/key hidden dimension |

二维旋转块使用：

$$
\begin{pmatrix}
\cos m\theta_i & -\sin m\theta_i\\
\sin m\theta_i & \cos m\theta_i
\end{pmatrix}
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $\theta_i$ | 第 $i$ 个维度 pair 的旋转频率 |
| $m\theta_i$ | position $m$ 对应的旋转角度 |
| $\cos,\sin$ | 实现二维旋转的三角函数 |

讲义强调 RoPE 与 sine embeddings 的差异：**RoPE 不是 additive，而且没有 cross terms。**

### 6.4 系统演进与接口绑定

**RoPE 接入 attention 的 Q/K projection 之后，而不是 embedding lookup 之后。**

实现位置：

```text
x -> Wq/Wk -> q,k
q,k -> apply RoPE cos/sin rotation
rotated q,k -> usual attention score/value aggregation
```

因此 RoPE 的接口绑定是：embedding 输出仍为 `[B,S,D]`，但每个 attention layer 内部的 Q/K 需要按 position index 旋转。这个点是 assignment 实现中最容易接错的位置。

---

## 7. Hyperparameters：保守共识与少数例外

### 7.1 技术路线与演进逻辑

**Lecture 03 的 hyperparameter 结论很反直觉：大模型看似复杂，但许多关键比例高度保守。**

讲义覆盖四类问题：

- FFN dimension 与 model dimension 的比例
- head dimension、num heads 与 model dimension 的比例
- depth/width aspect ratio
- vocabulary size 与 regularization

演进逻辑不是“找到唯一理论最优”，而是从大量模型经验中看到稳定 basin：默认值通常够好，例外可以工作但未必更优。

### 7.2 系统设计与资源权衡链

**这些 hyperparameters 本质上是参数量、FLOPs、并行性、latency 和 tokenizer 覆盖范围之间的资源分配。**

| 参数 | 常见共识 | 例外 | 系统含义 |
|---|---|---|---|
| $d_{ff}/d_{model}$ | ReLU/GeLU 约 4；GLU 约 $8/3$ | T5 11B 为 64；Gemma 2 为 8 | FFN 参数和 FLOPs 大头 |
| heads × head dim / model dim | 多数约 1 | T5 为 16，LaMDA 为 2，PaLM 为 1.48 | attention projection 宽度 |
| aspect ratio | 许多模型在约 100-200；讲义标出 128 附近 | T5 11B 为 33，Gemma 4 为 61 | 深度影响 latency 和并行性 |
| vocab size | 单语 30-50k；多语/生产 100-250k | 取决于语言覆盖 | embedding/output logits 成本 |
| dropout | 新模型多不用 dropout | 旧模型常 0.1 | 大数据单 pass 降低 memorization 风险 |

### 7.3 数学原理与推导链

**FFN multiplier 直接控制 FFN 中间维度，从而控制 FFN 参数和计算规模。**

标准 FFN 经验规则：

$$
d_{ff}=4d_{model}
$$

GLU variants 缩放规则：

$$
d_{ff}\approx \frac{8}{3}d_{model}
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $d_{ff}$ | FFN hidden/intermediate dimension |
| $d_{model}$ | Transformer hidden/model dimension |
| $4$ | 非 gated FFN 的常见 multiplier |
| $8/3$ | gated FFN 为控制参数量而采用的常见 multiplier |

T5 例外：

$$
d_{ff}=65{,}536,\quad d_{model}=1024
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $65{,}536$ | T5 11B 的 FFN intermediate dimension |
| $1024$ | T5 11B 的 model dimension |
| $65{,}536/1024=64$ | 讲义称为非常激进的 64-times multiplier |

### 7.4 系统演进与接口绑定

**Hyperparameters 是 block 内部 shape 的配置层：它们不改变 Transformer 的宏观接口，却决定每层 memory/FLOPs/latency 账本。**

从 Lecture 02 的公式看：

- 增大 $d_{model}$ 会放大 hidden states、projection matrices、attention head dimensions。
- 增大 $d_{ff}$ 会放大 FFN 的 projection FLOPs。
- 增大 vocab 会放大 embedding table 和 output softmax logits。
- 增加 depth 会增加串行层数，影响 parallelization 与 latency。

因此 Lecture 03 的经验共识不是“调参技巧列表”，而是资源分配模板。

---

## 8. Stability Tricks：Z-Loss、QK Norm、Logit Soft-Capping

### 8.1 技术路线与演进逻辑

**稳定性技巧集中处理 softmax 附近的数值风险，因为 exponentials 和除法会放大 logit 尺度问题。**

讲义明确提醒：Beware of softmaxes。现代训练关注 stable training，不要训练出曲线像讲义蓝线那样失控的模型。

三类技巧：

- output softmax z-loss：约束 partition function 的 log 值。
- attention softmax QK norm：在 attention logits 进入 softmax 前控制 Q/K 尺度。
- logit soft-capping：用 tanh 把 logits 限制到最大值附近。

### 8.2 系统设计与资源权衡链

**稳定性技巧用少量额外计算或约束换取训练可控性，但可能牺牲部分性能。**

| 技巧 | 作用位置 | 资源代价 | 讲义结论 |
|---|---|---|---|
| z-loss | output softmax | 增加一项 loss 计算 | PaLM 等使用，有稳定性收益 |
| QK norm | attention softmax 前 | 增加 normalization | 多个新模型使用 |
| soft-capping | logits | tanh cap | 防止 logit 爆炸，但可能有性能问题 |

### 8.3 数学原理与推导链

**z-loss 的动机是 output softmax 的归一化项 $Z(x)$ 过大或不稳定时，会破坏训练数值行为。**

Softmax log probability：

$$
\log(P(x))=\log\left(\frac{e^{U_r(x)}}{Z(x)}\right)
$$

$$
=U_r(x)-\log(Z(x))
$$

$$
Z(x)=\sum_{r'=1}^{|V|}e^{U_{r'}(x)}
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $x$ | 当前预测位置的 hidden/input context |
| $P(x)$ | 讲义中目标 token 的 softmax probability 写法 |
| $U_r(x)$ | 目标词 $r$ 的 raw logit |
| $Z(x)$ | softmax partition function / normalizer |
| $r'$ | 遍历 vocab 中所有 token 的索引 |
| $|V|$ | vocabulary size |

讲义给出的 z-loss 形式：

$$
L=\sum_i\left[\log(P(x_i))-\alpha(\log(Z(x_i))-0)^2\right]
$$

$$
=\sum_i\left[\log(P(x_i))-\alpha\log^2(Z(x_i))\right]
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $L$ | 讲义写出的 objective 形式；此处保留讲义符号，不改写为常规 negative loss |
| $i$ | training sequence / batch 中的位置索引 |
| $x_i$ | 第 $i$ 个预测位置的 context |
| $\alpha$ | z-loss weight |
| $\log(Z(x_i))$ | 被压向 0 的 log partition value |
| $-\alpha\log^2(Z(x_i))$ | 讲义形式中的稳定性惩罚项 |

Logit soft-capping 的讲义只给出机制说明：通过 Tanh 把 logits soft-cap 到某个最大值；未给完整公式，因此不补写。

### 8.4 系统演进与接口绑定

**这些技巧不改变模型主干 shape，但改变训练 objective 或 attention/logit 前处理。**

绑定点：

```text
output logits -> softmax -> z-loss term
Q,K -> optional QK norm -> attention logits -> softmax
raw logits -> tanh soft cap -> softmax
```

Lecture 02 的数值精度和 memory accounting 在这里变成稳定性问题：softmax 的 exponentials 对 logit scale 敏感，bf16/FP16 训练尤其需要控制尺度。

---

## 9. Attention Heads、MQA/GQA 与 Sparse/Sliding Attention

### 9.1 技术路线与演进逻辑

**Attention head 的主要现代变化不是追求更复杂表达，而是降低自回归推理时 KV Cache 的 memory movement。**

前置基线是 standard multi-head attention：每个 query head 对应自己的 key/value head。增量路线：

- MQA：multiple queries，共享一组 key/value。
- GQA：在 MHA 和 MQA 之间折中，多个 query heads 共享较少的 KV groups。
- Sparse / sliding window attention：限制 attention pattern，降低长上下文 quadratic 成本。
- Interleaved full + local attention：部分层 full attention，部分层 local/restricted attention。

### 9.2 系统设计与资源权衡链

**MQA/GQA 是典型的“牺牲部分 KV 表达自由度，换取推理带宽和 KV Cache 成本下降”。**

讲义给出 full attention 的 arithmetic intensity：

$$
O\left(\left(\frac{1}{k}+\frac{1}{bn}\right)^{-1}\right)
$$

增量生成的 arithmetic intensity：

$$
O\left(\left(\frac{n}{d}+\frac{1}{b}\right)^{-1}\right)
$$

MQA 的 memory access 与 arithmetic intensity：

$$
bnd+bn^2k+nd^2
$$

$$
O\left(\left(\frac{1}{d}+\frac{n}{dh}+\frac{1}{b}\right)^{-1}\right)
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $d$ | hidden dimension |
| $b$ | batch size |
| $n$ | sequence length，讲义设 $n<d$ |
| $h$ | number of heads |
| $k$ | head dimension，$k=d/h$ |
| $bnd$ | 与 batch、sequence、hidden 成正比的访问项 |
| $bn^2k$ | attention score/value 相关访问项 |
| $nd^2$ | projection weight 相关访问项 |

OCR 注意：full attention 的总 arithmetic operations / memory accesses 在讲义提取文本中有破损，本文只保留可读的 arithmetic intensity 形式。

### 9.3 数学原理与推导链

**incremental generation 的难点是不能并行生成未来 token，因此每步都要读取历史 KV Cache。**

讲义给出的 incremental 总量：

$$
\text{operations}=bnd^2
$$

$$
\text{memory accesses}=bn^2d+nd^2
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $b$ | batch size |
| $n$ | 已生成或上下文长度 |
| $d$ | hidden dimension |
| $bnd^2$ | incremental projection/attention 相关的总 arithmetic operations 表达 |
| $bn^2d$ | KV Cache 读取随 sequence length 增长的访问项 |
| $nd^2$ | weights 访问项 |

MQA 改写 memory access 后，$bn^2d$ 中与所有 heads 相关的 KV 访问被压缩到 $bn^2k$，其中 $k=d/h$，这就是 inference 成本下降的来源。

### 9.4 系统演进与接口绑定

**MQA/GQA 改的是 KV head layout，不改 attention 对外输出 shape。**

接口流：

```text
x [B,S,D] -> Q heads [B,S,H,K]
x [B,S,D] -> K/V heads
MHA: K/V heads ~= H
MQA: K/V heads = 1
GQA: K/V heads = G, 1 < G < H
attention output -> [B,S,D]
```

Sparse/sliding window attention 则改 attention mask：

```text
full attention: token i can attend broad context
sliding window: token i attends local window
interleaved: some layers full, most layers local/restricted
```

讲义给出当前标准技巧：Cohere Command A 每 4 层一个 full attention，3:1 的 SWA/full 比例；long-range info 通过 NoPE，short-range info 通过 RoPE + SWA。其他模型如 LLaMA 4、Gemma 3/4、OLMo 3 也使用 SWA + Full RoPE 相关设计。

---

## 10. 总结接口：Lecture 03 的现代 LLM 模板

**本讲最终给出的不是一套唯一架构，而是一套现代 dense LLM 的默认配置和可解释变体空间。**

可以把 Lecture 03 的输出压缩为下面的工程模板：

```text
token IDs
-> embeddings
-> repeated transformer blocks:
   pre/RMSNorm
   attention with RoPE on Q/K
   optional GQA/MQA and optional local/full attention pattern
   residual
   RMSNorm
   SwiGLU/GeGLU MLP, usually no bias
   residual
-> logits over vocab
-> softmax with possible z-loss / soft-capping stability tricks
```

关键结论：

- architecture variation 大多是局部替换，不是完全重写 Transformer。
- RMSNorm、no bias、GQA/MQA 与 Lecture 02 的 memory movement / arithmetic intensity 强绑定。
- RoPE 是 position embedding 的关键现代接口，它必须作用在每层 attention 的 Q/K 上。
- hyperparameters 呈现强共识：FFN multiplier、head ratio、aspect ratio、dropout practice 都比直觉更保守。
- major differences 仍集中在 position embeddings、activations、tokenization。
