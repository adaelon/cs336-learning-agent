# CS336 Lecture 03 Phase 2: Educator 认知脚手架

## 0. 全局知识树

**Lecture 03 的本质不是“教你怎么搭 Transformer”，而是“告诉你工业界已经在哪些维度上收敛、在哪些维度上分叉”。把它读完，你脑子里应该多出一张《现代 LLM 默认配置表》。**

```text
Lecture 02 继承
├── tensor 资源账本：FLOPs / memory / bandwidth / MFU
├── roofline：compute-bound vs memory-bound
└── 训练 memory = params + grads + optimizer states + activations

Lecture 03 新增（按“可选 knob”组织）
├── 现代 anchor：LLaMA-like block
│   ├── pre-norm           （vs 原始 post-norm）
│   ├── RMSNorm            （vs LayerNorm，叠加论文 RMSNorm 内核）
│   ├── SwiGLU             （vs ReLU / GeLU，*GLU 家族）
│   ├── no-bias            （Linear/Norm 全部去 bias）
│   └── RoPE               （vs sine / absolute / relative bias）
├── 拓扑选择
│   └── serial vs parallel block
├── 超参数共识
│   ├── d_ff / d_model = 4（GLU: 8/3），T5 11B 是 64× 异类
│   ├── head_dim · num_heads ≈ d_model
│   ├── aspect ratio d_model / n_layer ≈ 100–200
│   └── vocab：mono 30–50k / multi 100–250k
├── 正则化
│   └── 现代 LM：少 dropout、留 weight decay（与 cosine LR 互动）
├── Stability 三件套（围绕 softmax）
│   ├── z-loss              （output softmax）
│   ├── QK-norm             （attention softmax）
│   └── logit soft-capping  （兜底 tanh）
└── Attention head 变体
    ├── MQA / GQA / MLA     （减小 KV cache，针对推理 memory-bound）
    ├── Sliding Window      （减小 attention 计算 quadratic）
    └── interleaved full + SWA（每 k 层一次 full attention，保长程）
```

挂载提示：**所有“为什么这么选”的回答最终都要落到 Lecture 02 的一句话——“它在 FLOPs、memory、bandwidth、stability 里换了什么”**。读 Lecture 03 时，请在每个 knob 旁边默默问一句：「省的是 compute 还是 memory？换的是 stability 还是 expressiveness？」

---

## 1. 现代 anchor 的“四点 diff”：怎么看懂一篇 LLM 论文的架构表

### 1.1 溯源与关联拓扑

**「现代 LLM 架构」不是从头长出来的，而是原始 Transformer 的四个零件被逐一替换。**

挂载点：

- 母概念：Lecture 02 的 Transformer block（attention + FFN + norm + residual）。
- 兄弟概念：原始 Transformer（Vaswani 2017）。
- 因果链：每一项替换都对应一个 Lecture 02 中的资源指标。

来龙去脉：当模型尺寸从 GPT-2 (1.5B) 走到 GPT-3 (175B) 再到 70B-级 LLaMA，Lecture 02 中所有「memory-bound」「optimization 不稳」「MFU 上不去」的痛点先后出现；行业用四把小手术刀逐个修补——pre-norm 修 stability、RMSNorm 修 bandwidth、SwiGLU 修表达力、RoPE 修长程位置不变性。

### 1.2 直观类比与极简案例

**把现代 LLM block 想成「装修过的老房子」**：

- 老房子（原始 Transformer）：每层归一化在出门处（post-norm），照明用 ReLU，门牌按门牌号 sine 编码，每扇门都带把手（bias）。
- 现代版：归一化挪到进门处（pre-norm）、灯换成可调光 SwiGLU、门牌改为「相对楼层」RoPE、把手能省就省（no-bias）。

读论文时只要 diff 这四项，剩下的（attention pattern、KV cache 形式、stability tricks）才需要细读。

### 1.3 差异鉴别

| 替换项 | 它换掉了什么 | 在 Lecture 02 账本上 |
|---|---|---|
| pre-norm | post-norm | 改 gradient flow，不改 FLOPs |
| RMSNorm | LayerNorm | 减少 bandwidth + 减一组 bias 参数 |
| SwiGLU | ReLU FFN | 在等 FLOPs 下加入 gating |
| RoPE | sine PE | 让 attention inner product 只依赖 i-j |
| no-bias | bias term | 减少 params/grads/opt state |

### 1.4 认知陷阱与跨越难点

- **陷阱**：以为现代 LLM 是某种「全新架构」。它仍是 2017 Transformer，只是 4 个零件换了。
- **反直觉**：替换不是为了「让模型更聪明」，而是为了「让训练曲线更稳、推理更省、参数更紧凑」——和 Lecture 02 的 mindset 一脉相承。

---

## 2. Pre-norm vs Post-norm

### 2.1 溯源与关联拓扑

**Pre-norm 是“residual stream 保持 identity 通路”这条设计哲学的具体实现。**

挂载点：

- 母概念：residual connection（让深度网络梯度可传递的 backbone）。
- 兄弟概念：post-norm（原始 Transformer / BERT）。
- 因果链：要训得深、要 LR 大、要省 warmup → 必须让主 residual 路径不被 norm 打断。

### 2.2 直观类比与极简案例

**把 residual 想成一条主干公路，每层是公路边的“辅道维修站”。**

- post-norm：所有车（梯度）从主道开过来，必须先进维修站洗一遍再放回主道——洗多了车漆都褪了（gradient attenuation）。
- pre-norm：主道一路畅通；每个维修站只对**进站的车**做清洗，主道车流不受影响。

极简案例（伪代码）：

```python
# Post-norm
y = norm(x + sublayer(x))
# Pre-norm
y = x + sublayer(norm(x))
```

读到这两行时请在脑里画箭头：哪条线是 residual 主干，norm 在不在主干上。

### 2.3 差异鉴别

| 项 | Post-norm | Pre-norm |
|---|---|---|
| Norm 是否在 residual 主路径上 | 是 | 否 |
| 是否需要 warmup | 通常需要 | 通常不需要 |
| 在大 LR / 大模型上的表现 | 易出 gradient spike | 更稳 |
| 代表模型 | 原始 Transformer、BERT、OPT350M | 几乎全部 2024+ LLM |

新变体：**double-norm / non-residual post-norm**（Grok、Gemma 2、OLMo 2）= pre-norm 主路径 + residual 之外再加一次 norm，相当于「主道不动 + 维修站出口再洗一遍」。

### 2.4 认知陷阱与跨越难点

- **陷阱**：把 pre-norm 当成「数值技巧」。它真正改的是**反向传播的几何**——让 $\partial x_{\ell+1}/\partial x_\ell$ 接近 $I + J_F$，避免多层连乘衰减。
- **反直觉**：pre-norm 的 train-time loss 曲线**未必更好**，但 final stability 与 scalability 显著更好——这就是为什么所有大模型都用它。

---

## 3. LayerNorm → RMSNorm（关键模块，绑定论文卡片）

### 3.1 溯源与关联拓扑

**RMSNorm 不是“更聪明的 LayerNorm”，而是“砍掉 LayerNorm 中一个其实没用的部件”。**

挂载点：

- 母概念：normalization（把 activation 投影到固定 magnitude）。
- 兄弟概念：LayerNorm（保留均值与标准差）、BatchNorm（跨 batch 维度归一化）。
- 因果链：normalization 是 memory-bound op（Lecture 02）→ 越简单越好 → 哪一部分可以删？

来龙去脉（基于论文 RMSNorm, Zhang & Sennrich 2019）：作者押注 LayerNorm 起作用的核心是 **re-scaling invariance**（缩放不变性），而 **re-centering invariance**（平移不变性）几乎没贡献。把均值这一项删掉，理论性能不掉，工程时间显著下降。

### 3.2 直观类比与极简案例

**比喻**：LayerNorm 像“先把一组数字平移到 0 中心，再缩放到单位标准差”；RMSNorm 像“跳过平移，直接除以 RMS”。

数字案例（toy）：设 $\mathbf{a} = [3, 4, 5]$，$n=3$。

- mean $\mu = 4$；$\sigma = \sqrt{\tfrac{(3-4)^2+(4-4)^2+(5-4)^2}{3}} = \sqrt{2/3} \approx 0.816$。
- LayerNorm 输出（不计 gain/bias）：$[-1.22, 0, 1.22]$。
- RMS = $\sqrt{(9+16+25)/3} = \sqrt{50/3} \approx 4.08$。
- RMSNorm 输出：$[0.735, 0.980, 1.225]$。

两种输出**绝对值不同**，但都把向量映射到固定 magnitude 上；下游 affine 层会把这种偏移自动吸收。

### 3.3 差异鉴别

| 维度 | LayerNorm | RMSNorm |
|---|---|---|
| 处理维度 | 单 token 上 $d_{model}$ 维 | 同上 |
| 是否减均值 | **是** | **否** |
| 是否含 bias | 是（gain + bias） | **仅 gain** |
| 参数数 | $2 d_{model}$ | $d_{model}$ |
| 推理 reduction 次数 | 2（mean + variance） | 1（mean of squares） |
| 在 LLM 上的加速 | baseline | 7%–9%（Transformer）；25%–64%（RNN，论文实测） |
| 代表模型 | GPT-3/2/1, OPT, GPT-J, BLOOM | LLaMA, PaLM, Chinchilla, T5 |

**vs BatchNorm**（再补一个常被混淆的 sibling）：

| 维度 | BatchNorm | LayerNorm / RMSNorm |
|---|---|---|
| 归一化轴 | 跨 batch 维度 | 单样本内特征维度 |
| 在序列模型上的适用性 | 差（batch 内序列长不一） | 好 |
| 是否依赖 batch size | 是 | 否 |

### 3.4 认知陷阱与跨越难点

- **陷阱 1**：以为 RMSNorm 只是“写法简化”。它实际上**抛弃了 re-centering 不变性**这一数学假设——人们曾以为「零均值很重要」，事实证明在 LLM 里并不。
- **陷阱 2**：以为 RMSNorm 与 L2-norm 是一回事。论文明确指出，L2-norm 直接替换 LayerNorm 会**性能下降**（Test14: 22.4 vs 20.7），二者只差一个 $1/\sqrt{n}$ 因子，但这个因子决定了输出与 $d_{model}$ 的解耦。
- **反直觉 1**：RMSNorm 在 **异常初始化**（如权重均值偏到 0.2）下**比 LayerNorm 更稳**——本来以为 re-centering 是“防初始化偏移”的保险，实测反而让 LayerNorm 更敏感。
- **反直觉 2**：RMSNorm 自带**隐式学习率自适应**：$\partial \mathcal{L}/\partial \mathbf{W} \propto 1/\|\mathbf{W}\|$。权重越大，梯度越被缩小，等价于免费的 gradient clipping。
- **实现陷阱**：现代 LLM 用 **pre-RMSNorm**——`norm` 在 residual sublayer 内部、`norm.weight` 是 `[hidden_size]` 而非 `[2*hidden_size]`（没有 bias）。

---

## 4. 丢 bias：一个看似无聊的优化

### 4.1 溯源与关联拓扑

**`linear.bias` 在 LLM 里被普遍删除，是“RMSNorm 删 bias”这条逻辑的延伸。**

挂载点：bias 是「affine」的平移项，与 normalization 中的 re-centering 是“同款冗余”——下游有 affine 层就能吸收掉。

### 4.2 直观类比与极简案例

把 FFN 想成「车间」：

- 带 bias：每台机器都有一个固定的“出厂偏移”。
- 无 bias：偏移交给下游的 affine 层统一处理，省下每台机器自己存一份偏移的内存。

### 4.3 差异鉴别

- 与 RMSNorm 删 bias 完全同因：减少参数、grad、optimizer state；几乎不影响 final loss。

### 4.4 认知陷阱

- **陷阱**：在小模型 fine-tune 时仍按教材习惯加 bias。在 LLM pretraining 框架里 `linear.bias=None` 已是默认。

---

## 5. Activations：ReLU → GeLU → *GLU 家族

### 5.1 溯源与关联拓扑

**激活函数演进的核心不是“非线性变锋利”，而是“加门控”。**

挂载点：

- 母概念：FFN（attention 之间的逐 token 非线性处理）。
- 兄弟概念：ReLU、GeLU、Swish 等纯逐元素函数。
- 进阶概念：**GLU 家族**——把 FFN 第一段从「一个线性 + 非线性」改成「两条并行线性 + 逐元素相乘」。

来龙去脉：FFN 的容量受限于「单支非线性」，引入 gate 后等价于「学到一个软掩码」，能压制无关 channel、放大相关 channel。

### 5.2 直观类比与极简案例

**类比**：

- 没有 gate：每个 channel 通过同一种非线性「灯罩」。
- 有 gate（GLU）：每个 channel 在通过非线性之前先经过一个**可学习的开关**（gate），决定要不要进、进多少。

极简数字（极端简化的 2-D 例子）：

- 输入 $x = [1, 2]$，$xW_1 = [a, b]$，$xV = [0, 1]$。
- ReLU FFN 输出：$\mathrm{ReLU}([a, b])$。
- ReGLU 输出：$\mathrm{ReLU}([a, b]) \otimes [0, 1] = [0, \mathrm{ReLU}(b)]$。

可以直观看到 gate 把第一个 channel 关掉了——这就是 “gating”。

### 5.3 差异鉴别

| 名称 | 非线性 | gate？ | 代表模型 |
|---|---|---|---|
| ReLU FFN | ReLU | 无 | 原始 Transformer, T5, OPT |
| GeLU FFN | GeLU $= x\Phi(x)$ | 无 | GPT-1/2/3, BLOOM |
| ReGLU | ReLU | 有 | — |
| GeGLU | GeLU | 有 | T5 v1.1, Gemma 系列 |
| **SwiGLU** | Swish $= x\sigma(x)$ | 有 | LLaMA, PaLM, Mistral, OLMo |

**vs L1/L2 正则**（用来澄清 gate 不是 regularization）：gate 是**学习出来的乘性掩码**，与 L1/L2 这种「对参数本身的惩罚」完全不同。L1/L2 是固定地缩参数；GLU 是动态地缩 activation。

### 5.4 认知陷阱与跨越难点

- **陷阱**：以为 SwiGLU 必然导致 FFN 参数翻倍。**不会**——只要把 $d_{ff}$ 从 $4 d_{model}$ 缩到 $(8/3) d_{model}$，总参数与 ReLU FFN 持平。
- **反直觉**：在等 FLOPs 等参数下，仅仅给 FFN 加一支 gate 就能稳定带来 perplexity 收益（Shazeer 2020、Narang 2020 的核心证据）。这告诉我们「容量」不是唯一变量，「计算图形状」也重要。
- **陷阱**：把 SwiGLU 当成 attention 替代品。它只换 FFN，不动 attention。

---

## 6. Position embeddings：从“加性”走到“乘性”

### 6.1 溯源与关联拓扑

**RoPE 的诞生不是“又一个 PE”，而是「让 attention 数学上只依赖相对位置 $i-j$」的最干净实现。**

挂载点：

- 母概念：position encoding。
- 兄弟概念：sine、absolute learnable、relative bias。
- 因果链：sine/absolute 的 inner product 包含**非相对位置**的 cross terms → relative bias 解决相对性但不是 inner-product 形式 → RoPE 同时满足相对性与 inner product。

### 6.2 直观类比与极简案例

**类比**：想象每个位置在 2D 平面上「转一个角度」。位置 $i$ 转 $i\theta$，位置 $j$ 转 $j\theta$。两者做 inner product 时，旋转角度的差是 $(i-j)\theta$——绝对位置全部消去。

极简（2D）：

- $q$ 在位置 $i$ 旋转：$(q_0 \cos i\theta - q_1 \sin i\theta,\ q_0 \sin i\theta + q_1 \cos i\theta)$。
- $k$ 在位置 $j$ 同样旋转。
- $\langle q', k'\rangle$ 展开后**只含 $\cos((i-j)\theta)$ 与 $\sin((i-j)\theta)$**。

这就是 RoPE 的全部魔法：用 2D 旋转的正交性，把 inner product 自动“相对化”。

### 6.3 差异鉴别

| 方案 | 注入位置 | 形式 | inner product 只依赖 i-j？ |
|---|---|---|---|
| Sine | embedding（加性） | 加性 | 否（有 cross terms） |
| Absolute learnable | embedding（加性） | 加性 | 否 |
| Relative bias | attention logits | 加性 bias | 是，但不是 inner product 形式 |
| **RoPE** | Q/K（乘性） | 乘性 2D 旋转 | **是**，且仍是 inner product |

**vs sine PE 的微妙差异**：sine PE 注入到 embedding 后会一路传到所有层，且在 inner product 中会和 token embedding 形成 cross terms；RoPE **只在 attention 内部** 注入到 Q/K，token embedding 本身不带 position。

### 6.4 认知陷阱与跨越难点

- **陷阱 1**：以为 RoPE 是“sine PE 的改进”。它**不是同一家族**——一个是加性、一个是乘性，数学结构完全不同。
- **陷阱 2**：以为 RoPE 增加参数。**不增加**——cos/sin 表是预计算常数。
- **反直觉**：RoPE 必须**在每层 attention 都施加一次**（讲义 Page 35 强调），而不是只在 embedding 处加一次——因为 attention 内部的 Q/K 需要每次都带上位置信息。

---

## 7. Serial vs Parallel Block

### 7.1 溯源与关联拓扑

**Parallel block 是“为了 fuse 更大的 GEMM”而做的拓扑改造。**

挂载点：transformer block 的 dependency graph。

### 7.2 直观类比与极简案例

- Serial：attention 与 MLP 像「先做菜再装盘」，必须串行。
- Parallel：attention 与 MLP 同时下锅，最后一起端上（求和），但两者**看同一份食材**（attention 看不到 MLP 中间结果，反之亦然）。

### 7.3 差异鉴别

| 项 | Serial | Parallel |
|---|---|---|
| MLP 看得到 attention 的输出？ | 是 | 否 |
| Norm 可否共享 | 否 | 是 |
| QKV/MLP matmul 是否可融合 | 一定程度上 | 更激进可融合 |
| 代表模型 | 绝大多数 | GPT-J, PaLM, GPT-NeoX, Cohere Command A, Falcon 2 11B, Command R+ |

### 7.4 认知陷阱

- **陷阱**：以为 parallel block 更快是“理论可证”的——其实它的优势主要是**实现层的 kernel fusion**，不是模型本身的胜出。讲义指出主流仍选 serial。

---

## 8. 超参数：四条共识

### 8.1 溯源与关联拓扑

**这四条共识不是“理论最优”，而是“调过很多次之后大家都收敛到这里”——你可以违反，但默认从这里出发。**

挂载点：Lecture 02 的「训练前预算」+ Lecture 03 的「现代默认表」。

### 8.2 直观类比与极简案例

**类比**：选 LLM 超参数像装修房子选层高、过道宽度、窗户大小——有一组久经考验的“好用尺寸”，没有特殊需求就照搬。

四条共识：

1. $d_{ff} = 4 d_{model}$（无 GLU）/ $d_{ff} = (8/3) d_{model}$（GLU）。
2. head_dim · num_heads $= d_{model}$。
3. $d_{model} / n_{layer} \in [100, 200]$。
4. vocab：mono 30–50k / multi 100–250k。

### 8.3 差异鉴别

| 共识 | 典型违反者 | 它违反了什么 | 后果 |
|---|---|---|---|
| $d_{ff} = 4 d_{model}$ | T5 11B（64×） | FFN 异常宽 | 可训但次优；后继 T5 v1.1 回到 2.5× |
| head·dim = d_model | LaMDA (2×)、T5 (16×) | QKV 投影膨胀 | KV cache 同比放大，推理更重 |
| aspect ratio 100–200 | T5 11B (33) | 极深 | pipeline latency 高 |
| mono ≤ 50k vocab | mT5 250k、Gemma 4 262144 | 多语种刚需 | embedding/softmax 大幅膨胀 |

### 8.4 认知陷阱与跨越难点

- **陷阱 1**：以为「比例选 4 而不是 3 是科学问题」。Kaplan 2020 的图显示 1–10 都接近最优，4 是工程惯例。
- **陷阱 2**：误以为 num_heads 必须整除 $d_{model}$。**不必须**——只要 head_dim · num_heads = $d_{model}$ 即可，head_dim 与 num_heads 是独立旋钮。
- **反直觉**：aspect ratio 这个看似纯模型的选择，**实际上由 system parallelism 决定**——讲义明确指出「Systems concerns dictate the value」。

---

## 9. 现代 LLM 怎么“正则化”

### 9.1 溯源与关联拓扑

**LLM 里的 weight decay 已经不是“防过拟合”，而是“在 cosine LR schedule 下塑造 effective LR”。**

挂载点：Lecture 02 的 AdamW；本节关心的不是 optimizer state，而是 update rule 中的 $\lambda \theta$ 项。

### 9.2 直观类比与极简案例

**类比**：weight decay 像“慢慢把橡皮筋拉紧”——它给 cosine LR 中的有效步长加一个持续的、依赖参数 magnitude 的反向力。

不需要 dropout 的逻辑：训练 corpus 远大于模型容量，且 SGD 通常只单 epoch 过一遍数据，**根本没机会过拟合**。

### 9.3 差异鉴别

| 项 | 老式（GPT-2/3） | 现代（LLaMA、PaLM） |
|---|---|---|
| dropout | 0.1 | 0 |
| weight decay | 0.1 | 0.1 |
| 主要作用 | 过拟合控制 | optimization dynamics |

### 9.4 认知陷阱

- **陷阱**：把 LLM 的 weight decay 当成「防过拟合」继续讲。Andriushchenko 2023 明确指出它的作用机制变了。
- **反直觉**：weight decay 与 cosine LR schedule **耦合**——单独看 weight decay 不够，要看「LR × weight decay」的联合效应。

---

## 10. Softmax 稳定三件套

### 10.1 溯源与关联拓扑

**三件套都挂在同一个根：softmax 的指数算子在 bf16 下数值脆弱。**

挂载点：

- 母概念：softmax（output softmax for next-token, attention softmax for QK）。
- 兄弟概念：z-loss、QK-norm、logit soft-capping。
- 因果链：softmax 数值不稳 → 训练曲线 spike → 需要在 (a) 归一化项、(b) softmax 入口、(c) logits 上加约束。

### 10.2 直观类比与极简案例

**类比**：softmax 像一个非常敏感的天平——只要一边放上 outlier 重物，整个分布就倾覆。

三件套各自做的事：

1. **z-loss**：给 partition $\log Z$ 加一个二次惩罚，**不让秤盘整体浮动**。
2. **QK-norm**：在 attention softmax 之前，先把 Q/K **校准到相同 magnitude**，让 logits 量级稳定。
3. **logit soft-cap**：在 softmax 之前给 logits 套一层 $\tanh$ 上限，**防止单一 logit 爆炸**。

极简数字（z-loss）：设输出 vocab=3，logits = $[10, 12, 11]$，那么 $Z = e^{10}+e^{12}+e^{11} \approx 1.84\times 10^5$；$\log Z \approx 12.1$。z-loss 项 $\approx \alpha \cdot 146$，会把 loss 推到稍微在乎「logits 整体不要再高了」的方向。

### 10.3 差异鉴别

| 技巧 | 注入位置 | 控制什么 | 增加 FLOPs / 参数 |
|---|---|---|---|
| z-loss | output softmax | partition function 量级 | 一个标量加项 |
| QK-norm | attention 内部，Q/K 上 | QK logits 量级 | 每层多两次 RMSNorm + 两组 gain |
| Logit soft-cap | 任意 logits | 单一 logit 上限 | $\tanh$ 调用 |

**vs L2 regularization**（用来澄清 z-loss 不是 weight decay）：z-loss 惩罚的是 **activation 的归一化项**（$\log Z$），与「对参数惩罚」的 L2 完全不同。

### 10.4 认知陷阱与跨越难点

- **陷阱 1**：以为这三件套是“随便加都更稳”。logit soft-cap 在讲义中标了「might have perf issues?」——它可能轻微损 perplexity。
- **陷阱 2**：把 QK-norm 与 RoPE 搞混顺序——通常先做 RoPE 旋转，再做 norm（或反过来，模型间有差异）；这是实现细节，要看具体论文。
- **反直觉**：「stability trick」不只是工程兜底——在 bf16/fp16 + 大 LR 训练里它们是**让大模型能跑通**的必要件，不是可选优化。

---

## 11. Attention head 变体：为推理而生

### 11.1 溯源与关联拓扑

**MQA/GQA 的真实动机不在「训练精度」，而在「推理时 KV cache 太大」。**

挂载点：

- 母概念：Lecture 02 §7 roofline——decode 阶段是 matrix-vector，arithmetic intensity 低、memory-bound。
- 兄弟概念：MHA（baseline）、MQA、GQA、MLA、Sliding Window、Interleaved。
- 因果链：每次 decode 都要把整张 KV cache 从 HBM 搬上来 → KV cache 越小，每步搬运越少 → MQA/GQA 直接缩小 cache 的 head 维度。

### 11.2 直观类比与极简案例

**类比**：

- MHA：每个 query head 自己带一份 K/V 笔记本（笔记本数 = $H$）。
- MQA：所有 query head 共用一本 K/V 笔记（笔记本数 = 1）。
- GQA：分组共享，每组共用一本笔记（笔记本数 = $H/g$）。
- SWA：每个 query 只翻看最近 $w$ 页的笔记，不读整本。
- Interleaved：大部分层只翻最近 $w$ 页（SWA），每隔几层让其中一层翻完整本（full attention）保留长程信息。

极简 KV cache 占用对比（$H=32, S=8192, k=128$，bf16）：

```text
MHA  : B * 32 * 8192 * 128 * 2 bytes
MQA  : B *  1 * 8192 * 128 * 2 bytes  → 1/32 倍
GQA8 : B *  4 * 8192 * 128 * 2 bytes  → 1/8 倍
```

### 11.3 差异鉴别

| 变体 | KV cache 维度 | 推理 intensity 影响 | 训练精度损失 |
|---|---|---|---|
| MHA | full $H$ | baseline（差） | baseline |
| MQA | $1$ | 显著改善 | 轻微 perplexity 损失 |
| GQA | $H/g$ | 中等改善（可调 $g$） | 几乎无损（Ainslie 2023） |
| MLA | latent 维 | 进一步压缩 | 需要额外解压步骤（DeepSeek v2） |
| SWA | full $H$ 但只看 $w$ 步 | 长程能力下降 | 取决于窗口 |
| Interleaved | 混合 | full 层吃 KV cache，SWA 层省 | 取决于设计 |

**vs Lecture 02 中的 gradient accumulation / activation checkpointing**：

- gradient accumulation：在**训练**端用 micro-batch 换 activation memory。
- activation checkpointing：在**训练**端用 recompute 换 activation memory。
- MQA/GQA：在**推理**端用 KV cache 共享换 memory bandwidth。

三者都是「memory-bound 时的资源置换」，但发生在不同阶段。

### 11.4 认知陷阱与跨越难点

- **陷阱 1**：把 MQA/GQA 当成「模型质量」改造。它们的主要价值在**推理**——训练时通常仍 compute-bound。
- **陷阱 2**：以为 sliding window 是“损失长程能力”的纯妥协。Interleaved full+SWA 的玩法是「绝大多数层用 SWA + 少数 full 层 + 长程位置用 NoPE」，能兼顾长程与效率。
- **反直觉**：decode 阶段的瓶颈是**搬 KV cache 而不是算 attention**——所以「减少 attention FLOPs」未必加速推理；「减少 KV cache 大小」才是关键。这与 Lecture 02 的 roofline 反直觉点一脉相承（少做计算不等于跑得快）。

---

## 12. 本讲应留下的心智模型

**Lecture 03 最该记住的不是某个具体数字，而是一句话：「现代 LLM 的每个 architecture knob 都对应 Lecture 02 中的一个资源指标」。**

```text
training compute-bound 的胜负     -> Pre-norm / RMSNorm / no-bias / SwiGLU
inference memory-bound 的胜负     -> MQA / GQA / SWA / Interleaved
long-context 表达的胜负           -> RoPE / Full+SWA layered
softmax 数值稳定（bf16 大 LR）   -> z-loss / QK-norm / soft-cap
参数预算分配                      -> d_ff/d_model, head·dim/d_model, aspect ratio, vocab
正则化在 LLM 中的角色             -> 不是抗过拟合，而是 optimization dynamics
```

读论文 checklist（拿到一篇新模型的 architecture 表先问这几条）：

| 默认 | 该论文写的是什么 | 偏离了为什么 |
|---|---|---|
| pre-norm + RMSNorm | ？ | ？ |
| SwiGLU + no-bias | ？ | ？ |
| RoPE | ？ | ？ |
| serial block | ？ | ？ |
| $d_{ff} = (8/3) d_{model}$ | ？ | ？ |
| head·dim = $d_{model}$ | ？ | ？ |
| aspect ratio ≈ 128 | ？ | ？ |
| GQA | ？ | ？ |
| z-loss / QK-norm | ？ | ？ |

填完这张表，你就基本读懂了一篇 LLM 论文的 architecture section。

全局护栏：本产物只基于 `lecture/lecture_03.md`、`output/cs336/lecture_02/` 的历史本地产物、本讲 Phase 1 输出（`output/cs336/lecture_03/01_Phase1_Architect.md`）与 `paperAfterC/RootMeanSquareLayerNormalization_ConceptCard.md`，未使用外部搜索或外部教程。所有类比与差异鉴别都对齐 Phase 1 的数学/工程结论，未引入 Agent A 未覆盖的新事实。
