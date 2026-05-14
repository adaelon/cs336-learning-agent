# CS336 Lecture 02 Phase 2: Educator 认知脚手架

## 0. 全局知识树

**Lecture 02 的知识树只有一个根：把 Lecture 01 的“最大化效率”转成每一步都能数清楚的 tensor resource accounting。**

```text
Lecture 01 继承
├── raw text -> tokenizer -> token IDs
├── efficiency = 不浪费 compute / memory / bandwidth
└── systems mindset = 先算账，再训练

Lecture 02 新增
├── Tensor 是统一载体
│   ├── data
│   ├── parameters
│   ├── gradients
│   ├── optimizer states
│   └── activations
├── Memory accounting
│   ├── shape -> numel
│   ├── dtype -> bytes/value
│   ├── device -> CPU/GPU memory
│   └── mixed precision -> bf16 tensors + fp32 optimizer states
├── Compute accounting
│   ├── FLOPs vs FLOP/s
│   ├── matmul FLOPs = 2BDK
│   └── MFU = actual / promised
├── Roofline analysis
│   ├── arithmetic intensity = FLOPs / bytes
│   ├── accelerator intensity = FLOP/s / bytes/s
│   ├── elementwise -> memory-bound
│   ├── large matmul -> compute-bound
│   └── decode-like matvec -> memory-bound
├── Training accounting
│   ├── forward = 2 data points parameters
│   ├── backward = 4 data points parameters
│   ├── total = 6 data points parameters
│   └── training memory = params + grads + optimizer states + activations
└── Memory optimizations
    ├── gradient accumulation: smaller micro-batch activations
    └── activation checkpointing: recompute activations to save memory
```

---

## 1. 从 Lecture 01 到 Lecture 02 的桥

### 1.1 溯源与关联拓扑

**Lecture 01 说“要最大化效率”，Lecture 02 教你如何给效率记账。**

Lecture 01 的 Tokenizer 输出 `list[int]`。Lecture 02 不再停留在字符串或 token 层，而是进入 PyTorch tensor world：token IDs 会变成 batch tensors，模型参数是 tensors，forward 产生 activations，backward 产生 gradients，optimizer 维护 states。

### 1.2 直观类比与极简案例

**如果 Lecture 01 是设计训练工厂的蓝图，Lecture 02 就是工厂的电表、水表和库存表。**

极简链路：

```text
"Hello" -> tokenizer -> [token IDs]
[token IDs] -> tensor batch
tensor batch -> model parameters -> activations
loss.backward() -> gradients
optimizer.step() -> updated parameters
```

每个箭头都可以问：

```text
占多少 bytes？
做多少 FLOPs？
数据在 CPU 还是 GPU？
瓶颈是 compute 还是 memory bandwidth？
```

### 1.3 差异鉴别

| Lecture 01 概念 | Lecture 02 对应落点 |
|---|---|
| resources | compute, memory, bandwidth 的具体数字 |
| efficiency | MFU、arithmetic intensity、peak memory |
| tokenization | sequence length 进入 tensor shape |
| systems mindset | roofline analysis 与 memory optimization |

### 1.4 认知陷阱

**不要把 resource accounting 当成课后优化；讲义一开始就用 70B/15T/1024 H100 和 8 H100 memory 上限说明，训练计划在跑之前就要先算可行性。**

---

## 2. Napkin Math

### 2.1 溯源与关联拓扑

**Napkin math 是本讲最重要的 mindset：用粗略但量纲正确的公式快速判断训练计划是否荒唐。**

它挂在 Lecture 01 的 efficiency 根节点下。你不需要一开始就知道所有 kernel 细节，但必须知道 parameter 数、token 数、GPU peak FLOP/s、MFU、bytes/parameter 这些量级。

### 2.2 直观类比与极简案例

**Napkin math 像出门前估油量：不需要精确到每个红绿灯，但要知道能不能开到目的地。**

极简案例 1：

```text
70B model, 15T tokens
training FLOPs = 6 * 70e9 * 15e12 = 6.3e24
1024 H100, MFU=0.5
结果量级：约 144 天
```

极简案例 2：

```text
8 H100, 每张 80GB
AdamW 粗略 bytes/parameter = params 2 + grads 2 + states 8 = 12
讲义代码用 2 + 2 + (4 + 4) = 12 bytes/parameter
最大参数量上限约 64B
但没有计入 activations，所以只是上界
```

### 2.3 差异鉴别

| 问题 | 主要资源 | 常见漏项 |
|---|---|---|
| 训练多久 | FLOPs, FLOP/s, MFU, GPU 数 | MFU 不是 1 |
| 模型放不放得下 | GPU memory, bytes/parameter | activations |
| batch 能多大 | activation memory | sequence length 和 layer 数 |

### 2.4 认知陷阱

**最容易犯的错是只数 parameters，不数 gradients、optimizer states、activations。**

一个 70B model 的 parameter memory 只是账本的一部分；训练时还要保存 backward 需要的中间 activations 和 optimizer 的长期统计量。

---

## 3. Tensor：统一载体

### 3.1 溯源与关联拓扑

**Tensor 是所有资源账本的最小单位；不管概念上叫 data、parameter 还是 activation，落到 PyTorch 都是 tensor。**

它从 Lecture 01 的 token IDs 延伸而来：token IDs 是模型输入的离散表示，而 Lecture 02 关注的是这些输入和模型状态如何作为 tensors 存储和计算。

### 3.2 直观类比与极简案例

**Tensor 像多维表格；rank 是表格有多少个索引轴，shape 是每个轴有多长。**

极简案例：

```python
torch.zeros(4)        # rank 1: vector
torch.zeros(4, 8)     # rank 2: matrix
torch.zeros(4, 8, 2)  # rank 3
```

Transformer 常见 shape：

```text
x: [B, S, H, D]
B = batch size
S = sequence length
H = number of heads
D = hidden dimension per head
```

### 3.3 差异鉴别

| 名称 | 它是什么 | 资源影响 |
|---|---|---|
| rank | 维度数量 | 决定操作语义复杂度 |
| shape | 每个维度的 size | 决定 numel |
| dtype | 每个 value 的表示格式 | 决定 bytes/value 和稳定性 |
| device | CPU/GPU 位置 | 决定执行位置和搬运成本 |

### 3.4 认知陷阱

**不要把 shape 当成注释；shape 是资源公式的输入。**

`[B, S, H, D]` 中的 $S$ 来自 tokenizer 和 context length，$B$ 来自 batch strategy，$H,D$ 来自 architecture。它们每一个都会改变 memory 或 FLOPs。

---

## 4. dtype 与 Mixed Precision

### 4.1 溯源与关联拓扑

**dtype 是 memory 与数值稳定性的交换旋钮。**

父概念是 tensor memory：同样 numel，dtype 决定每个 value 占几个 bytes，也决定 underflow、dynamic range、resolution 等数值行为。

### 4.2 直观类比与极简案例

**dtype 像尺子的刻度格式：粗尺子省空间、快，但太小或太细的数可能量不准。**

极简 memory：

```text
shape = [4, 8]
numel = 32
fp32 element_size = 4 bytes
memory = 32 * 4 = 128 bytes
```

fp16 underflow：

```python
torch.tensor([1e-8], dtype=torch.float16) == 0
```

bf16 不 underflow：

```python
torch.tensor([1e-8], dtype=torch.bfloat16) != 0
```

### 4.3 差异鉴别

| dtype | memory | 优点 | 风险/代价 |
|---|---:|---|---|
| fp32 | 4 bytes | baseline，稳定 | memory 大 |
| fp16 | 2 bytes | memory 减半 | dynamic range 差，可能 underflow |
| bf16 | 2 bytes | fp32 dynamic range | resolution 更差 |
| fp8 | 1 byte | 更省、更快潜力 | 需要硬件/库支持与格式选择 |
| fp4/nvfp4 | 4 bits + scale | 极低 precision | 需要 block scale，用户控制有限 |

### 4.4 认知陷阱

**不要以为“更低 precision 一定更好”；训练稳定性要求 optimizer states 通常保留 fp32。**

讲义给出的 mixed precision 直觉是：

```text
parameters / activations / gradients: bf16
optimizer states: fp32
```

这是 memory 与 stability 的折中，而不是单纯把所有东西都压低。

---

## 5. CPU/GPU Device

### 5.1 溯源与关联拓扑

**GPU 不是魔法加速开关；tensor 必须在 GPU memory 上，相关 operation 才能利用 GPU parallelism。**

device 概念挂在 tensor storage 下。CPU tensor 和 GPU tensor 的 shape 可能一样，但资源位置完全不同。

### 5.2 直观类比与极简案例

**CPU 和 GPU 像两个仓库；货物在哪个仓库，工人就在哪个仓库干活。**

极简案例：

```python
x = torch.zeros(32, 32)
x.device == torch.device("cpu")

x = x.to(cuda_if_available())
```

也可以直接在 GPU 创建：

```python
with torch.device(device):
    x = torch.zeros(32, 32)
```

### 5.3 差异鉴别

| 操作 | 含义 |
|---|---|
| `x.to(device)` | 把已有 tensor 移动到目标 device |
| `with torch.device(device)` | 在目标 device 上创建新 tensor |
| `cuda_if_available()` | 有 CUDA 用 CUDA，否则 fallback |

### 5.4 认知陷阱

**不要只看代码里的 `torch` operation；如果 tensor 仍在 CPU，后面的 GPU FLOP/s 讨论就不适用。**

---

## 6. Einops

### 6.1 溯源与关联拓扑

**einops 的认知价值是把“第 -2 维”变成“seq/head/hidden 这样的命名维度”。**

它挂在 tensor operation 下。Lecture 02 不是引入一个新数学体系，而是给 tensor algebra 加上可读的 shape bookkeeping。

### 6.2 直观类比与极简案例

**einops 像给每个箱子贴标签；你不再靠箱子位置猜里面是什么。**

旧写法：

```python
z = x @ y.transpose(-2, -1)
```

新写法：

```python
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
```

这里 `hidden` 不在输出里，所以会被 summed over。

### 6.3 差异鉴别

| einops 操作 | 心智模型 | 例子 |
|---|---|---|
| `einsum` | 命名版矩阵乘/张量乘 | `"seq1 hidden, hidden seq2 -> seq1 seq2"` |
| `reduce` | 沿命名维度聚合 | `"... hidden -> ..."` |
| `rearrange` | 拆分或合并维度 | `"... (heads hidden1) -> ... heads hidden1"` |

### 6.4 认知陷阱

**不要把 einops 当成只为美观；它减少的是 shape bug，而 shape bug 在 Transformer 中常常会变成静默错误。**

例如 head 维和 hidden 维混掉，代码可能还能运行，但语义已经错了。

---

## 7. FLOPs 与 MFU

### 7.1 溯源与关联拓扑

**FLOPs 是计算量，FLOP/s 是硬件速度，MFU 是实际跑出来的效率。**

它们挂在 compute accounting 下，用来回答“这段代码到底做了多少计算，以及是否接近硬件峰值”。

### 7.2 直观类比与极简案例

**FLOPs 像工作总量，FLOP/s 像工厂每秒产能，MFU 像产能利用率。**

矩阵乘：

```python
x: [B, D]
w: [D, K]
y = x @ w
```

每个 `(b, d, k)` 对应一次 multiply 和一次 add：

```text
FLOPs = 2 * B * D * K
actual FLOP/s = FLOPs / measured_time
MFU = actual FLOP/s / promised FLOP/s
```

### 7.3 差异鉴别

| 术语 | 问的问题 |
|---|---|
| FLOPs | 总共做了多少 floating-point operations |
| FLOP/s | 每秒能做多少 operations |
| promised FLOP/s | spec sheet 上的峰值 |
| actual FLOP/s | benchmark 测出来的值 |
| MFU | actual / promised |

### 7.4 认知陷阱

**不要期待 MFU 接近 1；讲义说 MFU 约 0.5 通常已经很好。**

原因不一定是代码差，而是 memory movement、communication、kernel overhead、operation shape 等都会让硬件无法一直满负荷做 arithmetic。

---

## 8. Arithmetic Intensity 与 Roofline

### 8.1 溯源与关联拓扑

**Roofline 把“为什么跑不满”拆成两个限制：算得够不够多，搬得够不够少。**

父概念是 MFU。MFU 低时，下一步不是盲目优化代码，而是判断 operation 是 compute-bound 还是 memory-bound。

### 8.2 直观类比与极简案例

**Arithmetic intensity 像每搬一箱材料能生产多少产品；如果每箱材料只做一点点工，瓶颈就是搬运。**

判断流程：

```text
workload intensity = FLOPs / bytes
accelerator intensity = peak FLOP/s / memory bandwidth

workload < accelerator -> memory-bound
workload > accelerator -> compute-bound
```

H100 讲义数值：

```text
peak dense bf16 = 1979e12 / 2 FLOP/s
memory bandwidth = 3.35e12 bytes/s
accelerator intensity ≈ 295 FLOPs/byte
```

### 8.3 差异鉴别

| Operation | Arithmetic intensity | Bottleneck |
|---|---:|---|
| ReLU | about 1/4 | memory-bound |
| GeLU | about 5 | memory-bound |
| Dot product | about 1/2 | memory-bound |
| Matrix-vector | about 1 | memory-bound |
| Large matmul, n=1024 | about n/3 ≈ 341 | compute-bound |

### 8.4 认知陷阱

**最反直觉的一点是：ReLU 不一定比 GeLU 快，因为二者孤立执行时都可能被 memory bandwidth 限制。**

如果瓶颈是搬 bytes，少做一些 arithmetic 未必能减少总时间。真正要改善这类 operation，通常要减少 memory movement，例如 fusion。

---

## 9. Training FLOPs

### 9.1 溯源与关联拓扑

**训练比推理贵，因为 backward 不只是反向传一个数，还要同时计算 activation gradients 和 weight gradients。**

它挂在 compute accounting 下，并解释本讲开头 `6 * data points * parameters` 的来源。

### 9.2 直观类比与极简案例

**Forward 是把货物送到终点；Backward 是沿路回查每个站点和每段轨道该如何调整。**

简单 gradient：

```text
y = 0.5 * (x @ w - 5)^2
x = [1, 2, 3]
w = [1, 1, 1]
x @ w = 6
gradient wrt w = (6 - 5) * x = [1, 2, 3]
```

### 9.3 差异鉴别

| 阶段 | 做什么 | FLOPs 量级 |
|---|---|---|
| Forward | compute activations/loss | $2 * data points * parameters$ |
| Backward input grad | 把 gradient 传给上一层 | forward 同量级 |
| Backward weight grad | 计算 parameter gradients | forward 同量级 |
| Total backward | input grad + weight grad | $4 * data points * parameters$ |
| Training step | forward + backward | $6 * data points * parameters$ |

### 9.4 认知陷阱

**不要把 training cost 当成 inference cost；训练一步大约是 forward 的三倍 FLOPs，还要存 activations 和 optimizer states。**

这也是为什么同样模型在 inference 能放下，不代表 training 能放下。

---

## 10. Optimizer Memory

### 10.1 溯源与关联拓扑

**Optimizer state 是很多人第一次算 training memory 时漏掉的大项。**

它挂在 training accounting 下。parameters 是模型本身，gradients 是当前 step 的更新方向，optimizer states 是跨 step 保存的统计量。

### 10.2 直观类比与极简案例

**Optimizer state 像每个参数自己的历史账本；Adam 要记两个账本，AdaGrad 要记一个账本。**

AdaGrad：

```text
g2 += grad^2
p -= lr * grad / sqrt(g2 + 1e-5)
```

Memory 例子：

```text
bf16 parameter: 2 bytes/parameter
bf16 gradient: 2 bytes/parameter
AdaGrad fp32 state: 4 bytes/parameter
Adam fp32 states: 8 bytes/parameter
```

### 10.3 差异鉴别

| Tensor 类别 | 是否随 parameters 增长 | 典型 dtype in lecture | 何时存在 |
|---|---|---|---|
| Parameters | 是 | bf16 | 训练和推理 |
| Gradients | 是 | bf16 | 训练 backward 后 |
| Optimizer states | 是 | fp32 | 训练 optimizer 中 |
| Activations | 随 batch/depth/shape | bf16 | forward/backward 期间 |

### 10.4 认知陷阱

**不要只问“模型参数是多少 GB”；训练时更关键的是 peak memory，它包含 transient activations 和 persistent optimizer states。**

讲义中的 `8 H100 最大模型` 计算特意 caveat：没有计入 activations，所以只是上限。

---

## 11. Gradient Accumulation

### 11.1 溯源与关联拓扑

**Gradient accumulation 是 batch size 与 activation memory 之间的折中。**

它从 training loop 派生出来：我们想要 large batch 的稳定性，但不想一次 forward/backward 存下 large batch 的所有 activations。

### 11.2 直观类比与极简案例

**它像分批搬货但最后一起结账：每次只搬 micro-batch，占用空间小；累计够了再更新参数。**

流程：

```text
micro batch 1: forward/backward, keep gradients
micro batch 2: forward/backward, add gradients
...
after enough micro batches: optimizer.step()
zero gradients
```

Memory 对比：

```text
full batch activation memory = 2 * B * D * L
micro batch activation memory = 2 * micro_batch_size * D * L
```

### 11.3 差异鉴别

| 方法 | 改变什么 | 不改变什么 |
|---|---|---|
| 减小 batch size | 真的改变 training batch | optimization dynamics |
| Gradient accumulation | 减小每次 forward 的 micro-batch | 目标 effective batch |
| Activation checkpointing | 减少保存 activations | batch size 本身 |

### 11.4 认知陷阱

**不要在每个 micro-batch 后 `zero_grad`；讲义强调要 accumulate gradients，然后每隔 `batch_size / micro_batch_size` 次再 update 和 zero。**

---

## 12. Activation Checkpointing

### 12.1 溯源与关联拓扑

**Activation checkpointing 是 depth 与 memory 之间的折中：不存所有中间结果，需要时重新算。**

它挂在 activation memory 下。训练需要 backward，所以默认要保留许多 forward activations；推理不需要 gradients，所以只保留当前 layer activations 即可。

### 12.2 直观类比与极简案例

**它像长途路线只记几个检查点，不保存每一步细节；回程需要细节时，从最近检查点重新走一段。**

讲义示意：

```text
store all: x g1 h1 g2 h2 g3 h3 g4 h4
checkpointing: x h1 h2 h3 h4
```

PyTorch 心智模型：

```python
x = torch.utils.checkpoint.checkpoint(layer, x)
```

### 12.3 差异鉴别

| 策略 | Memory | Compute |
|---|---|---|
| Store all layers | $O(L)$ | no recomputation |
| Store no layers | $O(1)$ | $O(L^2)$ recomputation |
| Store every $\sqrt{L}$ layers | $O(\sqrt{L})$ | $O(L)$ recomputation |

### 12.4 认知陷阱

**最容易误解的是 checkpointing 会让训练“免费省内存”；实际上它明确用额外 compute 换 memory。**

如果你的瓶颈已经是 compute time，checkpointing 会增加运行时间；如果你的瓶颈是 memory OOM，它可能让本来不能训练的 batch/model 变得可训练。

---

## 13. 本讲应留下的核心心智模型

**Lecture 02 最该记住的不是某个 PyTorch API，而是“每个训练决策都能写成 bytes、FLOPs、bandwidth、MFU 的账”。**

```text
token IDs 进入 tensor world
-> shape 决定 numel
-> dtype 决定 bytes/value
-> device 决定在哪里算
-> operation 决定 FLOPs
-> FLOPs/bytes 决定 bottleneck
-> forward/backward/optimizer 决定 training memory
-> accumulation/checkpointing 在 memory 和 compute 间换资源
```

最小检查清单：

| 训练前问题 | 应该使用的账本 |
|---|---|
| 这个 run 要多久？ | $6ND$ 和 GPU FLOP/s/MFU |
| 模型能放下吗？ | params + grads + optimizer states + activations |
| 为什么 MFU 低？ | arithmetic intensity vs accelerator intensity |
| 为什么 inference 慢？ | decode 类 matrix-vector 易 memory-bound |
| batch 放不下怎么办？ | gradient accumulation |
| depth/activation 放不下怎么办？ | activation checkpointing |

全局护栏：本产物只基于 `lecture/lecture_02.md`、历史 `lecture_01` 本地产物和本地 deconstruction SOP，没有使用外部搜索或外部教程。
