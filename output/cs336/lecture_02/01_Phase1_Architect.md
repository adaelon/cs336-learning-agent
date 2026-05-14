# CS336 Lecture 02 Phase 1: Architect 技术结构拆解

## 0. 全局定位

**Lecture 02 把 Lecture 01 的效率主线落到可计算的资源账本：给定 tensor computation，明确它消耗多少 memory、多少 FLOPs、是否被 compute 或 memory bandwidth 限制。**

历史状态继承：

- Lecture 01 建立了总目标：在固定 resources 下最大化 model quality，即 `accuracy = efficiency x resources`。
- Lecture 01 的第一个具体接口是 Tokenizer：`raw string -> list[int] token IDs`。
- Lecture 02 接上这个接口：token IDs、parameters、gradients、optimizer states、activations、data 全部都变成 tensors，并在 GPU memory 与 accelerator compute units 之间流动。

本讲目标：

1. Memory accounting：tensor shape、dtype、device 决定 memory。
2. Compute accounting：tensor operations 决定 FLOPs。
3. Roofline analysis：用 arithmetic intensity 判断 memory-bound 或 compute-bound。
4. Training accounting：forward/backward/optimizer/activations 的 memory 与 FLOPs。
5. Memory optimizations：gradient accumulation 与 activation checkpointing。

---

## 1. Motivating Questions：Napkin Math for Resources

### 1.1 技术路线与演进逻辑

**本讲从两个粗算问题开始，是为了把“效率”从抽象价值观变成训练前就能估算的工程约束。**

问题 1：训练一个 **70B parameter model**，使用 **15T tokens**，在 **1024 H100s** 上需要多久？

问题 2：使用 AdamW，在 **8 H100s** 上最多能训练多大的模型？

前置基线是 Lecture 01 中的课程目标：训练最好模型。核心崩溃点是如果不理解 compute 和 memory，无法判断一个训练计划是否可行。破局机制是 back-of-the-envelope resource accounting。

### 1.2 系统设计与资源权衡链

**这两个问题分别暴露 compute budget 与 memory budget：前者决定训练时间，后者决定模型是否放得下。**

训练时间估算：

```python
total_flops = 6 * 70e9 * 15e12
h100_flop_per_sec = 1979e12 / 2
mfu = 0.5
flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24
days = total_flops / flops_per_day
```

内存上限估算：

```python
h100_bytes = 80e9
bytes_per_parameter = 2 + 2 + (4 + 4)
num_parameters = (h100_bytes * 8) / bytes_per_parameter
```

该上限没有计入 activations，因此是 optimistic upper bound。

### 1.3 数学原则与推导链

**训练 compute 的粗略公式来自 Lecture 01/02 共同使用的经验账本：每个 data point 每个 parameter 约 6 FLOPs。**

公式：

$$
C = 6ND
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $C$ | 总训练 FLOPs |
| $N$ | parameter 数量 |
| $D$ | training data points / tokens 数量，本讲问题中为 15T tokens |
| $6$ | forward 约 2，backward 约 4 的总系数 |

对第一个问题：

$$
C = 6 \times 70 \times 10^9 \times 15 \times 10^{12} = 6.3 \times 10^{24}
$$

硬件每天可用 FLOPs：

$$
F_{\text{day}} = F_{\text{H100}} \times \mathrm{MFU} \times 1024 \times 86400
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $F_{\text{H100}}$ | 单张 H100 的 dense peak FLOP/s，讲义取 $1979e12 / 2$ |
| $\mathrm{MFU}$ | Model FLOPs utilization，本例取 0.5 |
| $1024$ | GPU 数 |
| $86400$ | 每天秒数 |

得到量级约 **144 days**。

对第二个问题：

$$
N_{\max} = \frac{8 \times 80 \times 10^9}{2 + 2 + 4 + 4} \approx 64 \times 10^9
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $8$ | H100 数量 |
| $80 \times 10^9$ | 每张 H100 约 80GB memory |
| $2$ | bf16 parameters bytes/parameter |
| $2$ | bf16 gradients bytes/parameter |
| $4+4$ | AdamW 两个 fp32 optimizer states bytes/parameter |
| $N_{\max}$ | 未计 activations 的最大 parameter 上界 |

### 1.4 系统增量与接口绑定

**Lecture 02 的新增接口是 resource estimate：给定 model size、data size、dtype、GPU spec，输出训练时间和显存可行性。**

它对接 Lecture 01 的全局系统链：

```text
token IDs -> tensor batches -> forward tensors -> backward gradients
          -> optimizer states -> resource accounting -> feasible training plan
```

---

## 2. Tensor Basics：Rank, Shape, and Storage

### 2.1 技术路线与演进逻辑

**Tensor 是本讲所有资源账本的共同对象：data、parameters、gradients、optimizer states、activations 都是 tensors。**

本模块属于基础定义/前置概念，讲义未涉及特定历史演进痛点。它建立后续 memory/FLOPs 计算的统一表示。

### 2.2 系统设计与资源权衡链

**Tensor 的资源消耗由 shape、dtype、device 三个属性共同决定。**

讲义示例：

```python
x = torch.zeros(4)        # rank 1 tensor
x = torch.zeros(4, 8)     # rank 2 tensor
x = torch.zeros(4, 8, 2)  # rank 3 tensor
```

Transformer 常见 rank 4 tensor：

```python
B = 32  # Batch size
S = 16  # Sequence length
H = 16  # Number of heads
D = 64  # Hidden dimension per head
x = torch.zeros(B, S, H, D)
```

资源含义：

| 维度 | 工程含义 | 来自上一讲的连接 |
|---|---|---|
| $B$ | Batch size | training loop 的 data batch |
| $S$ | Sequence length | tokenizer 输出 token sequence 长度 |
| $H$ | Number of heads | Transformer architecture |
| $D$ | Hidden dimension per head | model shape / parameter size |

### 2.3 数学原则与推导链

**Tensor rank 是维度数量，numel 是所有维度 size 的乘积。**

对 shape $(d_1, d_2, \dots, d_r)$：

$$
\mathrm{rank}(x)=r
$$

$$
\mathrm{numel}(x)=\prod_{i=1}^{r}d_i
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $x$ | Tensor |
| $r$ | Tensor rank |
| $d_i$ | 第 $i$ 个维度的 size |
| $\mathrm{numel}(x)$ | tensor 中 scalar values 总数 |

对 rank 4 Transformer tensor：

$$
\mathrm{numel}(x)=B \times S \times H \times D
$$

### 2.4 系统增量与接口绑定

**Tensor basics 把 Lecture 01 的 token IDs 接入 PyTorch computation：token sequence 长度 $S$ 后续直接决定 activation memory 与 attention/MLP compute。**

---

## 3. Tensor Memory：dtype and Precision

### 3.1 技术路线与演进逻辑

**dtype 是 memory accounting 的第一把尺：同样 shape，fp32、fp16、bf16、fp8、fp4 的 bytes/value 不同，数值稳定性也不同。**

技术演进按讲义展开：

| dtype | memory | 讲义中的关键点 |
|---|---:|---|
| fp32 | 4 bytes/value | scientific computing baseline，deep learning 中内存压力大 |
| fp16 | 2 bytes/value | memory 减半，但 dynamic range 尤其小数不够 |
| bf16 | 2 bytes/value | memory 同 fp16，dynamic range 同 fp32，resolution 更差 |
| fp8 | 1 byte/value | H100 支持 E4M3/E5M2 两种 variant |
| fp4 / nvfp4 | 4 bits/value 加 scale factor | 讲义提到 Nemotron 3 Super 使用 NVFP4 |

### 3.2 系统设计与资源权衡链

**低精度用数值分辨率与潜在稳定性风险，换取 memory reduction 与更高硬件吞吐。**

讲义示例：

```python
x = torch.zeros(4, 8)
assert x.dtype == torch.float32
assert x.numel() == 4 * 8
assert x.element_size() == 4
assert get_memory_usage(x) == 4 * 8 * 4
```

GPT-3 feedforward layer 的一个 matrix：

```python
torch.empty(12288 * 4, 12288)
```

讲义给出 memory：**2.3 GB**。

fp16 underflow 示例：

```python
x = torch.tensor([1e-8], dtype=torch.float16)
assert x == 0
```

bf16 保留 dynamic range：

```python
x = torch.tensor([1e-8], dtype=torch.bfloat16)
assert x != 0
```

### 3.3 数学原则与推导链

**Tensor memory 由元素数量乘以每个元素 bytes 得到。**

公式：

$$
\mathrm{memory}(x)=\mathrm{numel}(x)\times \mathrm{element\_size}(x)
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $x$ | Tensor |
| $\mathrm{numel}(x)$ | tensor values 总数 |
| $\mathrm{element\_size}(x)$ | 当前 dtype 的 bytes/value |
| $\mathrm{memory}(x)$ | tensor storage bytes |

代码映射：

```python
def get_memory_usage(x: torch.Tensor):
    return x.numel() * x.element_size()
```

Mixed precision policy：

| Tensor 类别 | 讲义建议 |
|---|---|
| parameters | bf16 |
| activations | bf16 |
| gradients | bf16 |
| optimizer states | fp32 |

### 3.4 系统增量与接口绑定

**dtype 选择直接进入 training memory equation：parameter、gradient、activation 和 optimizer state 各自按不同 bytes/parameter 计账。**

后续 optimizer 模块使用：

```text
parameter_memory = 2 * num_parameters
gradient_memory = 2 * num_parameters
optimizer_state_memory = 4 * num_parameters  # AdaGrad
activation_memory = 2 * B * D * L
```

---

## 4. Tensor Device：CPU Memory vs GPU Memory

### 4.1 技术路线与演进逻辑

**Tensor 默认在 CPU memory；要利用 GPU parallelism，必须把 tensor 放到 GPU memory。**

本模块属于基础定义/前置概念，讲义未涉及特定历史演进痛点。它服务于后续 GPU FLOP/s 与 memory bandwidth 计算。

### 4.2 系统设计与资源权衡链

**GPU 加速不是自动发生的：tensor device 决定 operation 在 CPU 还是 GPU 上执行，也决定数据是否需要跨设备移动。**

讲义代码：

```python
x = torch.zeros(32, 32)
assert x.device == torch.device("cpu")

device = cuda_if_available()
x = x.to(device)
```

直接在 GPU 上创建：

```python
with torch.device(device):
    x = torch.zeros(32, 32)
    assert x.device == device
```

### 4.3 数学原则与推导链

**本模块偏向 PyTorch device semantics，讲义未给出数学推导。**

留白原则：不补写 CPU-GPU transfer bandwidth 或 PCIe/NVLink 公式，因为本讲没有展开。

### 4.4 系统增量与接口绑定

**Device placement 是后续 benchmark、MFU、roofline 的前置条件；没有 CUDA device 时，讲义函数会返回保守 fallback。**

`get_promised_flop_per_sec(dtype)` 会根据 GPU 名称 A100/H100/B200 与 dtype 返回 peak FLOP/s；否则返回 `None` 或 CPU fallback。

---

## 5. Einops：Named-Dimension Tensor Algebra

### 5.1 技术路线与演进逻辑

**einops 解决的是 tensor dimension bookkeeping 容易出错的问题，用命名维度替代 `transpose(-2, -1)` 这类脆弱写法。**

前置基线：

```python
z = x @ y.transpose(-2, -1)
```

核心崩溃点：`-2`、`-1` 难以从代码中看出语义，尤其在 batch/seq/head/hidden 交错时容易写错。破局机制是把维度名写进 operation。

### 5.2 系统设计与资源权衡链

**einops 是上层 tensor API 抽象，不显著改变底层 FLOPs 或 memory，但显著降低 shape 错误风险。**

讲义引入三个操作：

| 操作 | 作用 |
|---|---|
| `einsum` | generalized matrix multiplication with bookkeeping |
| `reduce` | 按命名维度聚合，如 sum/mean/max/min |
| `rearrange` | 拆分或合并维度 |

### 5.3 数学原则与推导链

**einsum 的规则是：输出中未出现的命名维度会被 summation reduced。**

矩阵乘法：

$$
z_{i,k}=\sum_{j}x_{i,j}y_{j,k}
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $x_{i,j}$ | `x` 中 `seq1 hidden` |
| $y_{j,k}$ | `y` 中 `hidden seq2` |
| $z_{i,k}$ | 输出 `seq1 seq2` |
| $j$ | `hidden` 维度，未出现在输出中，因此被求和 |

代码映射：

```python
z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")
```

batched sequence similarity：

$$
z_{b,i,k}=\sum_j x_{b,i,j}y_{b,k,j}
$$

代码映射：

```python
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
```

`reduce`：

$$
y_{b,s}=\sum_h x_{b,s,h}
$$

代码映射：

```python
y = reduce(x, "... hidden -> ...", "sum")
```

`rearrange` 拆合 head：

```python
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")
```

### 5.4 系统增量与接口绑定

**einops 为后续 FLOPs counting 和 gradient derivation 提供可读 shape contracts。**

例如 `h2 = einsum(h1, w2, "batch in, in out -> batch out")` 直接暴露了 FLOPs 中的 $B \times D \times D$ 三重循环。

---

## 6. FLOPs and MFU

### 6.1 技术路线与演进逻辑

**FLOPs 计量 computation done，FLOP/s 计量 hardware speed；MFU 计量模型实际利用了多少 promised hardware throughput。**

讲义强调两个同音 acronym：

| 术语 | 含义 |
|---|---|
| FLOPs | floating-point operations，做了多少计算 |
| FLOP/s 或 FLOPS | floating-point operations per second，硬件速度 |

### 6.2 系统设计与资源权衡链

**FLOP/s 不是常数，它强依赖 dtype、hardware、operation shape 和 memory movement。**

H100 讲义事实：

- peak performance: **1979 teraFLOP/s with sparsity**
- dense 估算取 **1979e12 / 2**

线性模型示例：

```python
x = torch.ones(B, D)
w = torch.randn(D, K)
y = x @ w
```

### 6.3 数学原则与推导链

**矩阵乘法 FLOPs 来自每个 $(i,j,k)$ triple 的一次 multiplication 和一次 addition。**

对 $x \in \mathbb{R}^{B \times D}$，$w \in \mathbb{R}^{D \times K}$，$y=xw$：

$$
y_{b,k}=\sum_{d=1}^{D}x_{b,d}w_{d,k}
$$

FLOPs：

$$
\mathrm{FLOPs}=2BDK
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $B$ | number of points / batch size |
| $D$ | input dimension |
| $K$ | number of outputs |
| $2$ | multiply + add |

实际吞吐：

$$
F_{\mathrm{actual}}=\frac{\mathrm{FLOPs}}{\mathrm{time}}
$$

MFU：

$$
\mathrm{MFU}=\frac{F_{\mathrm{actual}}}{F_{\mathrm{promised}}}
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $F_{\mathrm{actual}}$ | benchmark 测到的实际 FLOP/s |
| $F_{\mathrm{promised}}$ | GPU spec 中该 dtype 的 peak FLOP/s |
| $\mathrm{MFU}$ | Model FLOPs utilization |

讲义经验：MFU 约 **0.5** 通常已经很好。

### 6.4 系统增量与接口绑定

**FLOPs/MFU 将 model operation 接到 GPU spec：同一个 PyTorch expression 现在可以转化为实际硬件利用率。**

这为下一节 roofline 提供问题：为什么 MFU 不接近 1？

---

## 7. Arithmetic Intensity and Roofline Analysis

### 7.1 技术路线与演进逻辑

**Roofline analysis 回答 MFU 为什么受限：计算不仅需要 FLOPs，还需要把 bytes 从 memory 搬到 accelerator 再写回 memory。**

计算流程：

```text
send inputs from memory to accelerator
perform computation
send outputs from accelerator to memory
```

核心瓶颈由两个硬件指标决定：

- accelerator speed: FLOP/s
- memory bandwidth: bytes/s

### 7.2 系统设计与资源权衡链

**Arithmetic intensity 是工作负载的 FLOPs/byte；它决定 operation 是 memory-bound 还是 compute-bound。**

硬件示例：

```python
h100_flop_per_sec = 1979e12 / 2
h100_bytes_per_sec = 3.35e12
```

Accelerator intensity：

$$
I_{\mathrm{acc}}=\frac{F_{\mathrm{peak}}}{B_{\mathrm{mem}}}
$$

对 H100 dense bf16 约为：

$$
\frac{1979e12/2}{3.35e12} \approx 295 \ \mathrm{FLOPs/byte}
$$

判定：

| 条件 | 瓶颈 |
|---|---|
| $I_{\mathrm{work}} < I_{\mathrm{acc}}$ | memory-bound |
| $I_{\mathrm{work}} > I_{\mathrm{acc}}$ | compute-bound |

### 7.3 数学原则与推导链

**总时间由 communication time 与 computation time 的较大者决定，等价地由 arithmetic intensity 相对 accelerator intensity 决定。**

时间模型：

$$
t_{\mathrm{comm}}=\frac{\mathrm{bytes}}{B_{\mathrm{mem}}}
$$

$$
t_{\mathrm{comp}}=\frac{\mathrm{FLOPs}}{F_{\mathrm{peak}}}
$$

$$
t_{\mathrm{total}}=\max(t_{\mathrm{comm}},t_{\mathrm{comp}})
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $t_{\mathrm{comm}}$ | memory read/write 时间 |
| $t_{\mathrm{comp}}$ | accelerator 计算时间 |
| $B_{\mathrm{mem}}$ | memory bandwidth，bytes/s |
| $F_{\mathrm{peak}}$ | peak FLOP/s |

Arithmetic intensity：

$$
I_{\mathrm{work}}=\frac{\mathrm{FLOPs}}{\mathrm{bytes}}
$$

Roofline 近似 MFU：

$$
\mathrm{MFU}=\min\left(1,\frac{I_{\mathrm{work}}}{I_{\mathrm{acc}}}\right)
$$

#### ReLU

$$
\mathrm{bytes}=2n+2n=4n
$$

$$
\mathrm{FLOPs}=n
$$

$$
I_{\mathrm{ReLU}}=\frac{n}{4n}=\frac14
$$

结论：**memory-bound**。

#### GeLU

讲义用近似：

$$
\mathrm{FLOPs}=20n,\quad \mathrm{bytes}=4n
$$

$$
I_{\mathrm{GeLU}}=5
$$

结论：GeLU 比 ReLU arithmetic intensity 高，但仍 **memory-bound**；孤立执行时 ReLU 不一定比 GeLU 快。

#### Dot Product

$$
\mathrm{bytes}=2n+2n+2
$$

$$
\mathrm{FLOPs}=2n-1
$$

$$
I_{\mathrm{dot}}\approx\frac12
$$

结论：**memory-bound**。

#### Matrix-Vector Product

$$
\mathrm{bytes}=2n+2n^2+2n
$$

$$
\mathrm{FLOPs}=n(2n-1)
$$

$$
I_{\mathrm{matvec}}\approx 1
$$

结论：**memory-bound**。讲义指出 inference 中常见 matrix-vector，因此 inference memory-bound。

#### Matrix Multiplication

对 $n \times n$ matmul：

$$
\mathrm{bytes}=2n^2+2n^2+2n^2=6n^2
$$

$$
\mathrm{FLOPs}=n^2(2n-1)
$$

$$
I_{\mathrm{matmul}}\approx\frac{n}{3}
$$

当 $n=1024$ 时，$I \approx 341$，高于 H100 accelerator intensity，结论：**compute-bound**。

### 7.4 系统增量与接口绑定

**Roofline 把 operation-level accounting 接到 training/inference 设计：large matmuls 适合训练吞吐，decode 阶段 matrix-vector 更容易 memory-bound。**

与 Lecture 01 的 systems 模块对齐：

- training Transformers involves big matrix multiplications -> compute-bound。
- inference decode often resembles matrix-vector product -> memory-bound。
- elementwise operations isolated -> memory-bound，因此 kernel fusion 可减少 memory movement。

---

## 8. Deep Network Interface

### 8.1 技术路线与演进逻辑

**DeepNetwork 是讲义用来演示 resource accounting 的简化模型：每层是 linear + ReLU，便于把 parameters、activations、gradients 全部数清楚。**

本模块属于教学 scaffold，不是 Transformer 完整架构。它为后续推导 `6 * data points * parameters` 提供可验证例子。

### 8.2 系统设计与资源权衡链

**模型 depth $L$ 同时线性增加 parameters 与 activations；batch size $B$ 主要增加 activations 与 compute。**

代码结构：

```python
class Block(nn.Module):
    def __init__(self, dim: int):
        self.weight = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))

    def forward(self, x):
        x = x @ self.weight
        x = F.relu(x)
        return x
```

```python
class DeepNetwork(nn.Module):
    self.layers = nn.ModuleList([Block(dim) for i in range(num_layers)])
```

### 8.3 数学原则与推导链

**每层参数是一个 $D \times D$ matrix，$L$ 层总参数为 $D^2L$。**

公式：

$$
N_{\mathrm{params}}=D^2L
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $D$ | input/activation/output dimensionality |
| $L$ | number of layers |
| $N_{\mathrm{params}}$ | total trainable parameters |

对 batch input：

$$
x \in \mathbb{R}^{B \times D},\quad y \in \mathbb{R}^{B \times D}
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $B$ | batch size |
| $x$ | input tensor |
| $y$ | model output tensor |

### 8.4 系统增量与接口绑定

**DeepNetwork 是 training accounting 的最小可运行系统：forward 产生 activations，backward 产生 gradients，optimizer 产生 optimizer states。**

---

## 9. Gradients and Backward FLOPs

### 9.1 技术路线与演进逻辑

**Backward pass 是训练资源的主要增量：forward 只算输出，backward 还要为 activations 和 weights 分别算 gradients。**

前置基线是 forward tensor operations。核心崩溃点是训练不仅要推理，还要计算 gradients。破局机制是用 autograd 与显式 FLOPs accounting 拆开 backward cost。

### 9.2 系统设计与资源权衡链

**Backward pass 大约是 forward pass 的 2 倍 FLOPs，因此训练一步约为 forward 的 3 倍。**

简单 gradient 示例：

```python
x = torch.tensor([1., 2, 3])
w = torch.tensor([1., 1, 1], requires_grad=True)
pred_y = x @ w
loss = 0.5 * (pred_y - 5).pow(2)
loss.backward()
assert torch.equal(w.grad, torch.tensor([1, 2, 3]))
```

### 9.3 数学原则与推导链

**对线性层 $h_2=h_1W_2$，backward 需要两个矩阵乘法：一个算 input gradient，一个算 weight gradient。**

Forward：

$$
h_2 = h_1 W_2
$$

其中：

$$
h_1 \in \mathbb{R}^{B \times D},\quad W_2 \in \mathbb{R}^{D \times D},\quad h_2 \in \mathbb{R}^{B \times D}
$$

Forward FLOPs：

$$
F_{\mathrm{forward}}=2BD^2
$$

Backward input gradient：

$$
\frac{\partial \ell}{\partial h_1}=\frac{\partial \ell}{\partial h_2}W_2^\top
$$

Backward weight gradient：

$$
\frac{\partial \ell}{\partial W_2}=h_1^\top \frac{\partial \ell}{\partial h_2}
$$

Backward FLOPs：

$$
F_{\mathrm{backward}}=2BD^2+2BD^2=4BD^2
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $\ell$ | loss |
| $h_1$ | 当前 layer 输入 activations |
| $h_2$ | 当前 layer 输出 activations |
| $W_2$ | 当前 layer parameter matrix |
| $\partial \ell/\partial h_1$ | 传回上一层的 activation gradient |
| $\partial \ell/\partial W_2$ | optimizer 用于更新 parameter 的 gradient |

合计：

$$
F_{\mathrm{total}}=F_{\mathrm{forward}}+F_{\mathrm{backward}}=6BD^2
$$

推广到 all parameters：

$$
F_{\mathrm{train\ step}}=6 \times (\#\mathrm{data\ points}) \times (\#\mathrm{parameters})
$$

讲义说明：该结论对 MLPs 成立，对 short context Transformers 也是 good approximation。

### 9.4 系统增量与接口绑定

**Backward accounting 解释了本讲开头的 $6ND$，并把 training time estimate 接到每个 layer 的 matrix multiplication。**

---

## 10. Optimizer and Training Memory

### 10.1 技术路线与演进逻辑

**Optimizer 不只是 update rule，还引入 persistent optimizer states；AdamW/AdaGrad 的 memory 可能和 parameters 同量级甚至更大。**

讲义从 AdaGrad 出发，并回忆 optimizer 家族关系：

- momentum = SGD + exponential averaging of grad
- AdaGrad = SGD + averaging by grad^2
- RMSProp = AdaGrad but with exponential averaging of grad^2
- Adam = RMSProp + momentum

### 10.2 系统设计与资源权衡链

**Optimizer states 用 fp32 稳定长期累积统计量，代价是显存按 parameter 数线性增长。**

讲义 memory accounting：

```python
num_parameters = D * D * L
parameter_memory = 2 * num_parameters
gradient_memory = 2 * num_parameters
optimizer_state_memory = 4 * num_parameters
activation_memory = 2 * (B * D * L)
total_memory = parameter_memory + activation_memory + gradient_memory + optimizer_state_memory
```

Optimizer state memory：

| Optimizer | State | bytes/parameter |
|---|---|---:|
| AdaGrad | second moments | 4 |
| Adam | first and second moments | 8 |

### 10.3 数学原则与推导链

**AdaGrad 的状态是历史 squared gradients 的累积，用它缩放当前 gradient update。**

讲义代码：

```python
g2 = state.get("g2", torch.zeros_like(grad))
g2 += torch.square(grad)
state["g2"] = g2
p.data -= lr * grad / torch.sqrt(g2 + 1e-5)
```

公式：

$$
G_t = G_{t-1} + g_t^2
$$

$$
\theta_t = \theta_{t-1} - \eta \frac{g_t}{\sqrt{G_t+\epsilon}}
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $\theta_t$ | step $t$ 后的 parameter |
| $g_t$ | 当前 gradient |
| $G_t$ | AdaGrad optimizer state，历史 squared gradients 累积 |
| $\eta$ | learning rate，对应 `lr` |
| $\epsilon$ | 数值稳定项，对应 `1e-5` |

Training step FLOPs：

$$
F_{\mathrm{step}}=6B N_{\mathrm{params}}
$$

其中 $B$ 是 batch size，$N_{\mathrm{params}}=D^2L$。

### 10.4 系统增量与接口绑定

**Optimizer 模块把 gradients 转成 parameter updates，并把 training memory 从 parameters 扩展到 parameters + gradients + optimizer states + activations。**

Training loop 接口：

```text
get_batch -> forward loss -> backward gradients -> optimizer.step -> zero_grad
```

---

## 11. Gradient Accumulation

### 11.1 技术路线与演进逻辑

**Gradient accumulation 解决的是 large batch 想要训练稳定，但 activation memory 随 batch size 增长而可能 OOM。**

前置基线是一次性使用 full batch 做 forward/backward。核心崩溃点是 activation memory scales with batch size。破局机制是 micro batches：多次 backward 累积 gradients，达到等效大 batch 后再 update。

### 11.2 系统设计与资源权衡链

**Gradient accumulation 用更多 sequential micro-batch steps 和更晚的 optimizer update，换取更低 peak activation memory。**

讲义逻辑：

```text
Compute gradient on micro batches
Accumulate the gradients
Every batch_size / micro_batch_size steps, update parameters and zero gradients
```

Activation memory 从：

$$
2BDL
$$

变为：

$$
2B_{\mathrm{micro}}DL
$$

### 11.3 数学原则与推导链

**如果 loss/gradient 按 batch 平均方式一致处理，多个 micro-batch gradients 的累积可模拟 large batch gradient。**

讲义没有展开 normalization 细节，因此只保留资源公式：

$$
M_{\mathrm{activation}}=2BDL
$$

$$
M_{\mathrm{activation,micro}}=2B_{\mathrm{micro}}DL
$$

符号解释：

| 符号 | 工程含义 |
|---|---|
| $M_{\mathrm{activation}}$ | full batch activation memory |
| $M_{\mathrm{activation,micro}}$ | micro batch peak activation memory |
| $B$ | intended large batch size |
| $B_{\mathrm{micro}}$ | micro batch size |
| $D$ | activation dimension |
| $L$ | number of layers |
| $2$ | bf16 bytes/value |

### 11.4 系统增量与接口绑定

**Gradient accumulation 修改 training loop 的 update cadence，不修改 model forward semantics。**

系统变化：

```text
full batch: forward/backward once -> optimizer.step
accumulated: repeat micro forward/backward -> optimizer.step once
```

---

## 12. Activation Checkpointing

### 12.1 技术路线与演进逻辑

**Activation checkpointing 解决的是 backward 需要保存所有 layer activations 导致 memory 随 depth $L$ 增长。**

前置基线是训练时保存所有 activations。核心崩溃点是 deep networks 的 activation memory 大。破局机制是只保存部分 checkpoints，backward 时 recompute missing activations。

别名：

**Activation Checkpointing (激活检查点)**: 训练时只保存部分 layer activations，并在 backward 需要时重新计算未保存 activations，以 compute 换 memory。

讲义等价名：

```text
Activation checkpointing = gradient checkpointing = rematerialization
```

### 12.2 系统设计与资源权衡链

**Activation checkpointing 是显式 memory-compute tradeoff：少存 activation，就要在 backward 中多算 forward。**

讲义示意：

```text
Store all activations: x g1 h1 g2 h2 g3 h3 g4 h4
Activation checkpointing: x h1 h2 h3 h4
```

PyTorch 接口：

```python
x = torch.utils.checkpoint.checkpoint(layer, x)
```

更深网络的策略：

```text
Store all layers: | h1 h2 h3 h4 h5 h6 h7 h8 h9 |
Store no layers:  | |
Store some:       | h3 h6 h9 |
```

### 12.3 数学原则与推导链

**Checkpoint frequency 决定 memory complexity 与 recomputation complexity。**

讲义给出三种极端/折中：

| 策略 | Activation memory | Recomputation |
|---|---|---|
| store each layer | $O(L)$ | no recomputation |
| store no activations | $O(1)$ | $O(L^2)$ compute from start for each layer |
| store every $\sqrt{L}$ layers | $O(\sqrt{L})$ | $O(L)$ recomputation |

符号解释：

| 符号 | 工程含义 |
|---|---|
| $L$ | number of layers |
| $O(L)$ memory | 保存每层 activation |
| $O(1)$ memory | 几乎不保存中间 activation |
| $O(\sqrt{L})$ memory | 每隔 $\sqrt{L}$ 层保存 checkpoint |
| $O(L)$ recomputation | 额外 forward 重算量与网络深度同阶 |

### 12.4 系统增量与接口绑定

**Activation checkpointing 修改 autograd 保存策略，不改变 forward output；它与 gradient accumulation 都服务于更大 batch 或更深 model 的可训练性。**

系统变化：

```text
forward: save fewer activations
backward: recompute missing activations from checkpoints
memory decreases, compute increases
```

---

## 13. Lecture 02 总结接口

**Lecture 02 的核心产物是一个可复用的 resource accounting checklist：任何训练计划都要先数 tensor memory、operation FLOPs、roofline bottleneck、training state memory。**

关键公式与接口：

| 模块 | 输入 | 输出 | 核心公式/判断 |
|---|---|---|---|
| Tensor memory | shape, dtype | bytes | `numel * element_size` |
| Matmul FLOPs | $B,D,K$ | FLOPs | $2BDK$ |
| MFU | actual FLOP/s, promised FLOP/s | utilization | $F_{\mathrm{actual}}/F_{\mathrm{promised}}$ |
| Roofline | FLOPs, bytes, hardware spec | memory-bound or compute-bound | compare $I_{\mathrm{work}}$ and $I_{\mathrm{acc}}$ |
| Training FLOPs | batch/data points, parameters | FLOPs/step | $6 \times B \times N$ |
| Training memory | params, grads, optimizer, activations | peak memory estimate | sum of tensor categories |
| Gradient accumulation | large batch, micro batch | lower activation memory | $2B_{\mathrm{micro}}DL$ |
| Activation checkpointing | layer count, checkpoint spacing | memory-compute tradeoff | $O(\sqrt L)$ memory with $O(L)$ recompute |

全局护栏：以上内容严格来自 `lecture/lecture_02.md`、历史 `lecture_01` 本地产物与本地 SOP 文件；未使用外部搜索或外部教程补全。
