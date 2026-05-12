# Phase 2 - Agent B: Educator Cognitive Synthesis

目标参数：`lecture_01`

输入文件：

- `E:\allwork\cs336\lecture\lecture_01.md`
- `output/cs336/lecture_01/01_Phase1_Architect.md`

输出文件：`output/cs336/lecture_01/02_Phase2_Educator.md`

对齐约束：本产物以 Phase 1 的技术抽取为事实骨架；所有比喻和记忆模型只用于理解，不覆盖 Phase 1 的公式、模块边界和工程判断。

## 1. 核心冲突

**本讲的核心冲突是：语言模型越工业化，个人越无法复刻 frontier training；但越是这样，研究者越需要回到底层理解那些可迁移的机制和效率原则。**

讲义开头不是在介绍一个热门模型清单，而是在解释为什么只会 prompt API 不够。抽象层提高生产力，但语言模型的抽象不像编程语言或操作系统那样可靠封装。数据、架构、tokenizer、优化器、系统瓶颈和 post-training 都会泄漏到模型行为里。

课程真正要保住三类东西：

| 内容 | 如何理解 |
| --- | --- |
| Mechanics | 机器内部齿轮怎么咬合，例如 tokenizer、Transformer、parallelism |
| Mindset | 每个选择都要过 resource accounting，例如 compute、memory、bandwidth、data |
| Intuitions | 有用但不保证跨规模迁移，因为很多经验来自实验 |

本讲最短判断式是：

```text
accuracy = efficiency x resources
```

所以后面的所有技术都不是孤立技巧，而是在问同一个问题：固定资源下，怎样少浪费一点。

## 2. 全局心智模型：高成本炼油厂

**整门课可以记成一座“文本炼油厂”：raw text 是原料，tokenizer 是切料机，Transformer 是反应炉，training 是控制系统，systems 是管道和阀门，scaling laws 是小试装置，data 是原料筛选，alignment 是精炼工序。**

这座工厂的目标不是把机器造得最大，而是在预算内产出最高质量的模型。每个环节都收费：文本切得太碎，后面的反应炉要处理更多 items；词表太大，embedding 和 output head 会更稀疏；参数太大，HBM 和通信吃紧；数据太脏，昂贵计算被浪费；alignment rollout 太慢，训练系统被推理吞吐拖住。

模块关系如下：

| 模块 | 工厂隐喻 | 输出流向 |
| --- | --- | --- |
| Tokenization | 切料机 | raw text 变成 token ids，决定 `seq_len` 与 `vocab_size` |
| Architecture | 反应炉 | token 交互形成 hidden states 与 logits |
| Training | 控制系统 | 用 loss 与 optimizer 调整参数 |
| Systems | 管道/阀门 | 减少数据搬运，让 GPU 真正做计算 |
| Scaling laws | 小试装置 | 用小规模实验预测大规模配方 |
| Data | 原料筛选 | 决定 token budget 的质量与混合 |
| Alignment | 精炼工序 | 用偏好或 verifier 信号修正模型行为 |

这个隐喻解释了为什么 lecture_01 先讲课程动机，再讲技术史，再讲课程单元，最后落到 tokenizer。tokenizer 是第一台机器；它一旦切错，后面每个环节都会替它付账。

## 3. 技术谱系的认知重构

**讲义中的历史线不是论文年表，而是在说明 LLM 的能力来自可扩展机制的连续累积。**

pre-neural 时代用 Shannon entropy 和 n-gram 统计语言；neural ingredients 时代引入 LSTM、feedforward neural LM、seq2seq、Adam、attention、Transformer、MoE、parallelism；early foundation models 通过 ELMo、BERT、T5 展示 pretraining 和 task unification；scaling 时代通过 GPT-2、scaling laws、GPT-3、PaLM、Chinchilla 让“更大”变成可预测工程决策；open models 则让研究者重新获得观察和复现实验的材料。

记忆链：

```text
统计语言
  -> 神经表示
  -> 注意力和 Transformer
  -> 预训练 foundation model
  -> scaling laws 让规模可预测
  -> open models 让课程可教学、可研究
```

每个阶段都解决上一阶段的瓶颈，同时制造新瓶颈。Transformer 解决 RNN 串行性，但带来 attention quadratic cost；MoE 增加容量，但带来 routing 和 load balancing；open models 提升可研究性，但要求我们能读懂 recipe 中的数据、系统和训练选择。

## 4. Basics 的认知模型

**Basics 的学习任务是把“语言模型”还原成三个可实现部件：文本怎么进来、token 怎么交互、参数怎么被更新。**

第一步是 tokenization：模型不直接读 Python 字符串，而是读整数序列。第二步是 model architecture：Transformer 等结构让 token 之间交互。第三步是 training：loss、optimizer、initialization、learning-rate schedule、regularization、batch size 决定参数如何移动。

Basics 的记忆锚点是三角平衡：

| 角 | 过低时会怎样 |
| --- | --- |
| Expressivity | 模型表示不了复杂依赖 |
| Stability | 参数或梯度失控，训练不稳 |
| Efficiency | 能训练但硬件成本不可接受 |

Assignment 1 把这个三角压成代码：实现 BPE tokenizer、Transformer、cross-entropy、AdamW、training loop，再做 resource accounting。也就是说，课程先让你亲手搭出最小闭环，而不是先背现代 LLM recipe。

## 5. 架构模块的认知模型

**模型架构可以理解为“反应炉的内部结构”：每个 refinement 都在调整容量、稳定性或运行成本。**

SwiGLU 像是在 FFN 中加一个门，让信息不是单纯通过非线性，而是被另一路投影调制；但因为多了一组矩阵，讲义指出 hidden dimension 要调小到两矩阵版本的 `2/3`。RoPE 像把位置信息旋进向量空间，不是给 token 贴静态标签，而是在 attention 计算中让相对位置进入几何关系。RMSNorm、pre-norm、QK norm 都在控制信号尺度，避免深层网络的激活和梯度偏离稳定区。

attention 变体的记忆链：

```text
full attention 表达强
  -> sequence 长时 quadratic cost 太高
  -> sparse/local attention 限制连接
  -> GQA/MQA 减少 KV heads 提升 decode
  -> MLA 压缩 KV cache
  -> linear attention / SSM 尝试改变长序列成本结构
```

MoE 的直觉是“不是每个 token 都经过所有专家”。它用 sparse activation 增加总参数容量，但代价是 router、expert load balance 和跨设备通信。架构选择因此从来不是单点质量问题，而是“质量收益是否值得系统成本”。

## 6. Training 的认知模型

**Training 是控制系统：它不直接决定模型会什么，而是决定参数沿什么规则、以什么尺度、在什么噪声条件下移动。**

loss 是方向盘，optimizer 是传动系统，initialization 是起步姿态，LR schedule 是油门曲线，batch size 是一次看多少样本，regularization 是防止模型走偏的约束。MoE load balancing 则是特殊的调度问题：如果专家负载失衡，某些路径过载，某些路径闲置，训练既不稳定也不经济。

训练模块的记忆链：

```text
预测目标定义梯度方向
  -> optimizer 把梯度转成参数更新
  -> initialization 和 norm 控制尺度
  -> LR schedule 控制时间动态
  -> batch size 控制噪声与并行效率
  -> MoE 还要控制专家负载
```

muP 的认知位置尤其重要：它服务于“不能在大模型上随便调参”的现实。小模型调出来的超参如果能迁移，大模型训练就少一次昂贵盲试。

## 7. Systems 的认知模型

**Systems 的核心直觉是：GPU 很快，但数据搬运很慢；大模型工程的很多技巧都是为了少搬一次。**

讲义用 B200 的 FLOPs 和 memory bandwidth 说明 compute 与 memory 不是同一种资源。一个算子如果频繁读写 HBM，即使数学上 FLOPs 不多，也可能被 memory bandwidth 卡住。kernel fusion、tiling、FlashAttention 类思路，都是减少中间结果落回 HBM。

系统单元的推导链：

```text
参数和激活很大
  -> 必须从 HBM 搬到 SM 计算
  -> HBM 带宽有限
  -> 要减少读写和跨 GPU 通信
  -> 引出 fused kernels、tiling、parallelism、sharding
```

推理再加一层区分：prefill 一次处理 prompt tokens，像训练一样更偏 compute-bound；decode 每次只生成一个 token，更容易 memory-bound。推理优化不只是“让模型快一点”，而是要区分 prefill 与 decode 两种硬件状态。

## 8. Scaling Laws 的认知模型

**Scaling laws 的核心直觉是：不能在最终规模上试错，所以要先造小试装置，用小实验预测大训练。**

如果目标训练预算是 `1e25 FLOPs`，直接调超参太贵。讲义要求把思维从“一个模型规模”切换成“一个 scaling recipe”：不同 FLOPs 下模型大小、数据量、超参如何变化。然后在小规模上采样，拟合规律，外推到目标规模。

记忆链：

```text
full-scale 太贵
  -> 只能做 small-scale experiments
  -> 需要 hyperparameter transfer
  -> 拟合 loss 随 N、D、C 的变化
  -> 用预测而不是盲试来选大规模 recipe
```

Chinchilla 部分的心智模型是天平：一边是参数 `N`，一边是 tokens `D`。固定 compute 下，模型太大但数据太少会 undertrain；数据太多但模型太小会受容量限制。讲义给出的经验锚点是：

$$
D \approx 20N
$$

但要记住 caveat：这没有纳入 inference cost，真实部署时小模型可能更有吸引力。

## 9. Data 的认知模型

**Data 单元的核心直觉是：token budget 是燃料预算，低质量、重复或不匹配的数据会把昂贵训练烧在错误方向。**

数据不是“下载文本”这么简单。网页、PDF、代码仓库、书籍、论文都要先转成可训练文本；还要 filtering、deduplication、mixing，并考虑 copyright、licensing 与 contamination。evaluation 也分 internal 与 external：前者看开发过程的平滑改进，后者看真实 use case 的 absolute quality。

记忆链：

```text
模型能力取决于训练数据
  -> raw data 不是干净文本
  -> 需要 transform/filter/dedup/mix
  -> token budget 才不被浪费
  -> evaluation 反过来指导 data recipe
```

本模块与 scaling laws 的关系是：scaling laws 关心 `D` 的数量与 compute 分配，data 单元关心 `D` 的内容质量。同样的 token 数，不同数据混合会带来不同能力。

## 10. Alignment 的认知模型

**Alignment 的核心直觉是：当生成正确答案很难但评价好坏较容易时，可以用 critique 信号继续训练模型。**

讲义把 alignment 放在最后，是因为它假设模型已经通过 next-token prediction 学到基础能力。之后的步骤变成：让模型生成 responses，用 human、verifier 或 LM judge 打分，再更新模型偏好更好的 responses。

记忆链：

```text
next-token pretraining 得到基础模型
  -> 有些任务生成难、评价相对容易
  -> 收集偏好或 verifier 信号
  -> 用 PPO/DPO/GRPO 等方法更新模型
  -> 新瓶颈变成 RL 稳定性和 rollout 系统效率
```

本讲没有展开 PPO/DPO/GRPO 的数学推导，所以这里不补公式。要保留的工程事实是：alignment 不只是算法选择，还会引入异步推理、on-policyness 和系统吞吐之间的权衡。

## 11. Tokenization 的认知模型

**Tokenization 的核心直觉是：模型不是读文字，而是读离散零件；切得太碎浪费序列长度，切得太粗浪费词表并制造 OOV。**

四种 tokenizer 可以用“切料尺度”理解：

| 方案 | 像什么 | 崩溃边界 |
| --- | --- | --- |
| Character | 每个 Unicode 字符一块料 | 字符种类太多，稀有字符浪费词表 |
| Byte | 每个 UTF-8 byte 一块料 | 词表小但序列极长，attention 成本高 |
| Word | 每个词一块料 | 词表爆炸，新词需要 `UNK` |
| BPE | 从小块开始，按高频组合自动粘合 | 需要训练 merge table，朴素 encode 可能慢 |

BPE 的 aha moment 是：它没有先问“语言学上什么是词”，而是把 tokenization 变成压缩问题。

```text
从 bytes 开始
  -> 保证所有字符串都可表示、可逆
  -> 统计最常见相邻 pair
  -> 把高频 pair 合并成新 token
  -> 常见片段变短，罕见片段仍能拆成 bytes 表示
```

这就是为什么 BPE 同时满足三件事：

| 目标 | BPE 的方式 |
| --- | --- |
| 可逆 | `vocab` 存 token id 到 bytes，decode 拼 bytes 再 UTF-8 decode |
| 固定词表 | 初始 256 byte tokens 加有限次 merges |
| 压缩 | 高频 byte/token 序列合并为单 token |

## 12. 全局记忆压缩

**本讲可以压缩成一条因果链：封闭 frontier 迫使我们学习底层机制，资源约束迫使我们关心效率，tokenization 是第一个能亲手验证这种效率思维的组件。**

最短链条：

```text
frontier models 昂贵且不透明
  -> 不能只依赖 API 抽象
  -> 学 mechanics + mindset
  -> mindset = maximize efficiency under fixed resources
  -> raw text 必须变成 token ids
  -> tokenizer 决定 seq_len 与 vocab_size
  -> seq_len 影响 attention/context 成本
  -> vocab_size 影响 embedding/output head 成本
  -> bytes 太碎，words 太粗
  -> BPE 用 learned merges 找中间粒度
```

后续学习时，每遇到一个技术点都用同一组三个问题检查：

```text
它解决了哪个资源瓶颈？
它把成本转移到了哪里？
它是否能随 scale 继续工作？
```

## 13. Phase 3 初始化状态

**Phase 3 已具备上下文：后续问答应保持 Agent A/B 分工，先守住技术骨架，再做直觉解释。**

已生成产物：

- `output/cs336/lecture_01/01_Phase1_Architect.md`：结构化技术路线、系统设计、数学核心与数据流。
- `output/cs336/lecture_01/02_Phase2_Educator.md`：核心冲突、全局心智模型、模块间关系与记忆压缩链。

后续若进入作业代码讨论，严格执行 anti-spoonfeeding：不直接给完整 assignment 实现，只逐步 review 用户已有代码，定位接口、shape、资源核算和测试问题，并引导用户自己完成实现。

