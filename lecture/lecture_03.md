

--- Page 1 ---

Lecture 3
E ’
VERYTHING YOU DIDN T WANT TO KNOW ABOUT
LM
ARCHI TECTU RE AND HYPERPARAMETERS
CS336
Tatsu H

[TABLE]
| Lecture 3 E ’ VERYTHING YOU DIDN T WANT TO KNOW ABOUT LM ARCHI TECTU RE AND HYPERPARAMETERS CS336 Tatsu H |
| --- |
|  |


--- Page 2 ---

Outline and goals
❖ Quick recap of a modern transformer (what you implement)
❖ What do most of the large LMs have in common?
❖ What are common variations to the architecture / training process?
Today’s theme: the best way to learn is hands-on experience
the second best way is to try to learn from others’ experience

[TABLE]
|  |
| --- |
| Outline and goals ❖ Quick recap of a modern transformer (what you implement) ❖ What do most of the large LMs have in common? ❖ What are common variations to the architecture / training process? Today’s theme: the best way to learn is hands-on experience the second best way is to try to learn from others’ experience |


--- Page 3 ---

Starting point: the ‘original’ transformer
Review: choices in the standard transformer
Position embedding: sines and cosines
FFN: ReLU
Norm type: post-norm, LayerNorm

[TABLE]
|  |
| --- |
| Starting point: the ‘original’ transformer Review: choices in the standard transformer Position embedding: sines and cosines FFN: ReLU Norm type: post-norm, LayerNorm |


--- Page 4 ---

What you implemented – simple, modern variant
Differences:
• LayerNorm is in front of the block
• Rotary position embeddings (RoPE)
• FF layers use SwiGLU, not ReLU
• Linear layers (and layernorm) have no
bias (constant) terms
Why did we pick these?
What should you pick?

[TABLE]
|  |
| --- |
| What you implemented – simple, modern variant Differences: • LayerNorm is in front of the block • Rotary position embeddings (RoPE) • FF layers use SwiGLU, not ReLU • Linear layers (and layernorm) have no bias (constant) terms Why did we pick these? What should you pick? |


--- Page 5 ---

How should we think about architectures?
Lots of architecture. Just in 2024-2025..
Over 19 new dense model releases, many of them with minor architecture tweaks..

[TABLE]
|  |
| --- |
| How should we think about architectures? Lots of architecture. Just in 2024-2025.. Over 19 new dense model releases, many of them with minor architecture tweaks.. |

| None | None | None | None | ts of architecture. Ju | None | None | None | None | None | None | None | None | None | None |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| None | None | None | None | None | None | None | None | None | None | None |  | None | None | None |
| None | None | None |  |  | None | None | None | None | None | None | None | None | None | None |
| None | None | None | None | None | None | None | None | None | None | None | None |  | None | None |
| None |  |  |  | None | None | None | None | None |  | None |  | None | None | None |
| None | None | None | None | None | None | None | None | None |  | None |  |  | None | None |
| None | None | None | None | None |  | None | None | None |  | None |  |  | None |  |
| Over 1 | Over 1 | None | None | None | 9 new dense model releases, | None | None | None |  |  | None |  | None |  |
| None | None | None | None | None | None | None | l releases, | None | m | any of them with min | None |  | None | None |
| None | None | None | None | None | None | None | None | None | None | None | None | None |  | None |
| None | None | None | None | None | None | None | None | None | None | None | None | them with min | or architec | ture tw |
| None | None | None | None | None |  | None |  | None | None | None | None | None | None | None |
| None | None | None | None | None |  | None |  | None |  |  | None |  |  | None |
| None | None | None | None |  |  | None |  | None | None |  | None |  | None | None |
| None | None | None | None | None | None | None | None | None |  | None | None |  |  | None |
| None | None | None | None | None | None | None |  |  | None | None | None | None | None | None |
| None | None | None | None | None | None | None | None |  | None | None | None | None | None | None |
| None | None | None | None | None | None |  | None |  |  | None | None | None | None | None |


--- Page 6 ---

How should we think about architectures?
There can’t be that many LLMs released this year right?

[TABLE]
| None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None |  | None | None |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | None | None | None | None | None | None | None |  | None | None | None | None | None | None | None | None | None | None | None | None | None |  | None |  | None | None |
| How should we think about architectures? There can’t be that many LLMs released this year right? | None | None | None | None | None | None | None | we think about archite many LLMs released this year righ | None | None | None | None | None | None | None | None | None | None | None | None | None |  | None |  | None | None |
| None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | te | None | None | None | None |  | None | None |
| None | None | None | None | ow should we thi ere can’t be that many LL | None | None | None | we thi many LL | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None |
| None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | year r | None | igh | None | None | None | None | None | None | None |
| None | None | None |  |  |  | None | None | None | None | None | None |  | None | None | None | None | None | None | None | None | None | None | None | None | None | None |
| None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None |  | None | None | None | None | None |  |
| None | None | None | None | None | None | None | None | None |  | None | None |  | None | None | None | None |  | None | None | None | None | None | None | None | None | None |
| None |  | None |  | None | None | None | None | None |  |  | None | None | None | None | None | None | None | None | None | None | None | None |  | None | None |  |
| None | None | None | None | None | None | None | None | None |  | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None |
| None | None | None | None | None | None |  | None | None |  |  | None | None | None | None |  | None |  |  | None |  | None | None |  | None | None | None |
| None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None |  | None | None | None | None |  | None | None | None |
| None | None | None | None | None | None | None |  | None |  |  | None | None | None |  |  | None | None |  | None | None | None | None | None | None | None | None |
| None | None |  | None | None | None |  |  | None |  |  | None | None | None | None | None | None | None |  | None | None |  | None |  | None |  | None |
| None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None |  | None |
| None | None | None | None | None | None | None | None | None | None | None | None | None |  |  | None | None | None | None | None | None | None | None | None | None | None | None |
| None | None |  | None | None | None |  | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None |
| None |  |  | None | None | None |  |  | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None |
| None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None |  | None |  | None | None |  | None | None | None |  | None |
| None | None | None | None | None | None | None | None | None | None | None |  | None |  | None |  | None | None | None | None | None | None | None | None | None | None | None |
| None | None | None | None | None | None | None | None | None | None | None |  | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None |


--- Page 7 ---

Let’s look at the data (on dense architectures)
Learn from the many other models (and papers) out there
We will talk through many major
architecture and hyperparameter variants.
• What do all these models have in common?
• What parts vary?
• What can we learn from this?

[TABLE]
|  |
| --- |
| Let’s look at the data (on dense architectures) Learn from the many other models (and papers) out there We will talk through many major architecture and hyperparameter variants. • What do all these models have in common? • What parts vary? • What can we learn from this? |


--- Page 8 ---

What are we going to cover?
Common architecture variations
• Activations, FFN
• Attention variants
• Position embeddings
Hyperparameters that (do or don’t) matter
• What is ff_dim? Do multi_head dims always sum to model_dim?
• How many vocab elements?
Stability tricks

[TABLE]
|  |
| --- |
| What are we going to cover? Common architecture variations • Activations, FFN • Attention variants • Position embeddings Hyperparameters that (do or don’t) matter • What is ff_dim? Do multi_head dims always sum to model_dim? • How many vocab elements? Stability tricks |


--- Page 9 ---

Architecture variations..
Let’s think about the core architecture piece
High level view:
• Dominance of ‘LLaMA-
like’ architectures
• Trends over the years
(QK-norm, Hybrid
attention)

[TABLE]
|  |
| --- |
| Architecture variations.. Let’s think about the core architecture piece High level view: • Dominance of ‘LLaMA- like’ architectures • Trends over the years (QK-norm, Hybrid attention) |


--- Page 10 ---

Pre-vs-post norm
The one thing everyone agrees on (in 2024)
Set up LayerNorm so that it doesn’t affect the
main residual signal path (on the left)
Figure from Xiong 2020
Almost all modern LMs use pre-norm (but BERT was post-norm)
(One somewhat funny exception – OPT350M. I don’t know why this is post-norm)

[TABLE]
|  |
| --- |
| Pre-vs-post norm The one thing everyone agrees on (in 2024) Set up LayerNorm so that it doesn’t affect the main residual signal path (on the left) Figure from Xiong 2020 Almost all modern LMs use pre-norm (but BERT was post-norm) (One somewhat funny exception – OPT350M. I don’t know why this is post-norm) |


--- Page 11 ---

Pre-vs-post-norm, the data
Salazar and Ngyuen 2019
Figure from Xiong 2020

[TABLE]
|  |
| --- |
| Pre-vs-post-norm, the data Salazar and Ngyuen 2019 Figure from Xiong 2020 |


--- Page 12 ---

Pre-vs-post norm, explanations?
Gradient attenuation [Xiong 2020] Gradient spikes [Salazar and Ngyuen]
Original stated advantage– removing warmup.
Today – stability and larger LRs for large networks

[TABLE]
|  |
| --- |
| Pre-vs-post norm, explanations? Gradient attenuation [Xiong 2020] Gradient spikes [Salazar and Ngyuen] Original stated advantage– removing warmup. Today – stability and larger LRs for large networks |


--- Page 13 ---

New things – ‘double’ norm or non-residual postnorm
If putting LayerNorms in residual streams is bad.. Why not post-norm outside the stream?
Recent models: Grok, Gemma 2. Olmo 2 only does non-residual post norm

[TABLE]
|  |
| --- |
| New things – ‘double’ norm or non-residual postnorm If putting LayerNorms in residual streams is bad.. Why not post-norm outside the stream? Recent models: Grok, Gemma 2. Olmo 2 only does non-residual post norm |


--- Page 14 ---

LayerNorm vs RMSNorm
Original transformer: LayerNorm – normalizes Notable models:
the mean and variance across 𝑑
𝑚𝑜𝑑𝑒𝑙
GPT3/2/1, OPT, GPT-J, BLOOM
Many modern LMs: RMSNorm – does not
Notable models:
subtract mean or add a bias term
LLaMA-family, PaLM, Chinchilla, T5
𝑥
𝑦 = ∗ 𝛾
2
𝑥 + 𝜀
2

[TABLE]
|  |
| --- |
| LayerNorm vs RMSNorm Original transformer: LayerNorm – normalizes Notable models: the mean and variance across 𝑑 𝑚𝑜𝑑𝑒𝑙 GPT3/2/1, OPT, GPT-J, BLOOM Many modern LMs: RMSNorm – does not Notable models: subtract mean or add a bias term LLaMA-family, PaLM, Chinchilla, T5 𝑥 𝑦 = ∗ 𝛾 2 𝑥 + 𝜀 2 |


--- Page 15 ---

Why RMSNorm?
Modern explanation – it’s faster (and just as good).
• Fewer operations (no mean calculation)
• Fewer parameters (no bias term to store)
Does this explanation make sense?
Matrix multiplies are the vast majority of FLOPs (and memory)
[Ivanov et al 2023]

[TABLE]
|  |
| --- |
| Why RMSNorm? Modern explanation – it’s faster (and just as good). • Fewer operations (no mean calculation) • Fewer parameters (no bias term to store) Does this explanation make sense? Matrix multiplies are the vast majority of FLOPs (and memory) [Ivanov et al 2023] |


--- Page 16 ---

Why RMSNorm (2)
Important lesson: FLOPS are not runtime! (we will discuss this in far more detail later)
RMSNorm can still matter due to
the importance of data movement
Left top (”43G”) is FLOPS
Right top (“153”) is the FLOP-to-memory ratio
[Ivanov et al 2023]

[TABLE]
|  |
| --- |
| Why RMSNorm (2) Important lesson: FLOPS are not runtime! (we will discuss this in far more detail later) RMSNorm can still matter due to the importance of data movement Left top (”43G”) is FLOPS Right top (“153”) is the FLOP-to-memory ratio [Ivanov et al 2023] |


--- Page 17 ---

RMSNorm - validation
RMSNorm runtime (and surprisingly, perf) gains have been seen in papers
Narang et al 2020

[TABLE]
|  |
| --- |
| RMSNorm - validation RMSNorm runtime (and surprisingly, perf) gains have been seen in papers Narang et al 2020 |


--- Page 18 ---

More generally: dropping bias terms
Most modern transformers don’t have bias terms.
Original Transformer:
Most implementations (if they’re not gated):
𝐹𝐹𝑁 𝑥 = 𝜎 𝑥𝑊 𝑊
1 2
Reasons: memory (similar to RMSnorm) and optimization stability

[TABLE]
|  |
| --- |
| More generally: dropping bias terms Most modern transformers don’t have bias terms. Original Transformer: Most implementations (if they’re not gated): 𝐹𝐹𝑁 𝑥 = 𝜎 𝑥𝑊 𝑊 1 2 Reasons: memory (similar to RMSnorm) and optimization stability |


--- Page 19 ---

LayerNorm: recap
• Basically everyone does non-residual norm (often prenorm)
• Intuition – keep the good parts of residual connections
• Observations – nicer gradient propagation, fewer spike
• Some people add a second norm outside the residual stream
• Most people do RMSnorm
• In practice, works as well as LayerNorm
• But, has fewer parameters to move around, which saves on wallclock time
• People more generally drop bias terms since the compute/param tradeoffs are not
great.

[TABLE]
|  |
| --- |
| LayerNorm: recap • Basically everyone does non-residual norm (often prenorm) • Intuition – keep the good parts of residual connections • Observations – nicer gradient propagation, fewer spike • Some people add a second norm outside the residual stream • Most people do RMSnorm • In practice, works as well as LayerNorm • But, has fewer parameters to move around, which saves on wallclock time • People more generally drop bias terms since the compute/param tradeoffs are not great. |


--- Page 20 ---

Activations
A whole zoo of activations ..
ReLU, GeLU, Swish, ELU, GLU, GeGLU, ReGLU, SeLU, SwiGLU, LiGLU
What are these things? What do people use? Does it matter?

[TABLE]
|  |
| --- |
| Activations A whole zoo of activations .. ReLU, GeLU, Swish, ELU, GLU, GeGLU, ReGLU, SeLU, SwiGLU, LiGLU What are these things? What do people use? Does it matter? |


--- Page 21 ---

A few of the common activations
Notable models:
ReLU
𝐹𝐹 𝑥 = max 0,𝑥𝑊 𝑊
1 2
Original transformer, T5,
Gopher, Chinchilla, OPT
GeLU
𝐹𝐹 𝑥 = GELU 𝑥𝑊 𝑊 Notable models:
1 2
𝐺𝐸𝐿𝑈 𝑥 ≔ 𝑥Φ(𝑥)
GPT1/2/3, GPTJ, GPT-Neox,
BLOOM
Notable models:
SwiGLU / GeGLU (next slide..)
Llama, PaLM,T5 v1.1, most
models post 2023

[TABLE]
|  |
| --- |
| A few of the common activations Notable models: ReLU 𝐹𝐹 𝑥 = max 0,𝑥𝑊 𝑊 1 2 Original transformer, T5, Gopher, Chinchilla, OPT GeLU 𝐹𝐹 𝑥 = GELU 𝑥𝑊 𝑊 Notable models: 1 2 𝐺𝐸𝐿𝑈 𝑥 ≔ 𝑥Φ(𝑥) GPT1/2/3, GPTJ, GPT-Neox, BLOOM Notable models: SwiGLU / GeGLU (next slide..) Llama, PaLM,T5 v1.1, most models post 2023 |


--- Page 22 ---

Gated activations (*GLU)
GLUs modify the ‘first part’ of a FF layer
𝐹𝐹 𝑥 = max 0, 𝑥𝑊 𝑊
1 2
Instead of a linear + ReLU, augment the above with an (entrywise) linear term
max 0, 𝑥𝑊 → max 0, 𝑥𝑊 ⊗ (𝑥𝑉)
1 1
This gives the gated variant (ReGLU) – note that we have an extra parameter (V)
FF 𝑥 = (max 0, 𝑥𝑊 ⊗ 𝑥𝑉) 𝑊
ReGLU 1 2

[TABLE]
|  |
| --- |
| Gated activations (*GLU) GLUs modify the ‘first part’ of a FF layer 𝐹𝐹 𝑥 = max 0, 𝑥𝑊 𝑊 1 2 Instead of a linear + ReLU, augment the above with an (entrywise) linear term max 0, 𝑥𝑊 → max 0, 𝑥𝑊 ⊗ (𝑥𝑉) 1 1 This gives the gated variant (ReGLU) – note that we have an extra parameter (V) FF 𝑥 = (max 0, 𝑥𝑊 ⊗ 𝑥𝑉) 𝑊 ReGLU 1 2 |


--- Page 23 ---

Gated variants of standard FF layers
GeGLU Notable models:
T5 v1.1, mT5, LaMDA, Phi3,
Gemma 2, Gemma 3, Gemma 4
Notable models:
SwiGLU (swish is 𝑥 ∗ sigmoid(𝑥))
LLaMa 1/2/3, PaLM, Mistral,
OlMo, most models post 2023
Note: Gated models use smaller dimensions for the 𝑑 by 2/3
𝑓𝑓

[TABLE]
|  |
| --- |
| Gated variants of standard FF layers GeGLU Notable models: T5 v1.1, mT5, LaMDA, Phi3, Gemma 2, Gemma 3, Gemma 4 Notable models: SwiGLU (swish is 𝑥 ∗ sigmoid(𝑥)) LLaMa 1/2/3, PaLM, Mistral, OlMo, most models post 2023 Note: Gated models use smaller dimensions for the 𝑑 by 2/3 𝑓𝑓 |


--- Page 24 ---

Do gated linear units work?
Yes, fairly consistently so.
Shazeer 2020

[TABLE]
|  |
| --- |
| Do gated linear units work? Yes, fairly consistently so. Shazeer 2020 |


--- Page 25 ---

Do gated linear units work (2)?
Yes, with other works corroborating Shazeer 2020
Narang et al 2020

[TABLE]
|  |
| --- |
| Do gated linear units work (2)? Yes, with other works corroborating Shazeer 2020 Narang et al 2020 |


--- Page 26 ---

Gating, activations
• Many variations (ReLU, GeLU, *GLU) across models.
• *GLU isn’t necessary for a working model (see GPT3), but it’s rare to see others..
Some outlier models..
Nemotron 340B (Squared ReLU)
• Evidence points towards somewhat consistent gains from Swi/GeGLU

[TABLE]
|  |
| --- |
| Gating, activations • Many variations (ReLU, GeLU, *GLU) across models. • *GLU isn’t necessary for a working model (see GPT3), but it’s rare to see others.. Some outlier models.. Nemotron 340B (Squared ReLU) • Evidence points towards somewhat consistent gains from Swi/GeGLU |


--- Page 27 ---

Serial vs Parallel layers
Normal transformer blocks are serial – they compute attention, then the MLP
Could we parallelize the transformer block?

[TABLE]
|  |
| --- |
| Serial vs Parallel layers Normal transformer blocks are serial – they compute attention, then the MLP Could we parallelize the transformer block? |


--- Page 28 ---

Parallel layers
A few models (GPTJ, PaLM, GPT-NeoX) do parallel layers. Originally in GPT-J
If implemented right, LayerNorm can be shared, and matrix multiplies can be fused
Recent Models: Cohere Command A, Falcon 2 11B, Command R+

[TABLE]
|  |
| --- |
| Parallel layers A few models (GPTJ, PaLM, GPT-NeoX) do parallel layers. Originally in GPT-J If implemented right, LayerNorm can be shared, and matrix multiplies can be fused Recent Models: Cohere Command A, Falcon 2 11B, Command R+ |


--- Page 29 ---

Summary: architectures
Pre-vs-post norm:
• Everyone does non-residual norm (except
OPT350M), likely with good reason.
Layer vs RMSnorm:
• RMSnorm has clear compute wins,
sometimes even performance
Gating:
• GLUs are consensus now
Serial vs parallel layers:
• Most models now use serial layers

[TABLE]
|  |
| --- |
| Summary: architectures Pre-vs-post norm: • Everyone does non-residual norm (except OPT350M), likely with good reason. Layer vs RMSnorm: • RMSnorm has clear compute wins, sometimes even performance Gating: • GLUs are consensus now Serial vs parallel layers: • Most models now use serial layers |


--- Page 30 ---

Many variations in position embeddings
Sine embeddings: add sines and cosines that enable localization Notable models:
𝐸𝑚𝑏𝑒𝑑 𝑥,𝑖 = 𝑣 +𝑃𝐸
𝑥 𝑝𝑜𝑠 Original transformer
Absolute embeddings: add a position vector to the embedding Notable models:
𝐸𝑚𝑏𝑒𝑑 𝑥,𝑖 = 𝑣 +𝑢 GPT1/2/3, OPT
𝑥 𝑖
Relative embeddings: add a vector to the attention computation Notable models:
T5, Gopher, Chinchilla
Notable models:
GPTJ, PaLM, LLaMA
Rope embeddings (next slides..)
Most 2024+ models

[TABLE]
|  |
| --- |
| Many variations in position embeddings Sine embeddings: add sines and cosines that enable localization Notable models: 𝐸𝑚𝑏𝑒𝑑 𝑥,𝑖 = 𝑣 +𝑃𝐸 𝑥 𝑝𝑜𝑠 Original transformer Absolute embeddings: add a position vector to the embedding Notable models: 𝐸𝑚𝑏𝑒𝑑 𝑥,𝑖 = 𝑣 +𝑢 GPT1/2/3, OPT 𝑥 𝑖 Relative embeddings: add a vector to the attention computation Notable models: T5, Gopher, Chinchilla Notable models: GPTJ, PaLM, LLaMA Rope embeddings (next slides..) Most 2024+ models |


--- Page 31 ---

RoPE: rotary position embeddings
High level thought process: a relative position embedding should be some 𝑓(𝑥,𝑖) s.t.
𝑓 𝑥,𝑖 ,𝑓 𝑦,𝑗 = 𝑔(𝑥,𝑦,𝑖 − 𝑗)
That is, the attention function only gets to depend on the relative position (i-j). How do
existing embeddings not fulfill this goal?
• Sine: Has various cross-terms that are not relative
𝐸𝑚𝑏𝑒𝑑 𝑥,𝑖 ,𝐸𝑚𝑏𝑒𝑑 𝑦,𝑖 = 𝑣 ,𝑣 + 𝑃𝐸 ,𝑣 …
𝑥 𝑦 𝑖 𝑦
• Absolute: obviously not relative
• Relative embeddings: is not an inner product

[TABLE]
|  |
| --- |
| RoPE: rotary position embeddings High level thought process: a relative position embedding should be some 𝑓(𝑥,𝑖) s.t. 𝑓 𝑥,𝑖 ,𝑓 𝑦,𝑗 = 𝑔(𝑥,𝑦,𝑖 − 𝑗) That is, the attention function only gets to depend on the relative position (i-j). How do existing embeddings not fulfill this goal? • Sine: Has various cross-terms that are not relative 𝐸𝑚𝑏𝑒𝑑 𝑥,𝑖 ,𝐸𝑚𝑏𝑒𝑑 𝑦,𝑖 = 𝑣 ,𝑣 + 𝑃𝐸 ,𝑣 … 𝑥 𝑦 𝑖 𝑦 • Absolute: obviously not relative • Relative embeddings: is not an inner product |


--- Page 32 ---

RoPE: rotary position embeddings
How can we solve this problem?
• We want our embeddings to be invariant to absolute position
• We know that inner products are invariant to arbitrary rotation.
we we
know
know we
know
Embedding Embedding
Position independent
“we know that” “of course we know”
embedding
Rotate we by ‘0 positions’ Rotate we by ‘2 positions’
know by ‘1 positions’ Rotate know by ‘3 positions’

[TABLE]
|  |
| --- |
| RoPE: rotary position embeddings How can we solve this problem? • We want our embeddings to be invariant to absolute position • We know that inner products are invariant to arbitrary rotation. we we know know we know Embedding Embedding Position independent “we know that” “of course we know” embedding Rotate we by ‘0 positions’ Rotate we by ‘2 positions’ know by ‘1 positions’ Rotate know by ‘3 positions’ |


--- Page 33 ---

RoPE: rotary position embeddings
There are many rotations, which one do you pick?
Gemma 4 alternative: just first 2
[Su et al 2021]
Just pair up the coordinates and rotate them in 2d (motivation: complex numbers)

[TABLE]
|  |
| --- |
| RoPE: rotary position embeddings There are many rotations, which one do you pick? Gemma 4 alternative: just first 2 [Su et al 2021] Just pair up the coordinates and rotate them in 2d (motivation: complex numbers) |


--- Page 34 ---

The actual RoPE math
Multiply with sines and cosines
Difference with sine embeddings – not additive, no cross terms

[TABLE]
|  |
| --- |
| The actual RoPE math Multiply with sines and cosines Difference with sine embeddings – not additive, no cross terms |


--- Page 35 ---

Implementation and code for RoPE
Usual
attention stuff
Get the RoPE
matrix cos/sin
Multiply
query/key inputs …
Same stuff as the usual multi-head self attention below
Note: embedding at each attention operation to enforce position invariance

[TABLE]
|  |
| --- |
| Implementation and code for RoPE Usual attention stuff Get the RoPE matrix cos/sin Multiply query/key inputs … Same stuff as the usual multi-head self attention below Note: embedding at each attention operation to enforce position invariance |


--- Page 36 ---

Hyperparameters
Transformer hyperparameter questions you might have had in 224n..
• How much bigger should the feedforward size be compared to hidden size?
• How many heads, and should num_heads always divide hidden size?
• What should my vocab size be?
And other model setting questions
• Do people even regularize these huge LMs?
• How do people scale these models - very deep or very wide?

[TABLE]
|  |
| --- |
| Hyperparameters Transformer hyperparameter questions you might have had in 224n.. • How much bigger should the feedforward size be compared to hidden size? • How many heads, and should num_heads always divide hidden size? • What should my vocab size be? And other model setting questions • Do people even regularize these huge LMs? • How do people scale these models - very deep or very wide? |


--- Page 37 ---

Surprising (?) consensus hyperparameter 1
Feedforward – model dimension ratio.
There are two dimensions that are relevant – the feedforward dim (𝑑 ) and model dim
𝑓𝑓
(𝑑 ). What should their relationship be?
𝑚𝑜𝑑𝑒𝑙
𝒅 = 𝟒 𝒅
𝒇𝒇 𝒎𝒐𝒅𝒆𝒍
This is almost always true. There’s just a few exceptions.

[TABLE]
|  |
| --- |
| Surprising (?) consensus hyperparameter 1 Feedforward – model dimension ratio. There are two dimensions that are relevant – the feedforward dim (𝑑 ) and model dim 𝑓𝑓 (𝑑 ). What should their relationship be? 𝑚𝑜𝑑𝑒𝑙 𝒅 = 𝟒 𝒅 𝒇𝒇 𝒎𝒐𝒅𝒆𝒍 This is almost always true. There’s just a few exceptions. |


--- Page 38 ---

Exception #1 – GLU variants
Remember that GLU variants scale down by 2/3rd. This means most GLU variants have
8
𝑑 = 𝑑 . This is mostly what happens. Some notable such examples.
𝑓𝑓 𝑚𝑜𝑑𝑒𝑙
3
Model 𝒅 /𝒅
𝒇𝒇 𝒎𝒐𝒅𝒆𝒍
PaLM 4
Mistral 7B 3.5
LLaMA-2 70B 3.5
LLaMA 70B 2.68
Qwen 14B 2.67
DeepSeek 67B 2.68
Yi 34B 2.85
T5 v1.1 2.5
Models are roughly in this range, though PaLM, LLaMA2 and Mistral are slightly larger

[TABLE]
|  |
| --- |
| Exception #1 – GLU variants Remember that GLU variants scale down by 2/3rd. This means most GLU variants have 8 𝑑 = 𝑑 . This is mostly what happens. Some notable such examples. 𝑓𝑓 𝑚𝑜𝑑𝑒𝑙 3 Model 𝒅 /𝒅 𝒇𝒇 𝒎𝒐𝒅𝒆𝒍 PaLM 4 Mistral 7B 3.5 LLaMA-2 70B 3.5 LLaMA 70B 2.68 Qwen 14B 2.67 DeepSeek 67B 2.68 Yi 34B 2.85 T5 v1.1 2.5 Models are roughly in this range, though PaLM, LLaMA2 and Mistral are slightly larger |

|  |  |
| --- | --- |
| Model | 𝒅 /𝒅 𝒇𝒇 𝒎𝒐𝒅𝒆𝒍 |
| PaLM | 4 |
| Mistral 7B | 3.5 |
| LLaMA-2 70B | 3.5 |
| LLaMA 70B | 2.68 |
| Qwen 14B | 2.67 |
| DeepSeek 67B | 2.68 |
| Yi 34B | 2.85 |
| T5 v1.1 | 2.5 |


--- Page 39 ---

Exception #2 – T5
As we have (and will) see, most LMs are have boring, conservative hyperparameters.
One exception is T5 [Raffel et al 2020] which has some very bold settings.
In particular, for the 11B model, they set
𝑑 = 65,536
𝑓𝑓
𝑑 = 1024
𝑚𝑜𝑑𝑒𝑙
For an astounding 64-times multiplier.
Other, recent exceptions – Gemma 2 (8x), SmolLM/Gemma 3/Gemma 4 (4x, GLU)

[TABLE]
|  |
| --- |
| Exception #2 – T5 As we have (and will) see, most LMs are have boring, conservative hyperparameters. One exception is T5 [Raffel et al 2020] which has some very bold settings. In particular, for the 11B model, they set 𝑑 = 65,536 𝑓𝑓 𝑑 = 1024 𝑚𝑜𝑑𝑒𝑙 For an astounding 64-times multiplier. Other, recent exceptions – Gemma 2 (8x), SmolLM/Gemma 3/Gemma 4 (4x, GLU) |


--- Page 40 ---

Why this range of multipliers?
Empirically, there’s a basin between 1-10 where this hyperparameter is near-optimal
[Kaplan+ 2020]

[TABLE]
|  |
| --- |
| Why this range of multipliers? Empirically, there’s a basin between 1-10 where this hyperparameter is near-optimal [Kaplan+ 2020] |


--- Page 41 ---

What can we learn from the model-dim hyperparam?
• The ‘default’ choices of 𝑑 = 4𝑑 and 𝑑 = 2.66𝑑 have worked well for nearly
𝑓𝑓 𝑚𝑜𝑑𝑒𝑙 𝑓𝑓 𝑚𝑜𝑑𝑒𝑙
all modern LLMs.
• But T5 does show that even radical choices of 𝑑 = 64𝑑 can work. This
𝑓𝑓 𝑚𝑜𝑑𝑒𝑙
hyperparameter choice isn’t written in stone.
• That said, T5 has a follow-up model (T5 v1.1) that is ‘improved’ and uses a much more
standard 2.5 multiplier on GeGLU, so the 64-times multiplier is likely suboptimal.

[TABLE]
|  |
| --- |
| What can we learn from the model-dim hyperparam? • The ‘default’ choices of 𝑑 = 4𝑑 and 𝑑 = 2.66𝑑 have worked well for nearly 𝑓𝑓 𝑚𝑜𝑑𝑒𝑙 𝑓𝑓 𝑚𝑜𝑑𝑒𝑙 all modern LLMs. • But T5 does show that even radical choices of 𝑑 = 64𝑑 can work. This 𝑓𝑓 𝑚𝑜𝑑𝑒𝑙 hyperparameter choice isn’t written in stone. • That said, T5 has a follow-up model (T5 v1.1) that is ‘improved’ and uses a much more standard 2.5 multiplier on GeGLU, so the 64-times multiplier is likely suboptimal. |


--- Page 42 ---

Surprising (?) consensus hyperparameter 2
Head-dim*num-heads to model-dim ratio. As a reminder, slide from 224n.
This doesn’t have to be true: we can have head-dimensions > model-dim / num-heads.
But most models do follow this guideline

[TABLE]
|  |
| --- |
| Surprising (?) consensus hyperparameter 2 Head-dim*num-heads to model-dim ratio. As a reminder, slide from 224n. This doesn’t have to be true: we can have head-dimensions > model-dim / num-heads. But most models do follow this guideline |


--- Page 43 ---

How many heads, whats the model dim?
Some examples of this hyperparameter
Num heads Head dim Model dim Ratio
GPT3 96 128 12288 1
T5 128 128 1024 16
T5 v1.1 64 64 4096 1
LaMDA 128 128 8192 2
PaLM 48 258 18432 1.48
LLaMA2 64 128 8192 1
Qwen 3.5 (27B) 24 256 5120 1.2
Most models have ratios around 1 – notable exceptions by some google models.

[TABLE]
|  |
| --- |
| How many heads, whats the model dim? Some examples of this hyperparameter Num heads Head dim Model dim Ratio GPT3 96 128 12288 1 T5 128 128 1024 16 T5 v1.1 64 64 4096 1 LaMDA 128 128 8192 2 PaLM 48 258 18432 1.48 LLaMA2 64 128 8192 1 Qwen 3.5 (27B) 24 256 5120 1.2 Most models have ratios around 1 – notable exceptions by some google models. |

|  | Num heads | Head dim | Model dim | Ratio |
| --- | --- | --- | --- | --- |
| GPT3 | 96 | 128 | 12288 | 1 |
| T5 | 128 | 128 | 1024 | 16 |
| T5 v1.1 | 64 | 64 | 4096 | 1 |
| LaMDA | 128 | 128 | 8192 | 2 |
| PaLM | 48 | 258 | 18432 | 1.48 |
| LLaMA2 | 64 | 128 | 8192 | 1 |
| Qwen 3.5 (27B) | 24 | 256 | 5120 | 1.2 |


--- Page 44 ---

Aspect ratios
Should my model be deep or wide? How deep and how wide?
Most models are surprisingly consistent on this one too!
Model 𝒅 /𝒏
𝒎𝒐𝒅𝒆𝒍 𝒍𝒂𝒚𝒆𝒓
BLOOM 205
T5 v1.1 171
PaLM (540B) 156
GPT3/OPT/Mistral/Qwen 128
Sweet spot? /OLMo 3
LLaMA / LLaMA2 102
Gemma 3 87
Gemma 4 61
T5 (11B) 33

[TABLE]
|  |
| --- |
| Aspect ratios Should my model be deep or wide? How deep and how wide? Most models are surprisingly consistent on this one too! Model 𝒅 /𝒏 𝒎𝒐𝒅𝒆𝒍 𝒍𝒂𝒚𝒆𝒓 BLOOM 205 T5 v1.1 171 PaLM (540B) 156 GPT3/OPT/Mistral/Qwen 128 Sweet spot? /OLMo 3 LLaMA / LLaMA2 102 Gemma 3 87 Gemma 4 61 T5 (11B) 33 |

| Model | 𝒅 /𝒏 𝒎𝒐𝒅𝒆𝒍 𝒍𝒂𝒚𝒆𝒓 |
| --- | --- |
| BLOOM | 205 |
| T5 v1.1 | 171 |
| PaLM (540B) | 156 |
| GPT3/OPT/Mistral/Qwen /OLMo 3 | 128 |
| LLaMA / LLaMA2 | 102 |
| Gemma 3 | 87 |
| Gemma 4 | 61 |
| T5 (11B) | 33 |


--- Page 45 ---

Considerations about aspect ratio
Extremely deep models are harder to parallelize and have higher latency
[Tay et al 2021]

[TABLE]
|  |
| --- |
| Considerations about aspect ratio Extremely deep models are harder to parallelize and have higher latency [Tay et al 2021] |


--- Page 46 ---

Evidence on aspect ratio scaling
[Tay et al 2021]
[Kaplan et al 2020]

[TABLE]
|  |
| --- |
| Evidence on aspect ratio scaling [Tay et al 2021] [Kaplan et al 2020] |


--- Page 47 ---

What are typical vocabulary sizes?
Monolingual models – 30-50k vocab Multilingual / production systems 100-250k
Model Token count Model Token count
Original 37000 mT5 250000
transformer
PaLM 256000
GPT 40257
GPT4 100276
GPT2/3 50257
Gemma 4 262144
T5/T5v1.1 32128
DeepSeek 100000
LLaMA 32000
Qwen 15B 152064
Yi 64000
Monolingual vocabs don’t need to be huge, but multilingual ones do

[TABLE]
|  |
| --- |
| What are typical vocabulary sizes? Monolingual models – 30-50k vocab Multilingual / production systems 100-250k Model Token count Model Token count Original 37000 mT5 250000 transformer PaLM 256000 GPT 40257 GPT4 100276 GPT2/3 50257 Gemma 4 262144 T5/T5v1.1 32128 DeepSeek 100000 LLaMA 32000 Qwen 15B 152064 Yi 64000 Monolingual vocabs don’t need to be huge, but multilingual ones do |

| Model | Token count |
| --- | --- |
| Original transformer | 37000 |
| GPT | 40257 |
| GPT2/3 | 50257 |
| T5/T5v1.1 | 32128 |
| LLaMA | 32000 |

| Model | Token count |
| --- | --- |
| mT5 | 250000 |
| PaLM | 256000 |
| GPT4 | 100276 |
| Gemma 4 | 262144 |
| DeepSeek | 100000 |
| Qwen 15B | 152064 |
| Yi | 64000 |


--- Page 48 ---

Dropout and other regularization
Do we need regularization during pretraining?
Arguments against:
• There is a lot of data (trillions of tokens), more than parameters.
• SGD only does a single pass on a corpus (hard to memorize)
This is all quite reasonable.. but what do people do in practice?

[TABLE]
|  |
| --- |
| Dropout and other regularization Do we need regularization during pretraining? Arguments against: • There is a lot of data (trillions of tokens), more than parameters. • SGD only does a single pass on a corpus (hard to memorize) This is all quite reasonable.. but what do people do in practice? |


--- Page 49 ---

Dropout and weight decay in practice
Model Dropout* Weight decay
Original transformer 0.1 0
GPT2 0.1 0.1
T5 0.1 0
Many older models used
GPT3 0.1 0.1 dropout during pretraining
T5 v1.1 0 0
Newer models (except Qwen) rely
only on weight decay
PaLM 0 (variable)
OPT 0.1 0.1
LLaMA 0 0.1
Qwen 14B 0.1 0.1
* Most of the times papers just don’t discuss dropout. On open models, this closely matches not doing dropout.
This may not be true of closed models.

[TABLE]
|  |
| --- |
| Dropout and weight decay in practice Model Dropout* Weight decay Original transformer 0.1 0 GPT2 0.1 0.1 T5 0.1 0 Many older models used GPT3 0.1 0.1 dropout during pretraining T5 v1.1 0 0 Newer models (except Qwen) rely only on weight decay PaLM 0 (variable) OPT 0.1 0.1 LLaMA 0 0.1 Qwen 14B 0.1 0.1 * Most of the times papers just don’t discuss dropout. On open models, this closely matches not doing dropout. This may not be true of closed models. |

| Model | Dropout* | Weight decay |
| --- | --- | --- |
| Original transformer | 0.1 | 0 |
| GPT2 | 0.1 | 0.1 |
| T5 | 0.1 | 0 |
| GPT3 | 0.1 | 0.1 |
| T5 v1.1 | 0 | 0 |
| PaLM | 0 | (variable) |
| OPT | 0.1 | 0.1 |
| LLaMA | 0 | 0.1 |
| Qwen 14B | 0.1 | 0.1 |


--- Page 50 ---

Why weight decay LLMs?
[Andriushchenko et al 2023] has interesting observations about LLM weight decay
It’s not to control overfitting Weight decay interacts with learning rates (cosine schedule)

[TABLE]
|  |
| --- |
| Why weight decay LLMs? [Andriushchenko et al 2023] has interesting observations about LLM weight decay It’s not to control overfitting Weight decay interacts with learning rates (cosine schedule) |


--- Page 51 ---

Summary: hyperparameters
Feedforward
• Factor-of-4 rule of thumb (8/3 for GLUs) is
standard (with some evidence)
Head dim
• Head dim*Num head = D model is standard
– but low to no validation
Aspect ratio
• Wide range of ‘good’ values (100-200).
Systems concerns dictate the value
Regularization
• You still ‘regularize’ LMs but its effects are
primarily on optimization dynamics

[TABLE]
|  |
| --- |
| Summary: hyperparameters Feedforward • Factor-of-4 rule of thumb (8/3 for GLUs) is standard (with some evidence) Head dim • Head dim*Num head = D model is standard – but low to no validation Aspect ratio • Wide range of ‘good’ values (100-200). Systems concerns dictate the value Regularization • You still ‘regularize’ LMs but its effects are primarily on optimization dynamics |


--- Page 52 ---

Stability tricks
Recently, lots of attention on stable training
Don’t train models that look like the blue curve!

[TABLE]
|  |
| --- |
| Stability tricks Recently, lots of attention on stable training Don’t train models that look like the blue curve! |


--- Page 53 ---

Where do the issues arise? Beware of softmaxes!
Softmaxes – can be ill-behaved due to exponentials / divison by zero

[TABLE]
|  |
| --- |
| Where do the issues arise? Beware of softmaxes! Softmaxes – can be ill-behaved due to exponentials / divison by zero |


--- Page 54 ---

Output softmax stability – the ‘z-loss’
Recall the softmax calculation
[From Devlin 2014]
This is useful for stability! PaLM used this ‘z loss’ trick.
Other examples: Baichuan 2 (2023), DCLM (2024), OLMo 2 (2025), OLMo 3 (2025)

[TABLE]
|  |
| --- |
| Output softmax stability – the ‘z-loss’ Recall the softmax calculation [From Devlin 2014] This is useful for stability! PaLM used this ‘z loss’ trick. Other examples: Baichuan 2 (2023), DCLM (2024), OLMo 2 (2025), OLMo 3 (2025) |


--- Page 55 ---

Attention softmax stability – the ‘QK norm’
Norms
The query and keys are Layer (RMS) normed before going into the softmax operation.
Other examples: DCLM, OLMo2, Gemma 2, Qwen3, OLMo 3, Gemma 4
Originally from vision and multimodal models [Dehgani 2023, Idefcs, Chameleon]

[TABLE]
|  |
| --- |
| Attention softmax stability – the ‘QK norm’ Norms The query and keys are Layer (RMS) normed before going into the softmax operation. Other examples: DCLM, OLMo2, Gemma 2, Qwen3, OLMo 3, Gemma 4 Originally from vision and multimodal models [Dehgani 2023, Idefcs, Chameleon] |

|  |
| --- |
| Norms e Layer (RMS) normed before going into the |

|  |
| --- |
|  |


--- Page 56 ---

Logit soft-capping.
Soft-capping the logits to some maximum value via Tanh
Prevents logits from blowing up, but also might have perf issues?

[TABLE]
|  |
| --- |
| Logit soft-capping. Soft-capping the logits to some maximum value via Tanh Prevents logits from blowing up, but also might have perf issues? |


--- Page 57 ---

Attention heads
Most models don’t touch the attention heads much at all with a few minor exceptions..
GQA / MQA : Saving inference costs by reducing the number of heads
Sparse or sliding window attention (GPT4/Mistral): restricting the attention pattern
to reduce compute cost
Exotic SSM stuff (Jamba, Falcon 3, Qwen 3.5, etc): next lecture!

[TABLE]
|  |
| --- |
| Attention heads Most models don’t touch the attention heads much at all with a few minor exceptions.. GQA / MQA : Saving inference costs by reducing the number of heads Sparse or sliding window attention (GPT4/Mistral): restricting the attention pattern to reduce compute cost Exotic SSM stuff (Jamba, Falcon 3, Qwen 3.5, etc): next lecture! |


--- Page 58 ---

GQA/MQA – Reducing attention head cost
Let’s think about the compute involved for attention
d = hidden dim
b = batch
n = length (<d)
h = heads
k = head dim (d/h)
X softmax projection
Total arithmetric operations (𝑏𝑛𝑑2), total memory accesses (𝑏𝑛𝑑 + 𝑏ℎ𝑛2 + 𝑑2)
−1
1 1
Arithmetic intensity (compute/memory) is high 𝑂 + - we can keep our GPUs running
𝑘 𝑏𝑛

[TABLE]
|  |
| --- |
| GQA/MQA – Reducing attention head cost Let’s think about the compute involved for attention d = hidden dim b = batch n = length (<d) h = heads k = head dim (d/h) X softmax projection Total arithmetric operations (𝑏𝑛𝑑2), total memory accesses (𝑏𝑛𝑑 + 𝑏ℎ𝑛2 + 𝑑2) −1 1 1 Arithmetic intensity (compute/memory) is high 𝑂 + - we can keep our GPUs running 𝑘 𝑏𝑛 |


--- Page 59 ---

GQA/MQA – Reducing attention head cost
What about the incremental case when we generate text?
Key difference: can’t parallelize the generation process – needs to be step by step
In this case – we need to incrementally re-compute/update attention via the ‘KV cache’
[Animation from https://medium.com/@joaolages/kv-caching-explained-276520203249]

[TABLE]
|  |
| --- |
| GQA/MQA – Reducing attention head cost What about the incremental case when we generate text? Key difference: can’t parallelize the generation process – needs to be step by step In this case – we need to incrementally re-compute/update attention via the ‘KV cache’ [Animation from https://medium.com/@joaolages/kv-caching-explained-276520203249] |


--- Page 60 ---

GQA/MQA – Reducing attention head cost
What’s the incremental arithmetic intensity?
K, V projection
Total arithmetric operations (𝑏𝑛𝑑2), total memory accesses (𝑏𝑛2𝑑 + 𝑛𝑑2)
−1
𝑛 1
Arithmetic intensity is not good 𝑂 + - need large batches + short seq length
𝑑 𝑏
(n) or big model dimensions (d)
Is there some way around this? The n/d term is difficult to reduce.

[TABLE]
|  |
| --- |
| GQA/MQA – Reducing attention head cost What’s the incremental arithmetic intensity? K, V projection Total arithmetric operations (𝑏𝑛𝑑2), total memory accesses (𝑏𝑛2𝑑 + 𝑛𝑑2) −1 𝑛 1 Arithmetic intensity is not good 𝑂 + - need large batches + short seq length 𝑑 𝑏 (n) or big model dimensions (d) Is there some way around this? The n/d term is difficult to reduce. |


--- Page 61 ---

MQA – just have fewer key dimensions.
Key idea – have multiple queries, but just one dimension for keys and values
We have much fewer items to move in and out of memory (KV Cache)
−1
1 𝑛 1
Total memory access (𝑏𝑛𝑑 + 𝑏𝑛2𝑘 + 𝑛𝑑2), Arithmetic intensity 𝑂 + +
𝑑 𝑑ℎ 𝑏
[figure from https://blog.fireworks.ai/multi-query-attention-is-all-you-need-db072e758055]

[TABLE]
|  |
| --- |
| MQA – just have fewer key dimensions. Key idea – have multiple queries, but just one dimension for keys and values We have much fewer items to move in and out of memory (KV Cache) −1 1 𝑛 1 Total memory access (𝑏𝑛𝑑 + 𝑏𝑛2𝑘 + 𝑛𝑑2), Arithmetic intensity 𝑂 + + 𝑑 𝑑ℎ 𝑏 [figure from https://blog.fireworks.ai/multi-query-attention-is-all-you-need-db072e758055] |


--- Page 62 ---

Additional extensions – GQA
Don’t go all the way to one dimension of KV – have fewer dims
Simple knob to control expressiveness (key-query ratio) and inference efficiency
More recently – MLA (multihead latent attention) from deepseek v2

[TABLE]
|  |
| --- |
| Additional extensions – GQA Don’t go all the way to one dimension of KV – have fewer dims Simple knob to control expressiveness (key-query ratio) and inference efficiency More recently – MLA (multihead latent attention) from deepseek v2 |


--- Page 63 ---

Does MQA hurt? Sometimes..
Small PPL hit w/ MQA [Shazeer 2019] Low/no hit w/ GQA [Ainslie 2023]

[TABLE]
|  |
| --- |
| Does MQA hurt? Sometimes.. Small PPL hit w/ MQA [Shazeer 2019] Low/no hit w/ GQA [Ainslie 2023] |


--- Page 64 ---

Sparse / sliding window attention
Attending to the entire context can be expensive (quadratic).
Build sparse / structured attention that trades off expressiveness vs runtime (GPT3, GPT-
OSS, Gemma4)
[Child et al 2019]

[TABLE]
|  |
| --- |
| Sparse / sliding window attention Attending to the entire context can be expensive (quadratic). Build sparse / structured attention that trades off expressiveness vs runtime (GPT3, GPT- OSS, Gemma4) [Child et al 2019] |


--- Page 65 ---

Current standard trick – interleave ‘full’ and ‘LR’ attention
From Cohere Command A – Every 4th layer is a full attention
Long-range info via NoPE, short-range info via RoPE + SWA.
Other models – LLaMA 4, Gemma 3, Gemma 4, OLMo 3 does SWA+Full RoPE.

[TABLE]
|  |
| --- |
| Current standard trick – interleave ‘full’ and ‘LR’ attention From Cohere Command A – Every 4th layer is a full attention Long-range info via NoPE, short-range info via RoPE + SWA. Other models – LLaMA 4, Gemma 3, Gemma 4, OLMo 3 does SWA+Full RoPE. |


--- Page 66 ---

Other recent examples of interleaved attention
Gemma 4 Olmo 3 Qwen 3.5 / Qwen 3 Next
https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4

[TABLE]
|  |
| --- |
| Other recent examples of interleaved attention Gemma 4 Olmo 3 Qwen 3.5 / Qwen 3 Next https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4 |


--- Page 67 ---

Recap, conclusion, etc.
Many aspects (arch, hparams) of transformers are in common across the big LMs
Major differences? Position embeddings, activations, tokenization

[TABLE]
|  |
| --- |
| Recap, conclusion, etc. Many aspects (arch, hparams) of transformers are in common across the big LMs Major differences? Position embeddings, activations, tokenization |
