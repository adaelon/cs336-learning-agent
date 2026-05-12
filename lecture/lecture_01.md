lecture_01.py


import os



import regex


from abc import ABC


from dataclasses import dataclass


from collections import defaultdict


from edtrace import link, text, image


from lecture_util import article_link, post_link, video_link, get_local_url


from references import shannon_1950, lstm_1997, brants_2007, bengio_2003, glorot_2010, seq2seq_2014


from references import bahdanau_2015_attention, transformer_2017, gpt2_2019, t5_2019, kaplan_scaling_laws_2020, mup_2022


from references import dpo_2023, adamw_2017, adam_2014, grpo, ppo_2017, muon_2024


from references import large_batch_training_2018, wsd_2024, cosine_learning_rate_2017, moe_2017, switch_transformers_2021, auxfree_2024, mtp_2024


from references import megatron_lm_2019, shazeer_2020, elmo_2018, bert_2018


from references import rms_norm_2019, layernorm_2016, pre_post_norm_2020, qk_norm_2023


from references import rope_2021, soap_2024, sparse_transformer_2019, gqa_2023, mla_2024


from references import linear_attention_2020, mamba_2_2024, gdn_2024, mamba_3_2026


from references import megabyte_2023, byt5_2021, blt_2024, tfree_2024, hnet_2025, sennrich_2016, zero_2019, gpipe_2018


from references import regmix_2025, olmix_2026, wrap_2024



from references import gpt_3_2020, gpt_4_2023, instruct_gpt_2022


from references import the_pile_2020, gpt_j_2021, opt_175b_2022, bloom_2022, palm_2022, chinchilla_2022


from references import llama_2023, llama_2_2023, llama_3_2024


from references import mistral_7b_2023, mixtral_2024


from references import deepseek_67b_2024, deepseek_v2_2024, deepseek_v3_2024, deepseek_r1_2025, deepseek_v3_2_2025


from references import qwen_2_5_2024, qwen_3_2025


from references import kimi_1_5_2025, kimi_k2_5_2026


from references import glm_4_5_2025, glm_5_2026


from references import minimax_m2_5_2026


from references import xiaomi_mimo_v2_2026



from references import marin_8b_2025, marin_32b_2025


from references import olmo_7b_2024, olmo_2_2025, olmo_3_2025


from references import nemotron_15b_2024, nemotron_3_2025



import tiktoken



def main():


welcome()


why_this_course_exists()


current_lm_landscape()



what_is_this_program()



course_logistics()


course_syllabus()



tokenization() # First unit



Next time: resource accounting




def welcome():


## CS336: Language Models From Scratch (Spring 2026)



![Image](./Trace - lecture_01_files/course-staff.png)


...bringing you the 3rd offering of CS336.



Lectures from 2nd offering (Spring 2025) are on [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_).


What's new?


- Same 'from scratch' philosophy


- Prioritize high value-per-time concepts, don't lose the forest for the trees


- More coverage of modern LM ingredients (mixture of experts, long-context, agents)




def why_this_course_exists():


## Why did we make this course?



Problem: researchers are becoming **disconnected** from the underlying technology.


- 2016: researchers implemented and trained their own models.


- 2018: researchers downloaded models (e.g., BERT) and fine-tuned them.


- Today: researchers prompt API models (e.g., GPT/Claude/Gemini).



Moving up levels of abstraction boosts productivity, but


- These abstractions are leaky (in contrast to programming languages or operating systems).


- There is still fundamental research to be done that requires tearing up the stack.



**Full understanding** of this technology is necessary for **fundamental research**.



Philosophy of this course: **understanding via building**.


But there's one small problem...



## The industrialization of language models


![Image](./Trace - lecture_01_files/image-dda46aa409183107fcd9201cd89dac21-https_upload_wikimedia_org_wikipedia_commons_c_cc_Industrialisation_jpg)



Frontier models are really expensive:


- 2023: GPT-4 supposedly cost $100M to train.

[[article]](https://www.wired.com/story/openai-ceo-sam-altman-the-age-of-giant-ai-models-is-already-over/)

article


- 2025: xAI builds cluster with 230K GPUs for training Grok.

[[article]](https://x.com/elonmusk/status/1947701807389515912)

article



There are no public details on how frontier models are built.


From the GPT-4 technical report

[[OpenAI+ 2023]](https://arxiv.org/pdf/2303.08774.pdf)

GPT-4 Technical Report

[OpenAI] OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad... (271 more)... Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, Barret Zoph

2023-03-15

We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer-based model pre-trained to predict the next token in a document. The post-training alignment process results in improved performance on measures of factuality and adherence to desired behavior. A core component of this project was developing infrastructure and optimization methods that behave predictably across a wide range of scales. This allowed us to accurately predict some aspects of GPT-4's performance based on models trained with no more than 1/1,000th the compute of GPT-4.

No details on the data or model architecture.:


![Image](./Trace - lecture_01_files/gpt4-no-details.png)



Frontier models are out of reach for us.


We could build small language models (<1B parameters), but this might not be representative of large language models.



Example 1: fraction of FLOPs spent in attention versus MLP changes with scale.

[[post]](https://x.com/stephenroller/status/1579993017234382849)

post


![Image](./Trace - lecture_01_files/roller-flops.png)


Example 2: emergence of behavior with scale

[[Wei+ 2022]](https://arxiv.org/pdf/2206.07682)

Emergent Abilities of Large Language Models

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph... (6 more)... Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, William Fedus

2022-06-15

Scaling up language models has been shown to predictably improve performance and sample efficiency on a wide range of downstream tasks. This paper instead discusses an unpredictable phenomenon that we refer to as emergent abilities of large language models. We consider an ability to be emergent if it is not present in smaller models but is present in larger models. Thus, emergent abilities cannot be predicted simply by extrapolating the performance of smaller models. The existence of such emergence implies that additional scaling could further expand the range of capabilities of language models.


![Image](./Trace - lecture_01_files/wei-emergence-plot.png)



## What can we learn in this class that transfers to frontier models?


There are three types of knowledge:


- **Mechanics**: how things work (what a Transformer is, how model parallelism works)


- **Mindset**: squeezing the most out of the hardware, taking scaling seriously


- **Intuitions**: which data and modeling decisions yield good accuracy



We can teach mechanics and mindset (these do transfer).


We can only partially teach intuitions (do not necessarily transfer across scales).



## Intuitions? 🤷


Some design decisions are simply not (yet) justifiable and just come from experimentation.


Example: Noam Shazeer paper that introduced SwiGLU

[[Shazeer 2020]](https://arxiv.org/pdf/2002.05202.pdf)

GLU Variants Improve Transformer

[Google] Noam Shazeer

2020-02-12

Gated Linear Units (arXiv:1612.08083) consist of the component-wise product of two linear projections, one of which is first passed through a sigmoid function. Variations on GLU are possible, using different nonlinear (or even linear) functions in place of sigmoid. We test these variants in the feed-forward sublayers of the Transformer (arXiv:1706.03762) sequence-to-sequence model, and find that some of them yield quality improvements over the typically-used ReLU or GELU activations.

Experiments with different activation functions

Activation functions: ReLU, GeLU, Swish

Apply idea of gated units (GLU): ReGLU, GeGLU, SwiGLU

FFN-SwiGLU = Swish(x W1) * xV W2

Have 3 matrices now, so make hidden dimension 2/3 of the 2 matrix version


![Image](./Trace - lecture_01_files/divine-benevolence.png)



## The bitter lesson


Wrong interpretation: scale is all that matters, algorithms don't matter.


Right interpretation: algorithms that scale are what matter.


### accuracy = efficiency x resources


In fact, efficiency is way more important at larger scales (can't afford to be wasteful).


[[Hernandez+ 2020]](https://arxiv.org/abs/2005.04305)

Measuring the Algorithmic Efficiency of Neural Networks

Danny Hernandez, Tom B. Brown

2020-05-08

Three factors drive the advance of AI: algorithmic innovation, data, and the amount of compute available for training. Algorithmic progress has traditionally been more difficult to quantify than compute and data. In this work, we argue that algorithmic progress has an aspect that is both straightforward to measure and interesting: reductions over time in the compute needed to reach past capabilities. We show that the number of floating-point operations required to train a classifier to AlexNet-level performance on ImageNet has decreased by a factor of 44x between 2012 and 2019. This corresponds to algorithmic efficiency doubling every 16 months over a period of 7 years. By contrast, Moore's Law would only have yielded an 11x cost improvement. We observe that hardware and algorithmic efficiency gains multiply and can be on a similar scale over meaningful horizons, which suggests that a good model of AI progress should integrate measures from both.

showed 44x algorithmic efficiency on ImageNet between 2012 and 2019.



Framing: what is the best model one can build given a certain compute and data budget?


In other words, **maximize efficiency**!




def current_lm_landscape():


## Pre-neural (before 2010s)


- Language model to measure the entropy of English

[[Shannon 1950]](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf)

Prediction and Entropy of Printed English

Claude Shannon

1950-09-15


- N-gram language models (used in machine translation and speech recognition systems)

[[Brants+ 2007]](https://aclanthology.org/D07-1090.pdf)

Language Models in Machine Translation

[Google] Thorsten Brants, Ashok C. Popat, Peng Xu, Franz J. Och, Jeffrey Dean


Trained 5-gram model on 2T tokens



## Neural ingredients (2010s)


- Long-Short Term Memory (LSTM)

[[Hochreiter+ 1997]](https://www.bioinf.jku.at/publications/older/2604.pdf)

Long Short-Term Memory

Sepp Hochreiter, Jürgen Schmidhuber



- First neural language model

[[Bengio+ 2003]](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

A Neural Probabilistic Language Model

Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin

2003-02-01

Used a feedforward neural network over last n words to predict the next word in a sequence


- Sequence-to-sequence modeling (for machine translation)

[[Sutskever+ 2014]](https://arxiv.org/pdf/1409.3215.pdf)

Sequence to Sequence Learning with Neural Networks

[Google] Ilya Sutskever, Oriol Vinyals, Quoc V. Le

2014-09-10

Deep Neural Networks (DNNs) are powerful models that have achieved excellent performance on difficult learning tasks. Although DNNs work well whenever large labeled training sets are available, they cannot be used to map sequences to sequences. In this paper, we present a general end-to-end approach to sequence learning that makes minimal assumptions on the sequence structure. Our method uses a multilayered Long Short-Term Memory (LSTM) to map the input sequence to a vector of a fixed dimensionality, and then another deep LSTM to decode the target sequence from the vector. Our main result is that on an English to French translation task from the WMT'14 dataset, the translations produced by the LSTM achieve a BLEU score of 34.8 on the entire test set, where the LSTM's BLEU score was penalized on out-of-vocabulary words. Additionally, the LSTM did not have difficulty on long sentences. For comparison, a phrase-based SMT system achieves a BLEU score of 33.3 on the same dataset. When we used the LSTM to rerank the 1000 hypotheses produced by the aforementioned SMT system, its BLEU score increases to 36.5, which is close to the previous best result on this task. The LSTM also learned sensible phrase and sentence representations that are sensitive to word order and are relatively invariant to the active and the passive voice. Finally, we found that reversing the order of the words in all source sentences (but not target sentences) improved the LSTM's performance markedly, because doing so introduced many short term dependencies between the source and the target sentence which made the optimization problem easier.

Introduced seq2seq (encode entire sentence into one vector, decode translation)


- Adam optimizer

[[Kingma+ 2014]](https://arxiv.org/pdf/1412.6980.pdf)

Adam: A Method for Stochastic Optimization

Diederik P. Kingma, Jimmy Ba

2014-12-22

We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients. The hyper-parameters have intuitive interpretations and typically require little tuning. Some connections to related algorithms, on which Adam was inspired, are discussed. We also analyze the theoretical convergence properties of the algorithm and provide a regret bound on the convergence rate that is comparable to the best known results under the online convex optimization framework. Empirical results demonstrate that Adam works well in practice and compares favorably to other stochastic optimization methods. Finally, we discuss AdaMax, a variant of Adam based on the infinity norm.

Introduced Adam optimizer based on RMSProp and momentum


- Attention mechanism (for machine translation)

[[Bahdanau+ 2014]](https://arxiv.org/pdf/1409.0473.pdf)

Neural Machine Translation by Jointly Learning to Align and Translate

Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio

2014-09-01

Neural machine translation is a recently proposed approach to machine translation. Unlike the traditional statistical machine translation, the neural machine translation aims at building a single neural network that can be jointly tuned to maximize the translation performance. The models proposed recently for neural machine translation often belong to a family of encoder-decoders and consists of an encoder that encodes a source sentence into a fixed-length vector from which a decoder generates a translation. In this paper, we conjecture that the use of a fixed-length vector is a bottleneck in improving the performance of this basic encoder-decoder architecture, and propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word, without having to form these parts as a hard segment explicitly. With this new approach, we achieve a translation performance comparable to the existing state-of-the-art phrase-based system on the task of English-to-French translation. Furthermore, qualitative analysis reveals that the (soft-)alignments found by the model agree well with our intuition.

Introduced attention mechanism (for machine translation)


- Transformer architecture (for machine translation)

[[Vaswani+ 2017]](https://arxiv.org/pdf/1706.03762.pdf)

Attention Is All You Need

[Google] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

2017-06-12

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

Introduced Transformer (for machine translation)


- Mixture of experts

[[Shazeer+ 2017]](https://arxiv.org/pdf/1701.06538.pdf)

Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

[Google] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean

2017-01-23

The capacity of a neural network to absorb information is limited by its number of parameters. Conditional computation, where parts of the network are active on a per-example basis, has been proposed in theory as a way of dramatically increasing model capacity without a proportional increase in computation. In practice, however, there are significant algorithmic and performance challenges. In this work, we address these challenges and finally realize the promise of conditional computation, achieving greater than 1000x improvements in model capacity with only minor losses in computational efficiency on modern GPU clusters. We introduce a Sparsely-Gated Mixture-of-Experts layer (MoE), consisting of up to thousands of feed-forward sub-networks. A trainable gating network determines a sparse combination of these experts to use for each example. We apply the MoE to the tasks of language modeling and machine translation, where model capacity is critical for absorbing the vast quantities of knowledge available in the training corpora. We present model architectures in which a MoE with up to 137 billion parameters is applied convolutionally between stacked LSTM layers. On large language modeling and machine translation benchmarks, these models achieve significantly better results than state-of-the-art at lower computational cost.


- Model parallelism

[[Huang+ 2018]](https://arxiv.org/pdf/1811.06965.pdf)

GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism

[Google] Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen... (1 more)... HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, Zhifeng Chen

2018-11-16

Scaling up deep neural network capacity has been known as an effective approach to improving model quality for several different machine learning tasks. In many cases, increasing model capacity beyond the memory limit of a single accelerator has required developing special algorithms or infrastructure. These solutions are often architecture-specific and do not transfer to other tasks. To address the need for efficient and task-independent model parallelism, we introduce GPipe, a pipeline parallelism library that allows scaling any network that can be expressed as a sequence of layers. By pipelining different sub-sequences of layers on separate accelerators, GPipe provides the flexibility of scaling a variety of different networks to gigantic sizes efficiently. Moreover, GPipe utilizes a novel batch-splitting pipelining algorithm, resulting in almost linear speedup when a model is partitioned across multiple accelerators. We demonstrate the advantages of GPipe by training large-scale neural networks on two different tasks with distinct network architectures: (i) Image Classification: We train a 557-million-parameter AmoebaNet model and attain a top-1 accuracy of 84.4% on ImageNet-2012, (ii) Multilingual Neural Machine Translation: We train a single 6-billion-parameter, 128-layer Transformer model on a corpus spanning over 100 languages and achieve better quality than all bilingual models.

[[Rajbhandari+ 2019]](https://arxiv.org/abs/1910.02054)

ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

[Microsoft] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He

2019-10-04

Large deep learning models offer significant accuracy gains, but training billions to trillions of parameters is challenging. Existing solutions such as data and model parallelisms exhibit fundamental limitations to fit these models into limited device memory, while obtaining computation, communication and development efficiency. We develop a novel solution, Zero Redundancy Optimizer (ZeRO), to optimize memory, vastly improving training speed while increasing the model size that can be efficiently trained. ZeRO eliminates memory redundancies in data- and model-parallel training while retaining low communication volume and high computational granularity, allowing us to scale the model size proportional to the number of devices with sustained high efficiency. Our analysis on memory requirements and communication volume demonstrates: ZeRO has the potential to scale beyond 1 Trillion parameters using today's hardware. We implement and evaluate ZeRO: it trains large models of over 100B parameter with super-linear speedup on 400 GPUs, achieving throughput of 15 Petaflops. This represents an 8x increase in model size and 10x increase in achievable performance over state-of-the-art. In terms of usability, ZeRO can train large models of up to 13B parameters (e.g., larger than Megatron GPT 8.3B and T5 11B) without requiring model parallelism which is harder for scientists to apply. Last but not the least, researchers have used the system breakthroughs of ZeRO to create the world's largest language model (Turing-NLG, 17B parameters) with record breaking accuracy.

Introduced ZeRO optimizer, can train 100B parameter model over 400 GPUs

[[Shoeybi+ 2019]](https://arxiv.org/pdf/1909.08053.pdf)

Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

[NVIDIA] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro

2019-09-17

Recent work in language modeling demonstrates that training large transformer models advances the state of the art in Natural Language Processing applications. However, very large models can be quite difficult to train due to memory constraints. In this work, we present our techniques for training very large transformer models and implement a simple, efficient intra-layer model parallel approach that enables training transformer models with billions of parameters. Our approach does not require a new compiler or library changes, is orthogonal and complimentary to pipeline model parallelism, and can be fully implemented with the insertion of a few communication operations in native PyTorch. We illustrate this approach by converging transformer based models up to 8.3 billion parameters using 512 GPUs. We sustain 15.1 PetaFLOPs across the entire application with 76% scaling efficiency when compared to a strong single GPU baseline that sustains 39 TeraFLOPs, which is 30% of peak FLOPs. To demonstrate that large language models can further advance the state of the art (SOTA), we train an 8.3 billion parameter transformer language model similar to GPT-2 and a 3.9 billion parameter model similar to BERT. We show that careful attention to the placement of layer normalization in BERT-like models is critical to achieving increased performance as the model size grows. Using the GPT-2 model we achieve SOTA results on the WikiText103 (10.8 compared to SOTA perplexity of 15.8) and LAMBADA (66.5% compared to SOTA accuracy of 63.2%) datasets. Our BERT model achieves SOTA results on the RACE dataset (90.9% compared to SOTA accuracy of 89.4%).



## Early foundation models (late 2010s)


- ELMo: pretraining with LSTMs, fine-tuning improves downstream tasks

[[Peters+ 2018]](https://arxiv.org/abs/1802.05365)

Deep contextualized word representations

Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, Luke Zettlemoyer

2018-02-15

We introduce a new type of deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy). Our word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pre-trained on a large text corpus. We show that these representations can be easily added to existing models and significantly improve the state of the art across six challenging NLP problems, including question answering, textual entailment and sentiment analysis. We also present an analysis showing that exposing the deep internals of the pre-trained network is crucial, allowing downstream models to mix different types of semi-supervision signals.


- BERT: pretraining with Transformer, fine-tuning improves downstream tasks

[[Devlin+ 2018]](https://arxiv.org/abs/1810.04805)

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova

2018-10-11

We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).


- Google's T5 (11B): cast everything as text-to-text

[[Raffel+ 2019]](https://arxiv.org/pdf/1910.10683.pdf)

Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

[Google] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu

2019-10-23

Transfer learning, where a model is first pre-trained on a data-rich task before being fine-tuned on a downstream task, has emerged as a powerful technique in natural language processing (NLP). The effectiveness of transfer learning has given rise to a diversity of approaches, methodology, and practice. In this paper, we explore the landscape of transfer learning techniques for NLP by introducing a unified framework that converts all text-based language problems into a text-to-text format. Our systematic study compares pre-training objectives, architectures, unlabeled data sets, transfer approaches, and other factors on dozens of language understanding tasks. By combining the insights from our exploration with scale and our new ``Colossal Clean Crawled Corpus'', we achieve state-of-the-art results on many benchmarks covering summarization, question answering, text classification, and more. To facilitate future work on transfer learning for NLP, we release our data set, pre-trained models, and code.

Encoder-decoder, frames tasks as text-to-text

Introduced Colossal Cleaned Common Crawl (C4)

Filtering (Section 2.2)

11B parameters

Remove bias from feedforward layers



## Embracing scaling


- OpenAI's GPT-2 (1.5B): fluent text, first signs of zero-shot

[[Radford+ 2019]](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Language Models are Unsupervised Multitask Learners

[OpenAI] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever

2019-02-14

1.5B parameters

Pioneered stage release


- Scaling laws: provide hope / predictability for scaling

[[Kaplan+ 2020]](https://arxiv.org/pdf/2001.08361.pdf)

Scaling Laws for Neural Language Models

[OpenAI] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei

2020-01-23

We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude. Other architectural details such as network width or depth have minimal effects within a wide range. Simple equations govern the dependence of overfitting on model/dataset size and the dependence of training speed on model size. These relationships allow us to determine the optimal allocation of a fixed compute budget. Larger models are significantly more sample-efficient, such that optimally compute-efficient training involves training very large models on a relatively modest amount of data and stopping significantly before convergence.

Vary model size, dataset size, compute; get power laws

Larger models require fewer tokens


- OpenAI's GPT-3 (175B): in-context learning

[[Brown+ 2020]](https://arxiv.org/pdf/2005.14165.pdf)

Language Models are Few-Shot Learners

[OpenAI] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan... (21 more)... Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei

2020-05-28

Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT-3 can generate samples of news articles which human evaluators have difficulty distinguishing from articles written by humans. We discuss broader societal impacts of this finding and of GPT-3 in general.

Introduces GPT-3

Same as GPT-2, but alternating sparse and dense attention layers

175B parameters

Data: 300B tokens


- Google's PaLM (540B): massive scale, undertrained

[[Chowdhery+ 2022]](https://arxiv.org/pdf/2204.02311.pdf)

PaLM: Scaling Language Modeling with Pathways

[Google] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra... (57 more)... Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, Noah Fiedel

2022-04-05

Large language models have been shown to achieve remarkable performance across a variety of natural language tasks using few-shot learning, which drastically reduces the number of task-specific training examples needed to adapt the model to a particular application. To further our understanding of the impact of scale on few-shot learning, we trained a 540-billion parameter, densely activated, Transformer language model, which we call Pathways Language Model PaLM. We trained PaLM on 6144 TPU v4 chips using Pathways, a new ML system which enables highly efficient training across multiple TPU Pods. We demonstrate continued benefits of scaling by achieving state-of-the-art few-shot learning results on hundreds of language understanding and generation benchmarks. On a number of these tasks, PaLM 540B achieves breakthrough performance, outperforming the finetuned state-of-the-art on a suite of multi-step reasoning tasks, and outperforming average human performance on the recently released BIG-bench benchmark. A significant number of BIG-bench tasks showed discontinuous improvements from model scale, meaning that performance steeply increased as we scaled to our largest model. PaLM also has strong capabilities in multilingual tasks and source code generation, which we demonstrate on a wide array of benchmarks. We additionally provide a comprehensive analysis on bias and toxicity, and study the extent of training data memorization with respect to model scale. Finally, we discuss the ethical considerations related to large language models and discuss potential mitigation strategies.

Data: Social media conversations, webpages, books, GitHub, Wikipedia, news

540B parameters

SwiGLU, parallelize attention and feedforward layers, multi-query attention, RoPE, remove biases

hardware: 6144 TPUv4, 46.2% MFU

optimizer: Adafactor without factorization

Introduced the term model FLOPs utilization (MFU) metric (observed tokens/sec / theoretical max tokens/sec)


- DeepMind's Chinchilla (70B): compute-optimal scaling laws

[[Hoffmann+ 2022]](https://arxiv.org/pdf/2203.15556.pdf)

Training Compute-Optimal Large Language Models

[DeepMind] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai... (12 more)... Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre

2022-03-29

We investigate the optimal model size and number of tokens for training a transformer language model under a given compute budget. We find that current large language models are significantly undertrained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant. By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. We test this hypothesis by training a predicted compute-optimal model, Chinchilla, that uses the same compute budget as Gopher but with 70B parameters and 4

×

 more more data. Chinchilla uniformly and significantly outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B) on a large range of downstream evaluation tasks. This also means that Chinchilla uses substantially less compute for fine-tuning and inference, greatly facilitating downstream usage. As a highlight, Chinchilla reaches a state-of-the-art average accuracy of 67.5% on the MMLU benchmark, greater than a 7% improvement over Gopher.

Introduced the rigorous analysis scaling laws for language models

Key improvement over Kaplan: tune learning rate for the compute budget

Approach 1: for each model size, train with 4 learning rates, vary number of training tokens, fit lower envelope

Approach 2 (IsoFLOP): for each model size, train with 9 training budgets, take last point

Approach 3: fit parametric function L(N, D) = E + A/N^alpha + B/D^beta to data collected from approaches 1 and 2

Conclusion: model and data should scale up at same rate

Table 3: extrapolate to 10 trillion parameters

MassiveText, different data distribution (1.5 trillion tokens)

70B parameters



## Open models


Early attempts (attempts to replicate GPT-3):


- EleutherAI's open datasets (The Pile) and models (GPT-J)

[[Gao+ 2020]](https://arxiv.org/pdf/2101.00027.pdf)

The Pile: An 800GB Dataset of Diverse Text for Language Modeling

[EleutherAI] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe... (2 more)... Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, Connor Leahy

2020-12-31

Recent work has demonstrated that increased training dataset diversity improves general cross-domain knowledge and downstream generalization capability for large-scale language models. With this in mind, we present \textit{the Pile}: an 825 GiB English text corpus targeted at training large-scale language models. The Pile is constructed from 22 diverse high-quality subsets -- both existing and newly constructed -- many of which derive from academic or professional sources. Our evaluation of the untuned performance of GPT-2 and GPT-3 on the Pile shows that these models struggle on many of its components, such as academic writing. Conversely, models trained on the Pile improve significantly over both Raw CC and CC-100 on all components of the Pile, while improving performance on downstream evaluations. Through an in-depth exploratory analysis, we document potentially concerning aspects of the data for prospective users. We make publicly available the code used in its construction.

825GB text, 22 diverse subsets (CommonCrawl, PubMed, ArXiv, GitHub, StackExchange, USPTO, OpenWebText2, Books3, etc.)

[[Wang+ 2021]](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/)

GPT-J

[EleutherAI] Ben Wang, Aran Komatsuzaki

2021-06-04

6.7B parameters

Attention and feedforward layers put in parallel

v3 256 TPUs (5.4 PFLOPs) for 5 weeks


- Meta's OPT (175B): GPT-3 replication, lots of hardware issues

[[Zhang+ 2022]](https://arxiv.org/pdf/2205.01068.pdf)

OPT: Open Pre-trained Transformer Language Models

[Meta] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen... (9 more)... Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, Luke Zettlemoyer

2022-05-02

Large language models, which are often trained for hundreds of thousands of compute days, have shown remarkable capabilities for zero- and few-shot learning. Given their computational cost, these models are difficult to replicate without significant capital. For the few that are available through APIs, no access is granted to the full model weights, making them difficult to study. We present Open Pre-trained Transformers (OPT), a suite of decoder-only pre-trained transformers ranging from 125M to 175B parameters, which we aim to fully and responsibly share with interested researchers. We show that OPT-175B is comparable to GPT-3, while requiring only 1/7th the carbon footprint to develop. We are also releasing our logbook detailing the infrastructure challenges we faced, along with code for experimenting with all of the released models.

Data: The Pile, PushShift.io Reddit, deduplication

175B parameters

hardware: 992 A100 80GB for 2 months, lots of hardware failures

FSDP with Megatron-LM, fp16 with loss scaling


- Hugging Face / BigScience's BLOOM (176B): focused on data sourcing

[[Workshop+ 2022]](https://arxiv.org/abs/2211.05100)

BLOOM: A 176B-Parameter Open-Access Multilingual Language Model

[BigScience] BigScience Workshop, Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick... (383 more)... Zhongli Xie, Zifan Ye, Mathilde Bras, Younes Belkada, Thomas Wolf

2022-11-09

Large language models (LLMs) have been shown to be able to perform new tasks based on a few demonstrations or natural language instructions. While these capabilities have led to widespread adoption, most LLMs are developed by resource-rich organizations and are frequently kept from the public. As a step towards democratizing this powerful technology, we present BLOOM, a 176B-parameter open-access language model designed and built thanks to a collaboration of hundreds of researchers. BLOOM is a decoder-only Transformer language model that was trained on the ROOTS corpus, a dataset comprising hundreds of sources in 46 natural and 13 programming languages (59 in total). We find that BLOOM achieves competitive performance on a wide variety of benchmarks, with stronger results after undergoing multitask prompted finetuning. To facilitate future research and applications using LLMs, we publicly release our models and code under the Responsible AI License.

Model: BLOOM (176B parameters)

Data: ROOTS

Hardware: 48x8 A100s on Jean Zay supercomputer for 3.5 months

ZeRO stage 1



Credible open-weight models (weights + paper):


- Meta's Llama models

[[Touvron+ 2023]](https://arxiv.org/pdf/2302.13971.pdf)

LLaMA: Open and Efficient Foundation Language Models

[Meta] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux... (4 more)... Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, Guillaume Lample

2023-02-27

We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens, and show that it is possible to train state-of-the-art models using publicly available datasets exclusively, without resorting to proprietary and inaccessible datasets. In particular, LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, and LLaMA-65B is competitive with the best models, Chinchilla-70B and PaLM-540B. We release all our models to the research community.

Train only on open data (detailed recipe that is replicated by RedPajama)

Optimize for fast inference at 7B

Data: CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, StackExchange

Architecture: Pre-norm, SwiGLU, RoPE

Training: 2048 A100 80GB for 21 days

[[Touvron+ 2023]](https://arxiv.org/pdf/2307.09288.pdf)

Llama 2: Open Foundation and Fine-Tuned Chat Models

[Meta] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi... (58 more)... Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, Thomas Scialom

2023-07-18

In this work, we develop and release Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases. Our models outperform open-source chat models on most benchmarks we tested, and based on our human evaluations for helpfulness and safety, may be a suitable substitute for closed-source models. We provide a detailed description of our approach to fine-tuning and safety improvements of Llama 2-Chat in order to enable the community to build on our work and contribute to the responsible development of LLMs.

2T tokens

70B parameters

[[Grattafiori+ 2024]](https://arxiv.org/abs/2407.21783)

The Llama 3 Herd of Models

[Meta] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian... (550 more)... Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, Zhiyu Ma

2024-07-31

Modern artificial intelligence (AI) systems are powered by foundation models. This paper presents a new set of foundation models, called Llama 3. It is a herd of language models that natively support multilinguality, coding, reasoning, and tool usage. Our largest model is a dense Transformer with 405B parameters and a context window of up to 128K tokens. This paper presents an extensive empirical evaluation of Llama 3. We find that Llama 3 delivers comparable quality to leading language models such as GPT-4 on a plethora of tasks. We publicly release Llama 3, including pre-trained and post-trained versions of the 405B parameter language model and our Llama Guard 3 model for input and output safety. The paper also presents the results of experiments in which we integrate image, video, and speech capabilities into Llama 3 via a compositional approach. We observe this approach performs competitively with the state-of-the-art on image, video, and speech recognition tasks. The resulting models are not yet being broadly released as they are still under development.

15T tokens

405B parameters


- Mistral's models

[[Jiang+ 2023]](https://arxiv.org/pdf/2310.06825.pdf)

Mistral 7B

[Mistral] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot... (8 more)... Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed

2023-10-10

We introduce Mistral 7B v0.1, a 7-billion-parameter language model engineered for superior performance and efficiency. Mistral 7B outperforms Llama 2 13B across all evaluated benchmarks, and Llama 1 34B in reasoning, mathematics, and code generation. Our model leverages grouped-query attention (GQA) for faster inference, coupled with sliding window attention (SWA) to effectively handle sequences of arbitrary length with a reduced inference cost. We also provide a model fine-tuned to follow instructions, Mistral 7B -- Instruct, that surpasses the Llama 2 13B -- Chat model both on human and automated benchmarks. Our models are released under the Apache 2.0 license.

GQA, sliding window attention

[[Jiang+ 2024]](https://arxiv.org/pdf/2401.04088.pdf)

Mixtral of Experts

[Mistral] Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary... (16 more)... Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed

2024-01-08

We introduce Mixtral 8x7B, a Sparse Mixture of Experts (SMoE) language model. Mixtral has the same architecture as Mistral 7B, with the difference that each layer is composed of 8 feedforward blocks (i.e. experts). For every token, at each layer, a router network selects two experts to process the current state and combine their outputs. Even though each token only sees two experts, the selected experts can be different at each timestep. As a result, each token has access to 47B parameters, but only uses 13B active parameters during inference. Mixtral was trained with a context size of 32k tokens and it outperforms or matches Llama 2 70B and GPT-3.5 across all evaluated benchmarks. In particular, Mixtral vastly outperforms Llama 2 70B on mathematics, code generation, and multilingual benchmarks. We also provide a model fine-tuned to follow instructions, Mixtral 8x7B - Instruct, that surpasses GPT-3.5 Turbo, Claude-2.1, Gemini Pro, and Llama 2 70B - chat model on human benchmarks. Both the base and instruct models are released under the Apache 2.0 license.


- DeepSeek's models

[[DeepSeek-AI+ 2024]](https://arxiv.org/pdf/2401.02954.pdf)

DeepSeek LLM: Scaling Open-Source Language Models with Longtermism

[DeepSeek] DeepSeek-AI, Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen... (77 more)... Yao Zhao, Shangyan Zhou, Shunfeng Zhou, Qihao Zhu, Yuheng Zou

2024-01-05

The rapid development of open-source large language models (LLMs) has been truly remarkable. However, the scaling law described in previous literature presents varying conclusions, which casts a dark cloud over scaling LLMs. We delve into the study of scaling laws and present our distinctive findings that facilitate scaling of large scale models in two commonly used open-source configurations, 7B and 67B. Guided by the scaling laws, we introduce DeepSeek LLM, a project dedicated to advancing open-source language models with a long-term perspective. To support the pre-training phase, we have developed a dataset that currently consists of 2 trillion tokens and is continuously expanding. We further conduct supervised fine-tuning (SFT) and Direct Preference Optimization (DPO) on DeepSeek LLM Base models, resulting in the creation of DeepSeek Chat models. Our evaluation results demonstrate that DeepSeek LLM 67B surpasses LLaMA-2 70B on various benchmarks, particularly in the domains of code, mathematics, and reasoning. Furthermore, open-ended evaluations reveal that DeepSeek LLM 67B Chat exhibits superior performance compared to GPT-3.5.

Data: DeepSeek, The Stack, Reddit, etc. (2T tokens)

Architecture: LLaMA, but: for GQA increased depth, 67B parameters

Scaling laws: used non-embedding FLOPs with IsoFLOP

[[DeepSeek-AI+ 2024]](https://arxiv.org/abs/2405.04434)

DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang... (147 more)... Zhuoshu Li, Zihan Wang, Zihui Gu, Zilin Li, Ziwei Xie

2024-05-07

We present DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by economical training and efficient inference. It comprises 236B total parameters, of which 21B are activated for each token, and supports a context length of 128K tokens. DeepSeek-V2 adopts innovative architectures including Multi-head Latent Attention (MLA) and DeepSeekMoE. MLA guarantees efficient inference through significantly compressing the Key-Value (KV) cache into a latent vector, while DeepSeekMoE enables training strong models at an economical cost through sparse computation. Compared with DeepSeek 67B, DeepSeek-V2 achieves significantly stronger performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times. We pretrain DeepSeek-V2 on a high-quality and multi-source corpus consisting of 8.1T tokens, and further perform Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) to fully unlock its potential. Evaluation results show that, even with only 21B activated parameters, DeepSeek-V2 and its chat versions still achieve top-tier performance among open-source models.

[[DeepSeek-AI+ 2024]](https://arxiv.org/pdf/2412.19437.pdf)

DeepSeek-V3 Technical Report

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang... (190 more)... Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, Zizheng Pan

2024-12-27

We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities. Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models. Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training. In addition, its training process is remarkably stable. Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks. The model checkpoints are available at https://github.com/deepseek-ai/DeepSeek-V3.

[[DeepSeek-AI+ 2025]](https://arxiv.org/pdf/2501.12948.pdf)

DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song... (190 more)... Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, Zhen Zhang

2025-01-22

General reasoning represents a long-standing and formidable challenge in artificial intelligence. Recent breakthroughs, exemplified by large language models (LLMs) and chain-of-thought prompting, have achieved considerable success on foundational reasoning tasks. However, this success is heavily contingent upon extensive human-annotated demonstrations, and models' capabilities are still insufficient for more complex problems. Here we show that the reasoning abilities of LLMs can be incentivized through pure reinforcement learning (RL), obviating the need for human-labeled reasoning trajectories. The proposed RL framework facilitates the emergent development of advanced reasoning patterns, such as self-reflection, verification, and dynamic strategy adaptation. Consequently, the trained model achieves superior performance on verifiable tasks such as mathematics, coding competitions, and STEM fields, surpassing its counterparts trained via conventional supervised learning on human demonstrations. Moreover, the emergent reasoning patterns exhibited by these large-scale models can be systematically harnessed to guide and enhance the reasoning capabilities of smaller models.

[[DeepSeek-AI+ 2025]](https://arxiv.org/abs/2512.02556)

DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models

DeepSeek-AI, Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue... (254 more)... Yukun Zha, Zekai Zhang, Zhe Ju, Zhen Zhang, Zihua Qu

2025-12-02

We introduce DeepSeek-V3.2, a model that harmonizes high computational efficiency with superior reasoning and agent performance. The key technical breakthroughs of DeepSeek-V3.2 are as follows: (1) DeepSeek Sparse Attention (DSA): We introduce DSA, an efficient attention mechanism that substantially reduces computational complexity while preserving model performance in long-context scenarios. (2) Scalable Reinforcement Learning Framework: By implementing a robust reinforcement learning protocol and scaling post-training compute, DeepSeek-V3.2 performs comparably to GPT-5. Notably, our high-compute variant, DeepSeek-V3.2-Speciale, surpasses GPT-5 and exhibits reasoning proficiency on par with Gemini-3.0-Pro, achieving gold-medal performance in both the 2025 International Mathematical Olympiad (IMO) and the International Olympiad in Informatics (IOI). (3) Large-Scale Agentic Task Synthesis Pipeline: To integrate reasoning into tool-use scenarios, we developed a novel synthesis pipeline that systematically generates training data at scale. This methodology facilitates scalable agentic post-training, yielding substantial improvements in generalization and instruction-following robustness within complex, interactive environments.


- Alibaba's Qwen models

[[Qwen+ 2024]](https://arxiv.org/abs/2412.15115)

Qwen2.5 Technical Report

[Alibaba] Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui... (33 more)... Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zihan Qiu

2024-12-19

In this report, we introduce Qwen2.5, a comprehensive series of large language models (LLMs) designed to meet diverse needs. Compared to previous iterations, Qwen 2.5 has been significantly improved during both the pre-training and post-training stages. In terms of pre-training, we have scaled the high-quality pre-training datasets from the previous 7 trillion tokens to 18 trillion tokens. This provides a strong foundation for common sense, expert knowledge, and reasoning capabilities. In terms of post-training, we implement intricate supervised finetuning with over 1 million samples, as well as multistage reinforcement learning. Post-training techniques enhance human preference, and notably improve long text generation, structural data analysis, and instruction following. To handle diverse and varied use cases effectively, we present Qwen2.5 LLM series in rich sizes. Open-weight offerings include base and instruction-tuned models, with quantized versions available. In addition, for hosted solutions, the proprietary models currently include two mixture-of-experts (MoE) variants: Qwen2.5-Turbo and Qwen2.5-Plus, both available from Alibaba Cloud Model Studio. Qwen2.5 has demonstrated top-tier performance on a wide range of benchmarks evaluating language understanding, reasoning, mathematics, coding, human preference alignment, etc. Specifically, the open-weight flagship Qwen2.5-72B-Instruct outperforms a number of open and proprietary models and demonstrates competitive performance to the state-of-the-art open-weight model, Llama-3-405B-Instruct, which is around 5 times larger. Qwen2.5-Turbo and Qwen2.5-Plus offer superior cost-effectiveness while performing competitively against GPT-4o-mini and GPT-4o respectively. Additionally, as the foundation, Qwen2.5 models have been instrumental in training specialized models such as Qwen2.5-Math, Qwen2.5-Coder, QwQ, and multimodal models.

[[Yang+ 2025]](https://arxiv.org/abs/2505.09388)

Qwen3 Technical Report

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui... (50 more)... Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, Zihan Qiu

2025-05-14

In this work, we present Qwen3, the latest version of the Qwen model family. Qwen3 comprises a series of large language models (LLMs) designed to advance performance, efficiency, and multilingual capabilities. The Qwen3 series includes models of both dense and Mixture-of-Expert (MoE) architectures, with parameter scales ranging from 0.6 to 235 billion. A key innovation in Qwen3 is the integration of thinking mode (for complex, multi-step reasoning) and non-thinking mode (for rapid, context-driven responses) into a unified framework. This eliminates the need to switch between different models--such as chat-optimized models (e.g., GPT-4o) and dedicated reasoning models (e.g., QwQ-32B)--and enables dynamic mode switching based on user queries or chat templates. Meanwhile, Qwen3 introduces a thinking budget mechanism, allowing users to allocate computational resources adaptively during inference, thereby balancing latency and performance based on task complexity. Moreover, by leveraging the knowledge from the flagship models, we significantly reduce the computational resources required to build smaller-scale models, while ensuring their highly competitive performance. Empirical evaluations demonstrate that Qwen3 achieves state-of-the-art results across diverse benchmarks, including tasks in code generation, mathematical reasoning, agent tasks, etc., competitive against larger MoE models and proprietary models. Compared to its predecessor Qwen2.5, Qwen3 expands multilingual support from 29 to 119 languages and dialects, enhancing global accessibility through improved cross-lingual understanding and generation capabilities. To facilitate reproducibility and community-driven research and development, all Qwen3 models are publicly accessible under Apache 2.0.


- Moonshot's Kimi models

[[Kimi Team 2025]](https://arxiv.org/pdf/2501.12599.pdf)

Kimi k1.5: Scaling Reinforcement Learning with LLMs

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang... (86 more)... Zhiqi Huang, Zihao Huang, Ziyao Xu, Zonghan Yang, Zongyu Lin

2025-01-22

Language model pretraining with next token prediction has proved effective for scaling compute but is limited to the amount of available training data. Scaling reinforcement learning (RL) unlocks a new axis for the continued improvement of artificial intelligence, with the promise that large language models (LLMs) can scale their training data by learning to explore with rewards. However, prior published work has not produced competitive results. In light of this, we report on the training practice of Kimi k1.5, our latest multi-modal LLM trained with RL, including its RL training techniques, multi-modal data recipes, and infrastructure optimization. Long context scaling and improved policy optimization methods are key ingredients of our approach, which establishes a simplistic, effective RL framework without relying on more complex techniques such as Monte Carlo tree search, value functions, and process reward models. Notably, our system achieves state-of-the-art reasoning performance across multiple benchmarks and modalities -- e.g., 77.5 on AIME, 96.2 on MATH 500, 94-th percentile on Codeforces, 74.9 on MathVista -- matching OpenAI's o1. Moreover, we present effective long2short methods that use long-CoT techniques to improve short-CoT models, yielding state-of-the-art short-CoT reasoning results -- e.g., 60.8 on AIME, 94.6 on MATH500, 47.3 on LiveCodeBench -- outperforming existing short-CoT models such as GPT-4o and Claude Sonnet 3.5 by a large margin (up to +550%).

[[Kimi Team 2026]](https://arxiv.org/abs/2602.02276)

Kimi K2.5: Visual Agentic Intelligence

[Moonshot] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, S.H. Cai... (316 more)... Zhen Zhu, Jingze Zhuang, Weiyu Zhuang, Ying Zou, Xinxing Zu

2026-02-02

We introduce Kimi K2.5, an open-source multimodal agentic model designed to advance general agentic intelligence. K2.5 emphasizes the joint optimization of text and vision so that two modalities enhance each other. This includes a series of techniques such as joint text-vision pre-training, zero-vision SFT, and joint text-vision reinforcement learning. Building on this multimodal foundation, K2.5 introduces Agent Swarm, a self-directed parallel agent orchestration framework that dynamically decomposes complex tasks into heterogeneous sub-problems and executes them concurrently. Extensive evaluations show that Kimi K2.5 achieves state-of-the-art results across various domains including coding, vision, reasoning, and agentic tasks. Agent Swarm also reduces latency by up to

4.5×

 over single-agent baselines. We release the post-trained Kimi K2.5 model checkpoint to facilitate future research and real-world applications of agentic intelligence.


- Z.ai's GLM models

[[GLM-4.5 Team 2025]](https://arxiv.org/abs/2508.06471)

GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models

[Z.ai] GLM-4.5 Team, Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou... (161 more)... Minlie Huang, Hongning Wang, Juanzi Li, Yuxiao Dong, Jie Tang

2025-08-08

We present GLM-4.5, an open-source Mixture-of-Experts (MoE) large language model with 355B total parameters and 32B activated parameters, featuring a hybrid reasoning method that supports both thinking and direct response modes. Through multi-stage training on 23T tokens and comprehensive post-training with expert model iteration and reinforcement learning, GLM-4.5 achieves strong performance across agentic, reasoning, and coding (ARC) tasks, scoring 70.1% on TAU-Bench, 91.0% on AIME 24, and 64.2% on SWE-bench Verified. With much fewer parameters than several competitors, GLM-4.5 ranks 3rd overall among all evaluated models and 2nd on agentic benchmarks. We release both GLM-4.5 (355B parameters) and a compact version, GLM-4.5-Air (106B parameters), to advance research in reasoning and agentic AI systems. Code, models, and more information are available at https://github.com/zai-org/GLM-4.5.

[[GLM-5-Team 2026]](https://arxiv.org/abs/2602.15763)

GLM-5: from Vibe Coding to Agentic Engineering

[Z.ai] GLM-5-Team, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du... (176 more)... Minlie Huang, Hongning Wang, Juanzi Li, Yuxiao Dong, Jie Tang

2026-02-17

We present GLM-5, a next-generation foundation model designed to transition the paradigm of vibe coding to agentic engineering. Building upon the agentic, reasoning, and coding (ARC) capabilities of its predecessor, GLM-5 adopts DSA to significantly reduce training and inference costs while maintaining long-context fidelity. To advance model alignment and autonomy, we implement a new asynchronous reinforcement learning infrastructure that drastically improves post-training efficiency by decoupling generation from training. Furthermore, we propose novel asynchronous agent RL algorithms that further improve RL quality, enabling the model to learn from complex, long-horizon interactions more effectively. Through these innovations, GLM-5 achieves state-of-the-art performance on major open benchmarks. Most critically, GLM-5 demonstrates unprecedented capability in real-world coding tasks, surpassing previous baselines in handling end-to-end software engineering challenges. Code, models, and more information are available at https://github.com/zai-org/GLM-5.


- Minimax's models

[[[MiniMax M2.5]]](https://www.minimax.io/news/minimax-m25)

[MiniMax M2.5]



- Xiaomi's MIMO models

[[[Xiaomi MIMO v2]]](https://mimo.xiaomi.com/mimo-v2-pro)

[Xiaomi MIMO v2]



These models are approaching closed models (GPT, Claude, Gemini, etc.).



Open-source models (weights + paper + code + data):


- AI2's Olmo models

[[Groeneveld+ 2024]](https://arxiv.org/pdf/2402.00838.pdf)

OLMo: Accelerating the Science of Language Models

[AI2] Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney... (33 more)... Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A. Smith, Hannaneh Hajishirzi

2024-02-01

Language models (LMs) have become ubiquitous in both NLP research and in commercial product offerings. As their commercial importance has surged, the most powerful models have become closed off, gated behind proprietary interfaces, with important details of their training data, architectures, and development undisclosed. Given the importance of these details in scientifically studying these models, including their biases and potential risks, we believe it is essential for the research community to have access to powerful, truly open LMs. To this end, we have built OLMo, a competitive, truly Open Language Model, to enable the scientific study of language models. Unlike most prior efforts that have only released model weights and inference code, we release OLMo alongside open training data and training and evaluation code. We hope this release will empower the open research community and inspire a new wave of innovation.

Data: subset of Dolma (2.46T tokens, CommonCrawl, The Stack, Reddit, etc.)

Architecture: no biases, non-parametric layer norm, SwiGLU (8/3 d increased to closest multiple of 128)

Training: 256x4 AMD MI250X on LUMI supercomputer, 27x8 A100s, 800Gbps interconnect

[[Team OLMo 2024]](https://arxiv.org/abs/2501.00656)

2 OLMo 2 Furious

[AI2] Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo... (33 more)... Michael Wilson, Luke Zettlemoyer, Ali Farhadi, Noah A. Smith, Hannaneh Hajishirzi

2024-12-31

We present OLMo 2, the next generation of our fully open language models. OLMo 2 includes a family of dense autoregressive language models at 7B, 13B and 32B scales with fully released artifacts -- model weights, full training data, training code and recipes, training logs and thousands of intermediate checkpoints. In this work, we describe our modified model architecture and training recipe, focusing on techniques for achieving better training stability and improved per-token efficiency. Our updated pretraining data mixture introduces a new, specialized data mix called Dolmino Mix 1124, which significantly improves model capabilities across many downstream task benchmarks when introduced via late-stage curriculum training (i.e. specialized data during the annealing phase of pretraining). Finally, we incorporate best practices from Tülu 3 to develop OLMo 2-Instruct, focusing on permissive data and extending our final-stage reinforcement learning with verifiable rewards (RLVR). Our OLMo 2 base models sit at the Pareto frontier of performance to training compute, often matching or outperforming open-weight only models like Llama 3.1, Qwen 2.5, and Gemma 2 while using fewer FLOPs and with fully transparent training data, code, and recipe. Our fully open OLMo 2-Instruct models are competitive with open-weight only models of comparable size and even some proprietary models like GPT-3.5 Turbo and GPT 4o Mini.

[[Team Olmo 2025]](https://arxiv.org/abs/2512.13961)

Olmo 3

Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham... (58 more)... Luke Zettlemoyer, Pang Wei Koh, Ali Farhadi, Noah A. Smith, Hannaneh Hajishirzi

2025-12-15

We introduce Olmo 3, a family of state-of-the-art, fully-open language models at the 7B and 32B parameter scales. Olmo 3 model construction targets long-context reasoning, function calling, coding, instruction following, general chat, and knowledge recall. This release includes the entire model flow, i.e., the full lifecycle of the family of models, including every stage, checkpoint, data point, and dependency used to build it. Our flagship model, Olmo 3 Think 32B, is the strongest fully-open thinking model released to-date.


- NVIDIA's Nemotron models

[[Parmar+ 2024]](https://arxiv.org/pdf/2402.16819.pdf)

Nemotron-4 15B Technical Report

[NVIDIA] Jupinder Parmar, Shrimai Prabhumoye, Joseph Jennings, Mostofa Patwary, Sandeep Subramanian... (17 more)... Ashwath Aithal, Oleksii Kuchaiev, Mohammad Shoeybi, Jonathan Cohen, Bryan Catanzaro

2024-02-26

We introduce Nemotron-4 15B, a 15-billion-parameter large multilingual language model trained on 8 trillion text tokens. Nemotron-4 15B demonstrates strong performance when assessed on English, multilingual, and coding tasks: it outperforms all existing similarly-sized open models on 4 out of 7 downstream evaluation areas and achieves competitive performance to the leading open models in the remaining ones. Specifically, Nemotron-4 15B exhibits the best multilingual capabilities of all similarly-sized models, even outperforming models over four times larger and those explicitly specialized for multilingual tasks.

Data: 8T tokens, 70% English, 15% multilingual, 15% code

Architecture: RoPE, squared ReLU activations, no bias, no dropout, GQA (15B parameters)

Training: 384x8 H100s, After 8T tokens, train on higher quality sources + benchmark tasks

[[NVIDIA+ 2025]](https://arxiv.org/abs/2512.20856)

NVIDIA Nemotron 3: Efficient and Open Intelligence

NVIDIA, Aaron Blakeman, Aaron Grattafiori, Aarti Basant, Abhibha Gupta... (348 more)... Zhen Dong, Zhongbo Zhu, Zihan Liu, Zijia Chen, Zijie Yan

2025-12-24

We introduce the Nemotron 3 family of models - Nano, Super, and Ultra. These models deliver strong agentic, reasoning, and conversational capabilities. The Nemotron 3 family uses a Mixture-of-Experts hybrid Mamba-Transformer architecture to provide best-in-class throughput and context lengths of up to 1M tokens. Super and Ultra models are trained with NVFP4 and incorporate LatentMoE, a novel approach that improves model quality. The two larger models also include MTP layers for faster text generation. All Nemotron 3 models are post-trained using multi-environment reinforcement learning enabling reasoning, multi-step tool use, and support granular reasoning budget control. Nano, the smallest model, outperforms comparable models in accuracy while remaining extremely cost-efficient for inference. Super is optimized for collaborative agents and high-volume workloads such as IT ticket automation. Ultra, the largest model, provides state-of-the-art accuracy and reasoning performance. Nano is released together with its technical report and this white paper, while Super and Ultra will follow in the coming months. We will openly release the model weights, pre- and post-training software, recipes, and all data for which we hold redistribution rights.


- Marin's models (open development)

[[[Marin 8B retro]]](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/)

[Marin 8B retro]


[[[Marin 32B retro]]](https://marin.readthedocs.io/en/latest/reports/marin-32b-retro/)

[Marin 32B retro]




Openness is important for trust and innovation

[[Kapoor+ 2024]](https://arxiv.org/abs/2403.07918)

On the Societal Impact of Open Foundation Models

Sayash Kapoor, Rishi Bommasani, Kevin Klyman, Shayne Longpre, Ashwin Ramaswami... (15 more)... Victor Storchan, Daniel Zhang, Daniel E. Ho, Percy Liang, Arvind Narayanan

2024-02-27

Foundation models are powerful technologies: how they are released publicly directly shapes their societal impact. In this position paper, we focus on open foundation models, defined here as those with broadly available model weights (e.g. Llama 2, Stable Diffusion XL). We identify five distinctive properties (e.g. greater customizability, poor monitoring) of open foundation models that lead to both their benefits and risks. Open foundation models present significant benefits, with some caveats, that span innovation, competition, the distribution of decision-making power, and transparency. To understand their risks of misuse, we design a risk assessment framework for analyzing their marginal risk. Across several misuse vectors (e.g. cyberattacks, bioweapons), we find that current research is insufficient to effectively characterize the marginal risk of open foundation models relative to pre-existing technologies. The framework helps explain why the marginal risk is low in some cases, clarifies disagreements about misuse risks by revealing that past work has focused on different subsets of the framework with different assumptions, and articulates a way forward for more constructive debate. Overall, our work helps support a more grounded assessment of the societal impact of open foundation models by outlining what research is needed to empirically validate their theoretical benefits and risks.


Ideas from open models enable us to teach CS336.



What is a language model?


- 2018 (BERT): something you fine-tune


- 2020 (GPT-3): something you prompt


- 2022 (ChatGPT): something you talk to

[[example conversation]](https://huggingface.co/datasets/HuggingFaceTB/smoltalk/viewer/all/train?row=72&conversation-viewer=72)

example conversation


- 2026 (agents): something that acts autonomously

[[example trace]](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories/viewer/default/train?conversation-viewer=1)

example trace



The fundamentals are the same (attention, kernels, optimization).


The specs are different (longer context, inference efficiency matters even more).




def what_is_this_program():


This is an *executable lecture*, a program whose execution delivers the content of a lecture.


Executable lectures make it possible to:


- view and run code (since everything is code!),


total = 0


for x in [1, 2, 3]:


total += x


- see the hierarchical structure of the lecture




def course_logistics():


All information online:

[[course website]](https://stanford-cs336.github.io/spring2026/)

course website



This is a 5-unit class.


Comment from Spring 2024 course evaluation:


> *The entire assignment was approximately the same amount of work as all 5 assignments from CS 224n plus the final project. And that's just the first homework assignment.*



## Why you should take this course


- You have an obsessive need to understand how things work.


- You want to build up your research engineering muscles.



## Why you should not take this course


- You actually want to get research done this quarter. (Talk to your advisor.)


- You are interested in learning about the hottest new techniques in AI (e.g., multimodality, RAG, etc.). (You should take a seminar class for that.)


- You want to get good results on your own application domain. (You should just prompt or fine-tune an existing model.)



## How you can follow along at home


- All lecture materials and assignments will be posted online, so feel free to follow on your own.


- Lectures are recorded via [CGOE](https://cgoe.stanford.edu/).



## Assignments


- 5 assignments (basics, systems, scaling laws, data, alignment).


- No scaffolding code, but we provide unit tests and adapter interfaces to help you check correctness.


- Implement locally to test for correctness, then run on cluster for benchmarking (accuracy and speed).


- Leaderboard for some assignments (minimize perplexity given training budget).



## AI policy


- Coding agents can solve all the assignments, but you won't learn anything.


- AI can be tremendously useful for answering questions and tutoring.


- You must use our provided AGENTS.md file, which asks the AI to be pedagogically-minded.


- Please read our [AI policy guide](https://docs.google.com/document/d/1SZAlExB1qAc9izHt54gwunNpjKE6wXb8Y7yA_e-baK8/edit?tab=t.0).



## Compute


- Thanks to [Modal](https://modal.com/) for providing compute. 🙏


- Please read the [guide](https://docs.google.com/document/d/1cHE0iKVyXLJ3XpIs2XuXTmZ-HMmPk2hIPeCvy-AydMg/edit?tab=t.otis27tacaef) on how to access and use the compute.




def course_syllabus():


basics() # Assignment 1: tokenization, model architecture, training


systems() # Assignment 2: kernels, parallelism, inference


scaling_laws() # Assignment 3: scaling laws


data() # Assignment 4: evaluation, curation, transformation, filtering, deduplication, mixing


alignment() # Assignment 5: RLHF, RL algorithms, RL systems



Remember it's all about **efficiency**:


- Resources: data + hardware (compute, memory, communication bandwidth)


- How do you train the best model given a fixed set of resources?



Today, we are compute-constrained, so design decisions will reflect squeezing the most out of given hardware.


- Systems: clearly about efficiency


- Tokenization: working with raw bytes is elegant, but compute-inefficient with today's model architectures


- Model architecture: many changes motivated by reducing memory or FLOPs (e.g., sharing KV caches, sliding window attention)


- Data filtering: avoid wasting precious compute updating on bad / irrelevant data


- Scaling laws: use less compute on smaller models to do hyperparameter tuning



Tomorrow, we will become data-constrained...




class Tokenizer(ABC):


"""Abstract interface for a tokenizer."""


def encode(self, string: str) -> list[int]:


raise NotImplementedError



def decode(self, indices: list[int]) -> str:


raise NotImplementedError




def basics():


Goal: be able to train a basic language model


Components: tokenization, model architecture, training



## Tokenization


What are the atoms that the model operates on?


Formally: a tokenizer converts between raw inputs (bytes) and sequences of integers (tokens)


![Image](./Trace - lecture_01_files/tokenized-example.png)


Popular tokenizer: **Byte-Pair Encoding** (BPE)

[[Sennrich+ 2015]](https://arxiv.org/abs/1508.07909)

Neural Machine Translation of Rare Words with Subword Units

Rico Sennrich, Barry Haddow, Alexandra Birch

2015-08-31

Neural machine translation (NMT) models typically operate with a fixed vocabulary, but translation is an open-vocabulary problem. Previous work addresses the translation of out-of-vocabulary words by backing off to a dictionary. In this paper, we introduce a simpler and more effective approach, making the NMT model capable of open-vocabulary translation by encoding rare and unknown words as sequences of subword units. This is based on the intuition that various word classes are translatable via smaller units than words, for instance names (via character copying or transliteration), compounds (via compositional translation), and cognates and loanwords (via phonological and morphological transformations). We discuss the suitability of different word segmentation techniques, including simple character n-gram models and a segmentation based on the byte pair encoding compression algorithm, and empirically show that subword models improve over a back-off dictionary baseline for the WMT 15 translation tasks English-German and English-Russian by 1.1 and 1.3 BLEU, respectively.


Intuition: break input into frequently-occuring chunks


Efficiency lens


- Reduce context length (1000 bytes → ~250 tokens)


- Adaptive computation (more modeling capacity on interesting parts of input)



The dream: tokenizer-free model architectures, which operate directly on bytes

[[Xue+ 2021]](https://arxiv.org/abs/2105.13626)

ByT5: Towards a token-free future with pre-trained byte-to-byte models

Linting Xue, Aditya Barua, Noah Constant, Rami Al-Rfou, Sharan Narang, Mihir Kale, Adam Roberts, Colin Raffel

2021-05-28

Most widely-used pre-trained language models operate on sequences of tokens corresponding to word or subword units. By comparison, token-free models that operate directly on raw text (bytes or characters) have many benefits: they can process text in any language out of the box, they are more robust to noise, and they minimize technical debt by removing complex and error-prone text preprocessing pipelines. Since byte or character sequences are longer than token sequences, past work on token-free models has often introduced new model architectures designed to amortize the cost of operating directly on raw text. In this paper, we show that a standard Transformer architecture can be used with minimal modifications to process byte sequences. We characterize the trade-offs in terms of parameter count, training FLOPs, and inference speed, and show that byte-level models are competitive with their token-level counterparts. We also demonstrate that byte-level models are significantly more robust to noise and perform better on tasks that are sensitive to spelling and pronunciation. As part of our contribution, we release a new set of pre-trained byte-level Transformer models based on the T5 architecture, as well as all code and data used in our experiments.

[[Yu+ 2023]](https://arxiv.org/pdf/2305.07185.pdf)

MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers

Lili Yu, Dániel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, Mike Lewis

2023-05-12

Autoregressive transformers are spectacular models for short sequences but scale poorly to long sequences such as high-resolution images, podcasts, code, or books. We proposed Megabyte, a multi-scale decoder architecture that enables end-to-end differentiable modeling of sequences of over one million bytes. Megabyte segments sequences into patches and uses a local submodel within patches and a global model between patches. This enables sub-quadratic self-attention, much larger feedforward layers for the same compute, and improved parallelism during decoding -- unlocking better performance at reduced cost for both training and generation. Extensive experiments show that Megabyte allows byte-level models to perform competitively with subword models on long context language modeling, achieve state-of-the-art density estimation on ImageNet, and model audio from raw files. Together, these results establish the viability of tokenization-free autoregressive sequence modeling at scale.

[[Pagnoni+ 2024]](https://arxiv.org/abs/2412.09871)

Byte Latent Transformer: Patches Scale Better Than Tokens

Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller... (4 more)... Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, Srinivasan Iyer

2024-12-13

We introduce the Byte Latent Transformer (BLT), a new byte-level LLM architecture that, for the first time, matches tokenization-based LLM performance at scale with significant improvements in inference efficiency and robustness. BLT encodes bytes into dynamically sized patches, which serve as the primary units of computation. Patches are segmented based on the entropy of the next byte, allocating more compute and model capacity where increased data complexity demands it. We present the first FLOP controlled scaling study of byte-level models up to 8B parameters and 4T training bytes. Our results demonstrate the feasibility of scaling models trained on raw bytes without a fixed vocabulary. Both training and inference efficiency improve due to dynamically selecting long patches when data is predictable, along with qualitative improvements on reasoning and long tail generalization. Overall, for fixed inference costs, BLT shows significantly better scaling than tokenization-based models, by simultaneously growing both patch and model size.

[[Deiseroth+ 2024]](https://arxiv.org/abs/2406.19223)

T-FREE: Subword Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings

Björn Deiseroth, Manuel Brack, Patrick Schramowski, Kristian Kersting, Samuel Weinbach

2024-06-27

Tokenizers are crucial for encoding information in Large Language Models, but their development has recently stagnated, and they contain inherent weaknesses. Major limitations include computational overhead, ineffective vocabulary use, and unnecessarily large embedding and head layers. Additionally, their performance is biased towards a reference corpus, leading to reduced effectiveness for underrepresented languages. To remedy these issues, we propose T-FREE, which directly embeds words through sparse activation patterns over character triplets, and does not require a reference corpus. T-FREE inherently exploits morphological similarities and allows for strong compression of embedding layers. In our exhaustive experimental evaluation, we achieve competitive downstream performance with a parameter reduction of more than 85% on these layers. Further, T-FREE shows significant improvements in cross-lingual transfer learning.

[[Hwang+ 2025]](https://arxiv.org/abs/2507.07955)

Dynamic Chunking for End-to-End Hierarchical Sequence Modeling

Sukjun Hwang, Brandon Wang, Albert Gu

2025-07-10

Major progress on language models (LMs) in recent years has largely resulted from moving away from specialized models designed for specific tasks, to general models based on powerful architectures (e.g. the Transformer) that learn everything from raw data. Despite this trend, pre-processing steps such as tokenization remain a barrier to true end-to-end foundation models. We introduce a collection of new techniques that enable a dynamic chunking mechanism which automatically learns content- and context- dependent segmentation strategies learned jointly with the rest of the model. Incorporating this into an explicit hierarchical network (H-Net) allows replacing the (implicitly hierarchical) tokenization-LM-detokenization pipeline with a single model learned fully end-to-end. When compute- and data- matched, an H-Net with one stage of hierarchy operating at the byte level outperforms a strong Transformer language model operating over BPE tokens. Iterating the hierarchy to multiple stages further increases its performance by modeling multiple levels of abstraction, demonstrating significantly better scaling with data and matching the token-based Transformer of twice its size. H-Nets pretrained on English show significantly increased character-level robustness, and qualitatively learn meaningful data-dependent chunking strategies without any heuristics or explicit supervision. Finally, the H-Net's improvement over tokenized pipelines is further increased in languages and modalities with weaker tokenization heuristics, such as Chinese and code, or DNA sequences (nearly 4x improvement in data efficiency over baselines), showing the potential of true end-to-end models that learn and scale better from unprocessed data.


These are promising, but have not yet been scaled up to the frontier.



## Model architecture


Starting point: original Transformer

[[Vaswani+ 2017]](https://arxiv.org/pdf/1706.03762.pdf)

Attention Is All You Need

[Google] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

2017-06-12

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

Introduced Transformer (for machine translation)


![Image](./Trace - lecture_01_files/transformer-architecture.png)



Refinements:


- Activation functions: ReLU, SwiGLU

[[Shazeer 2020]](https://arxiv.org/pdf/2002.05202.pdf)

GLU Variants Improve Transformer

[Google] Noam Shazeer

2020-02-12

Gated Linear Units (arXiv:1612.08083) consist of the component-wise product of two linear projections, one of which is first passed through a sigmoid function. Variations on GLU are possible, using different nonlinear (or even linear) functions in place of sigmoid. We test these variants in the feed-forward sublayers of the Transformer (arXiv:1706.03762) sequence-to-sequence model, and find that some of them yield quality improvements over the typically-used ReLU or GELU activations.

Experiments with different activation functions

Activation functions: ReLU, GeLU, Swish

Apply idea of gated units (GLU): ReGLU, GeGLU, SwiGLU

FFN-SwiGLU = Swish(x W1) * xV W2

Have 3 matrices now, so make hidden dimension 2/3 of the 2 matrix version


- Positional encodings: sinusoidal, RoPE

[[Su+ 2021]](https://arxiv.org/pdf/2104.09864.pdf)

RoFormer: Enhanced Transformer with Rotary Position Embedding

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu

2021-04-20

Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets. Our experiments show that it consistently overcomes its alternatives. Furthermore, we provide a theoretical analysis to explain some experimental results. RoFormer is already integrated into Huggingface: \url{https://huggingface.co/docs/transformers/model_doc/roformer}.

Encodes absolute position with rotation matrix, incorporate relative position dependency in self-attention

Key: R W x, where R is a block-diagonal sequence of d/2 rotation matrices (equation 13)

Extrapolates to longer sequences


- Normalization: LayerNorm, RMSNorm, QK norm, pre-norm versus post-norm

[[Ba+ 2016]](https://arxiv.org/pdf/1607.06450.pdf)

Layer Normalization

Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton

2016-07-21

Training state-of-the-art, deep neural networks is computationally expensive. One way to reduce the training time is to normalize the activities of the neurons. A recently introduced technique called batch normalization uses the distribution of the summed input to a neuron over a mini-batch of training cases to compute a mean and variance which are then used to normalize the summed input to that neuron on each training case. This significantly reduces the training time in feed-forward neural networks. However, the effect of batch normalization is dependent on the mini-batch size and it is not obvious how to apply it to recurrent neural networks. In this paper, we transpose batch normalization into layer normalization by computing the mean and variance used for normalization from all of the summed inputs to the neurons in a layer on a single training case. Like batch normalization, we also give each neuron its own adaptive bias and gain which are applied after the normalization but before the non-linearity. Unlike batch normalization, layer normalization performs exactly the same computation at training and test times. It is also straightforward to apply to recurrent neural networks by computing the normalization statistics separately at each time step. Layer normalization is very effective at stabilizing the hidden state dynamics in recurrent networks. Empirically, we show that layer normalization can substantially reduce the training time compared with previously published techniques.

Introduced LayerNorm

[[Zhang+ 2019]](https://arxiv.org/abs/1910.07467)

Root Mean Square Layer Normalization

Biao Zhang, Rico Sennrich

2019-10-16

Layer normalization (LayerNorm) has been successfully applied to various deep neural networks to help stabilize training and boost model convergence because of its capability in handling re-centering and re-scaling of both inputs and weight matrix. However, the computational overhead introduced by LayerNorm makes these improvements expensive and significantly slows the underlying network, e.g. RNN in particular. In this paper, we hypothesize that re-centering invariance in LayerNorm is dispensable and propose root mean square layer normalization, or RMSNorm. RMSNorm regularizes the summed inputs to a neuron in one layer according to root mean square (RMS), giving the model re-scaling invariance property and implicit learning rate adaptation ability. RMSNorm is computationally simpler and thus more efficient than LayerNorm. We also present partial RMSNorm, or pRMSNorm where the RMS is estimated from p% of the summed inputs without breaking the above properties. Extensive experiments on several tasks using diverse network architectures show that RMSNorm achieves comparable performance against LayerNorm but reduces the running time by 7%~64% on different models. Source code is available at https://github.com/bzhangGo/rmsnorm.

[[Dehghani+ 2023]](https://arxiv.org/abs/2302.05442)

Scaling Vision Transformers to 22 Billion Parameters

Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek... (32 more)... Mario Lučić, Xiaohua Zhai, Daniel Keysers, Jeremiah Harmsen, Neil Houlsby

2023-02-10

The scaling of Transformers has driven breakthrough capabilities for language models. At present, the largest large language models (LLMs) contain upwards of 100B parameters. Vision Transformers (ViT) have introduced the same architecture to image and video modelling, but these have not yet been successfully scaled to nearly the same degree; the largest dense ViT contains 4B parameters (Chen et al., 2022). We present a recipe for highly efficient and stable training of a 22B-parameter ViT (ViT-22B) and perform a wide variety of experiments on the resulting model. When evaluated on downstream tasks (often with a lightweight linear model on frozen features), ViT-22B demonstrates increasing performance with scale. We further observe other interesting benefits of scale, including an improved tradeoff between fairness and performance, state-of-the-art alignment to human visual perception in terms of shape/texture bias, and improved robustness. ViT-22B demonstrates the potential for "LLM-like" scaling in vision, and provides key steps towards getting there.

[[Xiong+ 2020]](https://arxiv.org/pdf/2002.04745.pdf)

On Layer Normalization in the Transformer Architecture

Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tie-Yan Liu

2020-02-12

The Transformer is widely used in natural language processing tasks. To train a Transformer however, one usually needs a carefully designed learning rate warm-up stage, which is shown to be crucial to the final performance but will slow down the optimization and bring more hyper-parameter tunings. In this paper, we first study theoretically why the learning rate warm-up stage is essential and show that the location of layer normalization matters. Specifically, we prove with mean field theory that at initialization, for the original-designed Post-LN Transformer, which places the layer normalization between the residual blocks, the expected gradients of the parameters near the output layer are large. Therefore, using a large learning rate on those gradients makes the training unstable. The warm-up stage is practically helpful for avoiding this problem. On the other hand, our theory also shows that if the layer normalization is put inside the residual blocks (recently proposed as Pre-LN Transformer), the gradients are well-behaved at initialization. This motivates us to remove the warm-up stage for the training of Pre-LN Transformers. We show in our experiments that Pre-LN Transformers without the warm-up stage can reach comparable results with baselines while requiring significantly less training time and hyper-parameter tuning on a wide range of applications.


- Attention: full, sparse/local attention, group-query attention (GQA), multi-head latent attention (MLA)

[[Child+ 2019]](https://arxiv.org/pdf/1904.10509.pdf)

Generating Long Sequences with Sparse Transformers

[OpenAI] Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever

2019-04-23

Transformers are powerful sequence models, but require time and memory that grows quadratically with the sequence length. In this paper we introduce sparse factorizations of the attention matrix which reduce this to

O(nn). We also introduce a) a variation on architecture and initialization to train deeper networks, b) the recomputation of attention matrices to save memory, and c) fast attention kernels for training. We call networks with these changes Sparse Transformers, and show they can model sequences tens of thousands of timesteps long using hundreds of layers. We use the same architecture to model images, audio, and text from raw bytes, setting a new state of the art for density modeling of Enwik8, CIFAR-10, and ImageNet-64. We generate unconditional samples that demonstrate global coherence and great diversity, and show it is possible in principle to use self-attention to model sequences of length one million or more.

Local attention

[[Ainslie+ 2023]](https://arxiv.org/pdf/2305.13245.pdf)

GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

[Google] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai

2023-05-22

Multi-query attention (MQA), which only uses a single key-value head, drastically speeds up decoder inference. However, MQA can lead to quality degradation, and moreover it may not be desirable to train a separate model just for faster inference. We (1) propose a recipe for uptraining existing multi-head language model checkpoints into models with MQA using 5% of original pre-training compute, and (2) introduce grouped-query attention (GQA), a generalization of multi-query attention which uses an intermediate (more than one, less than number of query heads) number of key-value heads. We show that uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA.

Multi-query attention (MQA) speeds up, but less expressive

GQA: use an intermediate (more than one, less than number of heads) number of key-value heads

Experiments on T5

[[DeepSeek-AI+ 2024]](https://arxiv.org/abs/2405.04434)

DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang... (147 more)... Zhuoshu Li, Zihan Wang, Zihui Gu, Zilin Li, Ziwei Xie

2024-05-07

We present DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by economical training and efficient inference. It comprises 236B total parameters, of which 21B are activated for each token, and supports a context length of 128K tokens. DeepSeek-V2 adopts innovative architectures including Multi-head Latent Attention (MLA) and DeepSeekMoE. MLA guarantees efficient inference through significantly compressing the Key-Value (KV) cache into a latent vector, while DeepSeekMoE enables training strong models at an economical cost through sparse computation. Compared with DeepSeek 67B, DeepSeek-V2 achieves significantly stronger performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times. We pretrain DeepSeek-V2 on a high-quality and multi-source corpus consisting of 8.1T tokens, and further perform Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) to fully unlock its potential. Evaluation results show that, even with only 21B activated parameters, DeepSeek-V2 and its chat versions still achieve top-tier performance among open-source models.


- Recurrence/state-space models/linear attention: Mamba, Gated DeltaNet

[[Katharopoulos+ 2020]](https://arxiv.org/abs/2006.16236)

Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret

2020-06-29

Transformers achieve remarkable performance in several tasks but due to their quadratic complexity, with respect to the input's length, they are prohibitively slow for very long sequences. To address this limitation, we express the self-attention as a linear dot-product of kernel feature maps and make use of the associativity property of matrix products to reduce the complexity from

O(N2)

 to

O(N), where

N

 is the sequence length. We show that this formulation permits an iterative implementation that dramatically accelerates autoregressive transformers and reveals their relationship to recurrent neural networks. Our linear transformers achieve similar performance to vanilla transformers and they are up to 4000x faster on autoregressive prediction of very long sequences.

[[Dao+ 2024]](https://arxiv.org/abs/2405.21060)

Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

Tri Dao, Albert Gu

2024-05-31

While Transformers have been the main architecture behind deep learning's success in language modeling, state-space models (SSMs) such as Mamba have recently been shown to match or outperform Transformers at small to medium scale. We show that these families of models are actually quite closely related, and develop a rich framework of theoretical connections between SSMs and variants of attention, connected through various decompositions of a well-studied class of structured semiseparable matrices. Our state space duality (SSD) framework allows us to design a new architecture (Mamba-2) whose core layer is an a refinement of Mamba's selective SSM that is 2-8X faster, while continuing to be competitive with Transformers on language modeling.

[[Yang+ 2024]](https://arxiv.org/abs/2412.06464)

Gated Delta Networks: Improving Mamba2 with Delta Rule

Songlin Yang, Jan Kautz, Ali Hatamizadeh

2024-12-09

Linear Transformers have gained attention as efficient alternatives to standard Transformers, but their performance in retrieval and long-context tasks has been limited. To address these limitations, recent work has explored two distinct mechanisms: gating for adaptive memory control and the delta update rule for precise memory modifications. We observe that these mechanisms are complementary: gating enables rapid memory erasure while the delta rule facilitates targeted updates. Building on this insight, we introduce the gated delta rule and develop a parallel training algorithm optimized for modern hardware. Our proposed architecture, Gated DeltaNet, consistently surpasses existing models like Mamba2 and DeltaNet across multiple benchmarks, including language modeling, common-sense reasoning, in-context retrieval, length extrapolation, and long-context understanding. We further enhance performance by developing hybrid architectures that combine Gated DeltaNet layers with sliding window attention or Mamba2 layers, achieving both improved training efficiency and superior task performance.

[[Lahoti+ 2026]](https://arxiv.org/abs/2603.15569)

Mamba-3: Improved Sequence Modeling using State Space Principles

Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu

2026-03-16

Scaling inference-time compute has emerged as an important driver of LLM performance, making inference efficiency a central focus of model design alongside model quality. While the current Transformer-based models deliver strong model quality, their quadratic compute and linear memory make inference expensive. This has spurred the development of sub-quadratic models with reduced linear compute and constant memory requirements. However, many recent linear models trade off model quality and capability for algorithmic efficiency, failing on tasks such as state tracking. Moreover, their theoretically linear inference remains hardware-inefficient in practice. Guided by an inference-first perspective, we introduce three core methodological improvements inspired by the state space model (SSM) viewpoint of linear models. We combine: (1) a more expressive recurrence derived from SSM discretization, (2) a complex-valued state update rule that enables richer state tracking, and (3) a multi-input, multi-output (MIMO) formulation for better model performance without increasing decode latency. Together with architectural refinements, our Mamba-3 model achieves significant gains across retrieval, state-tracking, and downstream language modeling tasks. At the 1.5B scale, Mamba-3 improves average downstream accuracy by 0.6 percentage points compared to the next best model (Gated DeltaNet), with Mamba-3's MIMO variant further improving accuracy by another 1.2 points for a total 1.8 point gain. Across state-size experiments, Mamba-3 achieves comparable perplexity to Mamba-2 despite using half of its predecessor's state size. Our evaluations demonstrate Mamba-3's ability to advance the performance-efficiency Pareto frontier.


- MLP: dense, mixture of experts

[[Shazeer+ 2017]](https://arxiv.org/pdf/1701.06538.pdf)

Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

[Google] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean

2017-01-23

The capacity of a neural network to absorb information is limited by its number of parameters. Conditional computation, where parts of the network are active on a per-example basis, has been proposed in theory as a way of dramatically increasing model capacity without a proportional increase in computation. In practice, however, there are significant algorithmic and performance challenges. In this work, we address these challenges and finally realize the promise of conditional computation, achieving greater than 1000x improvements in model capacity with only minor losses in computational efficiency on modern GPU clusters. We introduce a Sparsely-Gated Mixture-of-Experts layer (MoE), consisting of up to thousands of feed-forward sub-networks. A trainable gating network determines a sparse combination of these experts to use for each example. We apply the MoE to the tasks of language modeling and machine translation, where model capacity is critical for absorbing the vast quantities of knowledge available in the training corpora. We present model architectures in which a MoE with up to 137 billion parameters is applied convolutionally between stacked LSTM layers. On large language modeling and machine translation benchmarks, these models achieve significantly better results than state-of-the-art at lower computational cost.

[[Fedus+ 2021]](https://arxiv.org/abs/2101.03961)

Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

[Google] William Fedus, Barret Zoph, Noam Shazeer

2021-01-11

In deep learning, models typically reuse the same parameters for all inputs. Mixture of Experts (MoE) defies this and instead selects different parameters for each incoming example. The result is a sparsely-activated model -- with outrageous numbers of parameters -- but a constant computational cost. However, despite several notable successes of MoE, widespread adoption has been hindered by complexity, communication costs and training instability -- we address these with the Switch Transformer. We simplify the MoE routing algorithm and design intuitive improved models with reduced communication and computational costs. Our proposed training techniques help wrangle the instabilities and we show large sparse models may be trained, for the first time, with lower precision (bfloat16) formats. We design models based off T5-Base and T5-Large to obtain up to 7x increases in pre-training speed with the same computational resources. These improvements extend into multilingual settings where we measure gains over the mT5-Base version across all 101 languages. Finally, we advance the current scale of language models by pre-training up to trillion parameter models on the "Colossal Clean Crawled Corpus" and achieve a 4x speedup over the T5-XXL model.


- Shape (hidden dimension, depth, number of heads, number of experts)



## Training


How do you set the parameters of the model?


- Loss function (e.g., multi-token prediction)

[[Gloeckle+ 2024]](https://arxiv.org/abs/2404.19737)

Better & Faster Large Language Models via Multi-token Prediction

Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, Gabriel Synnaeve

2024-04-30

Large language models such as GPT and Llama are trained with a next-token prediction loss. In this work, we suggest that training language models to predict multiple future tokens at once results in higher sample efficiency. More specifically, at each position in the training corpus, we ask the model to predict the following n tokens using n independent output heads, operating on top of a shared model trunk. Considering multi-token prediction as an auxiliary training task, we measure improved downstream capabilities with no overhead in training time for both code and natural language models. The method is increasingly useful for larger model sizes, and keeps its appeal when training for multiple epochs. Gains are especially pronounced on generative benchmarks like coding, where our models consistently outperform strong baselines by several percentage points. Our 13B parameter models solves 12 % more problems on HumanEval and 17 % more on MBPP than comparable next-token models. Experiments on small algorithmic tasks demonstrate that multi-token prediction is favorable for the development of induction heads and algorithmic reasoning capabilities. As an additional benefit, models trained with 4-token prediction are up to 3 times faster at inference, even with large batch sizes.

[[DeepSeek-AI+ 2024]](https://arxiv.org/pdf/2412.19437.pdf)

DeepSeek-V3 Technical Report

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang... (190 more)... Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, Zizheng Pan

2024-12-27

We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities. Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models. Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training. In addition, its training process is remarkably stable. Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks. The model checkpoints are available at https://github.com/deepseek-ai/DeepSeek-V3.


- Optimizer (e.g., AdamW, SOAP, Muon)

[[Kingma+ 2014]](https://arxiv.org/pdf/1412.6980.pdf)

Adam: A Method for Stochastic Optimization

Diederik P. Kingma, Jimmy Ba

2014-12-22

We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions, based on adaptive estimates of lower-order moments. The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients. The hyper-parameters have intuitive interpretations and typically require little tuning. Some connections to related algorithms, on which Adam was inspired, are discussed. We also analyze the theoretical convergence properties of the algorithm and provide a regret bound on the convergence rate that is comparable to the best known results under the online convex optimization framework. Empirical results demonstrate that Adam works well in practice and compares favorably to other stochastic optimization methods. Finally, we discuss AdaMax, a variant of Adam based on the infinity norm.

Introduced Adam optimizer based on RMSProp and momentum

[[Loshchilov+ 2017]](https://arxiv.org/pdf/1711.05101.pdf)

Decoupled Weight Decay Regularization

Ilya Loshchilov, Frank Hutter

2017-11-14

L


 regularization and weight decay regularization are equivalent for standard stochastic gradient descent (when rescaled by the learning rate), but as we demonstrate this is \emph{not} the case for adaptive gradient algorithms, such as Adam. While common implementations of these algorithms employ L


 regularization (often calling it "weight decay" in what may be misleading due to the inequivalence we expose), we propose a simple modification to recover the original formulation of weight decay regularization by \emph{decoupling} the weight decay from the optimization steps taken w.r.t. the loss function. We provide empirical evidence that our proposed modification (i) decouples the optimal choice of weight decay factor from the setting of the learning rate for both standard SGD and Adam and (ii) substantially improves Adam's generalization performance, allowing it to compete with SGD with momentum on image classification datasets (on which it was previously typically outperformed by the latter). Our proposed decoupled weight decay has already been adopted by many researchers, and the community has implemented it in TensorFlow and PyTorch; the complete source code for our experiments is available at https://github.com/loshchil/AdamW-and-SGDW

Improves Adam by decoupling weight decay

[[Vyas+ 2024]](https://arxiv.org/abs/2409.11321)

SOAP: Improving and Stabilizing Shampoo using Adam

Nikhil Vyas, Depen Morwani, Rosie Zhao, Mujin Kwun, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade

2024-09-17

There is growing evidence of the effectiveness of Shampoo, a higher-order preconditioning method, over Adam in deep learning optimization tasks. However, Shampoo's drawbacks include additional hyperparameters and computational overhead when compared to Adam, which only updates running averages of first- and second-moment quantities. This work establishes a formal connection between Shampoo (implemented with the 1/2 power) and Adafactor -- a memory-efficient approximation of Adam -- showing that Shampoo is equivalent to running Adafactor in the eigenbasis of Shampoo's preconditioner. This insight leads to the design of a simpler and computationally efficient algorithm:

S

hampo

O

 with

A

dam in the

P

reconditioner's eigenbasis (SOAP). With regards to improving Shampoo's computational efficiency, the most straightforward approach would be to simply compute Shampoo's eigendecomposition less frequently. Unfortunately, as our empirical results show, this leads to performance degradation that worsens with this frequency. SOAP mitigates this degradation by continually updating the running average of the second moment, just as Adam does, but in the current (slowly changing) coordinate basis. Furthermore, since SOAP is equivalent to running Adam in a rotated space, it introduces only one additional hyperparameter (the preconditioning frequency) compared to Adam. We empirically evaluate SOAP on language model pre-training with 360m and 660m sized models. In the large batch regime, SOAP reduces the number of iterations by over 40% and wall clock time by over 35% compared to AdamW, with approximately 20% improvements in both metrics compared to Shampoo. An implementation of SOAP is available at https://github.com/nikhilvyas/SOAP.

[[Keller 2024]](https://kellerjordan.github.io/posts/muon/)

Muon: An optimizer for hidden layers in neural networks

Jordan Keller

2024-12-08


- Initialization scale (e.g., Xavier init, muP)

[[Glorot+ 2010]](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

Understanding the difficulty of training deep feedforward neural networks

Xavier Glorot, Yoshua Bengio

2010-03-01

[[Yang+ 2022]](https://arxiv.org/abs/2203.03466)

Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer

Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao

2022-03-07

Hyperparameter (HP) tuning in deep learning is an expensive process, prohibitively so for neural networks (NNs) with billions of parameters. We show that, in the recently discovered Maximal Update Parametrization (muP), many optimal HPs remain stable even as model size changes. This leads to a new HP tuning paradigm we call muTransfer: parametrize the target model in muP, tune the HP indirectly on a smaller model, and zero-shot transfer them to the full-sized model, i.e., without directly tuning the latter at all. We verify muTransfer on Transformer and ResNet. For example, 1) by transferring pretraining HPs from a model of 13M parameters, we outperform published numbers of BERT-large (350M parameters), with a total tuning cost equivalent to pretraining BERT-large once; 2) by transferring from 40M parameters, we outperform published numbers of the 6.7B GPT-3 model, with tuning cost only 7% of total pretraining cost. A Pytorch implementation of our technique can be found at github.com/microsoft/mup and installable via `pip install mup`.


- Learning rate schedule (e.g., cosine, WSD)

[[Loshchilov+ 2016]](https://arxiv.org/pdf/1608.03983.pdf)

SGDR: Stochastic Gradient Descent with Warm Restarts

Ilya Loshchilov, Frank Hutter

2016-08-13

Restart techniques are common in gradient-free optimization to deal with multimodal functions. Partial warm restarts are also gaining popularity in gradient-based optimization to improve the rate of convergence in accelerated gradient schemes to deal with ill-conditioned functions. In this paper, we propose a simple warm restart technique for stochastic gradient descent to improve its anytime performance when training deep neural networks. We empirically study its performance on the CIFAR-10 and CIFAR-100 datasets, where we demonstrate new state-of-the-art results at 3.14% and 16.21%, respectively. We also demonstrate its advantages on a dataset of EEG recordings and on a downsampled version of the ImageNet dataset. Our source code is available at https://github.com/loshchil/SGDR

[[Hu+ 2024]](https://arxiv.org/pdf/2404.06395.pdf)

MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies

[Tsinghua] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui... (15 more)... Chao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, Maosong Sun

2024-04-09

The burgeoning interest in developing Large Language Models (LLMs) with up to trillion parameters has been met with concerns regarding resource efficiency and practical expense, particularly given the immense cost of experimentation. This scenario underscores the importance of exploring the potential of Small Language Models (SLMs) as a resource-efficient alternative. In this context, we introduce MiniCPM, specifically the 1.2B and 2.4B non-embedding parameter variants, not only excel in their respective categories but also demonstrate capabilities on par with 7B-13B LLMs. While focusing on SLMs, our approach exhibits scalability in both model and data dimensions for future LLM research. Regarding model scaling, we employ extensive model wind tunnel experiments for stable and optimal scaling. For data scaling, we introduce a Warmup-Stable-Decay (WSD) learning rate scheduler (LRS), conducive to continuous training and domain adaptation. We present an in-depth analysis of the intriguing training dynamics that occurred in the WSD LRS. With WSD LRS, we are now able to efficiently study data-model scaling law without extensive retraining experiments on both axes of model and data, from which we derive the much higher compute optimal data-model ratio than Chinchilla Optimal. Additionally, we introduce MiniCPM family, including MiniCPM-DPO, MiniCPM-MoE and MiniCPM-128K, whose excellent performance further cementing MiniCPM's foundation in diverse SLM applications. MiniCPM models are available publicly at https://github.com/OpenBMB/MiniCPM.


- Regularization (e.g., dropout, weight decay)


- Batch size (e.g., critical batch size)

[[McCandlish+ 2018]](https://arxiv.org/pdf/1812.06162.pdf)

An Empirical Model of Large-Batch Training

Sam McCandlish, Jared Kaplan, Dario Amodei, OpenAI Dota Team

2018-12-14

In an increasing number of domains it has been demonstrated that deep learning models can be trained using relatively large batch sizes without sacrificing data efficiency. However the limits of this massive data parallelism seem to differ from domain to domain, ranging from batches of tens of thousands in ImageNet to batches of millions in RL agents that play the game Dota 2. To our knowledge there is limited conceptual understanding of why these limits to batch size differ or how we might choose the correct batch size in a new domain. In this paper, we demonstrate that a simple and easy-to-measure statistic called the gradient noise scale predicts the largest useful batch size across many domains and applications, including a number of supervised learning datasets (MNIST, SVHN, CIFAR-10, ImageNet, Billion Word), reinforcement learning domains (Atari and Dota), and even generative model training (autoencoders on SVHN). We find that the noise scale increases as the loss decreases over a training run and depends on the model size primarily through improved model performance. Our empirically-motivated theory also describes the tradeoff between compute-efficiency and time-efficiency, and provides a rough model of the benefits of adaptive batch-size training.

Introduced critical batch size


- MoE specific: load balancing (e.g., aux-free)

[[Wang+ 2024]](https://arxiv.org/abs/2408.15664)

Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts

Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, Damai Dai

2024-08-28

For Mixture-of-Experts (MoE) models, an unbalanced expert load will lead to routing collapse or increased computational overhead. Existing methods commonly employ an auxiliary loss to encourage load balance, but a large auxiliary loss will introduce non-negligible interference gradients into training and thus impair the model performance. In order to control load balance while not producing undesired gradients during training, we propose Loss-Free Balancing, featured by an auxiliary-loss-free load balancing strategy. To be specific, before the top-K routing decision, Loss-Free Balancing will first apply an expert-wise bias to the routing scores of each expert. By dynamically updating the bias of each expert according to its recent load, Loss-Free Balancing can consistently maintain a balanced distribution of expert load. In addition, since Loss-Free Balancing does not produce any interference gradients, it also elevates the upper bound of model performance gained from MoE training. We validate the performance of Loss-Free Balancing on MoE models with up to 3B parameters trained on up to 200B tokens. Experimental results show that Loss-Free Balancing achieves both better performance and better load balance compared with traditional auxiliary-loss-controlled load balancing strategies.

[[DeepSeek-AI+ 2024]](https://arxiv.org/pdf/2412.19437.pdf)

DeepSeek-V3 Technical Report

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang... (190 more)... Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, Zizheng Pan

2024-12-27

We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token. To achieve efficient inference and cost-effective training, DeepSeek-V3 adopts Multi-head Latent Attention (MLA) and DeepSeekMoE architectures, which were thoroughly validated in DeepSeek-V2. Furthermore, DeepSeek-V3 pioneers an auxiliary-loss-free strategy for load balancing and sets a multi-token prediction training objective for stronger performance. We pre-train DeepSeek-V3 on 14.8 trillion diverse and high-quality tokens, followed by Supervised Fine-Tuning and Reinforcement Learning stages to fully harness its capabilities. Comprehensive evaluations reveal that DeepSeek-V3 outperforms other open-source models and achieves performance comparable to leading closed-source models. Despite its excellent performance, DeepSeek-V3 requires only 2.788M H800 GPU hours for its full training. In addition, its training process is remarkably stable. Throughout the entire training process, we did not experience any irrecoverable loss spikes or perform any rollbacks. The model checkpoints are available at https://github.com/deepseek-ai/DeepSeek-V3.



## Assignment 1 (basics)


[[GitHub]](https://github.com/stanford-cs336/assignment1-basics)

GitHub

[[PDF]](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2026_assignment1_basics.pdf)

PDF


- Implement BPE tokenizer


- Implement Transformer, cross-entropy loss, AdamW optimizer, training loop


- Do resource accounting


- Train on TinyStories and OpenWebText


- Leaderboard: minimize OpenWebText perplexity given 45 minutes on a B200

[[last year's leaderboard]](https://github.com/stanford-cs336/spring2025-assignment1-basics-leaderboard)

last year's leaderboard



High-level principle: everything is about balancing the following:


- Expressivity (can represent complex dependencies in the data)


- Stability (keep parameter and gradient norms in goldilocks zone)


- Efficiency (runs fast on hardware, both training and inference)




def systems():


Goal: squeeze the most out of the hardware (GPU or TPU)


Components: kernels, parallelism, inference



## Basics


- Resource accounting: memory and compute characteristics of a model


total_flops = 6 * 70e9 * 1e12 # Training 70B parameters on 1T tokens = 4.2e23 FLOPs


![Image](./Trace - lecture_01_files/compute-memory.png)


- Model parameters must be moved from memory (HBM) to the compute (SMs)


- Example: B200 can perform 2.25 PFLOP/sec (bf16) with 8TB/sec memory bandwidth


- Roofline analysis: understand whether we're compute-bound or memory-bound


- Benchmarking and profiling (nsight): see what happens in practice



[DGX B200](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html):


![Image](./Trace - lecture_01_files/image-d41d89a3b5f61b2597e9a608032479f0-https_docs_nvidia_com_dgx_dgxb200-user-guide__images_dgx-b200-system-topology_png)



## Kernels


- Kernel is a function that runs on GPU


- When using PyTorch, each primitive operation launches a standard kernel


- Can write custom kernels to make GPUs go brrr


- Principle: organize computation to minimize data movement


- Naive: read HBM; compute A; write HBM; read HBM; compute B; write HBM


- Fused: read HBM; compute A and B; write HBM


- Strategies: operator fusion (matmul + activation), tiling (FlashAttention)


- Warp divergence, memory coalescing, bank conflicts, occupancy, bulk-async memory transfers


- Write kernels in CUDA/**Triton**/CUTLASS/ThunderKittens



## Parallelism


- What if we have 1024 GPUs?


- Data movement between GPUs is even slower, but same 'minimize data movement' principle holds


- Use classic collective operations (e.g., gather, reduce, all-reduce)


- Shard memory (parameters, activations, gradients, optimizer states) across GPUs


- How to split computation: {data,tensor,pipeline,sequence,expert} parallelism



## Inference


Goal: generate tokens given a prompt (needed to actually use models!)


Inference is also needed for reinforcement learning, test-time compute, evaluation


Two phases: prefill and decode


![Image](./Trace - lecture_01_files/prefill-decode.png)


- Prefill (similar to training): tokens are given, can process all at once (compute-bound)


- Decode: need to generate one token at a time (memory-bound)


Methods to speed up decoding:


- Use cheaper model (via model pruning, quantization, distillation)


- Speculative decoding: use a cheaper "draft" model to generate multiple tokens, then use the full model to score in parallel (exact decoding!)


- Systems optimizations: fused kernels, continuous batching



## Assignment 2 (systems)


[[GitHub]](https://github.com/stanford-cs336/assignment2-systems)

GitHub

[[PDF from Spring 2025]](https://github.com/stanford-cs336/assignment2-systems/blob/spring2025/cs336_spring2025_assignment2_systems.pdf)

PDF from Spring 2025


- Implement a fused RMSNorm kernel in Triton


- Implement distributed data parallel training


- Implement optimizer state sharding


- Benchmark and profile the implementations



Recommended book: [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)


- Nicely lays out how to approach systems for LLMs conceptually


- From Google, so it foregrounds TPUs, but high-level concepts are similar




def scaling_laws():


Setting: if you had 1e25 FLOPs of compute, what hyperparameters would you use to train a good model?


Too expensive to do hyperparameter tuning at full scale!



Key conceptual shift: instead of a single scale, think of a **scaling recipe** (FLOPs → hyperparameters)


For a scaling recipe:


- Run experiments to compute the loss at various smaller scales (e.g., up to 1e24 FLOPs)


- Fit a scaling law to predict the loss of the scaling recipe at the target scale (e.g., 1e25 FLOPs)



Now you can:


1. Optimize the scaling recipe targeting a larger scale using smaller scale experiments


1. Predict the loss at the target scale before actually running the experiment!


Scaling laws don't happen automatically, they require careful construction of a scaling recipe.


Parameterize the model in a way to get **hyperparameter transfer**

[[Yang+ 2022]](https://arxiv.org/abs/2203.03466)

Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer

Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao

2022-03-07

Hyperparameter (HP) tuning in deep learning is an expensive process, prohibitively so for neural networks (NNs) with billions of parameters. We show that, in the recently discovered Maximal Update Parametrization (muP), many optimal HPs remain stable even as model size changes. This leads to a new HP tuning paradigm we call muTransfer: parametrize the target model in muP, tune the HP indirectly on a smaller model, and zero-shot transfer them to the full-sized model, i.e., without directly tuning the latter at all. We verify muTransfer on Transformer and ResNet. For example, 1) by transferring pretraining HPs from a model of 13M parameters, we outperform published numbers of BERT-large (350M parameters), with a total tuning cost equivalent to pretraining BERT-large once; 2) by transferring from 40M parameters, we outperform published numbers of the 6.7B GPT-3 model, with tuning cost only 7% of total pretraining cost. A Pytorch implementation of our technique can be found at github.com/microsoft/mup and installable via `pip install mup`.


Predictability is at least as important as optimality!



Question: given a FLOPs budget (C = 6 N D), use a bigger model (N) or train on more tokens (D)?


Classic compute-optimal scaling laws:

[[Kaplan+ 2020]](https://arxiv.org/pdf/2001.08361.pdf)

Scaling Laws for Neural Language Models

[OpenAI] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei

2020-01-23

We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude. Other architectural details such as network width or depth have minimal effects within a wide range. Simple equations govern the dependence of overfitting on model/dataset size and the dependence of training speed on model size. These relationships allow us to determine the optimal allocation of a fixed compute budget. Larger models are significantly more sample-efficient, such that optimally compute-efficient training involves training very large models on a relatively modest amount of data and stopping significantly before convergence.

Vary model size, dataset size, compute; get power laws

Larger models require fewer tokens

[[Hoffmann+ 2022]](https://arxiv.org/pdf/2203.15556.pdf)

Training Compute-Optimal Large Language Models

[DeepMind] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai... (12 more)... Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre

2022-03-29

We investigate the optimal model size and number of tokens for training a transformer language model under a given compute budget. We find that current large language models are significantly undertrained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant. By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. We test this hypothesis by training a predicted compute-optimal model, Chinchilla, that uses the same compute budget as Gopher but with 70B parameters and 4

×

 more more data. Chinchilla uniformly and significantly outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B) on a large range of downstream evaluation tasks. This also means that Chinchilla uses substantially less compute for fine-tuning and inference, greatly facilitating downstream usage. As a highlight, Chinchilla reaches a state-of-the-art average accuracy of 67.5% on the MMLU benchmark, greater than a 7% improvement over Gopher.

Introduced the rigorous analysis scaling laws for language models

Key improvement over Kaplan: tune learning rate for the compute budget

Approach 1: for each model size, train with 4 learning rates, vary number of training tokens, fit lower envelope

Approach 2 (IsoFLOP): for each model size, train with 9 training budgets, take last point

Approach 3: fit parametric function L(N, D) = E + A/N^alpha + B/D^beta to data collected from approaches 1 and 2

Conclusion: model and data should scale up at same rate

Table 3: extrapolate to 10 trillion parameters

MassiveText, different data distribution (1.5 trillion tokens)

70B parameters


- ISOFLOP curves: for multiple small FLOPs budgets, find optimal N


- Then fit a scaling law to extrapolate to large FLOPs budgets


![Image](./Trace - lecture_01_files/chinchilla-isoflop.png)


TL;DR: D = 20 N is roughly optimal (e.g., 70B parameter model should be trained on ~1.4T tokens)


Caveat: this doesn't take into account inference costs (want a smaller model)



Live example from Marin

[[post]](https://x.com/percyliang/status/2034367256277533100)

post


![Image](./Trace - lecture_01_files/image-49c56e974c417eacabafcd41ab0a90f0-https_pbs_twimg_com_media_HDuErvvbsAAQ5Yt_format_jpg_name_4096x4096)


Should be done training this week, should see how well we match the preregistered loss!



## Assignment 3 (scaling laws)


[[GitHub]](https://github.com/stanford-cs336/assignment3-scaling)

GitHub

[[PDF from Spring 2025]](https://github.com/stanford-cs336/assignment3-scaling/blob/master/cs336_spring2025_assignment3_scaling.pdf)

PDF from Spring 2025


- We define a training API (hyperparameters → loss) based on previous runs


- Submit "training jobs" (under a FLOPs budget) and gather data points


- Fit scaling laws to the data points


- Submit extrapolated hyperparameters and loss predictions


- Leaderboard: minimize loss given FLOPs budget




def data():


Question: What capabilities do we want the model to have?


Multilingual? Good at conversation? Agentic coding capabilities?



## Evaluation


What is the purpose of evaluation?


1. Internal: guide model development (smoothness across scales, relative performance matters)


1. External: measure absolute quality of a real use case (ecological validity matters)


Examples of evaluations:


1. Perplexity: ideally run on private documents not on Internet (avoid contamination)


1. Advanced use cases: GPQA, HLE, SWE-Bench, Terminal-Bench


LMs are general purpose, require a diverse set of evaluations!



## Data curation


- Data does not just fall from the sky.


- Sources: webpages crawled from the Internet, books, arXiv papers, GitHub code, etc.


![Image](./Trace - lecture_01_files/image-b3aebfa83a900cd491e70acf27806db3-https_ar5iv_labs_arxiv_org_html_2101_00027_assets_pile_chart2_png)


- Appeal to fair use to train on copyright data?

[[Henderson+ 2023]](https://arxiv.org/pdf/2303.15715.pdf)

Foundation Models and Fair Use

Peter Henderson, Xuechen Li, Dan Jurafsky, Tatsunori Hashimoto, Mark A. Lemley, Percy Liang

2023-03-28

Existing foundation models are trained on copyrighted material. Deploying these models can pose both legal and ethical risks when data creators fail to receive appropriate attribution or compensation. In the United States and several other countries, copyrighted content may be used to build foundation models without incurring liability due to the fair use doctrine. However, there is a caveat: If the model produces output that is similar to copyrighted data, particularly in scenarios that affect the market of that data, fair use may no longer apply to the output of the model. In this work, we emphasize that fair use is not guaranteed, and additional work may be necessary to keep model development and deployment squarely in the realm of fair use. First, we survey the potential risks of developing and deploying foundation models based on copyrighted content. We review relevant U.S. case law, drawing parallels to existing and potential applications for generating text, source code, and visual art. Experiments confirm that popular foundation models can generate content considerably similar to copyrighted material. Second, we discuss technical mitigations that can help foundation models stay in line with fair use. We argue that more research is needed to align mitigation strategies with the current state of the law. Lastly, we suggest that the law and technical mitigations should co-evolve. For example, coupled with other policy mechanisms, the law could more explicitly consider safe harbors when strong technical tools are used to mitigate infringement harms. This co-evolution may help strike a balance between intellectual property and innovation, which speaks to the original goal of fair use. But we emphasize that the strategies we describe here are not a panacea and more work is needed to develop policies that address the potential harms of foundation models.


- Might have to license data (e.g., Google with Reddit data)

[[article]](https://www.reuters.com/technology/reddit-ai-content-licensing-deal-with-google-sources-say-2024-02-22/)

article


- Raw data is HTML, PDF, directories (not text), requires processing



## Data processing


- Transformation: convert HTML/PDF to text (extract main content)


- Filtering: keep high quality data, remove harmful content (via classifiers)


- Deduplication: save compute, avoid memorization; use Bloom filters or MinHash


- Data mixing: how much to upweight/downweight each source?

[[Liu+ 2024]](https://arxiv.org/abs/2407.01492)

RegMix: Data Mixture as Regression for Language Model Pre-training

Qian Liu, Xiaosen Zheng, Niklas Muennighoff, Guangtao Zeng, Longxu Dou, Tianyu Pang, Jing Jiang, Min Lin

2024-07-01

The data mixture for large language model pre-training significantly impacts performance, yet how to determine an effective mixture remains unclear. We propose RegMix to automatically identify a high-performing data mixture by formulating it as a regression task. RegMix trains many small models on diverse data mixtures, uses regression to predict performance of unseen mixtures, and applies the best predicted mixture to train a large-scale model with orders of magnitude more compute. To empirically validate RegMix, we train 512 models with 1M parameters for 1B tokens to fit the regression model and predict the best data mixture. Using this mixture we train a 1B parameter model for 25B tokens (i.e. 1000x larger and 25x longer) which we find performs best among 64 candidate 1B parameter models with other mixtures. Furthermore, RegMix consistently outperforms human selection in experiments involving models up to 7B models trained on 100B tokens, while matching or exceeding DoReMi using just 10% of the computational resources. Our experiments also show that (1) Data mixtures significantly impact performance; (2) Web corpora rather than data perceived as high-quality like Wikipedia have the strongest positive correlation with downstream performance; (3) Domains interact in complex ways often contradicting common sense, thus automatic approaches like RegMix are needed; (4) Data mixture effects transcend scaling laws. Our code is available at https://github.com/sail-sg/regmix.

[[Chen+ 2026]](https://arxiv.org/abs/2602.12237)

Olmix: A Framework for Data Mixing Throughout LM Development

Mayee F. Chen, Tyler Murray, David Heineman, Matt Jordan, Hannaneh Hajishirzi, Christopher Ré, Luca Soldaini, Kyle Lo

2026-02-12

Data mixing -- determining the ratios of data from different domains -- is a first-order concern for training language models (LMs). While existing mixing methods show promise, they fall short when applied during real-world LM development. We present Olmix, a framework that addresses two such challenges. First, the configuration space for developing a mixing method is not well understood -- design choices across existing methods lack justification or consensus and overlook practical issues like data constraints. We conduct a comprehensive empirical study of this space, identifying which design choices lead to a strong mixing method. Second, in practice, the domain set evolves throughout LM development as datasets are added, removed, partitioned, and revised -- a problem setting largely unaddressed by existing works, which assume fixed domains. We study how to efficiently recompute the mixture after the domain set is updated, leveraging information from past mixtures. We introduce mixture reuse, a mechanism that reuses existing ratios and recomputes ratios only for domains affected by the update. Over a sequence of five domain-set updates mirroring real-world LM development, mixture reuse matches the performance of fully recomputing the mix after each update with 74% less compute and improves over training without mixing by 11.6% on downstream tasks.


- Rewriting / synthetic data: use LM to augment real data, more similar to downstream tasks

[[Maini+ 2024]](https://arxiv.org/abs/2401.16380)

Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling

Pratyush Maini, Skyler Seto, He Bai, David Grangier, Yizhe Zhang, Navdeep Jaitly

2024-01-29

Large language models are trained on massive scrapes of the web, which are often unstructured, noisy, and poorly phrased. Current scaling laws show that learning from such data requires an abundance of both compute and data, which grows with the size of the model being trained. This is infeasible both because of the large compute costs and duration associated with pre-training, and the impending scarcity of high-quality data on the web. In this work, we propose Web Rephrase Augmented Pre-training (

WRAP

) that uses an off-the-shelf instruction-tuned model prompted to paraphrase documents on the web in specific styles such as "like Wikipedia" or in "question-answer format" to jointly pre-train LLMs on real and synthetic rephrases. First, we show that using WRAP on the C4 dataset, which is naturally noisy, speeds up pre-training by

∼3x. At the same pre-training compute budget, it improves perplexity by more than 10% on average across different subsets of the Pile, and improves zero-shot question answer accuracy across 13 tasks by more than 2%. Second, we investigate the impact of the re-phrasing style on the performance of the model, offering insights into how the composition of the training data can impact the performance of LLMs in OOD settings. Our gains are attributed to the fact that re-phrased synthetic data has higher utility than just real data because it (i) incorporates style diversity that closely reflects downstream evaluation style, and (ii) has higher 'quality' than web-scraped data.



Types of data:


- Pretraining data: large and diverse


- Mid-training data: high quality, including long-context


- Post-training data: supervised fine-tuning (conversations, agentic traces with tool calling)



## Assignment 4 (data)


[[GitHub]](https://github.com/stanford-cs336/assignment4-data)

GitHub

[[PDF from Spring 2025]](https://github.com/stanford-cs336/assignment4-data/blob/spring2025/cs336_spring2025_assignment4_data.pdf)

PDF from Spring 2025


- Convert Common Crawl HTML to text


- Train classifiers to filter for quality and harmful content


- Deduplication using MinHash


- Leaderboard: minimize perplexity given token budget




def alignment():


So far, we have trained a model on full supervision (predict the next token).


Now that the model should be reasonable, we can improve it further from **weak supervision**.


Why weak supervision? When it is easier to critique than to generate.



Basic template:


1. Generate responses from the model.


1. Score responses with a {human, verifier, LM judge}.


1. Update the model to prefer better responses.



Algorithms:


- Proximal Policy Optimization (PPO) from reinforcement learning

[[Schulman+ 2017]](https://arxiv.org/pdf/1707.06347.pdf)

Proximal Policy Optimization Algorithms

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov

2017-07-20

We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a "surrogate" objective function using stochastic gradient ascent. Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates. The new methods, which we call proximal policy optimization (PPO), have some of the benefits of trust region policy optimization (TRPO), but they are much simpler to implement, more general, and have better sample complexity (empirically). Our experiments test PPO on a collection of benchmark tasks, including simulated robotic locomotion and Atari game playing, and we show that PPO outperforms other online policy gradient methods, and overall strikes a favorable balance between sample complexity, simplicity, and wall-time.

Introduced PPO (for RL)

[[Ouyang+ 2022]](https://arxiv.org/pdf/2203.02155.pdf)

Training language models to follow instructions with human feedback

[OpenAI] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright... (10 more)... Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe

2022-03-04

Making language models bigger does not inherently make them better at following a user's intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning. We then collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback. We call the resulting models InstructGPT. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, our results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.

Training language models to follow instructions with human feedback


- Direct Policy Optimization (DPO): for preference data, simpler

[[Rafailov+ 2023]](https://arxiv.org/pdf/2305.18290.pdf)

Direct Preference Optimization: Your Language Model is Secretly a Reward Model

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn

2023-05-29

While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training. Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF). However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model. In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss. The resulting algorithm, which we call Direct Preference Optimization (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning. Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.


- Group Relative Preference Optimization (GRPO): remove value function

[[Shao+ 2024]](https://arxiv.org/pdf/2402.03300.pdf)

DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song... (1 more)... Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo

2024-02-05

Mathematical reasoning poses a significant challenge for language models due to its complex and structured nature. In this paper, we introduce DeepSeekMath 7B, which continues pre-training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data. DeepSeekMath 7B has achieved an impressive score of 51.7% on the competition-level MATH benchmark without relying on external toolkits and voting techniques, approaching the performance level of Gemini-Ultra and GPT-4. Self-consistency over 64 samples from DeepSeekMath 7B achieves 60.9% on MATH. The mathematical reasoning capability of DeepSeekMath is attributed to two key factors: First, we harness the significant potential of publicly available web data through a meticulously engineered data selection pipeline. Second, we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO.



Challenges:


- RL algorithms are unstable and hard to tune


- At scale, this requires a lot of new infrastructure (inference with async rollouts)


- Constantly trading off systems efficiency and on-policyness



## Assignment 5 (alignment)


[[GitHub]](https://github.com/stanford-cs336/assignment5-alignment)

GitHub

[[PDF from Spring 2025]](https://github.com/stanford-cs336/assignment5-alignment/blob/spring2025/cs336_spring2025_assignment5_alignment.pdf)

PDF from Spring 2025


- Implement Direct Preference Optimization (DPO)


- Implement Group Relative Preference Optimization (GRPO)




############################################################


# Tokenization



def tokenization():


This unit was inspired by Andrej Karpathy's video on tokenization; check it out!

[[video]](https://www.youtube.com/watch?v=zduSFxRajkE)

video



intro_to_tokenization()


tokenization_examples()


character_tokenizer()


byte_tokenizer()


word_tokenizer()


bpe_tokenizer()



Summary:


- Tokenizer: strings ↔ tokens (indices)


- Character-based, byte-based, word-based tokenization are highly suboptimal


- BPE is an effective heuristic that is data-driven


- Tokenization is a separate step, maybe one day do it end-to-end from bytes...



But whatever solution needs to satisfy:


1. Model (e.g., Transformer) should operate on chunks (abstractions) of the sequence (text, video, DNA, etc.)


1. Chunks should be variable (allocate more model capacity to interesting chunks)




class CharacterTokenizer(Tokenizer):


"""Represent a string as a sequence of Unicode code points."""


def encode(self, string: str) -> list[int]:


return list(map(ord, string))



def decode(self, indices: list[int]) -> str:


return "".join(map(chr, indices))




class ByteTokenizer(Tokenizer):


"""Represent a string as a sequence of bytes."""


def encode(self, string: str) -> list[int]:


string_bytes = string.encode("utf-8")


indices = list(map(int, string_bytes))


return indices



def decode(self, indices: list[int]) -> str:


string_bytes = bytes(indices)


string = string_bytes.decode("utf-8")


return string




def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:


"""Return `indices`, but with all instances of `pair` replaced with `new_index`."""


new_indices = []


i = 0


while i < len(indices):


if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:


new_indices.append(new_index)


i += 2


else:


new_indices.append(indices[i])


i += 1


return new_indices




@dataclass(frozen=True)


class BPETokenizerParams:


"""All you need to specify a BPETokenizer."""


vocab: dict[int, bytes] # index -> bytes


merges: dict[tuple[int, int], int] # index1,index2 -> new_index





class BPETokenizer(Tokenizer):


"""BPE tokenizer given a set of merges and a vocabulary."""


def __init__(self, params: BPETokenizerParams):


self.params = params



def encode(self, string: str) -> list[int]:


indices = list(map(int, string.encode("utf-8")))


# Note: this is a very slow implementation


for pair, new_index in self.params.merges.items():


indices = merge(indices, pair, new_index)


return indices



def decode(self, indices: list[int]) -> str:


bytes_list = list(map(self.params.vocab.get, indices))


string = b"".join(bytes_list).decode("utf-8")


return string




def get_compression_ratio(string: str, indices: list[int]) -> float:


"""Given `string` that has been tokenized into `indices`, return the number of UTF-8 bytes per token.."""


num_bytes = len(bytes(string, encoding="utf-8"))


num_tokens = len(indices)


return num_bytes / num_tokens




def get_gpt5_tokenizer():


# Code: https://github.com/openai/tiktoken


return tiktoken.get_encoding("o200k_base")




def intro_to_tokenization():


Raw text is generally represented as Unicode strings.


string = "Hello, 🌍! 你好!"



A language model places a probability distribution over sequences of tokens (usually represented by integer indices).


indices = [15496, 11, 995, 0]



So we need a procedure that *encodes* strings into tokens.


We also need a procedure that *decodes* tokens back into strings.


A

[Tokenizer](https://cs336.stanford.edu/lectures?trace=lecture_01&showNotes=1&source=lecture_01.py&line=1#)

is a class that implements the encode and decode methods.




def tokenization_examples():


To get a feel for how tokenizers work, play with this

[[interactive site]](https://tiktokenizer.vercel.app/?encoder=gpt2)

interactive site



## Observations


- A word and its preceding space are part of the same token (e.g., " world").


- A word at the beginning and in the middle are represented differently (e.g., "hello hello").


- Numbers are tokenized into every few digits.



Here's the GPT-5 tokenizer from OpenAI (tiktoken) in action.


tokenizer = get_gpt5_tokenizer()


string = "Hello, 🌍! 你好!"



Check that encode() and decode() roundtrip:


indices = tokenizer.encode(string)


reconstructed_string = tokenizer.decode(indices)


assert string == reconstructed_string



Compression ratio: number of bytes per token


compression_ratio = get_compression_ratio(string, indices)


The larger the compression ratio, the shorter the sequence (good since attention is quadratic in sequence length).


One could increase compression ratio by increasing **vocabulary size** (number of possible token values increases), leading to sparsity.


vocabulary_size = tokenizer.n_vocab



Let's take a look at the actual vocabulary:

[[vocab]](https://github.com/stanford-cs336/lectures/blob/main/var/gpt5_tokenizer_vocab.txt)

vocab


output_tokenizer(tokenizer, "var/gpt5_tokenizer_vocab.txt")




def output_tokenizer(tokenizer, path: str):


"""Write out the vocabulary of `tokenizer` to `path`, one per line."""


if not os.path.exists(path):


vocab = [b.decode("utf-8", errors="replace") for b in tokenizer.token_byte_values()]


with open(path, "w") as f:


for token in vocab:


f.write(token + "\n")




def character_tokenizer():


A Unicode string is a sequence of Unicode characters.


Each character can be converted into a code point (integer) via `ord`.


assert ord("a") == 97


assert ord("🌍") == 127757


It can be converted back via `chr`.


assert chr(97) == "a"


assert chr(127757) == "🌍"



Now let's build a `Tokenizer` and make sure it round-trips:


tokenizer = CharacterTokenizer()


string = "Hello, 🌍! 你好!"


indices = tokenizer.encode(string) # call ord


reconstructed_string = tokenizer.decode(indices) # call chr


assert string == reconstructed_string



There are approximately 150K Unicode characters.

[[Wikipedia]](https://en.wikipedia.org/wiki/List_of_Unicode_characters)

Wikipedia


vocabulary_size = max(indices) + 1 # This is a lower bound


Problem 1: this is a very large vocabulary.


Problem 2: many characters are quite rare (e.g., 🌍), which is inefficient use of the vocabulary.


compression_ratio = get_compression_ratio(string, indices)


This tokenizer is the worst of both worlds (large vocabulary, low compression ratio).




def byte_tokenizer():


Unicode strings can be represented as a sequence of bytes, which can be represented by integers between 0 and 255.


The most common Unicode encoding is

[[UTF-8]](https://en.wikipedia.org/wiki/UTF-8)

UTF-8



Some Unicode characters are represented by one byte:


assert bytes("a", encoding="utf-8") == b"a"


Others take multiple bytes:


assert bytes("🌍", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"



Now let's build a `Tokenizer` and make sure it round-trips:


tokenizer = ByteTokenizer()


string = "Hello, 🌍! 你好!"


indices = tokenizer.encode(string)


reconstructed_string = tokenizer.decode(indices)


assert string == reconstructed_string



The vocabulary is nice and small: a byte can represent 256 values.


vocabulary_size = 256


What about the compression rate?


compression_ratio = get_compression_ratio(string, indices)


assert compression_ratio == 1


The compression ratio is terrible, which means the sequences will be too long.


Given that the context length of a Transformer is limited (since attention is quadratic), this is not looking great...




def word_tokenizer():


Another approach (closer to what was done classically in NLP) is to split strings into words.


string = "I'll say supercalifragilisticexpialidocious!"



chunks = regex.findall(r"\w+|.", string)


This regular expression keeps all alphanumeric characters together (words).



To turn this into a `Tokenizer`, we need to map these chunks into integers.


Then, we can build a mapping from each chunk into an integer.



What's good: each token is meaningful (since humans invented words).



vocabulary_size = "Number of distinct chunks in the training data"


compression_ratio = get_compression_ratio(string, chunks)


Compression ratio is good, but vocabulary size can be huge.



Moreover:


- Many words are rare and the model won't learn much about them.


- This doesn't obviously provide a fixed vocabulary size.


- New words we haven't seen during training get a special UNK token, which is ugly and can mess up perplexity calculations.




def bpe_tokenizer():


## Byte Pair Encoding (BPE)


The BPE algorithm was introduced by Philip Gage in 1994 for data compression.

[[article]](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM)

article


It was adapted to NLP for neural machine translation.

[[Sennrich+ 2015]](https://arxiv.org/abs/1508.07909)

Neural Machine Translation of Rare Words with Subword Units

Rico Sennrich, Barry Haddow, Alexandra Birch

2015-08-31

Neural machine translation (NMT) models typically operate with a fixed vocabulary, but translation is an open-vocabulary problem. Previous work addresses the translation of out-of-vocabulary words by backing off to a dictionary. In this paper, we introduce a simpler and more effective approach, making the NMT model capable of open-vocabulary translation by encoding rare and unknown words as sequences of subword units. This is based on the intuition that various word classes are translatable via smaller units than words, for instance names (via character copying or transliteration), compounds (via compositional translation), and cognates and loanwords (via phonological and morphological transformations). We discuss the suitability of different word segmentation techniques, including simple character n-gram models and a segmentation based on the byte pair encoding compression algorithm, and empirically show that subword models improve over a back-off dictionary baseline for the WMT 15 translation tasks English-German and English-Russian by 1.1 and 1.3 BLEU, respectively.


(Previously, papers had been using word-based tokenization.)


BPE was then used by GPT-2.

[[Radford+ 2019]](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Language Models are Unsupervised Multitask Learners

[OpenAI] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever

2019-02-14

1.5B parameters

Pioneered stage release



Basic idea: *train* the tokenizer on raw text to construct a vocabulary tailored to the data.


Intuition: common sequences of bytes are represented by a single token, rare sequences are represented by many tokens.



Sketch: start with each byte as a token, and successively merge the most common pair of adjacent tokens.



## Training the tokenizer


string = "the cat in the hat"


params = train_bpe(string, num_merges=3)



## Using the tokenizer


Now, given a new text, we can encode it.


tokenizer = BPETokenizer(params)


string = "the quick brown fox"


indices = tokenizer.encode(string)


reconstructed_string = tokenizer.decode(indices)


assert string == reconstructed_string



In Assignment 1, you will go beyond this in the following ways:


- encode() currently loops over all merges. Only loop over merges that matter.


- Detect and preserve special tokens (e.g., <|endoftext|>).


- Use pre-tokenization (e.g., the GPT-2 tokenizer regex).


- Try to make the implementation as fast as possible.




def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:


Start with the list of bytes of `string`.


indices = list(map(int, string.encode("utf-8")))


merges: dict[tuple[int, int], int] = {} # index1, index2 => merged index


vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)} # index -> bytes



for i in range(num_merges):


# Count the number of occurrences of each pair of tokens


counts = count_adjacent_pairs(indices)



# Find the most common pair


pair = max(counts, key=counts.get)



# Merge that pair


new_index = 256 + i


merges[pair] = new_index


vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]


indices = merge(indices, pair, new_index)



compression_ratio = get_compression_ratio(string, indices)



return BPETokenizerParams(vocab=vocab, merges=merges)




def count_adjacent_pairs(indices: list[int]) -> dict[tuple[int, int], int]:


"""Return a dictionary mapping each adjacent pair of tokens in `indices` to the number of times it occurs."""


counts = defaultdict(int)


for index1, index2 in zip(indices, indices[1:]):


counts[(index1, index2)] += 1


return counts




if __name__ == "__main__":


main()