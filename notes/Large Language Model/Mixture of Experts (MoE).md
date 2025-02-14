# Refers

1. [Adaptive Mixtures of Local Experts | MIT Press Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/6797059)（最早提出MoE架构）
2. [[2006.16668] GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)（Google 推出 GShard）
3. [[1701.06538] Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538)（首次将稀疏门控MoE引入深度学习，应用于LSTM）
4. [[2101.03961] Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)（Google提出新名字switch Transformers）
5. [48237d9f2dea8c74c2a72126cf63d933-Paper.pdf](https://proceedings.neurips.cc/paper/2021/file/48237d9f2dea8c74c2a72126cf63d933-Paper.pdf)（Google提出v-MoE）
6. [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://proceedings.mlr.press/v162/rajbhandari22a/rajbhandari22a.pdf)
7. [[2401.06066\] DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066)
8. [deepseek-ai/DeepSeek-VL2: DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding](https://github.com/deepseek-ai/DeepSeek-VL2)



# Core Concepts

## Experts

专家网络

每个专家是一个独立的子网络（通常是 FFN），在实际计算中只有部分专家会被激活参与处理。通过让多个专家分担不同数据子集的计算，模型在预训练时可以以较低的计算开销获得大参数量带来的表示能力。

在 DeepSeek‑v3等MOE大模型中，正是通过这种将 FFN 层替换为 MOE 层的设计，模型在拥有海量参数的同时，其实际计算量却与传统稠密模型相当，从而实现了高效预训练和快速推理。

## Router / Gating 

门控网络或者称为路由

该模块负责根据输入 token 的特征动态选择激活哪些专家。门控网络一般采用一个带 softmax 的简单前馈网络来计算每个专家的权重。经过训练后，门控网络会逐步学会将相似的输入路由到表现更好的专家。

## knowledge hybridity

- **定义**：知识混合是指在现有的MoE架构中，由于专家数量有限，分配给特定专家的token可能会涵盖多种不同的知识类型。这导致专家需要在其参数中整合不同类型的知识，这些知识可能难以同时有效地利用。
- **影响**：知识混合会降低专家的专业化程度，因为专家需要处理多种类型的知识，难以专注于特定领域的知识。这会限制模型的性能，使其无法达到理论上的上限性能。

> 来自 DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models

## knowledge redundancy

- **定义**：知识冗余是指在现有的MoE架构中，不同专家可能会学习到相同的或相似的知识。这导致多个专家的参数中存在重复的知识，从而浪费了模型的参数容量。
- **影响**：知识冗余会降低模型的效率和性能，因为多个专家学习到相同的知识，无法充分利用模型的参数容量来学习更多样化的知识。这也会限制专家的专业化程度，进一步影响模型的整体性能。

>  来自 DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models

## Expert Specialization

- **定义**：专家专业化是指每个专家专注于特定类型的知识或任务，从而提高模型的性能和效率。专家专业化是MoE架构的一个重要目标，但现有的MoE架构由于知识混合和知识冗余的问题，难以实现真正的专家专业化。
- **影响**：专家专业化程度的降低会限制模型的性能，使其无法达到理论上的上限性能。为了实现专家专业化，需要解决知识混合和知识冗余的问题，例如通过增加专家数量、优化专家分配策略等方法。

> 来自 DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models

## Activated Parameters

见[[Large Language Model/Metrics|Metrics]]



