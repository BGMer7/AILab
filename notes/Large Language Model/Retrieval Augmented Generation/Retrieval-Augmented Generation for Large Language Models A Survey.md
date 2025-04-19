# Introduction

- **背景知识**：大型语言模型（LLMs）在特定领域或知识密集型任务中面临挑战，如产生幻觉（hallucinations）、知识过时以及推理过程不透明、不可追溯。
- **研究动机**：RAG通过从外部数据库检索相关信息来增强LLMs，提高生成的准确性和可信度，特别适用于知识密集型任务。
- **研究意义**：RAG技术的发展为LLMs在实际应用中的广泛采用提供了关键技术支持，如聊天机器人等。

# Overview of RAG

## Naive RAG

### Indexing

### Retrieval

### Generation

### Notable Drawbacks

#### Retrieval Challenges

#### Generation Difficulties

#### Augmentation Hurdles

相似的信息在多个数据源，会导致检索出很多重复的信息。

## Advanced RAG

>  Advanced RAG refines its indexing techniques through the use of a sliding window approach, fine-grained segmentation, and the incorporation of metadata.

### Pre-Retrieval process

>  make the user’s original question clearer and more suitable for the retrieval task

#### query rewriting

#### query transformation

#### query expansion

### Post-Retrieval process

> The main methods in post-retrieval process include rerank chunks and context compressing.

>  Post-retrieval efforts concentrate on selecting the essential information, emphasizing critical sections, and shortening the context to be processed.

## Modular RAG

Modular RAG是检索增强生成（RAG）技术的最新研究范式，通过模块化设计提高了系统的灵活性和适应性。它不仅继承了Naive RAG和Advanced RAG的基本原理，还通过引入多种功能模块和灵活的交互模式，解决了传统RAG方法在复杂任务中的局限性。

Modular RAG的核心思想是通过模块化设计，使系统能够适应多种任务和场景。它通过引入多个功能模块，如搜索模块、记忆模块、路由模块和预测模块，增强了检索和处理能力。此外，Modular RAG还支持模块的替换和重新配置，以适应特定任务的需求。

- **灵活性**：支持多种模块的组合和替换，适应不同任务和场景。
- **扩展性**：通过新增模块或调整模块交互流程，提高系统的适应性和效率。
- **鲁棒性**：通过模块间的协同工作，提高系统的稳定性和抗噪能力。

### New Modules

#### Search Module

- **功能**：支持多种数据源的直接搜索，如搜索引擎、数据库和知识图谱。
- **应用场景**：适用于需要实时检索最新信息的场景，如新闻问答、动态知识库查询等。

#### Memory Module

- **功能**：利用LLMs的记忆能力，创建无界记忆池，通过迭代自增强提高检索质量。
- **应用场景**：适用于需要处理长上下文或复杂对话的场景，如多轮对话系统。

#### Routing Module

- **功能**：在多种数据源之间导航，选择最优路径。
- **应用场景**：适用于需要整合多种数据源的场景，如跨领域问答系统。

#### Predict Module

>  The Predict module aims to reduce redundancy and noise by generating context directly through the LLM, ensuring relevance and accuracy

- **功能**：通过LLMs生成上下文，减少冗余和噪声。
- **应用场景**：适用于需要生成高质量答案的场景，如专业领域问答。

#### Task Adapter Module

- **功能**：根据任务需求自动调整检索和生成策略。
- **应用场景**：适用于多任务处理场景，如同时处理问答和信息提取任务。

### New Patterns

#### Iterative Retrieval

- **流程**：通过多次检索和生成循环，逐步丰富和细化答案。
- **优点**：能够处理复杂问题，提供更全面的知识基础。
- **应用场景**：适用于需要多步骤推理的场景，如复杂问题解答。

#### Recursive Retrieval

- **流程**：通过逐步细化查询和分解问题，深入挖掘相关信息。
- **优点**：能够处理深度和相关性要求高的任务。
- **应用场景**：适用于需要深入分析的场景，如学术研究辅助。

#### Adaptive Retrieval

- **流程**：使LLMs能够主动决定检索的最佳时机和内容。
- **优点**：提高检索效率，减少不必要的计算开销。
- **应用场景**：适用于动态变化的场景，如实时问答系统。

#### Rewrite-Retrieve-Read

#### hypothetical document embeddings (HyDE)

HyDE（Hypothetical Document Embedding）是一种用于检索增强生成（RAG）的技术，旨在提高检索的准确性和相关性。HyDE通过构建假设文档（hypothetical documents）来优化检索过程，这些假设文档是基于原始查询生成的，用于更好地匹配和检索相关信息。

HyDE的核心思想是通过生成与原始查询相关的假设文档，然后计算这些假设文档与实际文档的嵌入相似性，而不是直接计算原始查询与文档的相似性。这种方法可以更好地捕捉查询的语义，提高检索的准确性和相关性。

**HyDE的工作流程**

1. **查询处理**：将原始查询输入到模型中。
2. **假设文档生成**：使用语言模型（LLM）生成与原始查询相关的假设文档。
3. **嵌入计算**：将生成的假设文档和实际文档转换为嵌入向量。
4. **相似性计算**：计算假设文档嵌入与实际文档嵌入的相似性。
5. **检索**：根据相似性分数检索出最相关的文档。

**HyDE的优势**

- **提高检索精度**：通过生成假设文档，HyDE能够更好地捕捉查询的语义，提高检索的准确性和相关性。
- **减少语义差距**：通过假设文档的生成，HyDE可以缩小查询与文档之间的语义差距，提高检索效果。
- **适应性强**：HyDE可以应用于多种检索任务，适应不同的应用场景。

**HyDE的应用场景**

HyDE特别适用于需要高精度检索的场景，如问答系统、信息提取和复杂问题解答等。它在处理复杂查询和多步骤推理任务时表现出色。

#### Flexible and Adaptive Retrieval Augmented Generation (FLARE)

FLARE（Flexible and Adaptive Retrieval Augmented Generation）是一种自适应检索增强生成方法，旨在通过灵活的检索策略提高RAG（Retrieval-Augmented Generation）系统的效率和相关性。FLARE通过监控生成过程的置信度，动态决定是否需要检索以及何时停止检索和生成，从而优化检索周期。

FLARE的核心思想是使大型语言模型（LLMs）能够主动判断是否需要检索外部知识，以及何时停止检索和生成。这种方法通过引入自适应机制，使模型能够根据生成过程的置信度动态调整检索行为，从而提高系统的效率和相关性。

FLARE的工作流程主要包括以下几个步骤：

1. **检索触发（Retrieval Triggering）**
- 置信度监控：FLARE通过监控生成过程的置信度（如生成词的概率）来判断是否需要检索。当生成词的概率低于某个阈值时，触发检索系统。
- 动态调整：根据生成过程的实时反馈，动态决定是否需要检索，避免不必要的检索操作。
2. **检索过程（Retrieval Process）**
- 检索执行：一旦触发检索，FLARE会从外部知识库中检索与当前生成内容相关的文档。
- 上下文整合：检索到的文档会被整合到生成模型的上下文中，以提供更丰富的信息支持。
3. **生成优化（Generation Optimization）**
- 生成调整：FLARE通过检索到的文档优化生成过程，确保生成的内容更加准确和相关。
- 反馈循环：生成的内容可以进一步反馈到检索系统，形成一个闭环优化过程。

FLARE特别适用于以下场景：

- 动态知识更新：需要实时检索最新信息的场景，如新闻问答、动态知识库查询。
- 复杂任务处理：需要多步骤推理或复杂问题解答的场景，如学术研究辅助。
- 资源受限环境：在计算资源有限的环境中，FLARE通过减少不必要的检索，提高资源利用效率。

FLARE的挑战

- 实现复杂度：FLARE的自适应机制增加了系统的复杂度，需要更多的开发和维护工作。
- 参数调优：需要仔细调整置信度阈值和其他参数，以确保系统的最佳性能。

#### Self-RAG

Self-RAG（Self-Retrieval Augmented Generation）是一种自检索增强生成方法，旨在通过模型自身的反思和评估机制，动态决定是否需要检索外部知识以及何时停止检索和生成。这种方法通过引入“反思令牌”（reflection tokens）使模型能够自我评估其输出，从而提高检索的效率和相关性。

Self-RAG的核心思想是赋予大型语言模型（LLMs）自我评估的能力，使其能够根据生成内容的置信度动态决定是否需要检索外部知识。这种方法通过引入特殊的“反思令牌”（如“retrieve”和“critic”），使模型能够自主判断何时需要检索以及何时停止检索。

- **自主决策**：模型能够自主决定是否需要检索外部知识。
- **动态调整**：根据生成内容的实时反馈，动态调整检索行为。
- **减少幻觉**：通过检索外部知识，减少生成内容中的幻觉问题。
- **高效资源利用**：通过减少不必要的检索，提高资源利用效率。

Self-RAG的工作流程主要包括以下几个步骤：

反思令牌（Reflection Tokens）

- **retrieve令牌**：当模型认为需要检索外部知识时，会生成“retrieve”令牌。
- **critic令牌**：当模型认为当前生成内容已经足够好，不需要进一步检索时，会生成“critic”令牌。

检索触发（Retrieval Triggering）

- **置信度评估**：模型通过评估生成内容的置信度来决定是否需要检索。例如，当生成内容的置信度低于某个阈值时，触发检索。
- **动态调整**：根据生成内容的实时反馈，动态决定是否需要检索。

检索执行（Retrieval Execution）

- **检索执行**：一旦触发检索，Self-RAG会从外部知识库中检索与当前生成内容相关的文档。
- **上下文整合**：检索到的文档会被整合到生成模型的上下文中，以提供更丰富的信息支持。

生成优化（Generation Optimization）

- **生成调整**：通过检索到的文档优化生成过程，确保生成的内容更加准确和相关。
- **反馈循环**：生成的内容可以进一步反馈到检索系统，形成一个闭环优化过程。

[[2310.11511] Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)

### RAG vs Fine-tuning

| **特性**      | **RAG**                 | **Fine-tuning**              |
| ----------- | ----------------------- | ---------------------------- |
| **外部知识需求**  | 高，依赖外部知识库进行检索增强。        | 低，主要依赖模型内部的知识和训练数据。          |
| **模型适应性需求** | 低，通常不需要对模型进行大规模修改。      | 高，需要对模型进行进一步训练以适应特定任务。       |
| **动态知识更新**  | 适合动态环境，能够实时检索最新信息。      | 不适合动态环境，需要重新训练以更新知识。         |
| **幻觉问题**    | 通过检索外部知识，能够有效减少幻觉问题。    | 幻觉问题可能仍然存在，尤其是在训练数据不足或任务复杂时。 |
| **计算资源需求**  | 较高，检索和生成过程需要额外的计算资源。    | 较低，但需要大量计算资源进行模型训练。          |
| **知识表示方式**  | 非参数化，依赖外部知识库。           | 参数化，通过模型的参数表示知识。             |
| **灵活性**     | 高，支持多种任务和数据类型，适应性强。     | 低，通常需要针对特定任务进行训练。            |
| **可解释性**    | 高，检索过程和生成过程可观察，便于验证和调试。 | 低，模型的推理过程通常是黑箱的，难以解释。        |

| **任务类型**   | **RAG适合的场景**                 | **Fine-tuning适合的场景**    |
| ---------- | ---------------------------- | ----------------------- |
| **动态知识更新** | 需要实时检索最新信息的场景，如新闻问答、动态知识库查询。 | 知识相对静态的场景，如固定领域问答。      |
| **复杂任务**   | 需要多步骤推理或复杂问题解答的场景，如学术研究辅助。   | 需要深度定制的场景，如特定领域问答或文本生成。 |
| **多模态任务**  | 需要处理多种数据类型的场景，如多模态问答系统。      | 通常不适用于多模态任务，除非结合其他技术。   |
| **资源受限环境** | 资源受限的环境，特别是需要快速部署的场景。        | 资源充足的环境，能够支持大规模训练。      |

# Retrieval

### Retrieval Source

检索源是RAG系统中用于检索信息的数据来源。选择合适的检索源对于提高检索质量和生成质量至关重要。

#### *Data Structure*

- **文本数据**：最常见的检索源，包括维基百科、新闻文章、学术论文等。
- **半结构化数据**：如PDF文档，包含文本和表格信息。
- **结构化数据**：如知识图谱（KG），提供经过验证的精确信息。
- **LLMs生成的内容**：利用LLMs内部知识生成的内容，用于补充或优化检索结果。

##### Unstructured Data

- **数据格式**：根据任务需求选择合适的数据格式，如文本、表格或知识图谱。
- **数据质量**：确保检索源的数据质量高，信息准确。
- **数据覆盖范围**：选择覆盖范围广的检索源，以提高检索的全面性。

###### *Open-Domain Question-Answering (ODQA) tasks*

[[1704.00051\] Reading Wikipedia to Answer Open-Domain Questions](https://arxiv.org/abs/1704.00051)

简称ODQA.中文翻译为开放式问答，意为基于涵盖广泛主题的文本集合给出问题答案[[2\]](https://zhuanlan.zhihu.com/p/93347083#ref_2)。

##### Semi-structured Data

> refers to data that contains a combination of text and table information, such as PDF.

半结构化数据通常包含文本和表格信息，如PDF文档。处理半结构化数据时，需要注意文本分割和表格语义的保持。

###### *data corruption*

##### Structured Data

###### *Prize-Collecting Steiner Tree (PCST)*

#### Retrieval Granularity

> In text, retrieval granularity ranges from fine to coarse, including Token, Phrase, Sentence, Proposition, Chunks, Document.

### Indexing Optimization

索引优化是提高检索效率和质量的关键步骤。通过优化索引结构和内容，可以更快、更准确地检索到相关信息。

> In the Indexing phase, documents will be processed, segmented, and transformed into Embeddings to be stored in a vector database. **The quality of index construction determines whether the correct context can be obtained in the retrieval phase.**

#### Chunking Strategy

> Larger chunks can capture more context, but they also generate more noise, requiring longer processing time and higher costs. 

While smaller chunks may not fully convey the necessary context, they do have less noise.

- **固定长度分块**：将文档分割成固定长度的片段，如100、256或512个标记。
- **滑动窗口方法**：使用滑动窗口技术，允许多个片段之间有重叠，以保持语义完整性。
- **小到大分块**：以句子为单位进行检索，同时提供上下文信息。

##### Small2Big

Small-to-Big Retrieval 技术试图解决这样一个矛盾：更大的 chunk 可以包含更多有用的信息，但其包含的较多无用文本又会掩盖 semantic representation 从而导致检索效果的变差。

主要的思路是：基于更小、更有针对性的 text chunk 进行 embedding 和 retrieval，但仍然使用较大的 text chunk 来为 LLM 提供更多的上下文信息。也就是在检索过程中使用较小的 text chunk，然后将检索到的文本的对应的更大的 text chunk 给 LLM。

> Therefore, methods like Small2Big have been proposed, where sentences (small) are used as the retrieval unit, and the preceding and following sentences are provided as (big) context to LLMs.

#### Metadata Attachments

> Chunks can be enriched with metadata information such as page number, file name, author,category timestamp. Subsequently, retrieval can be filtered based on this metadata, limiting the scope of the retrieval.
> 
> Assigning different weights to document timestamps during retrieval can achieve time-aware RAG, ensuring the freshness of knowledge and avoiding outdated information.

- **元数据丰富**：为片段附加元数据，如页码、文件名、作者、类别和时间戳。
- **过滤和排序**：基于元数据过滤和排序检索结果，提高检索的相关性。

通过元数据可以在搜索的时候根据结构化的数据结构过滤，不过同时也需要向量数据库支持。

##### Reverse HyDE

> adding summaries of paragraph, as well as introducing hypothetical questions.
> 
> using LLM to generate questions that can be answered by the document, then calculating the similarity between the original question and the hypothetical question during retrieval to reduce the semantic gap between the question and the answer.

#### Structural Index

或许将非结构化文本转换成markdown会是一个不错的选择。

- **层次结构**：构建文档的层次结构，提高检索和处理效率。
- **知识图谱索引**：利用知识图谱构建文档的层次结构，保持概念和实体之间的一致性。

##### Hierarchical index structure

##### Knowledge Graph index

### Query Optimization

查询优化通过改进用户原始查询，提高检索的准确性和相关性。

#### Query Expansion

##### Multi-Query

- **多查询**：通过提示工程扩展查询，生成多个相关查询并行执行。

##### Sub-Query

- **子查询**：将复杂查询分解为多个子查询，逐步构建完整答案。

##### Chain of Verification (CoVe)

- **链式验证（Chain-of-Verification, CoVe）**：通过验证扩展查询，减少幻觉。

[【AI论文学习笔记】链式验证减少了大型语言模型中的幻觉 - 知乎](https://zhuanlan.zhihu.com/p/657793950)

#### Query Transformation

> The core concept is to retrieve chunks based on a transformed query instead of the user’s original query.

##### Query Rewrite

- **查询重写**：使用LLMs或专用小模型重写原始查询，提高检索效果。
- **假设文档（HyDE）**：生成假设文档，基于答案的嵌入相似性进行检索。
- **抽象问题（Step-back Prompting）**：生成高层次概念问题，与原始查询结合进行检索。

##### Step-back Prompting

简而言之，Step-Back Prompting 包含两个简单的步骤：

- Abstraction：先让 LLM 根据 original question 提出一个更高层次概念的 step-back question，并检索这个 step-back question 的相关事实。

- Reasoning：基于高层次概念或原则的事实，LLM 就可以去推理原始问题的解决方案了。

[【LLM 论文】Step-Back Prompting：先解决更高层次的问题来提高 LLM 推理能力-CSDN博客](https://blog.csdn.net/qq_45668004/article/details/138683683)

[[2310.06117\] Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models](https://arxiv.org/abs/2310.06117)

#### Query Routing

##### Metadata Router/Filter

- **元数据路由**：基于查询中的关键词和元数据过滤，缩小搜索范围。

##### Semantic Router

- **语义路由**：利用查询的语义信息进行路由，提高检索的相关性。

##### Hybrid Router

- **混合路由**：结合元数据和语义信息进行查询路由，提高检索效率。

### Embedding

嵌入是将文本或文档转换为向量表示的过程，这些向量用于计算与查询的相似性，以实现检索。

#### Embedding Model

- **稀疏编码器**：如BM25，适用于基于关键词的检索。
- **密集检索器**：如基于BERT架构的预训练语言模型，适用于语义相似性检索。

**Dense embedding model** 和 **sparse embedding model** 都是将高维稀疏向量嵌入到低维稠密向量的技术，常用于NLP任务中。两种模型的主要区别在于它们如何表示嵌入向量：

##### Sparse embedding model

使用稀疏向量来表示每个单词或短语。只有少数维度的值是非零的，这些值代表该单词或短语在语义空间中的重要特征。例如，一个单词的嵌入向量可能只有几个非零维度，表示该单词与其他几个单词的语义相关性很强。

优点是参数数量更少，计算成本更低。但是，它们可能无法捕捉到单词或短语在语义空间中的所有信息。

###### *TF-IDF*

###### *BM25*

BM25（Best Matching 25）是一种经典的信息检索算法，是基于[Okapi TF-IDF](https://zhida.zhihu.com/search?content_id=237090732&content_type=Article&match_order=1&q=Okapi+TF-IDF&zhida_source=entity)算法的改进版本，旨在解决Okapi TF-IDF算法的一些不足之处。其被广泛应用于信息检索领域的排名函数，用于估计文档D与用户查询Q之间的相关性。它是一种基于概率检索框架的改进，特别是在处理长文档和短查询时表现出色。BM25的核心思想是基于词频(TF)和[逆文档频率](https://zhida.zhihu.com/search?content_id=237090732&content_type=Article&match_order=1&q=逆文档频率&zhida_source=entity)(IDF)来,同时还引入了文档的长度信息来计算文档D和查询Q之间的相关性。目前被广泛运用的搜索引擎ES就内置了BM25算法进行全文检索。

[RAG提效利器——BM25检索算法原理和Python实现 - 知乎](https://zhuanlan.zhihu.com/p/670322092#:~:text=BM25是一种基于词频和逆文档频率的信息检索算法，用于计算文档和查询的相关性得分。本文介绍了BM25的基本公式、参数、实现和优缺点，并给出了Python代码示例。)

##### Dense embedding model

使用稠密向量来表示每个单词或短语。每个维度的值代表该单词或短语在语义空间中对应方面的重要性。例如，一个维度的值可能表示该单词的积极性或消极性，另一个维度的值可能表示该单词的正式程度或非正式程度。

优点是能够捕捉到单词或短语在语义空间中的更细粒度信息。但是，它们的参数数量更多，计算成本也更高。

###### *BERT*

###### *Word2Vec*

###### *Global Vectors for Word Representation (GloVe)*

#### Mixed/Hybrid Retrieval

- **稀疏和密集检索结合**：利用稀疏检索提供初始结果，密集检索进一步优化。
- **预训练语言模型微调**：通过微调预训练语言模型，提高检索的准确性。

#### Embedding Model Fine-tuning

- **领域特定微调**：在特定领域数据集上微调嵌入模型，以适应专业领域。
- **对齐微调**：通过LLMs的输出作为监督信号，对齐检索器和生成器的偏好。

### Adapter

适配器是一种外部组件，用于优化LLMs的多任务能力，或解决因API集成或本地计算资源限制而带来的挑战。

Adapter的核心作用

- **优化多任务能力**：Adapter能够帮助LLMs更好地适应多种任务，通过调整模型的输入输出格式，使其能够处理不同类型的数据和任务需求。
- **解决资源限制**：在本地计算资源有限的情况下，Adapter可以作为外部组件，减少对模型内部结构的修改，从而降低资源消耗。

#### Lightweight Adapter

轻量级适配器

- **轻量级提示检索器（Lightweight Prompt Retriever）**：自动从预建的提示池中检索适合零样本任务的提示，提高模型在零样本任务中的表现。
- **通用适配器（Universal Adapter）**：设计用于适应多个下游任务的通用适配器，能够处理多种任务类型，减少针对每个任务单独调整的需要。

#### Pluggable Adapter

可插拔适配器

##### Pluggable Reward-driven Context Adapter (PRCA)

[[2310.18347\] PRCA: Fitting Black-Box Large Language Models for Retrieval Question Answering via Pluggable Reward-Driven Contextual Adapter](https://arxiv.org/abs/2310.18347)

可插拔奖励驱动上下文适配器：通过奖励信号动态选择和调整检索内容，提高模型在特定任务中的表现。

##### Bridge Model

桥接模型：在检索器和LLMs之间训练桥接模型，将检索到的信息转换为LLMs能够有效处理的格式，从而提高生成质量。

Adapter特别适用于以下场景：

- **多任务处理**：需要处理多种不同类型任务的场景，如同时处理问答和信息提取任务。
- **资源受限环境**：在计算资源有限的环境中，通过减少对模型内部结构的修改，提高资源利用效率。
- **动态知识更新**：需要实时检索最新信息的场景，如新闻问答、动态知识库查询。

**Adapter的实现方法**

- **提示工程（Prompt Engineering）**：通过设计特定的提示，引导模型产生期望的输出，而不需要对外部知识或模型进行大量修改。
- **微调（Fine-tuning）**：在特定数据集上对适配器进行微调，以适应特定任务需求。
- **模块化设计**：通过模块化设计，使适配器能够灵活地集成到不同的RAG系统中，适应不同的任务和场景。

# Generation

### Context Curation

在 RAG 的 Generation 阶段，直接将所有检索到的信息输入到 LLM 中可能会导致问题，比如冗余信息干扰生成结果，或者上下文过长导致 LLM 出现“Lost in the middle”问题（即模型倾向于忽略中间部分的信息）。因此，需要对检索到的内容进行优化处理。

#### Reranking

**目的**：重新排序检索到的文档片段，将最相关的内容放在前面，减少无关信息的干扰。

**作用**：

- **增强器（Enhancer）**：通过突出显示最相关的结果，提升检索结果的质量。
- **过滤器（Filter）**：通过筛选掉不相关的内容，减少无关信息的干扰。
- **精炼输入**：为后续的语言模型处理提供更精确的输入，从而生成更高质量的答案。

##### Rule-based Methods

**特点**：依赖预定义的指标（如多样性、相关性和 MRR）来评估和排序文档片段。

**应用场景**：适用于规则明确、结构化程度较高的任务。

**示例**：

- **Diversity（多样性）**：确保结果的多样性，避免重复信息。
- **Relevance（相关性）**：根据文档片段与查询的相关性进行排序。
- **MRR（Mean Reciprocal Rank）**：一种衡量检索结果排序质量的指标，重点关注第一个正确结果的位置。

##### Model-based Methods

**特点**：利用机器学习模型对文档片段进行排序，通常比基于规则的方法更灵活、更强大。

**应用场景**：适用于复杂、动态的任务，尤其是需要语义理解的场景。

**示例**：

- **Encoder-Decoder 模型（如 SpanBERT）**：通过编码器对文档片段进行语义编码，再通过解码器对结果进行排序。
- **专门的重排序模型（如 Cohere Rerank 或 bge-raranker-large）**：这些模型专门设计用于优化检索结果的排序。
- **通用大型语言模型**：利用 GPT 等模型的强大语言理解和生成能力，对文档片段进行排序。

#### Context Selection/Compression

> detect and remove unimportant tokens, transforming it into a form that is challenging for humans to comprehend but well understood by LLMs.

减少冗长上下文对 LLM 的干扰，提升生成结果的准确性。 

##### Small Language Models (SLM)

**基于小语言模型（SLM）的压缩**：例如使用 GPT-2 Small 或 LLaMA-7B 检测并移除不重要的信息。

利用小型语言模型（如 GPT-2 Small 或 LLaMA-7B）检测并移除不重要的标记（tokens），将上下文转换为一种对人类来说难以理解但对大型语言模型（LLMs）来说更容易处理的形式。

这种方法直接利用现有的 SLMs，避免了对 LLMs 进行额外训练的复杂性和成本。

##### PRCA

- **信息提取器**：通过训练信息提取器（如 PRCA 的信息提取器）或信息浓缩器（如 RECOMP 的对比学习方法）来压缩上下文。  

##### RECOMP

[[2310.04408\] RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation](https://arxiv.org/abs/2310.04408)

- **文档数量优化**：通过“Filter-Reranker”方法结合 SLM 和 LLM 的优势，减少输入文档的数量。

### LLM Fine-tuning

针对具体场景和数据特性对 LLM 进行微调，可以显著提升其性能。这是使用本地 LLM 的一大优势：

1. **领域知识补充**  
   
   - 当 LLM 在特定领域缺乏数据时，可以通过微调向模型注入额外的知识。例如，在医疗或法律领域，微调可以帮助模型更好地理解领域术语。

2. **输入输出格式调整**  
   
   - 微调可以调整 LLM 的输入和输出格式，使其适应特定任务。例如，SANTA 框架通过对比学习优化查询和文档的嵌入，使模型能够更好地处理结构化数据。

3. **强化学习与偏好对齐**  
   
   - 使用手动标注的生成结果并通过强化学习对齐 LLM 的输出与人类偏好。例如，通过标注最终生成的答案并提供反馈，提升模型的生成质量。

4. **Retriever 和 Generator 的协同微调**  
   
   - 通过 KL 散度对齐 Retriever 和 Generator 的评分函数，使两者在检索和生成过程中保持一致。

# Augmentation Process

## Iterative Retrieval

> The knowledge base is repeatedly searched based on the initial query and the text generated so far, providing a more comprehensive knowledge base for LLMs.

基于初始查询和目前已生成文本，反复对知识库进行搜索。每次检索都能为生成模型提供更全面、更有针对性的知识，弥补单次检索带来的信息局限性，从而提高后续答案生成的准确性与全面性。

- **优点** ：通过多次检索，拓宽了知识来源，使生成结果能更深入、更精准地贴合问题需求，尤其在复杂问题面前，能提供更充分的背景和细节支持。

- **缺点** ：一方面，可能出现语义不连贯 (semantic discontinuity) 的情况，即新检索到的信息与前文生成内容在逻辑、语义上难以完美衔接；另一方面，多次检索易累积无关信息 (the accumulation of irrelevant information)，干扰模型对关键内容的聚焦，降低生成效率与质量。

### ITER - RETGEN

采用 “检索增强生成” 与 “生成增强检索” 的协同策略。以输入任务所需内容为上下文基础检索知识，再依据新知识优化后续生成，适用于需要精确复现特定信息的任务，像专业文献复述、精准数据引用等场景。例如，当要求模型详细阐述某一历史事件的特定环节，ITER - RETGEN 可先检索该事件相关知识，依结果进一步确定细节信息，最终准确生成对应描述。

## Recursive Retrieval

### IRCoT

在CoT基础上，每次迭代都再检索一次。

先初步检索得到部分信息，再利用这些信息完善链式思考路径，反复迭代直至精准定位所需知识。比如在解决数学应用题时，先检索题目相关概念，依此梳理解题思路，不断细化检索，直至找到关键公式与解题步骤。

### ToC

> ToC creates a clarification tree that systematically optimizes the ambiguous parts in the Query.

> clarification tree：决策树，也可以叫分类树。简单地说，就是根据训练数据集构造一个类似树形的分类决策模型，然后用这个模型来辅助决策，每个节点用于判断一个很简单的条件，根据简单的判断结果进行到下一个步骤节点继续进行下一个简单的判断，直到做出最终的决策。

适合初始查询表意不清、信息需求复杂的情况，如用户询问 “如何提升工作效率”，ToC 可先分解为时间管理、工具使用等子分支，再逐层检索细化，给出针对性建议。

### multi-hop retrieval

> multi-hop retrieval is designed to delve deeper into graph-structured data sources, extracting interconnected information.

[[2401.15391] MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries](https://arxiv.org/abs/2401.15391)

## Adaptive Retrieval

使 RAG 系统具备自主判断能力，能自行决定是否需要检索外部知识，以及确定检索与生成的终止时机，让检索过程更灵活、高效，提升信息获取的相关性与适时性。

### FLARE

- **定义**：FLARE（Forward-Looking Active REtrieval-augmented generation）是一种检索增强型生成方法，它在生成过程中主动决定何时检索以及检索什么内容。
- **核心思想**：通过预测即将生成的句子来提前发现知识缺口，并以此作为查询检索相关文档，从而在生成长文本时动态地获取所需信息。

**算法过程**

1. **预测下一句**：模型基于输入生成一个临时的下一句。
2. **检查低置信度标记**：模型检查生成的每个标记的置信度，如果置信度低，说明模型对该部分不确定，可能需要更多信息。
3. **检索额外信息（如有必要）**：如果检测到知识缺口，模型会检索相关文档。
4. **高级查询构建**：可以采用隐式查询（如用[MASK]标记不确定部分）或显式查询（生成自然语言问题）。
5. **使用检索到的信息重新生成句子**：模型根据检索到的信息更新句子，确保生成内容的准确性。
6. **重复直到完成**：模型迭代重复上述过程，直到整个文本生成完成。

**算法特点**

- **按需检索**：与传统RAG方法（仅在生成前检索一次）或固定间隔检索方法不同，FLARE仅在需要时检索，避免了不必要的检索，提高了效率。
- **动态适应**：FLARE能够根据生成过程中不断变化的需求动态调整检索内容，增强了生成文本的准确性和上下文相关性。
- **减少幻觉**：通过主动检索相关知识，FLARE有效降低了生成文本中出现事实错误或“幻觉”的风险。 

**应用场景**

- **长文本生成**：FLARE特别适用于需要生成长文本的任务，如论文撰写、故事创作等，因为它能够在生成过程中动态获取所需信息。
- **知识密集型任务**：在需要准确知识的任务中，如事实问答、学术写作等，FLARE能够通过检索增强生成内容的准确性和可靠性。



### Self-RAG

以下是关于Self-RAG框架以及Self-Ask、AutoGPT、Toolformer、Graph-Toolformer等Agent的介绍：

- **定义**：Self-RAG是一种自反思的检索增强生成框架，旨在通过检索和自我反思提升语言模型的输出质量和事实性。
- **工作原理**：
  - **按需检索**：Self-RAG能够智能地判断何时需要检索外部信息，并按需检索相关文档，避免不必要的检索。
  - **反思机制**：通过特殊的反思标记（如`Retrieve`、`ISREL`、`ISSUP`、`ISUSE`），模型可以决定是否检索信息、评估检索内容的相关性和有用性，并据此调整生成内容。
  - **自我评估**：生成输出后，Self-RAG会通过自我评估机制检查内容的事实性和相关性，确保输出准确且符合任务要求。
- **优势**：
  - 提升事实性和准确性：显著优于传统语言模型和检索增强模型。
  - 提高效率：避免了不必要的检索，提升了系统效率。
  - 适应性强：可根据不同任务需求调整行为。
- **应用场景**：适用于需要高准确性和事实性的任务，如开放域问答、推理任务、长文本生成等。

#### Self-Ask Agent

- **定义**：Self-Ask Agent是一种通过分解复杂问题为多个中间问题来逐步解决问题的Agent。

- **工作原理**：
  
  - 接收复杂查询后，生成中间问题以获取更多信息。
  - 逐个回答这些子问题，最终形成全面的解决方案。

- **应用场景**：适用于需要逐步推理和分解问题的场景，如学术研究查询、教育工具中的逐步解答、决策支持系统等。

- **实现示例**：
  
  ```python
  from langchain.agents import create_self_ask_agent
  from langchain_community.chat_models import ChatAnthropic
  from langchain import hub
  
  prompt = hub.pull("hwchase17/self-ask-with-search")
  model = ChatAnthropic(model="claude-3-haiku-20240307")
  tools = [...]  # 应仅包含一个名为`Intermediate Answer`的工具
  agent = create_self_ask_agent(model, tools, prompt)
  agent_executor = AgentExecutor(agent=agent, tools=tools)
  agent_executor.invoke({"input": "hi"})
  ```

#### AutoGPT Agent

- **定义**：AutoGPT是一种能够自主执行任务的Agent，通常基于大型语言模型构建，能够通过自我对话和工具调用完成复杂任务。
- **工作原理**：
  - 自我对话：通过与自身的对话来逐步推理和解决问题。
  - 工具调用：根据需要调用外部工具（如搜索引擎、API等）以获取信息。
- **应用场景**：适用于需要自主决策和多步骤任务执行的场景，如自动化办公、复杂问题解决等。

#### Toolformer Agent

- **定义**：Toolformer是一种能够与外部工具（如API、数据库等）交互的Agent，用于增强语言模型的功能。
- **工作原理**：
  - 工具调用：根据任务需求，调用外部工具获取所需信息。
  - 结合生成：将工具返回的结果与语言模型的生成能力结合，生成更准确的输出。
- **应用场景**：适用于需要实时数据或外部信息的任务，如天气查询、股票价格查询等。

#### Graph-Toolformer Agent

- **定义**：Graph-Toolformer是基于图结构的Agent，能够将工具作为节点和边进行连接，构建复杂的多Agent系统。
- **工作原理**：
  - 图结构：通过图结构组织多个Agent和工具，实现复杂的任务流程。
  - 模块化设计：支持低层次的自定义和控制，能够灵活调整任务流程。
- **应用场景**：适用于需要多Agent协作和复杂任务流程的场景，如客户支持、教育、电子商务等。

# Task And Evaluation

> mainly introduce the main downstream tasks of RAG, datasets, and how to evaluate RAG systems.

## Downstream Task

### QA

#### Single/Multi hop QA

#### Domain-specific QA



## Evaluation Target

The evaluation of RAG (Retrieval-Augmented Generation) models has traditionally focused on their performance in specific downstream tasks, using metrics tailored to those tasks. However, there is a growing recognition of the need to evaluate the distinct characteristics of RAG models more comprehensively. Here is a detailed breakdown of the evaluation targets and methods historically used, as well as the emerging focus areas:

### Historical Evaluation Approach

#### Task-Specific Metrics

##### Question Answering (QA)

###### Exact Match (EM)

Measures the proportion of generated answers that exactly match the ground truth answers. It is a strict metric that evaluates the precision of the model's output.

###### F1 Score

Computes the harmonic mean of precision and recall, providing a balance between the two. It is particularly useful for evaluating the overlap between the generated answer and the reference answer.

##### Fact-Checking

###### Accuracy

Determines the proportion of generated statements that are factually correct. This is a critical metric for evaluating the reliability of the model in producing accurate information.

##### Text Generation Quality

###### BLEU (Bilingual Evaluation Understudy)

**BLEU** (其全称为**Bilingual Evaluation Understudy**)，其意思是双语评估替补。所谓Understudy (替补)，意思是代替人进行翻译结果的评估。尽管这项指标是为翻译而发明的，但它可以用于评估一组自然语言处理任务生成的文本。

[Bleu: a Method for Automatic Evaluation of Machine Translation - ACL Anthology](https://aclanthology.org/P02-1040/)



BLEU方法的实现是分别计算**candidate句**和**reference句**的**N-grams模型**，然后统计其匹配的个数来计算得到的。显然, 这种比较方法，是与语序无关的。

论文对匹配的 **N-grams** 计数进行了修改，以确保它考虑到reference文本中单词的出现，而非奖励生成大量合理翻译单词的候选结果。本文将其称为修正的 **N-grams** 精度。

此外, 有一种改进版的通过 **normalize N-grams** 的改进版BLEU。其目的是提升对多个句子组成的块(block)提升翻译效果。
在实践中得到完美的分数是不太可能的，因为翻译结果必须与**reference句**完全匹配。甚至人工翻译都不可能做到这一点。而且, 由于不同的数据集**reference句**的质量和数量不同, 所以跨数据集计算BLEU值可能带来一些麻烦。

Originally designed for machine translation, BLEU evaluates the similarity between the generated text and reference texts by comparing n-gram overlaps. It is commonly used to assess the fluency and adequacy of generated answers.

###### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Focuses on the recall of n-grams, unigrams, bigrams, etc., between the generated text and the reference text. It is particularly useful for evaluating summarization and other text generation tasks.



#### Automated Evaluation Tools

##### RALLE (Retrieval-Augmented Language Model Evaluation)

This tool is designed specifically for evaluating RAG applications. It leverages the aforementioned task-specific metrics to provide a comprehensive assessment of the model's performance in various downstream tasks.



### Retrieval Quality

传统搜索引擎的指标也可适用于 RAG 的检索模块评估。

> Standard metrics from the domains of search engines, recommendation systems, and information retrieval systems are employed to measure the performance of the RAG retrieval module. Metrics such as Hit Rate, MRR, and NDCG are commonly utilized for this purpose [161], [162].

#### Hit Rate

Measures the proportion of queries for which the retriever returns at least one relevant document.

#### Mean Reciprocal Rank (MRR)

**平均倒数排名**：正确检索结果值在检索结果中的排名来评估检索系统的性能。

$$
MRR = \frac{1}{Q} \sum_{i=1}^{|Q|} \frac{1}{rank_i}
$$



Evaluates the ranking quality of the retrieved documents by considering the reciprocal rank of the first relevant document. It rewards models that retrieve relevant documents higher in the ranking.

#### Normalized Discounted Cumulative Gain (NDCG)

归一化折损累计增益：在搜索和推荐任务中，系统常返回一个item列表。如何衡量这个返回的列表是否优秀呢？

> 例如，当我们检索【推荐排序】，网页返回了与推荐排序相关的链接列表。列表可能会是[A,B,C,G,D,E,F],也可能是[C,F,A,E,D]，现在问题来了，当系统返回这些列表时，怎么评价哪个列表更好？

NDCG就是用来评估排序结果的，搜索和推荐任务中比较常见。

https://zhuanlan.zhihu.com/p/371432647

- **Gain:** 表示一个列表中所有item的相关性分数。rel(i)表示item(i)相关性得分。

- **Cumulative Gain:** 表示对K个item的Gain进行累加。

- **Discounted Cumulative Gain:** 考虑排序顺序的因素，使得排名靠前的item增益更高，对排名靠后的item进行折损。

- **NDCG(Normalized DCG): 归一化折损累计增益**
  
  在NDCG之前，先了解一些IDGC(ideal DCG)--理想的DCG，IDCG的依据是：是根据rel(i)降序排列，即排列到最好状态。算出最好排列的DCG，就是IDCG。
  
  **IDCG=最好排列的DCG**
  
  对于上述的例子，按照rel(i)进行降序排列的最好状态为list_best=[B,D,A,C,E]
  
  | i     | rel(i) | log(i+1) | rel(i)/log(i+1) |
  | ----- | ------ | -------- | --------------- |
  | 1 = B | 0.9    | 1        | 0.9             |
  | 2 = D | 0.6    | 1.59     | 0.38            |
  | 3 = A | 0.5    | 2        | 0.25            |
  | 4 = C | 0.3    | 2.32     | 0.13            |
  | 5 = E | 0.1    | 2.59     | 0.04            |
  
  IDCG = list_best的DCG_best = 0.9+0.38+0.25+0.13+0.04=1.7 (理所当然，IDCG>DCG_1和DCG_2)



### Generation Quality

生成模块用于评估生成出来的内容的语义连续性和回答的相关性。

> The assessment of generation quality centers on the generator’s capacity to synthesize coherent and relevant answers from the retrieved context.

#### Unlabeled Content

##### Faithfulness

忠实度/真实性：评估生成答案是否与检索到的上下文一致，是否引入了事实性错误或无根据的声明。

> Evaluates whether the generated answers are consistent with the retrieved context and do not introduce factual errors or unsupported claims.

##### Relevance

相关度：评估生成答案是否针对输入查询或任务要求，确保输出内容与上下文相关。

> Assesses how well the generated answers address the input query or task requirements, ensuring that the output is pertinent to the context.

##### Non-Harmfulness

无害性：确保生成的内容不包含有害、有偏见或不适当的信息。

> Ensures that the generated content does not contain harmful, biased, or inappropriate information.

#### Labeled Content

- **Accuracy**: For tasks with labeled data, the focus is on the accuracy of the information produced by the model. This can be measured using metrics like EM and F1 score, similar to question answering evaluations.

- **Evaluation Methods**:
  - **Manual Evaluation**: Human evaluators can assess the coherence, relevance, and faithfulness of the generated answers, providing a qualitative assessment of the generation quality.
  - **Automatic Evaluation**: Using metrics like BLEU, ROUGE, and other natural language generation (NLG) evaluation metrics to automatically assess the quality of the generated text.

### Future Directions in Evaluation

- **Holistic Evaluation Frameworks**: There is a need to develop more comprehensive evaluation frameworks that integrate both retrieval and generation quality assessments. This would provide a more holistic view of the RAG model's performance.
- **Cross-Domain Evaluation**: Evaluating RAG models across multiple domains and tasks can provide insights into their generalizability and robustness. Metrics and benchmarks should be developed to support such cross-domain evaluations.
- **User-Centric Evaluation**: Incorporating user feedback and real-world usage scenarios into the evaluation process can help assess the practical utility and user satisfaction with RAG models.
- **Ethical and Social Impact Evaluation**: Assessing the ethical implications and potential biases in the generated content is crucial. Metrics and evaluation methods should be developed to ensure that RAG models produce fair, unbiased, and socially responsible outputs.

In summary, while traditional evaluations of RAG models have focused on task-specific metrics, there is a growing emphasis on evaluating the distinct characteristics of these models, particularly their retrieval and generation qualities. Developing more comprehensive and holistic evaluation frameworks will be essential for advancing the field and ensuring that RAG models meet the needs of diverse applications.



## Evaluation Aspects

### Quality Scores

#### Context Relevance

评估检索到的上下文的精确性和具体性，确保其与查询相关，并尽量减少因无关内容而产生的处理成本。

#### Answer Faithfulness

确保生成的答案与检索到的上下文保持一致，避免出现矛盾，维持真实性。

#### Answer Relevance

要求生成的答案直接与提出的问题相关，有效解决核心问题。



### Required Abilities

#### Noise Robustness

评估模型处理与问题相关但缺乏实质性信息的噪声文档的能力。

#### Negative Rejection

评估模型在检索到的文档中缺乏回答问题所需知识时，避免回答问题的判断力。

#### Information Integration

评估模型从多个文档中整合信息以回答复杂问题的能力。

#### Counterfactual Robustness

测试模型在已知文档中存在不准确信息的情况下，识别并忽略这些错误信息的能力，即使被提醒可能存在虚假信息。



**上下文相关性和抗噪声能力**对于评估检索质量至关重要，而**答案真实性、答案相关性、拒绝回答能力、信息整合能力以及反事实鲁棒性**则对于评估生成质量非常重要。

## Evaluation Benchmarks and Tools

> These instruments furnish quantitative metrics that not only gauge RAG model perfor- mance but also enhance comprehension of the model’s capabil- ities across various evaluation aspects.

### Benchmarks

#### RGB



#### Recall



#### CRUD

Create, Retrieve, Update, Delete

- **介绍**：CRUD基准通过模拟数据库操作的场景，评估RAG模型在处理不同类型任务时的表现。

- **特点**：
  
  - **任务模拟**：包括创建、检索、更新和删除等操作，模拟实际应用中的复杂场景。
  
  - **适应性评估**：通过多样化的任务设计，评估模型在不同任务类型下的适应能力。
  
  - **数据交互性**：强调模型与外部数据源的交互能力，评估其在动态数据环境中的表现。

### Tools

#### RAGAS

[[2309.15217] RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)

#### ARES

[[2311.09476] ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2311.09476)

# Discussion And Future Prospects
