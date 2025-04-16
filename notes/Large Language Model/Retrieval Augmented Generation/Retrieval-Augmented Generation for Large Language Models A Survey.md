
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



### RAG vs Fine-tuning

| **特性**           | **RAG**                                        | **Fine-tuning**                                          |
| ------------------ | ---------------------------------------------- | -------------------------------------------------------- |
| **外部知识需求**   | 高，依赖外部知识库进行检索增强。               | 低，主要依赖模型内部的知识和训练数据。                   |
| **模型适应性需求** | 低，通常不需要对模型进行大规模修改。           | 高，需要对模型进行进一步训练以适应特定任务。             |
| **动态知识更新**   | 适合动态环境，能够实时检索最新信息。           | 不适合动态环境，需要重新训练以更新知识。                 |
| **幻觉问题**       | 通过检索外部知识，能够有效减少幻觉问题。       | 幻觉问题可能仍然存在，尤其是在训练数据不足或任务复杂时。 |
| **计算资源需求**   | 较高，检索和生成过程需要额外的计算资源。       | 较低，但需要大量计算资源进行模型训练。                   |
| **知识表示方式**   | 非参数化，依赖外部知识库。                     | 参数化，通过模型的参数表示知识。                         |
| **灵活性**         | 高，支持多种任务和数据类型，适应性强。         | 低，通常需要针对特定任务进行训练。                       |
| **可解释性**       | 高，检索过程和生成过程可观察，便于验证和调试。 | 低，模型的推理过程通常是黑箱的，难以解释。               |



| **任务类型**     | **RAG适合的场景**                                        | **Fine-tuning适合的场景**                      |
| ---------------- | -------------------------------------------------------- | ---------------------------------------------- |
| **动态知识更新** | 需要实时检索最新信息的场景，如新闻问答、动态知识库查询。 | 知识相对静态的场景，如固定领域问答。           |
| **复杂任务**     | 需要多步骤推理或复杂问题解答的场景，如学术研究辅助。     | 需要深度定制的场景，如特定领域问答或文本生成。 |
| **多模态任务**   | 需要处理多种数据类型的场景，如多模态问答系统。           | 通常不适用于多模态任务，除非结合其他技术。     |
| **资源受限环境** | 资源受限的环境，特别是需要快速部署的场景。               | 资源充足的环境，能够支持大规模训练。           |



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
以下是关于 **Retrieval-Augmented Generation (RAG)** 中 **Generation 模块** 的学习文档，按照 ABCD 的结构依次展开：

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

### Iterative Retrieval
传统的 RAG 方法通常只进行一次检索，这在复杂问题中可能不足以提供足够的上下文信息。迭代检索通过多次检索和生成交替进行，逐步完善答案：

1. **过程**  
   - 基于初始查询和生成的文本，反复从知识库中检索相关信息，为 LLM 提供更全面的上下文。

2. **优势**  
   - 提供额外的上下文参考，增强后续生成的鲁棒性。例如，ITERRETGEN 模型通过“检索增强生成”和“生成增强检索”的协同作用，逐步优化答案。

3. **挑战**  
   - 可能受到语义不连续性或无关信息积累的影响，需要设计合理的机制来避免这些问题。

### Recursive Retrieval
递归检索通过逐步细化查询并分解问题，逐步解决复杂问题：

1. **过程**  
   - 根据前一次检索的结果，逐步优化查询，缩小搜索范围，直到找到最相关的信息。

2. **应用场景**  
   - 适用于复杂搜索场景，例如用户需求不明确或信息高度专业化的场景。例如，IRCoT 使用链式思考（Chain-of-Thought）指导检索过程，ToC 创建澄清树以优化模糊查询。

3. **与其他技术结合**  
   - 递归检索常与多跳检索结合，用于深入挖掘图结构数据源中的相关信息。

# Augmentation
- **迭代检索**：通过多次检索和生成循环，提供更全面的知识基础。
- **递归检索**：通过逐步细化查询和分解问题，深入挖掘相关信息。
- **自适应检索**：使LLMs能够主动决定检索的最佳时机和内容。

# Task And Evaluation
- **下游任务**：包括问答（QA）、信息提取（IE）、对话生成、代码搜索等。
- **评估目标**：检索质量和生成质量。
- **评估方面**：包括上下文相关性、答案忠实度、答案相关性、噪声鲁棒性、负拒绝、信息整合和反事实鲁棒性。
- **评估基准和工具**：如RGB、RECALL、RAGAS、ARES和TruLens等。

# Discussion And Future Prospects
- **RAG与长上下文**：尽管LLMs的上下文能力在扩展，RAG在处理长文档问答时仍具有不可替代的作用。
- **RAG鲁棒性**：提高RAG对噪声或矛盾信息的抵抗力成为研究热点。
- **混合方法**：将RAG与微调结合，探索最佳集成方式。
- **RAG的扩展规律**：研究RAG模型的参数扩展规律及其对性能的影响。
- **生产就绪的RAG**：提高检索效率、文档召回率和数据安全性。
- **多模态RAG**：扩展RAG到图像、音频、视频和代码等多模态数据。

## 八、结论
- **总结**：RAG通过结合LLMs的参数化知识和外部知识库的非参数化数据，显著提高了LLMs的能力。
- **未来研究方向**：提高RAG的鲁棒性、处理长上下文的能力，以及扩展到多模态领域。