学习 **tokenization（分词）** 可以从以下几个方面系统展开，涵盖其 **原理、算法、实现、使用、调优** 等知识维度。以下为详细分类：

------

## Concepts

### Token

- 什么是 token？（子词 / 单词 / 字符）
- 为什么不能直接用字符或单词训练模型？
- token 和 embedding 的关系

#### 什么是 token？

**Token** 是大语言模型中用于表示文本的最小单位之一。它不是固定的“一个字”或“一个词”，而是**模型分词器（tokenizer）切分文本后得到的基本片段**。

> 简单来说：**token 是模型理解文本的“拼图块”**，可以是一个字母、一个汉字、一个词，或者一个词的一部分（子词）。

------

**示例说明**

| 原始文本           | Tokenizer 类型     | Tokens（示例）                           |
| ------------------ | ------------------ | ---------------------------------------- |
| `ChatGPT is cool.` | 英文 BPE           | ["Chat", "G", "PT", " is", " cool", "."] |
| `今天天气不错`     | 中文 SentencePiece | ["今天", "天气", "不错"]                 |
| `unbelievable`     | BPE                | ["un", "believ", "able"]                 |

这些切分的单位就是 tokens。不同的 tokenizer 切分结果不同。

**Token 不是“词”或“字符”**

| 单位类型     | 特点                             |
| ------------ | -------------------------------- |
| 字符（char） | "C"、"h"、"a" 等单个字符         |
| 词（word）   | "ChatGPT"、"cool" 等完整单词     |
| ✅ **Token**  | 灵活的单位，可以是子词、字符或词 |

每个 token 对应：

- 一个字符串片段（如 "Chat"）
- 一个整数 id（如 15496）
- 一个向量（通过 embedding 映射出来）

------

**为什么使用 token 而不是字符或词？**

- **比词更泛化**：能拆词处理新词（比如 “re-invented”）
- **比字符更高效**：句子不会太长，训练收敛更快
- **通用性强**：适用于中英混合、多语言、拼写变化

**中文中的 token 是什么？**

中文没有空格，token 通常按 **字** 或 **词语** 切：

- 字粒度 token：["今", "天", "天", "气", "不", "错"]
- 词粒度 token（如使用 SentencePiece）：["今天", "天气", "不错"]

总结一句话：

> **Token 是模型理解文本的最小输入单位，既不是字符，也不是词，而是 tokenizer 设计后得到的“子词单位”**

#### 为何不能直接用字符级或单词级训练？

##### 字符级模型（Character-level）的问题

| 问题             | 说明                                                         |
| ---------------- | ------------------------------------------------------------ |
| 输入太长         | 一个句子可能包含几十上百个字符，导致 token 序列很长，Transformer 计算复杂度是 O(n²)，开销巨大 |
| 语言结构难捕捉   | 很难从字符中学习语义、词性等语言规律（比如 "information" 和 "data" 共享的意义模型很弱） |
| 收敛慢           | 因为模型从低级单位学习语言，学习路径很长，训练收敛极慢       |
| 不利于预训练迁移 | 字符层太低，难以捕捉通用知识抽象能力                         |

✅ 优点是：**字符集很小，适应性强，可处理拼写错误、新词**
 但缺点更致命：**太慢，表达能力差**

------

##### 单词级模型（Word-level）的问题

| 问题                         | 说明                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| 词表太大                     | 英文常见词有上百万，中文更甚，导致 embedding 层庞大，占内存且难训练 |
| OOV（Out-Of-Vocabulary）问题 | 训练时没见过的词（如“ChatGPT”），无法编码，模型失效          |
| 不支持新词组合               | 无法泛化组合词，如 "re-unzip-able" 这种组合式构词            |
| 多语言模型难统一             | 中文“打工人”、英文“hustler”、日文“サラリーマン”词表分裂，管理困难 |

✅ 优点是：**语义单位直接明确**
 但缺点是：**不具备泛化能力，词表不稳定**

------

##### 子词级（Subword-level）模型的优势（当前主流）

子词方法（如 BPE、WordPiece、Unigram）结合了字符级和词级的优点：

| 特点          | 说明                                                         |
| ------------- | ------------------------------------------------------------ |
| 高效泛化      | 词表较小（3万~5万），能自动组合新词，如 "unknowable" → "un" + "know" + "able" |
| 没有 OOV 问题 | 任何输入都能被拆分成子词（字符或已知片段）                   |
| 语义合理      | 子词比字符更语义化，比词更灵活                               |
| 适应多语言    | 可以训练多语言共享的词表（如中文拼音+英文+数字）             |



| 粒度       | 优点                     | 缺点                   | 主流应用模型                              |
| ---------- | ------------------------ | ---------------------- | ----------------------------------------- |
| 字符级     | 无OOV，词表小，适应强    | 序列长、语义弱、收敛慢 | 少数科研实验                              |
| 单词级     | 语义明确，句法结构清晰   | 词表大，OOV 问题严重   | 古老模型（如 early RNN）                  |
| **子词级** | ✅ 折中方案，结合两者优点 | 编码复杂度略高         | GPT/BERT/T5/LLaMA/Qwen 等几乎所有现代模型 |

------



#### Token 和 Embedding

很关键的一点是：**token id 本身 \*不包含\* 语义信息**，真正的语义是通过 **embedding 层** 赋予的。

------

##### token id 

token id 是 tokenizer 把文本转成的一串整数编号，例如：

```text
输入文本： "Hello world!"
Tokenizer 输出：
  tokens: ["Hello", " world", "!"]
  ids:    [15496, 2159, 0]
```

这些 `15496`、`2159` 等 **只是词表中的索引**，本质是整数，不含任何语义。不同模型对同一个 token 的 id 编号可能完全不同。

------

##### 语义来自哪儿？

**Embedding 层**会将每个 token id 映射为一个高维向量，如：

```text
token id 15496 → [0.23, -0.78, ..., 0.31] （假设是 4096 维）
```

这个向量经过训练，捕捉了词的上下文含义、语义关系。例如：

- "dog" 和 "puppy" 的 embedding 会在语义空间中靠近；
- "bank" 在金融语境和河流语境中会有不同向量表示（如果模型足够大或使用了 contextual embedding）。

所以：

> ✅ **token id 是“标号”，embedding 向量才是“语义”**

------

##### 训练过程中的关系

训练语言模型的大致流程是：

```text
文本 → tokenizer → token ids → embedding lookup → transformer → 预测下一个 token
```

这意味着：

- 模型学的是 **embedding 向量的组合与变化**；
- embedding 层是最先学会“词义”的组件（在预训练中不断调整）；
- 训练过程会把语义压进 embedding 中，而不是 token id 中。

##### Cases

| Token   | Token ID | Embedding 向量示意       |
| ------- | -------- | ------------------------ |
| "dog"   | 12345    | [0.15, -0.22, ..., 0.01] |
| "puppy" | 67890    | [0.14, -0.25, ..., 0.02] |
| "car"   | 54321    | [-0.33, 0.12, ..., 0.70] |

虽然它们的 ID 完全不同，但 embedding 向量可以学到 **"dog" ≈ "puppy"** 而不≈ "car"。

## ✅ 总结

| 问题                         | 回答                                            |
| ---------------------------- | ----------------------------------------------- |
| token id 含语义吗？          | ❌ 不含，它只是词表里的编号                      |
| 语义从哪里来？               | ✅ 来自 embedding 层的向量学习                   |
| embedding 是如何获得语义的？ | 通过大规模上下文学习（预测下一个词/填空任务等） |



### Tokenization

- 控制词表大小
- 提高稀有词的覆盖率
- 保留语言结构的表达能力

------

## Algorithm

### Character-level

- 每个字符是一个 token（极端细粒度）

### Word-level

- 基于空格或规则的词级分词
- 不适合中文或新词处理，词表庞大

### Subword-level

- **BPE（Byte Pair Encoding）**
- **WordPiece**（BERT 使用）
- **Unigram Language Model**（SentencePiece 默认）
- **SentencePiece（支持 Unicode、适合中文）**
- **Tiktoken（OpenAI 用于 GPT 模型）**

### Byte-level Tokenization

- GPT-2 开始使用（例如所有字符都映射成 UTF-8 byte）

------

## Training

- 语料准备（大规模文本、清洗）
- 设置参数：vocab size、coverage、特例 token（、 等）
- 训练工具：
  - `sentencepiece`（Google，主流）
  - `tokenizers`（Hugging Face，Rust 实现，高性能）
  - OpenAI `tiktoken`（只支持加载，不支持训练）

### 示例命令（SentencePiece）：

```bash
spm_train \
  --input=data.txt \
  --model_prefix=tokenizer \
  --vocab_size=32000 \
  --model_type=bpe
```

------

## Usage

### 与 LLM 模型对接

- 输入文本 → token ids → 模型 embedding 层
- output ids → 解码为文本（decode）

### 典型库使用

- Hugging Face `AutoTokenizer`
- `sentencepiece`
- `tokenizers`（更快）
- `tiktoken`（OpenAI GPT）

------

## Evaluation & Tuning

### token效率分析

- 平均每词 token 数（中文比英文高）
- token 长度对推理时间影响大

### vocabulary 设计策略

- vocab 大小的选择（小→更多 OOV，大→embedding 参数冗余）
- 多语言支持时需做特殊处理（如中英文共用词表）

------

## Case Studies

| 模型    | Tokenizer                 | 说明                                  |
| ------- | ------------------------- | ------------------------------------- |
| BERT    | WordPiece                 | 基于 WordPiece                        |
| GPT-2/3 | Byte-level BPE (Tiktoken) | byte-level BPE，支持所有 Unicode 字符 |
| T5      | SentencePiece             | 使用 Unigram                          |
| Qwen    | SentencePiece             | 中文支持优化                          |
| LLaMA   | BPE (custom tokenizer)    | 基于 BPE，自定义 vocab                |

------

## Reference

- 📘 [The Illustrated BPE](https://huggingface.co/learn/nlp-course/chapter6/6)
- 📗 SentencePiece GitHub: https://github.com/google/sentencepiece
- 📘 Hugging Face Tokenizers: https://github.com/huggingface/tokenizers
- 📗 Tiktoken (OpenAI): https://github.com/openai/tiktoken

------

## 总结：学习 Tokenization 的建议路径

1. 理解 Token 和 embedding 的基本关系
2. 学习主流子词分词算法（BPE / Unigram）
3. 学会用 SentencePiece 训练自己的 tokenizer
4. 学会如何对接 tokenizer 到 LLM 模型中
5. 分析不同模型的 tokenizer 差异
6. 掌握 tokenizer 的评估与优化技巧

