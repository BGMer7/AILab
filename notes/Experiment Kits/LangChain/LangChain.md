[[LangChain-core]]
## 🧭 一、LangChain 整体架构概览

LangChain 在 0.3.x 版本之后采用了 **模块化架构**（Modular Architecture）。  
每个功能模块被拆分为独立的 Python 包，便于解耦、按需安装。

整体可分为三大类：

|类别|模块示例|主要功能|
|---|---|---|
|**核心层（Core Layer）**|`langchain-core`|定义框架的基础接口与通用抽象，如 Prompt、Runnable、OutputParser|
|**框架层（Framework Layer）**|`langchain`、`langchain-community`、`langchain-text-splitters`|提供常用组件实现、社区集成、文本处理等|
|**集成层（Integration Layer）**|`langchain-openai`、`langchain-aws`、`langchain-huggingface` 等|对接各类第三方大模型、服务或API|

---

## 🧩 二、各模块详细说明

### 1️⃣ Core（核心层）

|模块|说明|学习重点|
|---|---|---|
|**langchain-core**|LangChain 的基础模块，定义所有核心抽象类与运行逻辑，包括：`Runnable`、`PromptTemplate`、`BaseChatModel`、`OutputParser` 等。|理解 LangChain 的「执行模型」与「数据流动」机制。是最值得深入研究的包。|

学习路径建议：

- 先看 `langchain-core` 文档中的 `Runnable` 与 `Chain` 概念。
    
- 理解「Prompt + Model + OutputParser」的组合思想。
    

---

### 2️⃣ Framework（框架层）

|模块|说明|学习重点|
|---|---|---|
|**langchain**|主框架包，提供“经典” Chain 架构、Memory、Agent、Tool 等高层功能。|熟悉 `LLMChain`、`SequentialChain`、`AgentExecutor`、`Memory` 等。|
|**langchain-community**|社区维护的接口集合，如连接第三方数据库、搜索引擎、API 等。|学习如何集成外部数据源。|
|**langchain-text-splitters**|文本切分工具，用于长文档的 chunk 分割。|理解 chunking 策略（如 `RecursiveCharacterTextSplitter`）。|
|**langchain-tests**|LangChain 官方测试模块，用于验证各包兼容性。|非学习重点，可跳过。|

---

### 3️⃣ Integrations（集成层）

|模块|对应平台|说明|
|---|---|---|
|**langchain-openai**|OpenAI|对 `ChatGPT`、`GPT-4`、`Embeddings` 等API的封装。|
|**langchain-anthropic**|Anthropic|对 `Claude` 系列模型的支持。|
|**langchain-google-vertexai**|Google VertexAI|对 Google Cloud AI 平台模型的接口封装。|
|**langchain-aws**|AWS Bedrock|集成 Amazon Bedrock 模型（如 Titan、Claude、Llama）。|
|**langchain-huggingface**|Hugging Face|支持本地或云端 Transformer 模型。|
|**langchain-mistralai**|Mistral AI|支持 Mistral 平台模型（如 Mixtral）。|

学习建议：

- 任选一种你常用的模型平台（如 `langchain-openai`）。
    
- 先理解 `ChatOpenAI`、`Embeddings`、`Tool` 的使用方式。
    
- 其他包之后可按需学习。
    

---

## 🧱 三、学习顺序推荐（实战导向）

|阶段|学习目标|推荐模块|学习内容|
|---|---|---|---|
|**阶段 1：入门**|理解 Chain 的概念与执行逻辑|`langchain-core`, `langchain`|PromptTemplate → LLMChain → SimpleSequentialChain|
|**阶段 2：强化理解**|学习 Memory、Agent、Tool 的组合方式|`langchain`|Memory + AgentExecutor + Tool 定制|
|**阶段 3：文本处理**|学习如何处理长文档|`langchain-text-splitters`, `langchain-community`|文本分割、文档检索、Retrieval QA|
|**阶段 4：集成实践**|使用特定模型平台|`langchain-openai`（或其他）|使用 GPT/Claude/Mistral 等模型完成任务|
|**阶段 5：系统设计**|构建复杂多链系统或Graph结构|`langgraph`（可选进阶）|学习 LangGraph 的 DAG 执行与状态管理|

---

## 🔍 四、LangChain 与 LangGraph 的关系

|项目|定位|核心区别|
|---|---|---|
|**LangChain**|「模块库」|提供模型调用、Prompt、Memory、Tool、Chain 等抽象组件。|
|**LangGraph**|「编排框架」|基于 LangChain Core 构建，用于编排和控制多节点（多Chain）的执行流程。|
|关系|LangGraph 依赖 LangChain Core。LangGraph = LangChain 的编排层。||

---

## ✅ 总结：学习路线图

```
Step 1. langchain-core   → 理解基础结构与数据流
Step 2. langchain        → 掌握 Chain / Memory / Agent
Step 3. langchain-openai → 实际运行模型
Step 4. langchain-text-splitters → 学习文本预处理
Step 5. langchain-community → 探索外部数据集成
Step 6. langgraph        → 学习复杂系统编排
```
