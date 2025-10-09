[[Runnable]]
## 🔍 LangChain Core 的定位

- `langchain-core` 是 LangChain 的基础抽象层，不依赖任何第三方模型或平台。它定义了整个生态中所有组件应遵循的接口和协议。([LangChain](https://python.langchain.com/api_reference/core/index.html "langchain-core: 0.3.78 —  LangChain  documentation"))
    
- 它包括：**可运行单元 (Runnables / Runnable 接口)**、**提示 / Prompt 模板 (PromptTemplate 等)**、**输出解析 (OutputParser)**、**消息 / Chat / LLM 抽象类 (BaseLanguageModel / BaseChatModel 等)**、**工具 / Tool 抽象** 等。([LangChain](https://python.langchain.com/api_reference/core/index.html "langchain-core: 0.3.78 —  LangChain  documentation"))
    
- 核心目标：让后续的 `langchain`、`langchain-openai`、`langgraph` 等包都能基于这些抽象进行一致的组合与执行。
    

---

## 🧱 核心模块与主要接口

下面是 `langchain-core` 中一些关键模块与概念，以及它们的职责和相互关系：

| 模块 / 概念                              | 主要接口 / 类                                                                                     | 责任 / 功能                      | 说明 /注意点                                                                                                                                                                                   |
| ------------------------------------ | -------------------------------------------------------------------------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Runnables / Runnable**             | `Runnable`, `RunnableSequence`, `RunnableParallel`, `RunnableBranch` 等                       | 抽象“可执行单元”，支持组合、流式、并行、分支等执行方式 | 这是核心的执行层抽象，几乎所有 Chain / Agent / Graph 最终都作为 Runnable 来调度执行。([LangChain](https://python.langchain.com/api_reference/core/index.html "langchain-core: 0.3.78 —  LangChain  documentation")) |
| **Prompt / PromptTemplate**          | `BasePromptTemplate`, `ChatPromptTemplate`, `StringPromptValue`, `ChatPromptValueConcrete` 等 | 定义“如何把输入转成模型能接收的 prompt”     | PromptTemplate 是把结构化输入（变量）填进模板后形成具体 prompt                                                                                                                                                |
| **Messages / Chat Message**          | `BaseMessage`, `ChatMessage`, `AIMessage`, `HumanMessage`, `SystemMessage` 等                 | 抽象对话消息的结构                    | 在 chat 模型场景下，用消息作输入 / 输出单位                                                                                                                                                                |
| **Language Models / Chat Models 抽象** | `BaseLanguageModel`, `BaseChatModel`                                                         | 定义调用模型的标准接口                  | 高层模型封装（如 OpenAI，Anthropic）会继承这些接口                                                                                                                                                         |
| **Output Parsers**                   | `BaseOutputParser`, `BaseLLMOutputParser`, `JsonOutputParser` 等                              | 把模型的“自由文本输出”解析成结构化数据         | 例如把 JSON 字符串解析成 dict、把列表形式的输出变成 Python list 等                                                                                                                                             |
| **Tools / Tool 抽象**                  | `BaseTool`, `StructuredTool` 等                                                               | 抽象“模型可以调用的外部函数 / 工具”         | 例如检索器、计算器、API 调用等，都可以通过 Tool 接入                                                                                                                                                           |
| **Retrievers**                       | `BaseRetriever`                                                                              | 抽象“从文档集合 / 向量库中检索相关文档”       | 在检索增强（RAG）场景下使用                                                                                                                                                                           |
| **Stores / 存储抽象**                    | `BaseStore`, `InMemoryStore`, `InMemoryByteStore`                                            | 抽象键值 / byte 存储               | 可用于缓存、状态存储等用途                                                                                                                                                                             |
| **Structured Query / 查询表达式**         | `StructuredQuery`, `Expr`, `Comparison` 等                                                    | 用于将逻辑查询表达成结构化形式              | 在一些索引 / 检索器接口中用于过滤或复杂检索                                                                                                                                                                   |
| **Callbacks / 事件 / 跟踪**              | `BaseCallbackHandler`, `CallbackManager`, `RunManager` 等                                     | 提供“执行过程”的钩子机制（日志、监控、调试）      | 用于中间状态跟踪、监控、可视化等扩展                                                                                                                                                                        |
| **Exceptions / 错误处理**                | `LangChainException`, `OutputParserException`, `TracerException` 等                           | 定义整个 core 层可能出现的标准错误         | 在扩展自定义组件时，按这些错误类型抛出可以保证一致性                                                                                                                                                                |

---

## 🔄 各核心抽象之间的关系（执行流程视图）

下面是一个简化的执行流程（pipeline）视角，帮助你理解这些核心抽象是如何协作的：

1. **输入变量 / PromptTemplate**
    
    - 用户提供结构化输入（如 `{"topic": "AI 在金融"}`
        
    - PromptTemplate 根据模板把这些变量填入生成一个具体提示文本（prompt）
        
2. **Runnable / 执行单元**
    
    - Prompt → 模型调用 → 得到原始响应（raw output）
        
    - 这个执行单元一般就是一个 `Runnable` 或一串 `RunnableSequence`
        
3. **模型调用（Language Model 抽象层）**
    
    - `BaseLanguageModel` / `BaseChatModel` 等接收 prompt 或 messages，发送请求到具体模型（如 OpenAI）
        
    - 返回一些结果结构，如 `LLMResult` / `ChatResult` / `ChatGeneration`
        
4. **Output Parser 解析**
    
    - 模型输出通常是自由文本，OutputParser 将其解析为结构化数据（比如 dict、list、Pydantic 对象等）
        
5. **Tool / Retriever 调用（可选）**
    
    - 如果模型在执行过程中需要调用外部工具（如检索、API）：
        
        - 这些工具会被抽象为 `BaseTool` 或 `StructuredTool`
            
        - Runnable 在执行时可能会触发工具调用，再将结果嵌入流程中继续执行
            
6. **Callbacks / Tracing**
    
    - 在执行各个阶段，系统可以触发 callback 事件（如“模型调用开始／结束”“tool 调用”“输出解析”等）
        
    - 这样可以用于日志、监控、调试、可视化
        
7. **State / Store / Cache（可选）**
    
    - 在某些场景下，需要把中间结果或状态存下来，用于重用、缓存或跨步传递
        
    - 这时候可用 `BaseStore`、`InMemoryStore` 等作为存储抽象
        

---

## ✅ 重点理解与学习建议

从 `langchain-core` 的结构来看，以下几方面是你在学习和扩展时一定要重点掌握的：

- **Runnable / 执行抽象**：几乎所有流程最终都要转化为 Runnable，理解它的组合能力（串行 / 并行 / 分支）非常关键
    
- **Prompt 与 Message 抽象**：要理解 “PromptTemplate + Message” 在不同模型类型（文本模型 / 聊天模型）下的适配方式
    
- **输出解析 (OutputParser)**：模型输出千差万别，一个好的解析器能显著提升系统稳定性
    
- **Tool 与 Retriever 抽象**：把外部能力（检索、数据库、API）以统一方式接入核心流程
    
- **Callbacks / Trace / 管理流程**：执行过程中插入监控 / 日志 / 可视化钩子，是开发 / 调试 / 监控系统必不可少的能力
    
