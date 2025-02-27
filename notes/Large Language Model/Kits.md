# Model Train Kits





# Model Inference Kits

## vLLM

vLLM是一个用于加速大语言模型推理的开源库，主要通过智能批处理和显存优化技术，使大模型在多GPU环境中高效运行。以下是vLLM的详细介绍：

### 核心特性
- **高吞吐量和低延迟**：vLLM采用动态KV缓存机制和PagedAttention算法，实现了KV缓存内存几乎零浪费，解决了大语言模型推理中的内存管理瓶颈问题。与Hugging Face Transformers相比，其吞吐量提升了24倍。
- **多硬件支持**：vLLM不仅支持Nvidia GPU，还支持AMD GPU、Intel GPU、AWS Neuron和Google TPU等市面上众多硬件架构。
- **易于使用**：vLLM提供了简单易用的API，支持离线批量推理和在线推理服务。
- **兼容多种模型**：vLLM支持超过40个模型架构，包括常见的开源大模型。
- **优化的CUDA内核**：vLLM集成了FlashAttention和FlashInfer等优化技术，进一步提升了推理性能。

### 技术原理
- **PagedAttention算法**：vLLM采用了PagedAttention算法，这是一种受操作系统虚拟内存和分页技术启发的注意力算法，可以有效地管理注意力键和值，从而提高内存利用率。
- **动态KV缓存机制**：vLLM通过动态KV缓存机制，实现了KV缓存内存的高效管理，减少了内存浪费。
- **多步调度和异步输出处理**：vLLM引入了多步调度和异步输出处理技术，优化了GPU的利用率并提高了处理效率。

### 功能
- **离线批量推理**：vLLM可以对数据集进行离线批量推理，生成输入提示列表的文本。
- **在线推理服务**：vLLM可以作为一个实现了OpenAI API协议的服务器进行部署，支持模型列表、创建聊天补全和创建完成等端点。
- **兼容OpenAI API**：vLLM可以作为使用OpenAI API的应用程序的直接替代品，支持相同的查询格式。

### 应用场景
- **高性能推理**：vLLM适用于需要高性能推理的场景，如在线问答系统、聊天机器人等。
- **多硬件环境部署**：vLLM支持多种硬件架构，适用于需要在不同硬件环境下部署大语言模型的场景。
- **大规模模型部署**：vLLM支持大规模模型的推理和部署，适用于需要处理大量请求的场景。

### 优势
- **高性能**：vLLM通过优化技术实现了高吞吐量和低延迟，提升了大语言模型的推理性能。
- **多硬件支持**：vLLM支持多种硬件架构，具有良好的硬件兼容性。
- **易于使用**：vLLM提供了简单易用的API，降低了使用门槛。
- **兼容性强**：vLLM兼容多种模型和API，具有良好的兼容性。

vLLM是一个高性能、易于使用的大语言模型推理库，适用于需要高性能推理和多硬件支持的场景。



## LightLLM



## DeepSpeed-MII



## TensorRT-LLM





# Model Deploy Kits

## Ollama

Ollama 是一个开源的大型语言模型（LLM）服务工具，旨在帮助用户在本地环境中轻松部署和运行各种 LLM 模型。通过简单的安装和配置，用户无需深入了解底层技术即可加载、运行和交互不同的模型。

### 主要特点

- **本地部署**：Ollama 支持在本地机器上运行模型，保障数据隐私和安全。
- **多平台支持**：兼容 macOS、Linux 和 Windows 操作系统，满足不同用户的需求。
- **丰富的模型库**：提供多种预置模型，如 Llama3.3、DeepSeek-R1、Gemma2 等，用户可根据需求选择合适的模型。
- **易于使用**：提供直观的命令行界面和服务器，简化模型的下载、运行和管理。
- **可扩展性**：支持自定义配置和模型导入，用户可以根据需要调整模型参数或导入自定义模型。

### 适用场景

- **个人研究**：研究人员可以在本地环境中测试和验证不同的 LLM 模型，进行学术研究。
- **企业应用**：企业可在内部部署 Ollama，开发和测试基于 LLM 的应用，确保数据不泄露。
- **教育培训**：教育机构可利用 Ollama 为学生提供实践平台，学习和体验前沿的 AI 技术。

### 提供的服务

- **模型管理**：下载、运行和管理各种开源 LLM 模型。
- **命令行接口**：通过简单的命令行操作与模型进行交互。
- **API 支持**：提供终端 API 和 Python API，方便开发者集成到其他应用中。

### 集成方法

1. **安装 Ollama**：
   
    - https://ollama.com/
    
2. **运行模型**：
   
    - 安装完成后，可通过以下命令下载并运行模型：
      
        ```bash
        ollama pull llama3.3
        ollama run llama3.3
        ```
        
    - 对于自定义模型，可创建 `Modelfile` 文件，指定模型路径和参数，然后使用以下命令创建并运行模型：
      
        ```bash
        ollama create my_custom_model -f Modelfile
        ollama run my_custom_model
        ```
    
3. **API 集成**：
   
    - Ollama 提供 REST API 和 Python API，方便开发者将其集成到其他应用中。
    - REST API 示例：
      
        ```bash
        curl http://localhost:11434/api/generate -d '{
          "model": "llama3.3",
          "prompt": "Hello, world!"
        }'
        ```
        
    - Python API 示例：
      
        ```python
        import requests
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.3",
                "prompt": "Hello, world!"
            }
        )
        print(response.json())
        ```
        

### 相关文档和资源

- **Ollama 官方网站**：[https://ollama.com](https://ollama.com/)
- **模型库**：[https://ollama.com/library](https://ollama.com/library)
- **GitHub 仓库**：[https://github.com/jmorganca/ollama](https://github.com/jmorganca/ollama)
- **中文教程**：[https://sspai.com/post/85193](https://sspai.com/post/85193)





## xinference

Xinference 是一个开源的分布式推理框架，旨在简化各种 AI 模型的运行和集成。通过 Xinference，用户可以在本地或云端环境中轻松部署和运行大语言模型（LLM）、嵌入模型、多模态模型等，创建强大的 AI 应用。

### 主要特点

- **多模型支持**：兼容多种模型类型，包括大语言模型、语音识别模型和多模态模型，满足不同应用需求。
- **分布式架构**：支持分布式部署，适用于从个人设备到大规模集群的多种环境。
- **高效推理**：集成多种推理引擎，如 Transformers、vLLM、Llama.cpp 和 SGLang，提供高性能的模型推理能力。
- **易于集成**：提供兼容 OpenAI 的 RESTful API、Python SDK、命令行工具和 Web 界面，方便开发者快速集成和使用。
- **灵活部署**：支持本地安装、Docker 容器和 Kubernetes 等多种部署方式，满足不同环境的需求。

### 适用场景

- **研究与开发**：研究人员和开发者可以使用 Xinference 部署和测试最新的 AI 模型，加速研究进程。
- **企业应用**：企业可以利用 Xinference 构建和部署 AI 服务，如智能客服、内容生成和数据分析等，提高业务效率。
- **教育培训**：教育机构可以借助 Xinference 为学生提供实践平台，学习和体验前沿 AI 技术。

### 提供的服务

- **模型管理**：支持模型的下载、部署、运行和管理，用户可以轻松加载和切换不同的模型。
- **多种接口**：提供兼容 OpenAI 的 RESTful API、Python SDK、命令行工具和 Web 界面，满足不同开发和使用习惯。
- **集群管理**：支持在分布式环境中部署和管理模型，适应大规模应用场景。

### 集成方法

1. **安装 Xinference**：

   - 使用 pip 安装：

     ```bash
     pip install xinference
     ```

   - 使用 Docker 部署：

     ```bash
     docker run -e XINFERENCE_MODEL_SRC=modelscope -p 9998:9997 --gpus all xprobe/xinference:<your_version> xinference-local -H 0.0.0.0 --log-level debug
     ```

     将 `<your_version>`替换为所需的 Xinference 版本，如 `v0.13.0`。

2. **运行模型**：

   - 启动本地服务：

     ```bash
     xinference-local --host 0.0.0.0 --port 9997
     ```

   - 部署模型： 以下示例展示了如何使用命令行工具部署 llama-2-chat 模型：

     ```bash
     xinference launch --model-engine <inference_engine> -u my-llama-2 -n llama-2-chat -s 13 -f pytorch
     ```

     其中，`<inference_engine>`指定推理引擎，如 `transformers` `vllm` 等。

3. **与模型交互**：

   - 使用 Python SDK：

     ```python
     from xinference.client import RESTfulClient
     
     client = RESTfulClient("http://127.0.0.1:9997")
     model = client.get_model("my-llama-2")
     response = model.chat(
         messages=[
             {"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": "What is the largest animal?"}
         ]
     )
     print(response)
     ```

   - 使用 cURL 命令：

     ```bash
     curl -X POST http://127.0.0.1:9997/v1/chat/completions \
       -H 'Content-Type: application/json' \
       -d '{
         "model": "my-llama-2",
         "messages": [
           {"role": "system", "content": "You are a helpful assistant."},
           {"role": "user", "content": "What is the largest animal?"}
         ]
       }'
     ```

### 相关文档和资源

- **Xinference 官方文档**：https://inference.readthedocs.io/zh-cn/latest/
- **GitHub 仓库**：https://github.com/xorbitsai/inference
- **中文教程**：https://zhuanlan.zhihu.com/p/685224526





## LangChain

LangChain 是一个用于开发由大型语言模型（LLMs）驱动的应用程序的框架。它旨在简化 LLM 应用程序的开发、生产化和部署过程。LangChain 的核心理念是通过将不同的组件“链”在一起，创建更高级的 LLM 应用程序。这些组件包括提示模板、大型语言模型、代理、工具和记忆模块等。

### 主要功能与特性

1. **Chains（链）**：核心组件，用于串联不同的处理步骤，可以是简单的顺序执行，也可以包含复杂的条件和循环逻辑。Chains 可以嵌套和组合，构建复杂的流程，如顺序执行、条件判断和循环等。

2. **Memory（记忆）**：支持短期和长期记忆，允许在任务或会话中存储和检索信息，增强模型的上下文理解能力。Memory 模块用于在会话或任务中存储信息，使 LLM 能够在后续的交互中引用先前的信息。

3. **Prompt Templates（提示模板）**：提供灵活的模板系统，支持参数化和动态生成，方便构建适合不同场景的模型输入。Prompt Templates 是不同类型提示的模板，例如“chatbot”样式模板、ELI5 问答等。

4. **Agents（代理）**：智能决策模块，能够根据当前状态和目标，动态选择和调用适当的工具或动作来完成任务。Agents 使用 LLMs 决定应采取的操作，可以使用诸如网络搜索或计算器之类的工具，并将所有工具包装成一个逻辑循环的操作。

5. **Tools（工具）**：可执行的功能单元，封装了具体的操作，如查询数据库、调用 API、执行计算等，供代理和链调用。Tools 是可调用的功能单元，封装了具体的操作，如数据库查询、API 调用、数学计算等。

6. **LLMs（大型语言模型）集成**：与各种主流的大型语言模型无缝对接，支持 OpenAI、Hugging Face 等平台，方便模型的替换和比较。LLMs 是整个框架的核心，负责理解输入、生成响应和执行语言相关的任务。

7. **数据连接器**：预置了对常见数据源的支持，如文件系统、数据库、网络请求等，方便数据的获取和存储。数据连接器用于将 LLM 与其他数据源连接在一起，使应用程序能够访问和处理外部数据。

8. **错误处理与调试**：内置了异常捕获和日志记录机制，帮助开发者及时发现和解决问题。LangChain 提供了完善的异常处理和容错机制，包括异常捕获、错误分类与处理策略、重试与降级机制、日志记录与监控等。

### 架构设计与模块划分

1. **Chain（链）**：核心流程控制单元，负责串联不同的组件和步骤，定义应用程序的执行逻辑。Chains 可以传递上下文和数据，从而使不同的模块之间能够共享信息。

2. **Memory（记忆）**：用于在会话或任务中存储信息，使 LLM 能够在后续的交互中引用先前的信息。Memory 模块管理对话的状态和历史，提高模型的连贯性和上下文理解。

3. **Prompt Templates（提示模板）**：提供统一的模板来生成 LLM 的输入，确保提示的一致性和质量。Prompt Templates 支持在模板中使用占位符和变量，根据上下文动态生成实际的提示内容。

4. **Agents（代理）**：决策引擎，负责根据当前的状态和目标，决定下一步的操作，如调用哪个工具或执行何种动作。Agents 可以自主选择最合适的策略和工具来完成任务。

5. **Tools（工具）**：功能扩展单元，封装了具体的操作，如数据库查询、API 调用、数学计算等。Tools 提供标准接口，方便代理和链进行调用。

6. **LLMs（大型语言模型）**：核心处理器，负责理解输入、生成响应和执行语言相关的任务。LLMs 支持与各种主流的 LLM 平台集成，如 OpenAI GPT-4、Hugging Face 等。

### 工作流程与执行过程

1. **异常处理**：LangChain 在链、代理和工具的内部对可能发生的异常进行捕获和处理，如网络超时、数据格式错误等。对于未被捕获的异常，系统会进行统一的处理，防止程序崩溃。

2. **错误分类与处理策略**：对于已知的错误类型，提供明确的错误信息和解决方案提示。对于未预料到的错误，记录详细的日志，返回通用的错误信息，防止敏感信息泄露。

3. **重试与降级机制**：对于临时性错误，如网络波动，可进行一定次数的自动重试。在某些功能不可用时，提供简化的替代方案，确保核心功能的可用性。

4. **日志记录与监控**：记录系统的运行状态和异常信息，方便后续的调试和分析。结合监控工具，实时跟踪系统的性能和错误，及时发现和处理问题。

5. **用户友好性**：在发生异常时，提供清晰、易懂的错误信息，指导用户进行下一步操作。允许用户报告错误或问题，促进系统的持续改进。

6. **资源清理与恢复**：在异常发生后，及时释放占用的资源，如内存、文件句柄等。在可能的情况下，恢复到安全的系统状态，避免数据损坏或进一步的错误。

### 生态系统

1. **LangChain-core**：基础抽象和 LangChain 表达式语言。

2. **LangChain-community**：第三方集成。

3. **合作伙伴库**：如 `langchain-openai`、`langchain-anthropic` 等，一些集成已进一步拆分为自己的轻量级库，仅依赖于 `langchain-core`。

4. **LangChain**：构成应用程序认知架构的链、代理和检索策略。

5. **LangGraph**：通过将步骤建模为图中的边和节点，构建强大且有状态的多参与者应用程序。与 LangChain 无缝集成，但也可以单独使用。

6. **LangServe**：将 LangChain 部署为 REST API。

7. **LangSmith**：一个开发人员平台，允许您调试、测试、评估和监控 LLM 应用程序。

### 入门指南

- **安装 LangChain**：可以通过 pip 安装 LangChain 及其依赖项。
- **快速入门**：通过构建第一个 LangChain 应用程序来熟悉该框架。
- **文档与教程**：查阅 LangChain 的官方文档和教程，了解如何使用各个组件和模块。



LangChain 是一个功能强大且灵活的框架，适用于开发各种由大型语言模型驱动的应用程序。它提供了丰富的组件和模块，支持多种应用场景，并且具有良好的扩展性和可维护性。通过使用 LangChain，开发者可以更高效地创建复杂的自然语言处理应用程序。



## FastGPT

FastGPT 是一个基于大型语言模型（LLM）的知识库问答系统，专注于高效对话和任务处理。以下是关于 FastGPT 的详细介绍：

### 功能

- **专属 AI 客服**：通过导入文档或问答对进行训练，AI 模型可以根据文档内容以交互式对话方式回答问题。
- **可视化工作流编排**：基于 Flow 模块，用户可以设计复杂的工作流，实现自动化和智能化的处理流程。
- **自动数据预处理**：支持多种文档格式（如 Word、PDF、Excel、Markdown 等）的导入，自动完成文本预处理、向量化和问答分割，节省手动训练时间。
- **强大的 API 集成**：API 接口对齐 OpenAI 官方接口，可以轻松集成到企业微信、公众号、飞书等平台。
- **多模型兼容性**：支持 GPT、Claude、文心一言等多种 LLM 模型。

### 特点

- **高效性**：经过优化，能够快速生成高质量内容，适合实时性要求较高的场景。
- **开源性**：遵循附加条件的 Apache License 2.0 开源协议，用户可以进行二次开发。
- **个性化与定制化**：可以根据具体业务需求定制训练，调整语言风格、语气或行业专用术语。
- **数据处理能力强**：能够整合大量非结构化数据，自动优化答案质量和查询效率。
- **用户友好**：对话流畅，具备自然的语义理解能力，适合长对话和复杂任务。

### 应用场景

- **企业场景**：
  - **文档管理与信息检索**：帮助企业管理和检索大量文档，如业务报告、销售数据报告、员工档案等。
  - **企业业务流程支持**：通过工作流编排实现复杂的问答场景，如查询数据库、查询库存、预约实验室等。
  - **特定业务需求解答**：在金融、医疗等领域提供特定业务需求的解答，如信用评估、欺诈检测、医疗影像诊断等。
- **个人场景**：
  - **个人知识管理**：帮助个人管理和检索自己的文档、笔记等，提高学习和工作效率。
  - **个人创作辅助**：在内容创作方面提供辅助支持，如创意构思、语法检查、词汇拓展等。

### 技术细节

- **数据预处理**：将文本拆分成更小的片段，如句子、段落，并进行向量化处理。
- **索引构建**：将生成的语义向量构建成索引，通常包括向量索引和倒排索引，以提高检索速度。
- **检索与匹配**：当用户查询时，查询内容也会被向量化，并在索引中匹配相关的语义向量。
- **AI 对话**：利用大模型的上下文理解能力，理解用户需求，生成自然回答。

FastGPT 是一个功能强大且灵活的工具，适用于多种场景，能够显著提高信息检索和处理的效率。
