高质量的、经过标注的指令微调数据集是训练出强大 Planner 的“养料”。这个领域发展非常迅速，已经涌现出了一批非常著名且公开的数据集。

这些数据集的核心特点是，它们不再是简单的“问-答”对，而是包含了 **“目标 -> 推理过程 -> 行动序列”** 的完整轨迹。

以下是一些在学术界和工业界都非常有影响力的著名数据集，它们大多可以在 Hugging Face 等平台上找到：

## 常见的指令微调开源数据集

### 1. ToolAlpaca / ToolInstruct

这是早期探索中非常经典和基础的数据集，它将指令微调的思想从通用对话扩展到了工具使用上。

- **核心思想**：通过“知识蒸馏”的方式，使用一个强大的教师模型（如 OpenAI 的 GPT-3.5/4）来生成大量的“如何使用工具”的示例，然后用这些示例来微调一个开源的小模型。
  
- **数据格式**：通常包含用户指令、可用的工具API描述、以及模型生成的包含思考过程和API调用的完整响应。
  
    - **示例**:
      
        ```json
        {
          "instruction": "帮我查一下今天上海的天气怎么样？",
          "input": "可用工具: search(query: str)",
          "output": "我需要查询天气，所以我应该使用搜索工具。search(query='上海今天天气')"
        }
        ```
    
- **著名项目**:
  
    - **ToolAlpaca**: [https://github.com/TangCoT/ToolAlpaca](https://www.google.com/search?q=https://github.com/TangCoT/ToolAlpaca)
      
    - **ToolInstruct**: Hugging Face 上有多个以此命名的变体，它们遵循类似的思想。
    
- **为何著名**：它是将 Alpaca 的指令微调范式成功应用到 Agent 工具使用领域的先行者，证明了通过合成数据微调小模型来学习工具使用的可行性。

### 2. ToolBench / ToolLLaMA

这是一个规模巨大、影响力极强的项目，旨在构建一个能够使用数千个真实世界 API 的 Agent。它的数据集是目前最全面、最接近真实应用场景的之一。

- **核心思想**：构建一个包含大量真实、多样化 API 的工具集，并自动生成高质量的指令树来覆盖这些 API 的复杂调用场景，最终训练出能熟练使用这些工具的 Agent (ToolLLaMA)。
  
- **数据格式**：数据以“对话树”或“决策树”的形式存在。每一步都包含模型的思考（`thought`）、得出的结论（`conclusion`）、以及要执行的 API 调用（`tool_code`）。
  
- **数据集**: **ToolBench** (数据集和评测基准)
  
    - **Hugging Face**: [https://huggingface.co/datasets/ToolBench/ToolBench_Dataset](https://www.google.com/search?q=https://huggingface.co/datasets/ToolBench/ToolBench_Dataset)
      
    - **项目主页**: [https://toolbench.github.io/](https://www.google.com/search?q=https://toolbench.github.io/)
    
- **为何著名**：
  - **规模宏大**：覆盖了16000+ 个真实世界的 API。
      
    - **高质量合成**：它提出了一种名为 "Depth-First Search-based Instruction Generation" (DFSIGN) 的方法，可以自动探索和生成复杂的工具调用链，数据质量非常高。
      
    - **端到端**：它不仅提供了数据集，还提供了评测环境和经过训练的模型（ToolLLaMA），是一个完整的生态。

### 3. AgentInstruct

这是一个混合了多种 Agent 任务的高质量、人工策划的数据集，旨在提升 Agent 在多种场景下的综合能力。

- **核心思想**：研究人员发现，现有的 Agent 数据集往往只专注于单一类型的任务（如只用 API，或只浏览网页）。AgentInstruct 则将不同来源、不同格式的 Agent 交互数据统一成一种标准格式，进行混合训练。
  
- **数据构成**：它融合了至少6个不同的 Agent 任务领域，包括：
  
    1. 工具使用 (Tool Usage)
       
    2. 网页浏览 (Web Browse - 来自 Mind2Web)
       
    3. 数据库操作 (SQL Generation)
       
    4. 代码生成 (Code Generation)
       
    5. 数学推理 (Reasoning)
       
    6. 一些开放式对话任务
    
- **数据集**: **AgentInstruct**
  - **Hugging Face**: [https://huggingface.co/datasets/WizardLM/AgentInstruct](https://www.google.com/search?q=https://huggingface.co/datasets/WizardLM/AgentInstruct)
    
- **为何著名**：它验证了一个非常重要的观点——**通过在多样化的 Agent 任务上进行联合训练，可以显著提升模型作为通用 Agent 的泛化能力**。训练出的模型在各项 Agent 基准测试上都表现优异。

### 4. Mind2Web

这个数据集专注于训练和评测那些需要在真实网站上完成复杂任务的 **Web Agent**。

- **核心思想**：在大量真实网站上，由人类标注员记录下完成特定任务（如“在亚马逊上找到价格低于50美元的蓝色跑鞋并加入购物车”）的完整操作轨迹。
  
- **数据格式**：每一条数据都是一个目标指令，对应着一系列在网页DOM元素上的操作（如 `CLICK`, `TYPE`, `SELECT`）。它不仅包含了操作，还包含了操作前的思考过程。
  
- **数据集**: **Mind2Web**
  
    - **Hugging Face**: [https://huggingface.co/datasets/osunlp/Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web)
      
    - **项目主页**: [https://mind2web.github.io/](https://www.google.com/search?q=https://mind2web.github.io/)
    
- **为何著名**：它是目前 Web Agent 领域规模最大、最权威的基准之一。由于其数据来源于真实世界、任务复杂度高，能够非常有效地评测和训练 Agent 在复杂网页环境中的导航和交互能力。

---

### 如何寻找更多数据集？

1. **Hugging Face Datasets Hub**: 这是寻找数据集的宝库。你可以使用关键词搜索，如 `agent`, `tool use`, `instruction`, `reasoning`。
   
2. **Papers with Code**: 当你读到一篇关于 Agent 的新论文时，可以去这个网站上查找该论文，通常它会链接到相关的代码库和数据集。
   
3. **GitHub**: 直接在 GitHub 搜索，很多研究团队会将他们的数据集和代码开源。
   

**总结**：这些公开数据集极大地推动了 AI Agent 领域的发展，使得研究人员和开发者不必从零开始标注数据，而是可以站在巨人的肩膀上，快速地训练和验证自己的 Agent 模型和框架。



## 在开源数据集基础上研究的相关论文

### 方向一：构建开源、可复现的 Agent 框架

这类研究的目标是利用现有数据集，构建一个完整的、开源的 Agent 系统，让社区可以方便地使用、复现和扩展。

#### 1. **OpenAgents: An Open Platform for Language Agents in the Wild**

- 核心贡献:

  这篇论文发布了一个名为 OpenAgents 的开源平台，集成了三种不同类型的 Agent：

  1. **数据 Agent (Data Agent)**：用于处理数据分析任务，如操作表格、绘图。
  2. **插件 Agent (Plugins Agent)**：类似于 ChatGPT 的插件系统，能够使用超过 200 个日常工具。
  3. **网页 Agent (Web Agent)**：用于在真实网站上执行复杂操作。

- **如何使用数据集**:

  - **ToolBench**: 论文使用 ToolBench 数据集来训练其 **Plugins Agent**，使其具备强大的、多样的工具调用能力。
  - **其他**: 同时它也借鉴了 WebShop 和 Mind2Web 等的思想来构建其 Web Agent。

- **为何值得关注**: 它不是一个孤立的算法，而是一个集大成的 **开源平台**。对于想要快速搭建和体验强大 Agent 能力的开发者来说，这篇论文及其附带的代码库 (https://github.com/xlang-ai/OpenAgents) 是一个绝佳的起点。

- **论文链接**: https://arxiv.org/abs/2310.10634

------



### 方向二：改进 Agent 的规划、反思与微调方法

这类研究专注于提升 Agent 的“大脑”（Planner），让它能进行更优的规划、从失败中学习，或者通过更高效的微调方法来掌握新技能。

#### 2. **FireAct: Toward Language Agent Fine-tuning**



- 核心贡献:

  FireAct 提出了一种新的、更高效的 Agent 微调框架。它认为简单的模仿学习（Behavior Cloning）是不够的，Agent 需要更深刻地理解任务的 “可解性” (solvability) 和 “轨迹质量” (trajectory quality)。它通过特定的Token来标记任务是否可解，并让模型学习区分高质量和低质量的解决路径。

- **如何使用数据集**:

  - **ToolBench**: FireAct 在 ToolBench 数据集上对其方法进行了广泛的评测，证明了其微调框架能够显著提升 LLaMA 等模型在工具使用任务上的性能。
  - **WebShop**: 同时也在 WebShop 这一电商购物环境的基准上验证了其有效性。

- **为何值得关注**: 它深入到了 Agent 微调的细节，超越了简单的模仿。这篇论文告诉我们，**如何“聪明地”利用已有的数据集进行微调**，是提升 Agent 能力的关键。

- **论文链接**: https://arxiv.org/abs/2310.05915

### 方向三：提升 Web Agent 的理解与交互能力

这类研究专注于 Web 场景，探索如何让 Agent 更好地理解网页的结构和视觉信息，从而完成更复杂的操作。

### 总结与阅读建议

| 论文           | 核心方向        | 使用的主要数据集 | 亮点                                      |
| -------------- | --------------- | ---------------- | ----------------------------------------- |
| **OpenAgents** | 开源 Agent 平台 | ToolBench        | 提供了立即可用的 Agent 系统和代码         |
| **FireAct**    | 高效微调方法    | ToolBench        | 探索如何更聪明地利用数据进行训练          |
| **Agent-Pro**  | 规划与反思      | ToolBench        | 提出了“计划-优化”的新范式，提升长任务性能 |
| **WebAgent**   | 多模态网页浏览  | Mind2Web         | 融合视觉和语言，解决复杂网页交互问题      |

**如何有效阅读这些论文？**

1. **先看摘要和结论**：快速了解论文的核心思想和主要成果。
2. **重点关注“方法论” (Methodology)**：这部分会详细描述其 Agent 架构、Prompt 设计、训练细节。
3. **细读“实验” (Experiments)**：查看他们是如何使用 ToolBench、Mind2Web 等数据集进行评测的，关注他们与 ReAct、Plan-and-Solve 等基线方法的对比结果。
4. **查找代码库**：如果论文开源了代码，一定要去 GitHub 上看一看。阅读代码和运行示例是理解其思想最深刻的方式。