[[2210.03629] ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

Agent在落地应用中技术经验分享 - YWTAI的文章 - 知乎
https://zhuanlan.zhihu.com/p/1917203229382542865

![img](https://pic1.zhimg.com/v2-57dc30d3a639f6ccc663d57ae217c93e_r.jpg)

![img](https://picx.zhimg.com/v2-b3eee6d52a58d6584517c84b188edde5_r.jpg)

![img](https://pic1.zhimg.com/v2-1a70d2fb167bcadaa140084f519f353a_r.jpg)



### ReAct 模式：推理与行动的动态循环

**ReAct** 是 “**Re**asoning and **Act**ing” 的缩写，意为“推理与行动”。这个框架的核心思想是让大语言模型（LLM）不仅仅是生成文本，更是模仿人类解决问题的方式：**边想边做，边观察边调整**。

ReAct 框架将任务处理过程分解为一个由三个步骤组成的动态循环：

1. **Thought (思考):** Agent 首先分析当前的目标和已有信息，进行推理，决定下一步应该采取什么具体行动来获取更多信息或接近目标。
2. **Action (行动):** 基于思考的结果，Agent 会执行一个具体的动作。这个动作通常是调用一个外部工具（Tool），例如进行一次网络搜索、查询数据库、调用计算器或访问某个API。
3. **Observation (观察):** Agent 接收并“观察”执行动作后返回的结果。这个结果（例如搜索到的信息、计算结果等）会成为下一轮“思考”的输入，帮助 Agent 评估进展并调整后续策略。

这个 “思考 → 行动 → 观察” 的循环会不断重复，直到 Agent 认为已经收集到足够的信息并可以得出最终答案为止。

**简单示例：** Agent 被要求回答 “提出《星际穿越》电影想法的人，他还导演了哪些评分超过8.0的电影？”

- **Thought 1:** 我需要先找出是谁提出了《星下穿越》的想法。
- **Action 1:** `search("Interstellar movie concept origin")`
- **Observation 1:** 搜索结果显示，这个想法最初由理论物理学家基普·索恩（Kip Thorne）提出，剧本由乔纳森·诺兰（Jonathan Nolan）编写，并由克里斯托弗·诺兰（Christopher Nolan）执导。问题问的是导演，所以关键人物是克里斯托弗·诺兰。
- **Thought 2:** 现在我知道了导演是克里斯托弗·诺兰。我需要查找他导演的所有电影以及它们的评分，然后筛选出评分超过8.0的。
- **Action 2:** `search("Christopher Nolan directed movies and ratings")`
- **Observation 2:** 搜索结果返回一个列表，例如：《盗梦空间》(Inception, 9.3)、《黑暗骑士》(The Dark Knight, 9.0)、《致命魔术》(The Prestige, 8.5)、《信条》(Tenet, 7.8) 等。
- **Thought 3:** 我已经获得了所有需要的信息。现在我可以根据观察到的评分列表，筛选出超过8.0的电影并给出最终答案。
- **Final Answer:** 克里斯托弗·诺兰还导演了《盗梦空间》、《黑暗骑士》、《致命魔术》等评分超过8.0的电影。

**核心优势：** ReAct 模式非常灵活，适应性强，特别适合处理那些信息不完整、需要与外部世界动态交互才能解决的问题。它能有效处理意外情况，并根据中间步骤的结果即时调整策略。

### Plan-and-Execute 模式：先规划，后执行

**Plan-and-Execute**（规划与执行）则采用了更为结构化的两步走策略。

1. **Planning (规划):** 首先，Agent 会全面分析用户的初始请求，并一次性制定出一个详尽的、分步骤的行动计划（Plan）。这个计划会列出解决整个问题所需的所有关键步骤和它们的执行顺序。
2. **Execution (执行):** 计划一旦制定完成，Agent 就会严格按照这个计划列表，一步一步地执行所有动作。在这个阶段，通常不会再进行动态的推理和计划调整。它会顺序执行完所有规划好的步骤，然后汇总结果。

**简单示例：** 同样是上面的问题。

- **Planning Phase:**
  1. 搜索《星际穿越》的导演是谁。
  2. 搜索该导演执导的全部电影列表。
  3. 获取列表中每部电影的评分。
  4. 筛选出评分超过8.0的电影。
  5. 整合筛选结果，形成最终答案。
- **Execution Phase:** Agent 会依次执行上述1到5步，中间不做停顿或重新规划，最后直接输出结果。

**核心优势：** 对于那些步骤明确、依赖关系清晰的复杂任务，Plan-and-Execute 模式非常可靠且高效。由于它预先生成了完整的计划，可以更好地预估任务的复杂度和执行成本，并且在执行阶段的调用开销（token消耗）通常更低。

### ReAct vs. Plan-and-Execute：核心区别对比

| 特征         | ReAct 模式                                                 | Plan-and-Execute 模式                                        |
| ------------ | ---------------------------------------------------------- | ------------------------------------------------------------ |
| **决策风格** | 迭代式、动态调整                                           | 前瞻性、一次性规划                                           |
| **架构**     | 思考、行动、观察的紧密循环                                 | 规划和执行两个独立且清晰的阶段                               |
| **适应性**   | **非常高**。能灵活应对环境变化和意外结果。                 | **较低**。如果计划中的某一步出错，可能导致整个任务失败，需要重新规划。 |
| **适用场景** | 动态交互、信息检索、需要实时反馈的任务。                   | 流程固定、步骤明确、依赖关系清晰的复杂任务（如代码生成、报告撰写）。 |
| **潜在缺点** | 对于长任务，可能因过多的循环和思考导致效率较低、成本较高。 | 对任务的初始理解要求很高，计划一旦出错则难以纠正。           |



### 触发Tool的具体流程

在ReAct框架中，**Tool（工具）的触发并非由大语言模型（LLM）直接执行，而是通过一个“约定好的格式化文本”作为信号，由外部的Agent执行框架来解析和调度的。**

我们可以把整个系统想象成一个“大脑”和一双“手”的协作：

- **大脑 (The Brain):** 大语言模型 (LLM)，负责思考、推理和决策。
- **手 (The Hands):** 具体的工具代码 (Tool)，例如网络搜索API、计算器程序、数据库查询函数等。
- **中枢神经系统 (The Central Nervous System):** Agent执行框架或编排器 (Orchestrator/Agent Framework)，负责连接大脑和手，传递指令。

所谓的“触发”，就是“大脑”通过“神经系统”向“手”下达指令的过程。这个过程具体分为以下几个关键步骤：

#### 第1步：在提示（Prompt）中定义规则

一切的起点是Prompt。在任务开始时，Agent框架会构建一个特殊的Prompt发给LLM。这个Prompt里不仅包含了用户的目标，最关键的是**定义了LLM与工具交互的规则**，包括：

1. **可用工具列表：** 明确告知LLM它有哪些“手”可以用，以及每只“手”的功能。例如：
   - `search(query)`: 用于在互联网上搜索信息。
   - `calculator(expression)`: 用于计算数学表达式。
2. **输出格式约定：** 强制要求LLM必须遵循特定的格式来输出它的思考过程和行动指令。这就是ReAct的核心格式：
   - `Thought:` 描述LLM的思考过程和下一步计划。
   - `Action:` 指定要调用的工具名称和所需参数。

#### 第2步：LLM生成“行动指令”文本

当LLM接收到这个包含规则的Prompt后，它会进行推理。当它认为需要使用工具时，它不会去执行代码，而是会**生成一段严格符合约定格式的文本**。

例如，对于问题“苹果公司的CEO现在是谁？他几岁了？”，LLM可能会生成如下文本字符串：

Plaintext

```
Thought: 我需要先找出苹果公司的现任CEO是谁。我可以使用搜索工具。
Action: search(query="who is the current CEO of Apple")
```

**请注意：** 上面的 `Action: search(...)` **仅仅是文本**，是LLM的输出结果，它本身没有任何执行能力。

#### 第3步：框架解析（Parse）文本

这是**最关键的触发环节**。Agent的执行框架（例如用Python编写的外部程序）会接收到LLM生成的完整文本。然后，它会像一个解析器一样：

1. **检查文本内容：** 寻找关键词 `Action:`。
2. **提取指令：** 如果找到了 `Action:`，它会用正则表达式或其他字符串处理方法，从这行文本中提取出两部分信息：
   - **工具名称 (Tool Name):** `search`
   - **工具参数 (Tool Arguments):** `query="who is the current CEO of Apple"`

#### 第4步：调度（Dispatch）并执行真正的工具代码

框架在成功解析出工具名称和参数后，会进行“调度”：

1. **查找工具：** 框架内部维护着一个“工具注册表”（通常是一个字典或哈希表），它将工具名称（字符串）映射到实际可执行的函数代码上。

   Python

   ```
   # 示例：一个简单的工具注册表
   def search_implementation(query):
       # ... 这里是调用搜索引擎API的真实代码 ...
       return "Tim Cook is the current CEO of Apple."
   
   tool_registry = {
       "search": search_implementation,
       # "calculator": calculator_function,
   }
   ```

2. **调用函数：** 框架在注册表中找到 `search` 对应的 `search_implementation` 函数，然后将解析出的参数 `query="who is the current CEO of Apple"` 传递给这个函数并执行它。

   Python

   ```
   tool_name = "search"
   tool_args = {"query": "who is the current CEO of Apple"}
   
   # 查找并执行
   function_to_call = tool_registry[tool_name]
   result = function_to_call(**tool_args)
   ```

#### 第5步：将执行结果反馈给LLM

工具代码执行完毕后会返回一个结果（例如，`"Tim Cook is the current CEO of Apple."`）。框架会将这个结果包装成`Observation:`的形式，并连同之前的历史记录一起，再次发送给LLM，开启下一轮的“思考-行动-观察”循环。

Plaintext

```
Observation: Tim Cook is the current CEO of Apple.
Thought: 我知道了CEO是蒂姆·库克。现在我需要找出他的年龄。我将再次使用搜索工具。
Action: search(query="How old is Tim Cook")
```

...然后重复上述第2-5步。