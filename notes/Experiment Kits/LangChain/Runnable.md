# LangChain Core — Runnable 抽象详解

## 目录

1. 概览：为什么需要 Runnable
   
2. Runnable 的核心能力与方法
   
3. 组合原语（LCEL）——Sequence / Parallel / Branch / Router
   
4. 常用 Runnable 实现（Lambda, Generator, Parallel, Sequence, Retry, Fallbacks 等）
   
5. 配置、Tracing 与回调（RunnableConfig / callbacks）
   
6. 流式（Streaming）与 `transform` 的角色
   
7. 如何实现自定义 Runnable（模板 + 注意事项）
   
8. 调试、测试与性能优化建议
   
9. 代码示例集
   
10. 常见问题与陷阱
    

---

## 1. 概览：为什么需要 Runnable

Runnable 是 LangChain Core 中的**通用执行抽象**，用来表示“一个可执行的单元（unit of work）”。它将“调用模型 / 工具 / 函数 / 子链”抽象为统一接口，保证：

- 支持**同步/异步**（sync/async）调用；
  
- 支持**批量**（batch）执行以提高吞吐；
  
- 支持**流式输出**（streaming），用于低延迟用户体验；
  
- 支持**组合**（把多个 Runnable 串/并联）形成复杂的管道（LCEL）。
  

简言之：Runnable 是把不同执行体 "包装" 为可编排、可监控、可重用的执行单元。文档里把它定义为“a unit of work that can be invoked, batched, streamed, transformed and composed”。

---

## 2. Runnable 的核心能力与方法

Runnable 暴露了一组标准方法（同步 + 异步对）用于不同的执行场景：

- `invoke(input, config=None)` / `ainvoke(input, config=None)`
  
    - 把单个输入映射为一个输出（最常用的 API）。
    
- `batch(inputs, config=None)` / `abatch(inputs, config=None)`
  
    - 针对多输入的高效执行。默认实现会并行调用 `invoke`/`ainvoke`（线程池 / asyncio.gather），但子类可以覆写以利用上游 API 的原生批量接口。
    
- `stream(input, config=None)` / `astream(input, config=None)`
  
    - 返回（同步 / 异步）流式输出；用于 LLM 的边生成边消费场景。
    
- `astream_log(input, config=None)`
  
    - 在流式场景下同时返回中间日志 / 事件（便于追踪）。
    
- `abatch_as_completed(...)`
  
    - 并行运行 `ainvoke`，并以完成顺序返回结果（适用于不关心顺序的高并发场景）。
      

此外，Runnable 还提供一组“装饰/改造”方法，使得任何 Runnable 都能方便地添加策略：

- `.with_retry(...)`：添加重试策略（常见与 `tenacity` 参数对齐）。
  
- `.with_config(...)`：为该 runnable 绑定默认 config（例如 tags、max_concurrency）。
  
- `.assign(...)`：在输入/输出上进行赋值/扩展操作。
  
- `.as_tool(...)`：把 Runnable 转为可供 Agent/Tools 调用的结构化工具。
  
- `.bind(...)`：绑定某些参数（类似 partial 应用）。
  

> 注意：所有方法都接受可选 `config`，用于控制运行时行为（标签、追踪 metadata、并发上限等）。

---

## 3. 组合原语（LCEL）——Sequence / Parallel / Branch / Router

LangChain Expression Language（LCEL）为 Runnable 提供了声明式组合语法。最常见的组合原语：

- `RunnableSequence`（`|` 操作符）
  
    - 顺序执行：前一步的输出作为后一步的输入。
      
    - 自动继承 sync/async/batch/stream 的能力（当组成组件都支持时）。
    
- `RunnableParallel`（在 sequence 中使用 dict literal）
  
    - 并行执行：同一输入并发调用多个子 Runnable，返回 key->value 的字典结果。
    
- `RunnableBranch` / `RouterRunnable`
  
    - 条件分支或路由选择，基于输入或某个 key 决定执行哪个分支。
      

组合的核心好处：

- 用户只需关注单个可组合单元，复杂的运行逻辑（并行、批处理、流式）会由组合层自动适配；
  
- 你可以用 `RunnableLambda` 把任意 Python 函数引入管道，用 `RunnableGenerator` 支持 streaming generator。
  

---

## 4. 常用 Runnable 实现（何时用哪个）

- `RunnableLambda(func, afunc=None)`
  
    - 把普通 Python callable 封装为 Runnable（默认将 sync impl 委托给线程池以支持 `ainvoke`）。
      
    - 适合小函数 / 轻逻辑（**不建议用于流式**）。
    
- `RunnableGenerator(transform)`
  
    - 支持把 generator/iterator 转为可流式处理的 runnable，适用于 chunked streaming 场景。
    
- `RunnableSequence`（最常用）
  
    - 串联多个 Runnable，常见于 prompt -> llm -> parser 的链路。
    
- `RunnableParallel` / `RunnableMap`
  
    - 用于同时把输入送到多条子链（eg. 同时跑多个模型或多个后处理器）。
    
- `RunnableWithFallbacks` / `RunnableRetry` / `RunnableAssign` / `RunnablePassthrough` / `RunnablePick`
  
    - 提供健壮性（retry / fallback）、轻量变换（assign/pick/passthrough）等。
    
- `RunnableSerializable` / `DynamicRunnable` / `RunnableConfigurable*`
  
    - 支持序列化/动态配置（在运行时调整行为或序列化到 JSON 的场景）。
      

---

## 5. 配置、Tracing 与回调（RunnableConfig / callbacks）

Runnable 的 `config` 是一个强大且灵活的机制：

- `config` 可以是单个 `RunnableConfig`，也可以是每步不同的 `RunnableConfig` 列表（序列化组合时很有用）。
  
- 常见的 config 字段：`tags`, `metadata`, `max_concurrency`（控制并发），`callbacks`（定制回调处理器）等。
  

**Tracing / 回调**：

- LangChain 提供 CallbackManager/RunManager 体系，可以把日志、监控、trace hook 注入任何 Runnable 的执行周期。
  
- 常见用法：通过 `config={'callbacks': [ConsoleCallbackHandler()]}` 将中间结果打印或上报到监控系统；或把 trace 数据发送到 LangSmith。
  

---

## 6. 流式（Streaming）与 `transform` 的角色

Streaming 是 Runnable 的一等能力：

- 如果整个 Sequence 的各个组件都实现了 `transform`（把流输入映射到流输出），那么序列整体可以从上游边生成边下游消费，实现端到端流式。
  
- 若其中某个组件不支持 `transform`，那么整个序列会在该组件执行完成后再继续流式（streaming 会被阻塞）。因此在设计链路时要小心：**把不支持流式的步骤放在序列末端，或用专门的 Generator 类型替代**。
  

注意：`RunnableLambda` 默认**不**支持 `transform`，因此若需要流式请使用 `RunnableGenerator` 或手动实现 `transform`。

---

## 7. 如何实现自定义 Runnable（快速模板）

当内置的 Lambda/Generator/Sequence/Parallel 不能满足时，你可以通过继承 `Runnable` 实现自定义逻辑。

**简易模板（同步 + 异步 + 流式）：**

```python
from langchain_core.runnables import Runnable

class MyRunnable(Runnable):
    # 可选：定义输入/输出类型（pydantic 模型）
    # input_schema = ...
    # output_schema = ...

    def invoke(self, input, config=None):
        # 同步实现
        # 处理并返回结果
        return {'result': str(input)}

    async def ainvoke(self, input, config=None):
        # 推荐实现：若逻辑本身是同步的，可以委托给默认行为
        return await super().ainvoke(input, config=config)

    def stream(self, input, config=None):
        # 返回 generator，yield 中间 chunk
        yield "部分结果 1"
        yield "部分结果 2"

    async def astream(self, input, config=None):
        # async generator
        yield "async chunk 1"
```

**实现要点**：

- 如果可以，优先实现 `ainvoke`/`astream` 的原生异步版本（比默认线程池委托更高效）；
  
- 若需要支持端到端流式：实现 `transform`（或 `stream`/`astream`），并确保上游/下游也支持流式；
  
- 通过 `config_schema`、`input_schema`、`output_schema` 提供可验证的类型声明，利于自动化工具/可视化追踪。
  

---

## 8. 调试、测试与性能优化建议

### 调试 & Trace

- 全局打开 debug：
  

```python
from langchain_core.globals import set_debug
set_debug(True)
```

- 或者在单个调用上传入回调处理器：
  

```python
chain.invoke(..., config={'callbacks': [ConsoleCallbackHandler()]})
```

- 将关键环节打点并使用 LangSmith/ConsoleCallback 等工具查看中间输出。
  

### 性能/可靠性技巧

- **批量优先**：为高吞吐场景自定义 `batch`/`abatch`，如果下游 API 支持原生批量接口，使用它而不是默认并行 `invoke`。
  
- **异步优先**：若函数/模型支持原生 async，请提供 `afunc` 或覆写 `ainvoke`，避免默认线程池。
  
- **并发控制**：使用 `max_concurrency` 配置控制并发量，避免过载第三方 API。
  
- **重试与回退**：使用 `.with_retry(...)` 与 `RunnableWithFallbacks` 把不稳定调用包起来。
  
- **流式策略**：把不支持 `transform` 的步骤放到序列末尾，或改写该步骤以支持 `transform`。
  

---

## 9. 代码示例集

### 示例 1 — 最小 Sequence

```python
from langchain_core.runnables import RunnableLambda

r1 = RunnableLambda(lambda x: x + 1)
r2 = RunnableLambda(lambda x: x * 2)
seq = r1 | r2
print(seq.invoke(1))  # 4
```

### 示例 2 — 并行字典（map）

```python
from langchain_core.runnables import RunnableLambda

seq = RunnableLambda(lambda x: x + 1) | {
    'mul_2': RunnableLambda(lambda x: x * 2),
    'mul_5': RunnableLambda(lambda x: x * 5),
}
print(seq.invoke(1))  # {'mul_2': 4, 'mul_5': 10}
```

### 示例 3 — 重试策略

```python
from langchain_core.runnables import RunnableLambda

r = RunnableLambda(lambda x: buggy_fn(x)).with_retry(stop_after_attempt=5)
print(r.invoke(10))
```

### 示例 4 — 流式（伪代码，结合 LLM）

```python
from langchain_core.runnables import RunnableSequence
# 假设 prompt | model | parser 已构成 chain
chain = prompt | model | parser
async for chunk in chain.astream({'topic': 'colors'}):
    print(chunk)
```

### 示例 5 — 自定义 Runnable

```python
from langchain_core.runnables import Runnable

class ToUpperRunnable(Runnable):
    def invoke(self, input, config=None):
        return str(input).upper()

r = ToUpperRunnable()
print(r.invoke('hello'))  # 'HELLO'
```

---

## 10. 常见问题与陷阱

- **把 RunnableLambda 放在需要流式的位置**：`RunnableLambda` 默认不支持 `transform`，会阻断上游 streaming；
  
- **依赖默认 batch 行为**：默认 `batch` 是并行调用 `invoke`，但某些 API（如 OpenAI 的嵌入批量接口）更高效，应该实现专门的 `batch`；
  
- **异步实现不足**：若只实现同步逻辑，`ainvoke` 默认会委托到线程池，可能导致不必要的线程开销；
  
- **回调滥用**：回调是强大工具，但滥用可能暴露敏感数据到日志，注意 `metadata` 与 `tags` 的设定。
  

---

## 参考（阅读顺序建议）

1. `runnables` 概览页面（模块总体设计）
   
2. `Runnable` API 说明（方法签名、标准方法、调试/trace 节）
   
3. `RunnableSequence`、`RunnableLambda` 等实现页面（示例与注意事项）
   

---

> 如果你愿意，我可以：
> 
> - 把上面的示例打包成一个可运行的 Python 脚本；
>     
> - 把文档转成 PPT（适合培训/分享）；
>     
> - 或者在文档中加入更多关于 `RunnableConfig` 字段的完整说明（包括 schema 字段）。

refe