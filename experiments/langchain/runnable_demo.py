import asyncio
from typing import AsyncIterator, Iterator, List

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# --- 步骤 1: 创建我们自己的 Runnable 类 ---
# 我们继承自 Runnable，并实现其核心方法。
# 这是一个更完整的实现，同时展示了同步和异步的流式处理。


class ReverseWords(Runnable):
    """
    一个自定义的 Runnable，它接收一个字符串，
    并以流式或一次性调用的方式，返回颠倒了单词顺序的字符串。
    """

    # InputType 和 OutputType 是可选的，但有助于类型提示和文档。
    @property
    def InputType(self):
        return str

    @property
    def OutputType(self):
        return str

    def invoke(self, input: str, config: RunnableConfig | None = None) -> str:
        """
        同步、一次性调用。
        """
        print("\n--- 调用了 ReverseWords.invoke ---")
        words = input.split()
        reversed_words = " ".join(words[::-1])
        return reversed_words

    async def ainvoke(self, input: str, config: RunnableConfig | None = None) -> str:
        """
        异步、一次性调用。
        """
        print("\n--- 调用了 ReverseWords.ainvoke ---")
        # 对于这个简单例子，同步和异步逻辑相同。
        return self.invoke(input, config)

    def stream(self, input: str, config: RunnableConfig | None = None) -> Iterator[str]:
        """
        同步、流式调用。我们逐字 yield 出来。
        """
        print("\n--- 调用了 ReverseWords.stream ---")
        words = input.split()
        reversed_words_list = words[::-1]
        for word in reversed_words_list:
            yield word + " "  # 逐个单词地流式输出

    async def astream(
        self, input: str, config: RunnableConfig | None = None
    ) -> AsyncIterator[str]:
        """
        异步、流式调用。
        """
        print("\n--- 调用了 ReverseWords.astream ---")
        words = input.split()
        reversed_words_list = words[::-1]
        for word in reversed_words_list:
            yield word + " "
            await asyncio.sleep(0.05)  # 模拟异步IO操作


# --- 步骤 2: 在 LCEL 链中使用我们自定义的 Runnable ---

# 假设你已经设置了 OPENAI_API_KEY 环境变量
model = ChatOpenAI(
    api_key="sk-243921467da54efe9e3c8e8f38e58a6f",
    model="qwen1.5-1.8b-chat",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
prompt = ChatPromptTemplate.from_template("写一句关于 {topic} 的简短英文名言。")
parser = StrOutputParser()

# 实例化我们自己的 Runnable
custom_reverser = ReverseWords()

# 构建包含自定义 Runnable 的链
chain = prompt | model | parser | custom_reverser

# --- 步骤 3: 运行并观察结果 ---

print("====== 演示 invoke() ======")
# 链的 invoke() 会自动调用我们自定义的 invoke()
result = chain.invoke({"topic": "artificial intelligence"})
print(f"最终结果: {result}")


print("\n\n====== 演示 stream() ======")
# 链的 stream() 会自动调用我们自定义的 stream()
final_streamed_output = ""
for chunk in chain.stream({"topic": "artificial intelligence"}):
    # 注意：前面的步骤（prompt, model, parser）也会流式输出
    print(chunk, end="", flush=True)
    final_streamed_output += chunk

print(f"\n最终流式结果: {final_streamed_output.strip()}")

# 你可以尝试运行 chain.ainvoke 和 chain.astream 来观察异步版本的行为
