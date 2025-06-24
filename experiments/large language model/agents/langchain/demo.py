import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

# --- 1. 检查环境变量 ---
# if "DEEPSEEK_API_KEY" not in os.environ:
#     print(os.environ)
#     raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

# --- 2. 初始化基于DeepSeek的LLM和工具 ---
# 通过配置base_url和model_name来使用DeepSeek
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="",
    base_url="https://api.deepseek.com/v1",
    temperature=0
)

# 初始化一个搜索工具（使用免费的DuckDuckGo）
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]

# --- 3. 创建Agent ---
# 从LangChain Hub获取一个预设的ReAct prompt模板
prompt = hub.pull("hwchase17/react")

# 创建Agent，它负责决定使用哪个工具
agent = create_react_agent(llm, tools, prompt)

# 创建Agent执行器，它负责实际运行Agent并调用工具
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 4. 运行Agent ---
task_prompt = "调研并总结有关人工智能在医疗领域应用的最新进展。请用中文回答。"
result = agent_executor.invoke({"input": task_prompt})

print("\n--- 最终研究结果 (LangChain x DeepSeek) ---")
print(result["output"])