import os
import json
import time
import random
import re
from dashscope import Generation
import dashscope

# ================= 配置区 =================
# 🔴 请确认 API Key 正确
dashscope.api_key = "sk-e0ea6eb13bbf44ed910748b72c011ab2"

# 目标生成总数
TARGET_COUNT = 600
# 批次大小 (调小一点可以减少单次生成的Token量，降低被截断的风险)
BATCH_SIZE = 5
OUTPUT_FILE = "train_data.json"

SYSTEM_PROMPT = """
你是一个构建【RAG 上下文判断数据】的专家。
我们要训练一个小模型，判断用户的 Query 是否需要查询【个人自选股数据库】。

【判断逻辑】
1. TRUE (Need Context): 问"我的"、"持仓"、"赚了没"、"自选股里的xx表现"、"账户浮盈"。
2. FALSE (No Context): 问大盘、公有数据(如"茅台股价")、闲聊、或"添加/删除自选"的操作指令、宏观政策。

【输出要求】
生成 5 条不同的 Query (包含正例和负例)。
严格输出一个 JSON List。不要包含 Markdown 标记 (```json)。不要有任何开场白或结束语。
List 中每个元素包含：
- instruction: "判断是否需要检索用户的自选股数据来辅助回答。"
- input: 用户模拟提问
- output: JSON字符串 {"need_watchlist_context": true/false}

【格式示例】
[
  {"instruction": "判断是否需要检索用户的自选股数据来辅助回答。", "input": "看看我的持仓", "output": "{\\"need_watchlist_context\\": true}"},
  {"instruction": "判断是否需要检索用户的自选股数据来辅助回答。", "input": "茅台现在多少钱", "output": "{\\"need_watchlist_context\\": false}"}
]
"""


def extract_json_from_text(text):
    """
    鲁棒的 JSON 提取器：
    1. 移除 Markdown 标记
    2. 寻找最外层的 [ ... ]
    """
    try:
        # 1. 简单清洗
        text = text.replace("```json", "").replace("```", "").strip()

        # 2. 尝试直接解析
        return json.loads(text)
    except json.JSONDecodeError:
        # 3. 如果失败，尝试暴力提取 List
        # 找到第一个 '[' 和最后一个 ']'
        start = text.find("[")
        end = text.rfind("]")

        if start != -1 and end != -1:
            json_str = text[start : end + 1]
            try:
                return json.loads(json_str)
            except:
                pass
        return None


def generate_full_dataset():
    existing_data = []

    # 断点续传：先读取已有数据
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except:
            print("⚠️ 旧文件格式有误或为空，将重新开始。")
            existing_data = []

    print(f"📊 当前已有数据: {len(existing_data)} 条。目标: {TARGET_COUNT} 条。")

    fail_count = 0

    while len(existing_data) < TARGET_COUNT:
        try:
            print(f"⏳ 正在生成 Batch (进度 {len(existing_data)}/{TARGET_COUNT})...")

            messages = [
                {"role": "system", "content": "You are a dataset generator."},
                {
                    "role": "user",
                    "content": SYSTEM_PROMPT
                    + f"\n\n请生成一批新的数据。Random Seed: {random.random()}",
                },
            ]

            response = Generation.call(
                model="qwen-max",
                messages=messages,
                result_format="message",
                temperature=0.85,
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content

                # 使用增强版提取器
                batch_data = extract_json_from_text(content)

                if batch_data and isinstance(batch_data, list):
                    valid_items = []
                    # 二次校验数据结构
                    for item in batch_data:
                        if "input" in item and "output" in item:
                            valid_items.append(item)

                    if valid_items:
                        existing_data.extend(valid_items)
                        # 实时写入磁盘
                        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                            json.dump(existing_data, f, ensure_ascii=False, indent=2)
                        print(f"✅ 成功写入 {len(valid_items)} 条。")
                        fail_count = 0  # 重置失败计数
                    else:
                        print("⚠️ 提取到了JSON但不符合字段要求，跳过。")
                else:
                    print("❌ 解析JSON失败，打印原始内容片段供调试:")
                    print(content[:100] + "...")  # 只打印前100个字符看看
                    fail_count += 1
            else:
                print(f"🌐 API 请求失败: {response.code} - {response.message}")
                time.sleep(2)

            # 如果连续失败太多次，休息一下
            if fail_count > 5:
                print("😴 连续失败多次，暂停 10 秒...")
                time.sleep(10)
                fail_count = 0

        except Exception as e:
            print(f"💥 发生未知异常: {e}")
            time.sleep(1)

    print(f"\n🎉 任务完成！共收集 {len(existing_data)} 条数据。请检查: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_full_dataset()
