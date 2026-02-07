import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ================= 配置 =================
# 🔴 确认你的模型路径
MODEL_PATH = r"D:\Learning\Notes\AILab\experiments\fine-tuning\LLaMA-Factory\saves\Custom\full\train_2026-02-05-21-21-29"
DATA_FILE = "dataset_test.json"
TEST_SIZE = 50


def load_model():
    print("⏳ 正在加载模型 (可能需要几分钟)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        print("✅ 模型加载完成！")
        return tokenizer, model
    except Exception as e:
        print(f"❌ 模型加载失败，请检查路径。错误信息: {e}")
        exit()


def parse_json_output(text):
    """尝试从模型输出中提取 JSON"""
    try:
        # 1. 简单清洗 Markdown
        text = text.replace("```json", "").replace("```", "").strip()
        # 2. 尝试解析
        return json.loads(text)
    except:
        return None


def run_benchmark():
    # 1. 读取数据
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_data = data[:TEST_SIZE]
    print(f"📊 开始测试前 {len(test_data)} 条数据...")

    tokenizer, model = load_model()

    correct_count = 0
    format_error_count = 0

    # 2. 循环测试
    # 注意：为了避免 tqdm 进度条和 print 混在一起，这里可以把 tqdm 去掉，或者忍受一下刷屏
    print("\n" + "=" * 50)
    print("🚀 开始逐条推理展示")
    print("=" * 50 + "\n")

    for i, item in enumerate(test_data):
        user_query = item["input"]

        # 获取标准答案标签，用于对比
        try:
            gt_json = json.loads(item["output"])
            gt_label = gt_json.get("need_watchlist_context")
        except:
            gt_label = "未知"

        # # 构造一个强力 System Prompt
        # strong_system_prompt = f"""
        # {item['instruction']}

        # 【严格约束】
        # 1. 这是一个API接口，不要进行任何思考或对话。
        # 2. 必须且只能输出一个合法的 JSON 字符串。
        # 3. 格式必须是：{{"need_watchlist_context": true}} 或 {{"need_watchlist_context": false}}
        # 4. 禁止输出 <think> 标签，禁止输出 markdown，禁止输出任何解释。
        # """

        # messages = [
        #     {"role": "system", "content": strong_system_prompt},
        #     {"role": "user", "content": user_query},
        # ]

        messages = [
            {"role": "system", "content": item["instruction"]},
            {"role": "user", "content": user_query},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs, max_new_tokens=128, do_sample=False
            )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        # ==========================================
        # 🟢 新增：强制打印每一条的详细输出
        # ==========================================
        print(f"[{i+1}/{TEST_SIZE}] Query: {user_query}")
        print(f"👉 Model Output (Raw): \n{response_text}")
        print("-" * 30)
        # ==========================================

        # 3. 结果判定
        pred_json = parse_json_output(response_text)

        if pred_json is None:
            format_error_count += 1
            print(f"❌ 格式判定: 失败 (Not JSON)")
        else:
            pred_label = pred_json.get("need_watchlist_context")
            if pred_label == gt_label:
                correct_count += 1
                print(f"✅ 结果判定: 正确")
            else:
                print(f"❌ 结果判定: 错误 (预期 {gt_label} vs 实际 {pred_label})")

        print("\n")  # 空一行，方便阅读

    # 4. 输出最终报告
    accuracy = (correct_count / len(test_data)) * 100
    print("=" * 30)
    print(f"📉 测试总结")
    print(f"✅ 正确: {correct_count}")
    print(f"⚠️ 格式错误: {format_error_count}")
    print(f"🏆 准确率: {accuracy:.2f}%")
    print("=" * 30)


if __name__ == "__main__":
    run_benchmark()
