import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ================= 配置 =================
# 🔴 指向你刚才下载的模型路径
MODEL_PATH = "./models/qwen/Qwen2___5-3B-Instruct"
DATA_FILE = "dataset_test.json"
TEST_SIZE = 50  # 只测试前50条，节省时间。如果数据不够50条则全测。


def load_model():
    print("⏳ 正在加载模型 (可能需要几分钟)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        # device_map="auto" 会自动调用显卡
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,  # 节省显存
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
        # 去掉 Markdown 标记
        text = text.replace("```json", "").replace("```", "").strip()
        # 尝试直接解析
        return json.loads(text)
    except:
        return None


def run_benchmark():
    # 1. 读取数据
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 截取测试集
    test_data = data[:TEST_SIZE]
    print(f"📊 开始测试前 {len(test_data)} 条数据...")

    tokenizer, model = load_model()

    correct_count = 0
    format_error_count = 0

    # 2. 循环测试
    for i, item in enumerate(tqdm(test_data)):
        user_query = item["input"]

        # 解析标准答案 (Ground Truth)
        try:
            ground_truth = json.loads(item["output"])
            gt_label = ground_truth.get("need_watchlist_context")
        except:
            print(f"⚠️ 第 {i+1} 条数据标准答案格式错误，跳过。")
            continue

        # 构造 Prompt (必须和微调时的 Instruction 保持一致)
        messages = [
            {"role": "system", "content": item["instruction"]},
            {"role": "user", "content": user_query},
        ]

        # 强力 Prompt
        strong_system_prompt = """
        你是一个意图识别助手。请判断用户的 Query 是否需要查询【自选股数据】。
        请务必只输出一个 JSON 对象，不要包含任何其他解释。
        格式要求：{"need_watchlist_context": true/false, "reason": "..."}
        """

        messages = [
            {"role": "system", "content": strong_system_prompt},
            {"role": "user", "content": user_query},
        ]

        # 转换成模型输入
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 推理
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=128,
                do_sample=False,  # <--- 修改这里：关闭采样，启用贪婪解码
            )

        # 解码输出
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        # 3. 结果判定
        pred_json = parse_json_output(response_text)

        if pred_json is None:
            # 格式错误（没输出JSON）
            format_error_count += 1
            print(f"\n❌ [格式错误] Query: {user_query}")
            print(f"   Model Output: {response_text}")
        else:
            pred_label = pred_json.get("need_watchlist_context")

            if pred_label == gt_label:
                correct_count += 1
            else:
                # 打印错误案例，方便你分析
                print(f"\n❌ [预测错误] Query: {user_query}")
                print(f"   预期: {gt_label} | 实际: {pred_label}")
                print(f"   理由: {pred_json.get('reason', '无理由')}")

    # 4. 输出最终报告
    accuracy = (correct_count / len(test_data)) * 100
    print("\n" + "=" * 30)
    print(f"📉 基线测试报告 (Baseline Report)")
    print(f"=" * 30)
    print(f"测试总数: {len(test_data)}")
    print(f"✅ 正确数量: {correct_count}")
    print(f"⚠️ 格式错误: {format_error_count} (模型没按JSON输出)")
    print(f"🏆 准确率: {accuracy:.2f}%")
    print(f"=" * 30)


if __name__ == "__main__":
    run_benchmark()
