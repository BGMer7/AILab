import json
import random

# 读取你生成的 600 条数据
INPUT_FILE = "train_data.json"


def split_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"原始数据总量: {len(data)}")

    # 打乱顺序，保证随机性
    random.shuffle(data)

    # 切分：最后 50 条做测试，剩下的做训练
    test_set = data[-50:]
    train_set = data[:-50]

    # 保存训练集 (给 LLaMA-Factory 用)
    with open("dataset_train.json", "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)

    # 保存测试集 (给 benchmark_baseline.py 用)
    with open("dataset_test.json", "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    print(f"✅ 切分完成！")
    print(
        f"   📂 训练集: dataset_train.json ({len(train_set)} 条) -> 放进 data 目录训练"
    )
    print(f"   🧪 测试集: dataset_test.json ({len(test_set)} 条) -> 用来跑基准测试")


if __name__ == "__main__":
    split_data()
