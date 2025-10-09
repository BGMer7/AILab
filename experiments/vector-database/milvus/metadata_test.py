import ollama  # 导入 ollama 库
from ollama import Client

from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# --- 1. 配置信息 (重要：修改了维度) ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
OLLAMA_HOST = "localhost"
OLLAMA_PORT = "11434"
OLLAMA_MODEL_NAME = "bge-m3"  # 指定要使用的 Ollama 模型
COLLECTION_NAME = "my_articles_bge_m3"  # 建议为新模型使用新集合
DIMENSION = 1024  # ⚠️ 关键修改：bge-m3 的向量维度是 1024

# --- 2. 连接到 Milvus ---
print("Connecting to Milvus...")
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
print("✅ Connected to Milvus.")

# --- 3. 创建集合 (如果不存在) ---
# 集合的 schema 定义与之前相同，只是维度变了
if not utility.has_collection(COLLECTION_NAME):
    print(f"Collection '{COLLECTION_NAME}' does not exist. Creating now...")
    fields = [
        FieldSchema(
            name="chunk_id", dtype=DataType.INT64, is_primary=True, auto_id=True
        ),
        FieldSchema(name="original_doc_url", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="publish_date", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="chunk_vector", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
    ]
    schema = CollectionSchema(
        fields, "A collection using bge-m3 embeddings from Ollama"
    )
    collection = Collection(COLLECTION_NAME, schema)

    print("Creating index for scalar fields...")
    collection.create_index(field_name="original_doc_url")
    collection.create_index(field_name="publish_date")

    print("Creating index for vector field...")
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="chunk_vector", index_params=index_params)
    print("✅ Collection and indexes created.")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists.")
    collection = Collection(COLLECTION_NAME)

# --- 4. 准备数据并存入 Milvus ---
print("\nPreparing and inserting data using Ollama...")

# 模拟两篇文章
articles = [
    {
        "content": "Milvus is a powerful open-source vector database built for AI applications. It's highly scalable and can handle billions of vectors.",
        "metadata": {"url": "https://milvus.io/docs", "publish_date": "2024-05-20"},
    },
    {
        "content": "The new AI model, released yesterday, shows incredible performance on text generation tasks. This model was trained on a massive dataset.",
        "metadata": {
            "url": "https://example.com/new-ai-model",
            "publish_date": "2025-07-09",
        },
    },
]

entities_to_insert = []
ollama_client = Client(host=OLLAMA_HOST)

# 简单的按句子切块
for article in articles:
    chunks = article["content"].split(". ")
    try:
        response = ollama_client.embed(model=OLLAMA_MODEL_NAME, input=chunks)
        embeddings = response["embeddings"]

        entities_to_insert.extend(
            {
                "original_doc_url": article["metadata"]["url"],
                "publish_date": article["metadata"]["publish_date"],
                "chunk_text": chunk,
                "chunk_vector": embedding,
            }
            for chunk, embedding in zip(chunks, embeddings)
        )
    except Exception as e:
        print(f"Error getting embedding from Ollama: {e}")

if entities_to_insert:
    collection.insert(entities_to_insert)
    collection.flush()
    print(f"✅ Inserted {len(entities_to_insert)} chunks into Milvus.")
    print(f"Total entities in collection: {collection.num_entities}")

# --- 5. 执行搜索！ ---
print("\n--- Performing Search ---")
collection.load()

query_text = "What is a vector database?"
# ⚠️ 关键修改：同样使用 Ollama 生成查询向量
query_vector = ollama_client.embed(model=OLLAMA_MODEL_NAME, input=query_text)[
    "embeddings"
][0]

# 场景1: 纯向量相似度搜索
print("\n🔍 Scenario 1: Pure vector similarity search...")
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(
    data=[query_vector],
    anns_field="chunk_vector",
    param=search_params,
    limit=3,
    output_fields=["chunk_text", "original_doc_url"],
)
for hit in results[0]:
    print(
        f"  - URL: {hit.entity.get('original_doc_url')}, Text: '{hit.entity.get('chunk_text')}', Distance: {hit.distance:.4f}"
    )

# 场景2: 向量搜索 + 元数据过滤
print("\n🔍 Scenario 2: Search with metadata filter (date = 2024-05-20)...")
filter_expression = 'publish_date == "2024-05-20"'
results_filtered = collection.search(
    data=[query_vector],
    anns_field="chunk_vector",
    param=search_params,
    limit=3,
    expr=filter_expression,
    output_fields=["chunk_text", "original_doc_url", "publish_date"],
)
if not results_filtered[0]:
    print("  No results found for this filter.")
else:
    for hit in results_filtered[0]:
        print(
            f"  - URL: {hit.entity.get('original_doc_url')}, Date: {hit.entity.get('publish_date')}, Text: '{hit.entity.get('chunk_text')}', Distance: {hit.distance:.4f}"
        )

# --- 6. 释放资源 ---
# collection.release()
connections.disconnect("default")
print("\n✅ Done. Disconnected from Milvus.")
