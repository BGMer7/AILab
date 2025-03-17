from pymilvus import model
from pymilvus import connections
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection

# 连接到Milvus服务
connections.connect("default", host="localhost", port="19530")
print(connections.list_connections())

# 定义集合模式
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=256),
]
schema = CollectionSchema(
    fields=fields,
    description="A collection for storing text documents and their embeddings.",
)

# 创建集合
collection = Collection(name="text_collection", schema=schema)

embedding_fn = model.DefaultEmbeddingFunction()

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

embeddings = embedding_fn.encode_documents(docs)
print("Dim:", embedding_fn.dim, embeddings[0].shape)  # Dim: 768 (768,)

data = [
    {"id": i, "vector": embeddings[i], "text": docs[i], "subject": "history"}
    for i in range(len(embeddings))
]

print("Data has", len(data), "entities, each with fields: ", data[0].keys())
print("Embedding dim:", len(data[0]["vector"]))

ids = [i for i in range(len(docs))]
subjects = ["history" for _ in range(len(docs))]

collection.insert([ids, docs, embeddings, subjects])
index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 100}}
collection.create_index("vector", index_params)

# 加载集合到内存中
collection.load()

# 准备搜索参数
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

query_embedding = embedding_fn.encode_documents(["Alan Turing"])[0]
print(query_embedding)

# 执行搜索
results = collection.search(
    data=[query_embedding],  # 查询向量
    anns_field="vector",     # 指定向量字段
    param=search_params,     # 搜索参数
    limit=10,                # 返回的最相似结果数量
    output_fields=["text"]   # 返回的字段
)

print(results)
