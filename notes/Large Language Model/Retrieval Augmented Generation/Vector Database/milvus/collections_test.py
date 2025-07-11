from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530")

# 3.1. Create schema
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

# 3.2. Add fields to schema
schema.add_field(field_name="my_id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="my_vector", datatype=DataType.FLOAT_VECTOR, dim=5)
schema.add_field(field_name="my_varchar", datatype=DataType.VARCHAR, max_length=512)

# 3.3. Prepare index parameters
index_params = client.prepare_index_params()

# 3.4. Add indexes
index_params.add_index(field_name="my_id", index_type="AUTOINDEX")

index_params.add_index(
    field_name="my_vector", index_type="AUTOINDEX", metric_type="COSINE"
)

# 3.5. Create a collection with the index loaded simultaneously
client.create_collection(
    collection_name="customized_setup_1", schema=schema, index_params=index_params
)

res = client.get_load_state(collection_name="customized_setup_1")
print(res)

# 3.6. Create a collection and index it separately
client.create_collection(
    collection_name="customized_setup_2",
    schema=schema,
)

res = client.get_load_state(collection_name="customized_setup_2")
print(res)

res = client.list_collections()
print(res)

res = client.describe_collection(collection_name="customized_setup_2")
print(res)

client.load_collection(collection_name="customized_setup_2")
res = client.get_load_state(collection_name="customized_setup_2")
print(res)

# With TTL
client.create_collection(
    collection_name="customized_setup_1",
    schema=schema,
    properties={
        "collection.ttl.seconds": 1209600
    }
)

client.create_collection(
    collection_name="customized_setup_2",
    schema=schema,
    properties={
        "collection.ttl.seconds": 1209600
    }
)
