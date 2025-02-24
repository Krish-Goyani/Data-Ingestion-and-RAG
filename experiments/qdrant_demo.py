'''from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue

#initialize
client = QdrantClient(url = "http://localhost:6333")

#create collection
collection = client.create_collection(
    collection_name="test_collection",
    vectors_config= VectorParams(size=4, distance= Distance.DOT)
)

# insert points
operation_info = client.upsert(
    collection_name= "test_collection",
    points=[
        PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
        PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
        PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
        PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
        PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
        PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
    ],
    wait= True

    
)
print(operation_info)

# query points
result = client.query_points(
    collection_name="test_collection",
    query = [0.2, 0.1, 0.9, 0.7],
    limit=3,
    with_payload =True,
    query_filter= Filter(
        must= [FieldCondition(key="city" , match= MatchValue(value = "Berlin"))]
    )
).points


print(result)'''
from qdrant_client import AsyncQdrantClient, models
import numpy as np
import asyncio


async def main():
    client = AsyncQdrantClient(url="http://localhost:6333")
    
    if not await client.collection_exists("async_collection"):
         await client.create_collection(
             collection_name= "async_collection",
             vectors_config= models.VectorParams(size=10, distance= models.Distance.COSINE),
             
         )
         
         
    await client.upsert(
        collection_name="async_collection",
        points=[
            models.PointStruct(
               id=i,
               vector=np.random.rand(10).tolist(),
            )
            for i in range(100)
      ],
    ) 
    res = await client.search(
        collection_name="async_collection",
        query_vector=np.random.rand(10).tolist(),
        limit=10
    )
    print(res)
    
    
    
    
asyncio.run(main())