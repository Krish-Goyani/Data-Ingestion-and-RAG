import os

os.environ["PINECONE_API_KEY"] = "pcsk_345tD4_4PC4nTwKFbe2Vivjf3W9D9ZMGH9GjkqLu5XzKk1K2c8t9p2ZhDbeurFoj6ERgXw"

import asyncio
from pinecone import PineconeAsyncio, ServerlessSpec


'''async def main():
        pc = PineconeAsyncio(api_key= os.getenv("PINECONE_API_KEY"))
        data=[
                {
                    "id": "1",
                    "title": "The Great Gatsby",
                    "author": "F. Scott Fitzgerald",
                    "description": "The story of the mysteriously wealthy Jay Gatsby and his love for the beautiful Daisy Buchanan.",
                    "year": 1925,
                },
                {
                    "id": "2",
                    "title": "To Kill a Mockingbird",
                    "author": "Harper Lee",
                    "description": "A young girl comes of age in the segregated American South and witnesses her father's courageous defense of an innocent black man.",
                    "year": 1960,
                },
                {
                    "id": "3",
                    "title": "1984",
                    "author": "George Orwell",
                    "description": "In a dystopian future, a totalitarian regime exercises absolute control through pervasive surveillance and propaganda.",
                    "year": 1949,
                },
            ]
        
        
        embeddings = await pc.inference.embed(
            model= "multilingual-e5-large",
            inputs= [x["description"] for x in data],
            parameters={
                    "input_type" : "passage",
                    "truncate" :  "END"
                })
        records = []
        
        for d,e in zip(data, embeddings):
            records.append(
                {
                    "id"  : d["id"],
                    "values" : e["values"],
                    "metadata"  : {
                        "title" : 123
                    }
                }
            )
        print(len(embeddings))
        idx_name  ="async-idx"
        if not await pc.has_index(idx_name):
            await pc.create_index(
                name= idx_name,
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                ),
                dimension=1024,
                metric="cosine"
            )
        async with pc.IndexAsyncio(host="https://async-idx-auoio4m.svc.aped-4627-b74a.pinecone.io") as idx:
            await idx.upsert_records(
                namespace="user0",
                records=records
            )'''

import time
import os
import asyncio
from pinecone import PineconeAsyncio, ServerlessSpec, VectorType

async def main():
    index_name = "example-index"
    
    # Step 1: Create/manage index asynchronously
    async with PineconeAsyncio(api_key=os.getenv("PINECONE_API_KEY")) as pc:
        # Check if the index exists
        if not await pc.has_index(index_name):
            print(f"Index '{index_name}' not found. Creating index...")
            await pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                deletion_protection="disabled",
                tags={"environment": "development"},
                vector_type=VectorType.DENSE
            )
            print("Index created successfully.")
        else:
            print(f"Index '{index_name}' already exists.")
        #time.sleep(3)
        # Step 2: Upsert records asynchronously into the index
        # Replace 'INDEX_HOST' with the actual host endpoint for your index.
        async with pc.IndexAsyncio(host="https://example-index-auoio4m.svc.aped-4627-b74a.pinecone.io") as idx:
            data=[
                {
                    "id": "4",
                    "title": "The Great Gatsby",
                    "author": "F. Scott Fitzgerald",
                    "description": "The story of the mysteriously wealthy Jay Gatsby and his love for the beautiful Daisy Buchanan.",
                    "year": 1925,
                },
                {
                    "id": "5",
                    "title": "To Kill a Mockingbird",
                    "author": "Harper Lee",
                    "description": "A young girl comes of age in the segregated American South and witnesses her father's courageous defense of an innocent black man.",
                    "year": 1960,
                },
                {
                    "id": "6",
                    "title": "1984",
                    "author": "George Orwell",
                    "description": "In a dystopian future, a totalitarian regime exercises absolute control through pervasive surveillance and propaganda.",
                    "year": 1949,
                },
            ]
            
            embeddings = await pc.inference.embed(
            model= "llama-text-embed-v2",
            inputs= [x["description"] for x in data],
            parameters={
                    "input_type" : "passage",
                    "truncate" :  "END"
                })
            
            records = []
            
            for d,e in zip(data, embeddings):
                records.append(
                    {
                        "id"  : d["id"],
                        "values" : e["values"],
                        "metadata"  : {
                            "title" : d["title"],
                            "description" : d["description"]
                        }
                    }
                )
                
            print(len(embeddings))
            await idx.upsert(vectors=records)
            print("Records upserted successfully.")

if __name__ == '__main__':
    asyncio.run(main())
