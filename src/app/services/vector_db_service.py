from src.app.config.settings import settings
import asyncio
from pinecone import PineconeAsyncio, ServerlessSpec, VectorType
import uuid
import time
import asyncio
import uuid
from qdrant_client import AsyncQdrantClient, models
from src.app.config.settings import settings
from pymilvus import AsyncMilvusClient, MilvusClient, DataType
from pinecone import Pinecone, SparseValues, Vector
import uuid
import time


class VectorDBService:
    def __init__(self) -> None:
        self.index_name = "idx001"
        self.token_limit_per_minute = 250000  # Pinecone's token limit
        self.batch_size = 50  # Adjust based on your use case
        self.collection_name = "rag_collection"
        self.qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL)
        self.milvus_async_client = None  # Example: "sqlite:///:memory:"
        #self.milvus_client = MilvusClient(uri="http://localhost:19530/") # Example: ""
        self.sparse_index_name = "sparse001"

    
    async def pinecone_generate_and_store_embeddings(self, chunks):
        async with PineconeAsyncio(api_key= settings.PINECONE_API_KEY) as pc:
        # Check if the index exists
            if not await pc.has_index(self.index_name):
                print(f"Index '{self.index_name}' not found. Creating index...")
                await pc.create_index(
                    name=self.index_name,
                    dimension=1024,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                    deletion_protection="disabled",
                    tags={"environment": "development"},
                    vector_type=VectorType.DENSE
                )
                print("Index created successfully.")
            else:
                print(f"Index '{self.index_name}' already exists.")
                
        
            index_info = await pc.describe_index(name= self.index_name)  
            
            try:    
                async with pc.IndexAsyncio(host= index_info.host) as idx:            
                    total_tokens_used = 0
                    e_total_tokens_used = 0
                    # Process in batches
                    for i in range(0, len(chunks), self.batch_size):
                        batch = chunks[i:i + self.batch_size]
                        
                        # **Count tokens by splitting text on spaces**
                        batch_token_count = sum(len(text.split()) for text in batch)

                        # If token limit exceeded, wait
                        if total_tokens_used + batch_token_count > self.token_limit_per_minute:
                            print("Rate limit reached, sleeping for 60 seconds...")
                            time.sleep(60)  # Wait before resuming
                            total_tokens_used = 0  # Reset counter

                        # Generate embeddings
                        embeddings = await pc.inference.embed(
                            model="llama-text-embed-v2",
                            inputs=batch,
                            parameters={
                                "input_type": "passage",
                                "truncate": "END",
                                "dimension" : 1024
                            }
                            
                        )
                        e_total_tokens_used += embeddings.usage["total_tokens"]
                        records = [
                            {
                                "id": str(uuid.uuid4()),  # Ensure ID is a string
                                "values": e["values"],  # Correct embedding extraction
                                "metadata": {"text": d},
                            }
                            for d, e in zip(batch, embeddings)
                        ]

                        # Upsert records into Pinecone
                        await idx.upsert(vectors=records, namespace="textportion")
                        print(f"Batch {i // self.batch_size + 1} upserted successfully.")

                        # Update total token usage
                        total_tokens_used += batch_token_count
                return e_total_tokens_used                    
            except Exception as e:
                print(e) 

    async def qdrant_setup_collection(self, vector_size: int):
        """
        Ensures the collection exists in Qdrant. Creates it if missing.
        """
        exists = await self.qdrant_client.collection_exists(self.collection_name)
        if not exists:
            await self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,  # Define based on your embedding model
                    distance=models.Distance.COSINE
                )
            )
            print(f"Collection '{self.collection_name}' created.")

    async def qdrant_store_embeddings(self, embeddings: list[dict]):
        """
        Stores embeddings into Qdrant.
        
        :param embeddings: List of dicts containing "text" and "embedding".
        """
        if not embeddings:
            print("No embeddings provided to store.")
            return

        # Ensure collection exists
        await self.qdrant_setup_collection(vector_size=len(embeddings[0]["embedding"]))

        # Prepare data for upsert
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),  # Unique ID for each vector
                vector=embedding["embedding"],  # Vector representation
                payload={"text": embedding["text"]}  # Store original text as metadata
            )
            for embedding in embeddings
        ]

        # Insert data into Qdrant
        await self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Stored {len(points)} embeddings in Qdrant.")
        
    
    '''async def milvus_initialize_client(self):
        """Initialize the Milvus Async Client inside an event loop"""
        self.milvus_async_client = AsyncMilvusClient(uri="http://localhost:19530/")
        
    async def milvus_setup_collection(self, vector_dim: int):
        """
        Ensures the collection exists in Milvus. Creates it if missing.
        """
        if self.milvus_client.has_collection(self.collection_name):
            return  # Collection already exists

        schema = self.milvus_async_client.create_schema(
            auto_id=False,
            description="RAG storage schema"
        )
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=vector_dim)
        schema.add_field("text", DataType.VARCHAR, max_length=1024)

        await self.milvus_async_client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            metric_type= "COSINE"
        )

        print(f"Collection '{self.collection_name}' created.")

    async def milvus_store_embeddings(self, embeddings: list[dict]):
        """
        Stores embeddings into Milvus.

        :param embeddings: List of dicts containing "text" and "embedding".
        """
        if not embeddings:
            print("No embeddings provided to store.")
            return
        if not self.milvus_async_client:
            await self.milvus_initialize_client()
            
        # Ensure collection exists
        await self.milvus_setup_collection(vector_dim=len(embeddings[0]["embedding"]))

        # Prepare data for insertion
        data = [
            {
                "id": i + 1,
                "vector": embedding["embedding"],  # Vector representation
                "text": embedding["text"],  # Store original text
            }
            for i, embedding in enumerate(embeddings)
        ]

        # Insert data into Milvus
        await self.milvus_async_client.insert(self.collection_name, data)
        print(f"Stored {len(data)} embeddings in Milvus.")'''
        
    async def pinecone_store_sparse_embeddings(self, chunks,embeddings):
        async with PineconeAsyncio(api_key= settings.PINECONE_API_KEY) as pc:
        # Check if the index exists
            if not await pc.has_index(self.sparse_index_name):
                print(f"Index '{self.sparse_index_name}' not found. Creating index...")
                await pc.create_index(
                    name=self.sparse_index_name,
                    metric="dotproduct",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                    deletion_protection="disabled",
                    tags={"environment": "development"},
                    vector_type="sparse"
                )
                print("Index created successfully.")
            else:
                print(f"Index '{self.sparse_index_name}' already exists.")
                
        
            index_info = await pc.describe_index(name= self.sparse_index_name)  
            
            try:    
                async with pc.IndexAsyncio(host= index_info.host) as idx:  

                    vectors = []
                    for d,e in zip(chunks, embeddings):
                        vec = Vector(
                            id= str(uuid.uuid4()),
                            sparse_values=SparseValues(
                                values=e["values"],
                                indices=e["indices"]
                            ),
                            metadata={"text": d}
                        )
                        vectors.append(vec) 
                        
                    await idx.upsert(vectors = vectors)
                
            except Exception as e:
                print(e)
                