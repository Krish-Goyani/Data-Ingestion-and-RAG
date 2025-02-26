from src.app.config.settings import settings
from pinecone import PineconeAsyncio
from src.app.services.vector_db_service import VectorDBService
from qdrant_client import AsyncQdrantClient, models
from src.app.config.settings import settings
from pymilvus import AsyncMilvusClient, MilvusClient, DataType
from src.app.services.sparse_embedding_service import SparseEmbeddingsService
from fastapi import Depends
from src.app.repositories.images_repository import ImageMappingRepo

class RetrieveChunksService:
    def __init__(self, query_embedding_service = Depends(SparseEmbeddingsService), images_repository = Depends(ImageMappingRepo)):
        self.index_name = "idx001"
        self.collection_name = "rag_collection"
        self.qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL)
        self.milvus_async_client = None
        self.sparse_index_name = "sparse001"
        self.query_emebedding_service= query_embedding_service
        self.images_repository = images_repository
        

    async def pinecone_retrieve_similar_chunks(self, query: str, top_k: int = 5):
        """
        Retrieves top-k most similar chunks to the user query from Pinecone.
        """
        async with PineconeAsyncio(api_key=settings.PINECONE_API_KEY) as pc:
            if not await pc.has_index(self.index_name):
                raise ValueError(f"Index '{self.index_name}' does not exist.")
            
            index_info = await pc.describe_index(name=self.index_name)

            try:

                async with pc.IndexAsyncio(host=index_info.host) as idx:
                    # Generate embedding for the query
                    embedding_result = await pc.inference.embed(
                        model="llama-text-embed-v2",
                        inputs=[query],
                        parameters={
                            "input_type": "query",
                            "truncate": "END",
                            "dimension" : 1024
                        }
                    )
                    query_token_usage = embedding_result.usage["total_tokens"]
                    
                    # Perform similarity search
                    results = await idx.query(
                        vector=embedding_result[0].values,
                        top_k=top_k,
                        include_metadata=True,
                        namespace="textportion"
                    )

                    # Extract retrieved chunks
                    retrieved_chunks = [match["metadata"]["text"] for match in results["matches"]]
                    
                    return retrieved_chunks, query_token_usage, results.usage["read_units"]

            except Exception as e:
                print(f"Error retrieving chunks: {e}")
                return []
            
    async def generate_query_embedding(self, query: str):
        """Generates an embedding for the query using Pinecone's hosted model."""
        async with PineconeAsyncio(api_key=settings.PINECONE_API_KEY) as pc:
            
            embedding_result = await pc.inference.embed(
                    model="llama-text-embed-v2",
                    inputs=[query],
                    parameters={
                        "input_type": "query",
                        "truncate": "END",
                        "dimension" : 1024
                    }
                )
        
        return embedding_result[0].values  # Extracting embedding vector

    async def search_qdrant(self, query: str, limit: int = 10):
        """Searches Qdrant for the most relevant chunks based on the query embedding."""
        query_vector = await self.generate_query_embedding(query)

        # Ensure collection exists
        if not await self.qdrant_client.collection_exists(self.collection_name):
            raise ValueError(f"Collection '{self.collection_name}' does not exist in Qdrant.")

        # Perform search
        points = await self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        chunks = [point.payload["text"] for point in points if "text" in point.payload]

        return chunks

    
    '''async def milvus_initialize_client(self):
        """Initialize the Milvus Async Client inside an event loop"""
        self.milvus_async_client = AsyncMilvusClient(uri="http://localhost:19530/")
        
    async def search_milvus(self, query: str, limit: int =10):
        try: 
            """
            Search for relevant chunks in Milvus based on query embeddings.
            
            :param query: User's search query.
            :param search_type: "dense" for dense vectors, "sparse" for sparse vectors.
            :param field: Field name in Milvus collection to search within.
            :return: Retrieved chunks containing relevant text.
            """
            # Generate query embeddings
            if not self.milvus_async_client:
                await self.milvus_initialize_client()
            client = MilvusClient()
            index_params = client.prepare_index_params()

            index_params.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="COSINE",  params= {"nlist": 128})

            await self.milvus_async_client.create_index(self.collection_name, index_params)
            
            query_embedding = await self.generate_query_embedding(query)

            await self.milvus_async_client.load_collection(self.collection_name)
            # Perform vector search
            search_results = await self.milvus_async_client.search(
                collection_name=self.collection_name,
                data=[query_embedding],  # Query vector must be a list
                anns_field="vector",
                output_fields=["text"],  # Retrieve relevant text chunks
                limit=limit,
                search_params={"metric_type": "COSINE", "nprobe" : 12} # Adjust the limit as needed
            )
            
            print(search_results)
            # Extract and return retrieved chunks
            retrieved_chunks = [hit.entity.get("text", "") for hit in search_results]
            await self.milvus_async_client.release_collection(self.collection_name)
            await self.milvus_async_client.close()  # if your version provides a close method
            client.close()
            return retrieved_chunks
        
        except Exception as e:
            print(e)'''
            
            
    async def pinecone_retrieve_similar_chunks_s(self, query: str, top_k: int = 5):
        """
        Retrieves top-k most similar chunks to the user query from Pinecone.
        """
        async with PineconeAsyncio(api_key=settings.PINECONE_API_KEY) as pc:
            if not await pc.has_index(self.sparse_index_name):
                raise ValueError(f"Index '{self.sparse_index_name}' does not exist.")
            
            index_info = await pc.describe_index(name=self.sparse_index_name)

            try:
                async with pc.IndexAsyncio(host=index_info.host) as idx:
                    vector = self.query_emebedding_service.generate_query_embedding(query)
                    # Perform similarity search
                    results = await idx.query(
                        sparse_vector =  vector,
                        top_k=top_k,
                        include_metadata=True,
                        include_values= False,
                        
                    )

                    # Extract retrieved chunks
                    retrieved_chunks = [match["metadata"]["text"] for match in results["matches"]]
                    return retrieved_chunks, results.usage["read_units"]

            except Exception as e:
                print(f"Error retrieving chunks: {e}")
                return []
            
    async def pinecone_retrieve_similar_chunks_images(self, query: str, top_k: int = 3):
        """
        Retrieves top-k most similar chunks to the user query from Pinecone.
        """
        async with PineconeAsyncio(api_key=settings.PINECONE_API_KEY) as pc:
            if not await pc.has_index("idx001"):
                raise ValueError(f"Index {'idx001'} does not exist.")
            
            index_info = await pc.describe_index(name="idx001")

            try:
                async with pc.IndexAsyncio(host=index_info.host) as idx:
                    vector = await self.generate_query_embedding(query)
                    # Perform similarity search
                    results = await idx.query(
                        vector = vector,
                        top_k=top_k,
                        include_metadata=True,
                        include_values= False,
                        namespace="imgportion"
                    )

                    # Extract retrieved chunks
                    retrieved_record_ids = [match["metadata"]["record_id"] for match in results["matches"] if match["score"]>0.50]
                    
                    
                    retrieved_images  = await self.images_repository.fetch_base64_images(retrieved_record_ids)
                    return retrieved_images
                    #return retrieved_chunks, results.usage["read_units"]

            except Exception as e:
                print(f"Error retrieving chunks: {e}")
                return []
