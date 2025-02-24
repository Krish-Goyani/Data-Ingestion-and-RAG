from src.app.services.file_conversion_service import FileConversionService
from fastapi import Depends
from src.app.services.text_chunking_service import TextSplitters
from src.app.services.vector_db_service import VectorDBService
from src.app.services.retrieve_chunks_service import RetrieveChunksService
from src.app.services.llm_response_service import LLMResponseService
from src.app.services.dense_embedding_service import EmbeddingService
from src.app.services.sparse_embedding_service import SparseEmbeddingsService
from src.app.services.rrf import ReciprocalRankFusionService
from src.app.services.re_ranking_service import ReRanker
from src.app.utils.cost_tracking import CostTracker
from src.app.repositories.usage_repository import CostStorageRepo

class FileUploadUsecase:
    def __init__(self, file_conversion_service = Depends(FileConversionService), text_splitter = Depends(TextSplitters), vector_db_service = Depends(VectorDBService), retrieve_chunks_service = Depends(RetrieveChunksService), llm_response_service = Depends(LLMResponseService), embedding_service  =Depends(EmbeddingService), sprase_embedding_service = Depends(SparseEmbeddingsService), rrf_service= Depends(ReciprocalRankFusionService),
                 re_ranking_service= Depends(ReRanker), cost_tracker = Depends(CostTracker), cost_storage_repo = Depends(CostStorageRepo)) -> None:
        
        self.file_conversion_service = file_conversion_service
        self.text_splitter = text_splitter
        self.vector_db_service = vector_db_service
        self.retrieve_chunks_service = retrieve_chunks_service
        self.llm_response_service = llm_response_service
        self.embedding_service = embedding_service
        self.sprase_embedding_service = sprase_embedding_service
        self.rrf_service = rrf_service
        self.re_ranking_service = re_ranking_service
        self.cost_tracker = cost_tracker
        self.cost_storage_repo = cost_storage_repo
        
        
    async def process_file(self,file_bytes : bytes,chunk_size,chunk_overlap, query):
        text =  await self.file_conversion_service.convert_to_text(file_bytes)
        chunks =  self.text_splitter.recursive_text_splitter(text, chunk_size, chunk_overlap)
        #embedding_tokens = await self.vector_db_service.pinecone_generate_and_store_embeddings(chunks)
        #self.cost_tracker.add_embedding_tokens(embedding_tokens)
        
        pinecone_chunks, embedding_tokens, read_units = await self.retrieve_chunks_service.pinecone_retrieve_similar_chunks(query, 10)
        self.cost_tracker.add_read_units(read_units)
        self.cost_tracker.add_embedding_tokens(embedding_tokens)
        
        #response, llm_usage  = await self.llm_response_service.generate_response(pinecone_chunks, query)
        #self.cost_tracker.add_llm_tokens(llm_usage)
        
        #dense_embeddings,embedding_tokens  = await self.embedding_service.generate_embeddings(chunks)
        #self.cost_tracker.add_embedding_tokens(embedding_tokens)

        #await self.vector_db_service.qdrant_store_embeddings(embeddings)
        #await self.vector_db_service.milvus_store_embeddings(embeddings)
        #qdrant_chunks = await self.retrieve_chunks_service.search_qdrant(query, 20)
        #milvus_chunks = await self.retrieve_chunks_service.search_milvus(query, 10)
        sparse_embeddings = self.sprase_embedding_service.generate_sparse_embeddings(chunks)
        #await self.vector_db_service.pinecone_store_sparse_embeddings(chunks, sparse_embeddings)
        
        sparse_chunks, read_units = await self.retrieve_chunks_service.pinecone_retrieve_similar_chunks_s(query, 10)
        self.cost_tracker.add_read_units(read_units)
    
        sorted_items, sorted_documents = self.rrf_service.fuse(pinecone_chunks, sparse_chunks)
        final_chunks, rerank_units = await self.re_ranking_service.re_ranker(query, sorted_documents)
        self.cost_tracker.add_rerank_units(rerank_units)
        await self.cost_storage_repo.store_cost_details(self.cost_tracker.to_dict())
        return final_chunks, pinecone_chunks, sparse_chunks
    
    
        