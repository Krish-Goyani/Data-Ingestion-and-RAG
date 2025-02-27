from fastapi import Depends
from src.app.services.retrieve_chunks_service import RetrieveChunksService
from src.app.services.rrf import ReciprocalRankFusionService
from src.app.services.re_ranking_service import ReRanker
from src.app.services.llm_response_service import LLMResponseService
from src.app.services.delete_index import DeleteIndex
from src.app.services.vector_db_service import VectorDBService

class RetrievalAndResponseUsecase:
    def __init__(
        self,
        retrieve_chunks_service: RetrieveChunksService = Depends(RetrieveChunksService),
        rrf_service: ReciprocalRankFusionService = Depends(ReciprocalRankFusionService),
        re_ranking_service: ReRanker = Depends(ReRanker),
        llm_response_service: LLMResponseService = Depends(LLMResponseService),
        delete_index: DeleteIndex = Depends(DeleteIndex),
        vector_db_service: VectorDBService = Depends(VectorDBService)
    ) -> None:
        self.retrieve_chunks_service = retrieve_chunks_service
        self.rrf_service = rrf_service
        self.re_ranking_service = re_ranking_service
        self.llm_response_service = llm_response_service
        self.delete_index = delete_index
        self.vector_db_service = vector_db_service

    async def retrieve_and_generate(self, query: str):
        try:
            # Retrieve similar images.
            retrieved_images = await self.retrieve_chunks_service.pinecone_retrieve_similar_chunks_images(query, top_k=3)
            
            # Retrieve dense chunks.
            dense_chunks, embedding_tokens, read_units = await self.retrieve_chunks_service.pinecone_retrieve_similar_chunks(query, 10)
            
            # Retrieve sparse chunks.
            sparse_chunks, read_units_sparse = await self.retrieve_chunks_service.pinecone_retrieve_similar_chunks_s(query, 10)
            
            # Fuse the results using RRF.
            sorted_items, sorted_documents = self.rrf_service.fuse(dense_chunks, sparse_chunks)
            
            # Re-rank the sorted documents.
            final_chunks_with_score, final_chunks, = await self.re_ranking_service.re_ranker(query, sorted_documents)
            
            # Optionally, you can generate an LLM response if needed:
            response = await self.llm_response_service.generate_response_gemini(final_chunks, query, retrieved_images)
            
            # Delete all indexes.
            #self.delete_index.delete_all_index()
            
            # Return the final results.
            return response, final_chunks_with_score, retrieved_images
        
        
        except Exception as e:
            print(e)
            raise e
