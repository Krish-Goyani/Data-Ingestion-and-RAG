from fastapi import Depends
from src.app.services.retrieve_chunks_service import RetrieveChunksService
from src.app.services.rrf import ReciprocalRankFusionService
from src.app.services.re_ranking_service import ReRanker
from src.app.services.llm_response_service import LLMResponseService
from src.app.services.delete_index import DeleteIndex
from src.app.services.vector_db_service import VectorDBService
from src.app.services.query_decomposition_service import QueryDecompositioneService
import asyncio


class RetrievalAndResponseUsecase:
    def __init__(
        self,
        retrieve_chunks_service: RetrieveChunksService = Depends(RetrieveChunksService),
        rrf_service: ReciprocalRankFusionService = Depends(ReciprocalRankFusionService),
        re_ranking_service: ReRanker = Depends(ReRanker),
        llm_response_service: LLMResponseService = Depends(LLMResponseService),
        delete_index: DeleteIndex = Depends(DeleteIndex),
        vector_db_service: VectorDBService = Depends(VectorDBService),
        query_decomposition_service = Depends(QueryDecompositioneService)
    ) -> None:
        self.retrieve_chunks_service = retrieve_chunks_service
        self.rrf_service = rrf_service
        self.re_ranking_service = re_ranking_service
        self.llm_response_service = llm_response_service
        self.delete_index = delete_index
        self.vector_db_service = vector_db_service
        self.query_decomposition_service= query_decomposition_service

    async def retrieve_and_generate(self, query: str):
        try:
            sub_queries = await self.query_decomposition_service.decompose_query(query)
            
            tasks = [self._retrieve_and_generate_for_subquery(sub_q) for sub_q in sub_queries]
            results = await asyncio.gather(*tasks)

            all_final_chunks_with_score = []
            sub_responses = []
            for res in results:
                sub_response, final_chunks_with_score =  res
                all_final_chunks_with_score.append(final_chunks_with_score)
                sub_responses.append(sub_response)
            response = await self.llm_response_service.generate_final_response(query, sub_queries, sub_responses)
            # Delete all indexes.
            #self.delete_index.delete_all_index()
            # Return the final results.
            return response, all_final_chunks_with_score
        
        except Exception as e:
            print(e)
            raise e
        
    async def _retrieve_and_generate_for_subquery(self, sub_query: str):
        try:
            retrieved_images = await self.retrieve_chunks_service.pinecone_retrieve_similar_chunks_images(sub_query, top_k=3)
            dense_chunks, _, _ = await self.retrieve_chunks_service.pinecone_retrieve_similar_chunks(sub_query, 5)
            sparse_chunks, _ = await self.retrieve_chunks_service.pinecone_retrieve_similar_chunks_s(sub_query, 5)
            sorted_items, sorted_documents = self.rrf_service.fuse(dense_chunks, sparse_chunks)
            final_chunks_with_score, final_chunks = await self.re_ranking_service.re_ranker(sub_query, sorted_documents)
            response = await self.llm_response_service.generate_response_gemini(final_chunks, sub_query, retrieved_images)
            return response, final_chunks_with_score
            
        except Exception as e:
            print(e)
            raise e
