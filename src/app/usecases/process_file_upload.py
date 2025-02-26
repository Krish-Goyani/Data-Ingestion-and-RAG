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
from src.app.services.delete_index import DeleteIndex
import time
from src.testing.ragas_testing import RAGAsTest
import pandas as pd
from src.app.services.unstructured_api_service import UnstructuredAPIService
from src.app.services.image_summary_service import ImageSummaryService


class FileUploadUsecase:
    def __init__(self, file_conversion_service = Depends(FileConversionService), text_splitter = Depends(TextSplitters), vector_db_service = Depends(VectorDBService), retrieve_chunks_service = Depends(RetrieveChunksService), llm_response_service = Depends(LLMResponseService), embedding_service  =Depends(EmbeddingService), sprase_embedding_service = Depends(SparseEmbeddingsService), rrf_service= Depends(ReciprocalRankFusionService),
                 re_ranking_service= Depends(ReRanker), cost_tracker = Depends(CostTracker), cost_storage_repo = Depends(CostStorageRepo), delete_index = Depends(DeleteIndex), ragas_tesing = Depends(RAGAsTest), unstructured_api_service = Depends(UnstructuredAPIService), image_summary_service = Depends(ImageSummaryService)) -> None:
        
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
        self.delete_index = delete_index
        self.ragas_tesing =  ragas_tesing
        self.unstructured_api_service = unstructured_api_service
        self.image_summary_service = image_summary_service
        
        
    def get_questions_answers_and_text(self,document_indices,
                                    multi_passage_file='RAG_Evaluation_Dataset/multi_passage_answer_questions.csv',
                                    single_passage_file='RAG_Evaluation_Dataset/single_passage_answer_questions.csv',
                                    no_answer_file='RAG_Evaluation_Dataset/no_answer_questions.csv',
                                    documents_file='RAG_Evaluation_Dataset/documents.csv'):
        """
        Reads CSV files using pandas and returns:
        - questions: List of questions for the given document indices.
        - answers: Corresponding answers or a dummy message if no answer is available.
        - document_text: A single string that merges the text of all documents with the given indices.

        :param document_indices: A list of document indices to look up.
        :param multi_passage_file: Path to multi_passage_answer_questions.csv.
        :param single_passage_file: Path to single_passage_answer_questions.csv.
        :param no_answer_file: Path to no_answer_questions.csv.
        :param documents_file: Path to documents.csv containing "index" and "text".
        :return: (questions, answers, document_text)
        """
        # Convert document indices to strings for consistency
        doc_indices_set = set(map(str, document_indices))
        
        # Load and filter multi_passage_answer_questions.csv
        multi_df = pd.read_csv(multi_passage_file, encoding='utf-8')
        multi_df = multi_df[multi_df['document_index'].astype(str).isin(doc_indices_set)]
        
        # Load and filter single_passage_answer_questions.csv
        single_df = pd.read_csv(single_passage_file, encoding='utf-8')
        single_df = single_df[single_df['document_index'].astype(str).isin(doc_indices_set)]
        
        # Load and filter no_answer_questions.csv, and assign a dummy answer
        no_answer_df = pd.read_csv(no_answer_file, encoding='utf-8')
        no_answer_df = no_answer_df[no_answer_df['document_index'].astype(str).isin(doc_indices_set)]
        no_answer_df = no_answer_df.assign(answer="No answer provided")
        
        # Combine all question-answer pairs from the three DataFrames
        combined_df = pd.concat([multi_df, single_df, no_answer_df], ignore_index=True)
        
        # Extract questions and answers as lists (filling missing values if necessary)
        questions = combined_df['question'].fillna('').tolist()
        answers = combined_df['answer'].fillna('No answer found').tolist()
        
        # Load and filter documents.csv for the relevant document indices
        documents_df = pd.read_csv(documents_file, encoding='utf-8')
        filtered_docs = documents_df[documents_df['index'].astype(str).isin(doc_indices_set)]
        
        # Merge all document texts into a single string (separated by newline)
        document_text = "\n".join(filtered_docs['text'].fillna('').tolist())
        
        return questions, answers, document_text
    
    
    async def process_file(self,file_bytes : bytes,chunk_size,chunk_overlap, query):
        try:
            text =  await self.file_conversion_service.convert_to_makedown(file_bytes)
            chunks =  self.text_splitter.hierarchical_markdown_chunker(text, chunk_size, chunk_overlap)
            
            embedding_tokens = await self.vector_db_service.pinecone_generate_and_store_embeddings(chunks)
            time.sleep(15)
            #self.cost_tracker.add_embedding_tokens(embedding_tokens)
            
            base64s =  await self.unstructured_api_service.process_file(file_bytes)
            await self.image_summary_service.summarize_images(base64s)
            
            sparse_embeddings = self.sprase_embedding_service.generate_sparse_embeddings(chunks)
            await self.vector_db_service.pinecone_store_sparse_embeddings(chunks, sparse_embeddings)
            time.sleep(15)
            
            
            dense_chunks, embedding_tokens, read_units = await self.retrieve_chunks_service.pinecone_retrieve_similar_chunks(query, 10)
            #self.cost_tracker.add_read_units(read_units)
            #self.cost_tracker.add_embedding_tokens(embedding_tokens)
            sparse_chunks, read_units = await self.retrieve_chunks_service.pinecone_retrieve_similar_chunks_s(query, 10)
            #self.cost_tracker.add_read_units(read_units)
            
            sorted_items, sorted_documents = self.rrf_service.fuse(dense_chunks, sparse_chunks)
            final_chunks = await self.re_ranking_service.re_ranker(query, sorted_documents)
            #self.cost_tracker.add_rerank_units(rerank_units)
            
            
            response, llm_usage  = await self.llm_response_service.generate_response(final_chunks, query)
            responses.append(response)
            relevant_doc.append(final_chunks)
            #self.cost_tracker.add_llm_tokens(llm_usage)
            
            
            
            #dense_embeddings,embedding_tokens  = await self.embedding_service.generate_embeddings(chunks)
            #self.cost_tracker.add_embedding_tokens(embedding_tokens)

            #await self.vector_db_service.qdrant_store_embeddings(embeddings)
            #await self.vector_db_service.milvus_store_embeddings(embeddings)
            #qdrant_chunks = await self.retrieve_chunks_service.search_qdrant(query, 20)
            #milvus_chunks = await self.retrieve_chunks_service.search_milvus(query, 10)
                    
            #await self.cost_storage_repo.store_cost_details(self.cost_tracker.to_dict())
            self.delete_index.delete_all_index()
            return 'final_chunks', 'pinecone_chunks', 'sparse_chunks', chunks
        except Exception as e:
            print(e)
        
            