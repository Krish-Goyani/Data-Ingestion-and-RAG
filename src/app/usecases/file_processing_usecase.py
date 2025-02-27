# src/app/usecases/file_processing_usecase.py
import time
from fastapi import Depends
from src.app.services.file_conversion_service import FileConversionService
from src.app.services.text_chunking_service import TextSplitters
from src.app.services.vector_db_service import VectorDBService
from src.app.services.unstructured_api_service import UnstructuredAPIService
from src.app.services.sparse_embedding_service import SparseEmbeddingsService
from src.app.services.image_summary_service import ImageSummaryService

class FileProcessingUsecase:
    def __init__(
        self,
        file_conversion_service: FileConversionService = Depends(FileConversionService),
        text_splitter: TextSplitters = Depends(TextSplitters),
        vector_db_service: VectorDBService = Depends(VectorDBService),
        unstructured_api_service: UnstructuredAPIService = Depends(UnstructuredAPIService),
        sparse_embedding_service: SparseEmbeddingsService = Depends(SparseEmbeddingsService),
        image_summary_service: ImageSummaryService = Depends(ImageSummaryService)
    ) -> None:
        self.file_conversion_service = file_conversion_service
        self.text_splitter = text_splitter
        self.vector_db_service = vector_db_service
        self.unstructured_api_service = unstructured_api_service
        self.sparse_embedding_service = sparse_embedding_service
        self.image_summary_service = image_summary_service

    async def process_file_storage(self, file_bytes: bytes, chunk_size: int, chunk_overlap: int):
        try:
            # Convert file to markdown and chunk it.
            text = await self.file_conversion_service.convert_to_makedown(file_bytes)
            chunks = self.text_splitter.hierarchical_markdown_chunker(text, chunk_size, chunk_overlap)
            
            # Generate and store dense embeddings.
            await self.vector_db_service.pinecone_generate_and_store_embeddings(chunks)
            time.sleep(15)
            
            # Process the file through the unstructured API to extract base64 images.
            base64s = await self.unstructured_api_service.process_file(file_bytes)
            await self.image_summary_service.summarize_images(base64s)
            
            # Generate sparse embeddings and store them.
            sparse_embeddings = self.sparse_embedding_service.generate_sparse_embeddings(chunks)
            await self.vector_db_service.pinecone_store_sparse_embeddings(chunks, sparse_embeddings)
            time.sleep(45)
            
            return 

        except Exception as e:
            print(e)
            raise e
