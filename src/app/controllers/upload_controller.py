from fastapi import Depends
from src.app.usecases.file_processing_usecase import FileProcessingUsecase
from src.app.usecases.retrieval_and_response_usecase import RetrievalAndResponseUsecase



class FileuploadController:
    def __init__(self, file_upload_usecase = Depends(FileProcessingUsecase), retrieval_and_response_usecase = Depends(RetrievalAndResponseUsecase)) -> None:
        self.file_upload_usecase = file_upload_usecase
        self.retrieval_and_response_usecase = retrieval_and_response_usecase
        
        
    async def process_file(self,file_bytes : bytes,chunk_size,chunk_overlap ):
        return await self.file_upload_usecase.process_file_storage(file_bytes, chunk_size, chunk_overlap)
    
    async def generate_response(self, query : str):
        return await self.retrieval_and_response_usecase.retrieve_and_generate(query)
    
        