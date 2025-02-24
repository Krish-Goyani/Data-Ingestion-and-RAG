from fastapi import Depends
from src.app.usecases.process_file_upload import FileUploadUsecase




class FileuploadController:
    def __init__(self, file_upload_usecase = Depends(FileUploadUsecase)) -> None:
        self.file_upload_usecase = file_upload_usecase
        
        
    async def process_file(self,file_bytes : bytes,chunk_size,chunk_overlap, query ):
        return await self.file_upload_usecase.process_file(file_bytes, chunk_size, chunk_overlap, query)
    
        