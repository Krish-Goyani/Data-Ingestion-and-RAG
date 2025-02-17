from src.app.services.file_conversion_service import FileConversionService
from fastapi import Depends



class FileUploadUsecase:
    def __init__(self, file_conversion_service = Depends(FileConversionService)) -> None:
        self.file_conversion_service = file_conversion_service
        
    async def process_file(self,file_bytes : bytes):
        return await self.file_conversion_service.convert_to_markdown(file_bytes)
    
    
        