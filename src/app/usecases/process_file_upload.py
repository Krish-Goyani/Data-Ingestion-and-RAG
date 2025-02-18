from src.app.services.file_conversion_service import FileConversionService
from fastapi import Depends
from src.app.services.text_chunking_service import TextSplitters


class FileUploadUsecase:
    def __init__(self, file_conversion_service = Depends(FileConversionService), text_splitter = Depends(TextSplitters)) -> None:
        self.file_conversion_service = file_conversion_service
        self.text_splitter = text_splitter
        
    async def process_file(self,file_bytes : bytes,chunk_size,chunk_overlap):
        text =  await self.file_conversion_service.convert_to_text(file_bytes)
        chunks =  self.text_splitter.recursive_text_splitter(text, chunk_size, chunk_overlap)
        return chunks
    
    
        