from fastapi import APIRouter, Depends, UploadFile

from src.app.controllers.upload_controller import FileuploadController
from src.app.utils.error_handler import error_handler

file_upload_router = APIRouter()


@file_upload_router.post("/upload/")
@error_handler
async def create_upload_file(
    file: UploadFile,
    chunk_size,
    chunk_overlap,
    file_upload_controller=Depends(FileuploadController),
):
    if not file:
        return {"message": "No upload file sent"}
    else:
        chunks = await file_upload_controller.process_file(
            file,
            int(chunk_size),
            int(chunk_overlap)
        )
        return {"filename": file.filename, "chunks": chunks}
