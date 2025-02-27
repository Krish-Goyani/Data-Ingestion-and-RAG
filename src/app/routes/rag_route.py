from fastapi import APIRouter, Depends, UploadFile, File, Form
from src.app.controllers.upload_controller import FileuploadController
from src.app.utils.error_handler import error_handler

file_upload_router = APIRouter()

@file_upload_router.post("/upload/")
@error_handler
async def create_upload_file(
    chunk_size: int = Form(...),
    chunk_overlap: int = Form(...),
    file: UploadFile = File(...),
    file_upload_controller=Depends(FileuploadController),
):
    if not file:
        return {"message": "No upload file sent"}
    else:
        await file_upload_controller.process_file(file, chunk_size, chunk_overlap)
        return {"message": "File processed successfully!"}

@file_upload_router.post("/query/")
@error_handler
async def generate_response(
    query: str = Form(...),
    file_upload_controller=Depends(FileuploadController)
):
    response, final_chunks, retrieved_images = await file_upload_controller.generate_response(query)
    return {"response": response, "final_chunks": final_chunks, "retrieved_images" : retrieved_images}
