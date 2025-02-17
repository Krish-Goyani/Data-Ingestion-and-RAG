from fastapi import FastAPI, UploadFile, APIRouter, Depends
from pydantic import BaseModel
from src.app.utils.error_handler import error_handler
from src.app.controllers.upload_controller import FileuploadController
file_upload_router = APIRouter()

@file_upload_router.post("/upload/")
@error_handler
async def create_upload_file(file: UploadFile | None = None, file_upload_controller = Depends(FileuploadController) ):
    if not file:
        return {"message": "No upload file sent"}
    else:
        md_text = await file_upload_controller.process_file(file)
        return {"message": f"{md_text}"}
