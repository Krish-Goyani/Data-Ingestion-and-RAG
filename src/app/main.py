from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from src.app.routes.upload_route import file_upload_router

app = FastAPI()
app.include_router(file_upload_router)


