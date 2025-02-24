from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from src.app.routes.upload_route import file_upload_router
from src.app.config.database import mongodb_database
from contextlib import asynccontextmanager


@asynccontextmanager
async def db_lifespan(app: FastAPI):
    mongodb_database.connect()
    yield
    mongodb_database.disconnect()


app = FastAPI(lifespan=db_lifespan)
app.include_router(file_upload_router)


