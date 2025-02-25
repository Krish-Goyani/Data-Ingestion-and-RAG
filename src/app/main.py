import asyncio
import sys

# This will ensure the standard event loop policy is used
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Add this block to forcibly remove uvloop from the modules
# This is more aggressive but should prevent uvloop from being used anywhere
try:
    if 'uvloop' in sys.modules:
        del sys.modules['uvloop']
except:
    pass

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