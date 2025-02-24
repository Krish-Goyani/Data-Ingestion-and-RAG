from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PINECONE_API_KEY  : str
    GROQ_API_KEY : str
    QDRANT_URL : str
    MILVUS_URI : str
    MONGO_URI : str
    MONGODB_COLLECTION_NAME : str
    MONGODB_DB_NAME : str
        

    class Config:
        env_file = "src/.env"


settings = Settings()