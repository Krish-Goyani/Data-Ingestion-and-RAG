from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PINECONE_API_KEY: str
    GROQ_API_KEY: str
    QDRANT_URL: str
    MILVUS_URI: str
    MONGO_URI: str
    MONGODB_COLLECTION_NAME: str
    MONGODB_DB_NAME: str
    RAGAS_APP_TOKEN: str
    GEMINI_API_KEY: str
    UNSTRUCTURED_API_URL: str
    UNSTRUCTURED_API_KEY: str

    class Config:
        env_file = "src/.env"


settings = Settings()
