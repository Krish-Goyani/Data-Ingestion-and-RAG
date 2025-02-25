from pinecone.grpc import PineconeGRPC as Pinecone

from src.app.config.settings import settings

# Initialize Pinecone (replace with your API key and environment)
class DeleteIndex:
    def __init__(self) -> None:
        pass
    

    def delete_all_index(self):
        try:
            pc =  Pinecone(api_key=settings.PINECONE_API_KEY)
            # List all indexes
            response = pc.list_indexes()
            indexes = [index["name"] for index in response]
            # Delete each index
            for index in indexes:
                pc.delete_index(name = index)

            
        except Exception as e:
            print(e)
