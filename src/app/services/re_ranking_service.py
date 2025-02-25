from src.app.config.settings import settings
from pinecone import PineconeAsyncio, ServerlessSpec, VectorType


class ReRanker:
    def __init__(self) -> None:
        self.reranker_model = "bge-reranker-v2-m3"
        
    async def re_ranker(self,query, chunks):
        async with PineconeAsyncio(api_key= settings.PINECONE_API_KEY) as pc:
            results = await pc.inference.rerank(
                model= self.reranker_model,
                query= query,
                documents= chunks,
                return_documents= True,
                top_n=5,
                parameters={
                    "truncate": "END"
                }
            )
        docs =    [entry["document"]["text"] for entry in results.data]
        return docs
         
        #return [{"score" : entry["score"], "document": entry["document"]["text"]} for entry in results.data], results.usage["rerank_units"]
        
        