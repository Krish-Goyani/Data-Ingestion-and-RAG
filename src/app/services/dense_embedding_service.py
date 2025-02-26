import asyncio
from pinecone import PineconeAsyncio
from src.app.config.settings import settings


class EmbeddingService:
    def __init__(self) -> None:
        self.token_limit_per_minute = 250000  # Pinecone's token limit
        self.batch_size = 50  # Adjust based on your use case

    async def generate_embeddings(self, chunks: list[str]) -> list[dict]:
        """
        Generates embeddings for the given chunks asynchronously.
        
        :param chunks: List of text chunks.
        :return: List of dictionaries containing embeddings.
        """
        async with PineconeAsyncio(api_key=settings.PINECONE_API_KEY) as pc:
            try:
                total_tokens_used = 0
                all_embeddings = []
                e_tokens_used = 0
                # Process in batches
                for i in range(0, len(chunks), self.batch_size):
                    batch = chunks[i:i + self.batch_size]

                    # **Count tokens by splitting text on spaces**
                    batch_token_count = sum(len(text.split()) for text in batch)

                    # If token limit exceeded, wait
                    if total_tokens_used + batch_token_count > self.token_limit_per_minute:
                        print("Rate limit reached, sleeping for 60 seconds...")
                        await asyncio.sleep(60)  # Async sleep before resuming
                        total_tokens_used = 0  # Reset counter

                    # Generate embeddings
                    embeddings = await pc.inference.embed(
                        model="llama-text-embed-v2",
                        inputs=batch,
                        parameters={
                            "input_type": "passage",
                            "truncate": "END",
                            "dimension" :  1024
                        }
                    )
                    e_tokens_used += embeddings.usage["total_tokens"]
                    # Store embeddings in a structured format
                    batch_embeddings = [
                        {"text": text, "embedding": emb["values"]}
                        for text, emb in zip(batch, embeddings)
                    ]

                    all_embeddings.extend(batch_embeddings)

                    # Update total token usage
                    total_tokens_used += batch_token_count
                    
                return all_embeddings
            
            except Exception as e:
                print(f"Error in generating embeddings: {e}")
                return []
