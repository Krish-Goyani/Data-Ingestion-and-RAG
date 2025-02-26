import os
import base64
import asyncio
from typing import List, Dict
from google import genai
from google.genai import types
from src.app.config.settings import settings
from src.app.services.dense_embedding_service import EmbeddingService
from fastapi import Depends
from src.app.config.database import mongodb_database
from pinecone import PineconeAsyncio, ServerlessSpec, VectorType
import uuid
from src.app.repositories.images_repository import ImageMappingRepo


class ImageSummaryService:
    
    def __init__(self,dense_embedding_service = Depends(EmbeddingService), images_repossitory = Depends(ImageMappingRepo) ):
        # Use the provided API key or fall back to an environment variable.
        self.api_key = settings.GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("Gemini API key must be provided or set in the environment variable 'GEMINI_API_KEY'.")
        
        # Initialize the Gemini client for asynchronous calls.
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.0-flash-exp"
        # Prompt to ask for a summary of the image.
        self.prompt = '''You are an assistant tasked with summarizing tables and images particularly for semantic retrieval.
                        These summaries will be embedded and used to retrieve the raw images or table elements
                        Give a detailed summary of the table or images below that is well optimized for retrieval.
                        For any tables also add in a one line description of what the table is about besides the summary.
                        in the final response Do not add additional words like Summary: etc.'''
                        
        self.dense_embedding_service= dense_embedding_service
        self.images_repository = images_repossitory
        self.index_name = "idx001"
        

    async def _process_single_image(self, image_base64: str):
        """
        Process a single base64 encoded image: decode, call Gemini API, and return summary.
        
        Args:
            image_base64 (str): Base64 encoded image.
        
        Returns:
            dict: A dictionary with keys 'image_base64' and 'summary'.
        """
        try:
            # Decode the base64 string to bytes.
            image_bytes = base64.b64decode(image_base64)
            
            # Create a Part for the image using the decoded bytes.
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            
            # Prepare the contents list: prompt + image part.
            contents = [self.prompt, image_part]
            
            # Asynchronously call the Gemini API to generate content.
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents
            )
            
            # Extract the summary from the response text.
            summary = response.text.strip() if response.text else "No summary generated."
            
            return summary
        
        except Exception as e:
            # Log the error or return an error message as summary.
            return {"image_base64": image_base64, "summary": f"Error: {str(e)}"}

    async def pinecone_generate_and_store_embeddings(self, summaries):
        async with PineconeAsyncio(api_key= settings.PINECONE_API_KEY) as pc:
            index_info = await pc.describe_index(name= self.index_name)  
            try:    
                async with pc.IndexAsyncio(host= index_info.host) as idx: 
                    
                    embeddings = await  self.dense_embedding_service.generate_embeddings(summaries)      
                    records = [
                        {
                            "id": str(uuid.uuid4()),  
                            "values": e["embedding"],  
                            "metadata": {"text":e["text"] ,"record_id": str(uuid.uuid4())},
                        }
                        for e in embeddings
                    ]

                    # Upsert records into Pinecone
                    await idx.upsert(vectors=records, namespace="imgportion")
                return records                    
            except Exception as e:
                print(e) 
                
                
                
    async def summarize_images(self, base64_images: List[str]):
            """
            Processes a list of base64 images concurrently and returns summaries.
            
            Args:
                base64_images (List[str]): List of base64 encoded images.
            
            Returns:
                List[Dict[str, str]]: List of dictionaries containing original base64 image and its summary.
            """
            tasks = [self._process_single_image(img) for img in base64_images]
            results = await asyncio.gather(*tasks)
            records = await self.pinecone_generate_and_store_embeddings(results)
            await self.images_repository.store_image_mappings([x["metadata"]["record_id"] for x in records], base64_images)