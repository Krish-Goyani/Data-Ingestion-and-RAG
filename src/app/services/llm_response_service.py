import os
import asyncio
import textwrap
from groq import AsyncGroq
import os
import base64
from google import genai
from google.genai import types

from src.app.config.settings import settings  # Ensure settings has the GROQ_API_KEY

class LLMResponseService:
    def __init__(self) -> None:
        #self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)  # Groq API client
        #self.model = "llama-3.3-70b-versatile"  # Choose an optimal Groq model
        self.client = genai.Client(api_key= settings.GEMINI_API_KEY)
        self.model = "gemini-2.0-flash-exp"

    async def generate_response_groq(self, retrieved_chunks: list[str], user_query: str) -> str:
        """
        Generates a response using Groq LLM based on retrieved document chunks.
        
        :param retrieved_chunks: List of relevant document chunks retrieved from vector DB.
        :param user_query: User's original query.
        :return: LLM-generated response.
        """
        
        # Format context chunks
        formatted_chunks = "\n\n".join(textwrap.fill(chunk, width=80) for chunk in retrieved_chunks)
        

        # Industry-grade RAG prompt
        prompt = f"""
        You are an advanced AI assistant specializing in answering queries based on given document contexts.
        ----
        **Context Information:**  
        {formatted_chunks}
        
        ----
        **User Query:**  
        {user_query}
        
        ----
        **Instructions:**  
        - Provide a detailed yet concise response.
        - If the context lacks relevant information, state that clearly.
        - Keep the response professional and industry-standard.
        
        ----
        **Answer:**
        """

        try:
            # Call Groq LLM asynchronously
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": "You are a helpful AI assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.7,  # Balanced creativity
                max_tokens=1024
            )

            return response.choices[0].message.content.strip(), response.usage.to_dict()
        
        except Exception as e:
            return f"Error generating LLM response: {str(e)}"
        
    async def generate_response_gemini(self, retrieved_chunks: list[str], query: str, base64_images: list[str] = None) -> str:
        """
        Generate an LLM response using Gemini API given retrieved text chunks, a user query,
        and optionally a list of base64-encoded images.
        
        Args:
            retrieved_chunks (list[str]): List of text chunks used as context.
            query (str): The user's query.
            base64_images (list[str], optional): List of base64 encoded images (if available).
        
        Returns:
            str: The generated response from the Gemini API.
        """
        # Build the RAG prompt with context and query.
        context = "\n\n".join(textwrap.fill(chunk, width=80) for chunk in retrieved_chunks)
        prompt = (f"""
            You are an advanced AI assistant specializing in answering queries based on given document contexts and images.
            ----
            **Context Information:**  
            {context}
            
            ----
            **User Query:**  
            {query}
            
            ----
            **Instructions:**  
            - Provide a detailed yet concise response.
            - If the context lacks relevant information, state that clearly.
            - Keep the response professional and industry-standard.
            - if you recieve the image in the input than at the end of the response return "image recieved".
            ----
            **Answer:**
            """)
        
        # Prepare the contents list for Gemini API.
        contents = [prompt]
        print(prompt)
        # If provided, add each image as a Gemini types.Part.
        if base64_images:
            for b64 in base64_images:
                try:
                    image_bytes = base64.b64decode(b64)
                    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                    contents.append(image_part)
                except Exception as e:
                    # Log the error and continue processing other images.
                    print(f"Error processing an image: {e}")
        print(f"===================={len(contents)}")
        # Call the Gemini API asynchronously.
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents
        )
        
        return response.text.strip()
