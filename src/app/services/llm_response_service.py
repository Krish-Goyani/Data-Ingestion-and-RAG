import os
import asyncio
import textwrap
from groq import AsyncGroq

from src.app.config.settings import settings  # Ensure settings has the GROQ_API_KEY

class LLMResponseService:
    def __init__(self) -> None:
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)  # Groq API client
        self.model = "llama-3.3-70b-versatile"  # Choose an optimal Groq model

    async def generate_response(self, retrieved_chunks: list[str], user_query: str) -> str:
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
