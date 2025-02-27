from typing import List
import ast
from groq import AsyncGroq

from src.app.config.settings import settings  # Ensure settings has the GROQ_API_KEY


class QueryDecompositioneService:
    def __init__(self) -> None:
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)  # Groq API client
        self.model = "llama-3.3-70b-versatile"  # Choose an optimal Groq model
        # self.client = genai.Client(api_key= settings.GEMINI_API_KEY)
        # self.model = "gemini-1.5-pro-002"

    async def decompose_query(self, user_query: str):
        try:
            # Call Groq LLM asynchronously
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful assistant that prepares queries that will be sent to a search component.
                Sometimes, these queries are very complex.
                Your job is to simplify complex queries into multiple queries that can be answered
                in isolation to eachother. and strictly maintain the consistent response format. 
                If the query is simple, then keep it as it is.""",
                    },
                    {
                        "role": "user",
                        "content": "Did Microsoft or Google make more money last year?",
                    },
                    {
                        "role": "assistant",
                        "content": """['How much profit did Microsoft make last year?','How much profit did Google make last year?']""",
                    },
                    {"role": "user", "content": "What is the capital of France?"},
                    {"role": "assistant", "content": "What is the capital of France?"},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.7,  # Balanced creativity
                max_tokens=1024,
            )

            return ast.literal_eval(response.choices[0].message.content.strip())
        except Exception as e:
            return f"Error generating LLM response: {str(e)}"
