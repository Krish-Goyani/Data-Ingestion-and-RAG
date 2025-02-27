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
        self.model = "gemini-1.5-pro-002"
        
    def create_combined_answer_prompt(self, original_query, sub_queries, sub_answers):
        """
        Create a prompt that instructs an LLM to generate a combined answer by analyzing
        multiple sub-query results.
        
        Args:
            original_query (str): The original user query
            sub_queries (list): List of sub-queries that were generated
            sub_answers (list): List of answers corresponding to each sub-query
        
        Returns:
            str: A formatted prompt for the LLM
        """
        # Validate inputs
        if len(sub_queries) != len(sub_answers):
            raise ValueError("The number of sub-queries must match the number of sub-answers")
        
        # Format the sub-query and answer pairs
        query_answer_pairs = ""
        for i, (query, answer) in enumerate(zip(sub_queries, sub_answers), 1):
            query_answer_pairs += f"\nSUB-QUERY {i}: {query}\n"
            query_answer_pairs += f"SUB-ANSWER {i}: {answer}\n"
        
        # Create the full prompt
        prompt_template = f"""
                You are an expert assistant that synthesizes information to provide comprehensive answers.

                ORIGINAL QUERY:
                {original_query}

                To answer this query thoroughly, it was broken down into the following sub-queries, each with its own answer:
                {query_answer_pairs}

                INSTRUCTIONS:
                1. Analyze all the sub-answers provided above.
                2. Synthesize this information into a single, coherent response that fully addresses the original query.
                3. Ensure your answer is comprehensive but avoids unnecessary repetition.
                4. If there are conflicting pieces of information, acknowledge them and provide the most accurate synthesis.
                5. If some sub-answers contain information about images, incorporate that visual information appropriately.
                6. Maintain a natural, conversational tone.
                7. Format your response for readability.
                8. Do not reference these instructions or mention sub-queries in your final answer.

                YOUR SYNTHESIZED ANSWER:
                """
        
        return prompt_template

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

            return response.choices[0].message.content.strip()
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
        prompt = (f"""Answer the user's query using all provided context and images (if available). 
                  When images are provided, extract any relevant information from them and incorporate it into your response but also combine the knoledge of both context and image. 
                  If no images are included, rely solely on the text context. Ensure your answer is clear, accurate, and directly addresses the user's query.
            
                <context>
                {context}
                </context>
                
                <user_query>
                {query}
                </user_query>
                
                <answer>""")
        
        # Prepare the contents list for Gemini API.
        contents = [prompt]
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
        # Call the Gemini API asynchronously.
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents
        )
        
        return response.text.strip()


    async def generate_final_response(self, original_query, sub_queries, sub_answers):
        prompt = self.create_combined_answer_prompt(original_query, sub_queries, sub_answers)
        response = await self.client.aio.models.generate_content(
            model= self.model,
            contents= [prompt]
        )
        
        return response.text.strip()