

class TextSplitters:
    def __init__(self) -> None:
        pass
    
    def recursive_text_splitter(self, text, chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]):
        """
        Splits text recursively based on a list of separators until the chunks are small enough.
        
        Parameters:
            text (str): The input text to be split.
            chunk_size (int): Maximum size of each chunk.
            chunk_overlap (int): Overlapping characters between chunks.
            separators (list): List of separators to use for splitting.
        
        Returns:
            List[str]: List of text chunks.
        """
        try: 
            # If text is already within chunk size, return as a single chunk
            if len(text) <= chunk_size:
                return [text]

            # Try splitting using different separators
            for separator in separators:
                if separator in text:
                    parts = text.split(separator)

                    # Reconstruct chunks ensuring they fit within chunk_size
                    chunks = []
                    current_chunk = ""
                    
                    for part in parts:
                        if current_chunk and len(current_chunk) + len(part) + len(separator) > chunk_size:
                            chunks.append(current_chunk)
                            current_chunk = part  # Start new chunk
                        else:
                            current_chunk = current_chunk + separator + part if current_chunk else part
                    
                    if current_chunk:
                        chunks.append(current_chunk)  # Append last chunk
                    
                    # Apply recursion if any chunk is still too large
                    final_chunks = []
                    for chunk in chunks:
                        if len(chunk) > chunk_size:
                            final_chunks.extend(self.recursive_text_splitter(chunk, chunk_size, chunk_overlap, separators))
                        else:
                            final_chunks.append(chunk)

                    # Add overlap for continuity
                    for i in range(1, len(final_chunks)):
                        final_chunks[i] = final_chunks[i - 1][-chunk_overlap:] + final_chunks[i]
                    return final_chunks

            # If no separators work, use brute-force splitting
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
        
        except Exception as e:
            raise e


'''
text_class = TextSplitters()
text = """FastAPI is a modern, fast (high-performance) web framework for building APIs with Python.
It is built on standard Python type hints, making it easy to use and very developer-friendly.

Many companies use FastAPI for high-performance API services. It is based on Starlette and Pydantic.
This allows FastAPI to be extremely efficient while maintaining simplicity.

Natural Language Processing (NLP) models often require high-performance API layers.
FastAPI is commonly used in AI applications due to its speed and ease of use."""


chunks = text_class.recursive_text_splitter(text, chunk_size=100, chunk_overlap=20)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")
'''