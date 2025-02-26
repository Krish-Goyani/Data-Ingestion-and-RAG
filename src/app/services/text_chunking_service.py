import re

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
        
        
    def split_content_with_header(self, prefix: str, content: str, max_chunk_size: int, chunk_overlap: int):
        """
        Split a content block (with a header prefix already added) into chunks that do not exceed
        max_chunk_size (in characters). This function tries to avoid breaking sentences.
        
        Parameters:
        - prefix: The header path string to prepend to every chunk.
        - content: The text content to split.
        - max_chunk_size: Maximum total size of a chunk (including the prefix).
        - chunk_overlap: Number of characters from the end of the previous chunk to include at the beginning of the next.
        
        Returns:
        - A list of chunk strings.
        """
        allowed_size = max_chunk_size - len(prefix)
        # Use a simple regex to split content into sentences.
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        chunks = []
        current_chunk = ""
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            # Decide if we can add the sentence to the current chunk.
            candidate = sentence if not current_chunk else current_chunk + " " + sentence
            if len(candidate) <= allowed_size:
                current_chunk = candidate
                i += 1
            else:
                # If the current chunk is empty (i.e. single sentence too long), force-split it.
                if not current_chunk:
                    current_chunk = sentence[:allowed_size]
                    # Update the sentence to the remaining part.
                    sentences[i] = sentence[allowed_size:]
                # Create the chunk by prepending the header path.
                chunk_text = prefix + current_chunk
                chunks.append(chunk_text)
                # Prepare overlap: take the last 'chunk_overlap' characters from current_chunk (if possible)
                if chunk_overlap > 0 and len(current_chunk) > chunk_overlap:
                    current_chunk = current_chunk[-chunk_overlap:]
                else:
                    current_chunk = ""
        # Append any remaining text as a final chunk.
        if current_chunk:
            chunk_text = prefix + current_chunk
            chunks.append(chunk_text)
        
        return chunks


    def hierarchical_markdown_chunker(self,markdown_text: str, max_chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Split a markdown document into chunks while preserving the hierarchical header context.
        For each content block, the full header path is prepended so that each chunk is self-contained.
        
        The algorithm:
        1. Parses the markdown line-by-line.
        2. Uses header lines (starting with '#') to maintain a header stack.
        3. When content is encountered, it is accumulated until the next header.
        4. The full header path (e.g., "[H1] Main Topic > [H2] Subtopic") is prepended to the content,
            and then the content is split into chunks (without breaking sentences if possible).
        
        Parameters:
        - markdown_text: The complete markdown document as a string.
        - max_chunk_size: Maximum size (in characters) for each chunk (including header path).
        - chunk_overlap: Number of overlapping characters between consecutive chunks.
        
        Returns:
        - A list of text chunks.
        """
        lines = markdown_text.splitlines()
        header_stack = []  # List of tuples: (header_level, header_text)
        current_content_lines = []
        chunks = []
        
        def flush_current_content():
            nonlocal current_content_lines, header_stack, chunks
            if current_content_lines:
                content = "\n".join(current_content_lines).strip()
                if content:
                    # Build the header breadcrumb string from the current header stack.
                    header_prefix = " > ".join(f"[H{level}] {text}" for level, text in header_stack)
                    if header_prefix:
                        header_prefix += "\n\n"
                    # Split the content into chunks, each with the header prefix.
                    sub_chunks = self.split_content_with_header(header_prefix, content, max_chunk_size, chunk_overlap)
                    chunks.extend(sub_chunks)
                current_content_lines = []
        
        for line in lines:
            stripped = line.lstrip()
            # Check if the line is a markdown header.
            if stripped.startswith("#"):
                # Flush any accumulated content before updating the header context.
                flush_current_content()
                # Determine header level (count of '#' characters)
                level = len(stripped) - len(stripped.lstrip("#"))
                header_text = stripped[level:].strip()
                # Remove headers from the stack that are at the same or deeper level.
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()
                # Add the new header to the stack.
                header_stack.append((level, header_text))
            else:
                # Normal content: accumulate the line.
                current_content_lines.append(line)
        
        # Flush any remaining content after processing all lines.
        flush_current_content()
        return chunks

