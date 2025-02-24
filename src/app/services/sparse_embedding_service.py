from pinecone_text.sparse import BM25Encoder
from typing import List, Dict

class SparseEmbeddingsService:
    def __init__(self):
        """
        Initializes the BM25 encoder.
        """
        self.bm25 = BM25Encoder()
        self.is_fitted = False  # Track if corpus is fitted

    def fit_corpus(self, chunks: List[str]):
        """
        Fits the BM25 model on the provided list of text chunks.

        :param chunks: List of text chunks.
        """
        if not chunks:
            raise ValueError("The corpus (chunks) must not be empty.")
        
        self.bm25.fit(chunks)
        self.is_fitted = True  # Mark as fitted

    def generate_sparse_embeddings(self, chunks: List[str]) -> List[Dict]:
        """
        Generates sparse embeddings for each chunk after fitting the corpus.

        :param chunks: List of text chunks.
        :return: List of sparse embeddings for each chunk.
        """
        self.fit_corpus(chunks)
        if not self.is_fitted:
            raise RuntimeError("BM25 model is not fitted. Call `fit_corpus` first.")
        
        if not chunks:
            raise ValueError("Chunks list must not be empty.")
        
        return [self.bm25.encode_documents(chunk) for chunk in chunks]
    
    def generate_query_embedding(self, query):
        return self.bm25.encode_queries(query)
