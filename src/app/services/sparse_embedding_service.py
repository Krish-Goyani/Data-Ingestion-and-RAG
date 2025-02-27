import os
import pickle
from pinecone_text.sparse import BM25Encoder
from typing import List, Dict

class SparseEmbeddingsService:
    def __init__(self, model_path: str = "bm25_model.pkl"):
        """
        Initializes the BM25 encoder.
        """
        self.model_path = model_path
        self.bm25 = None
        self.is_fitted = False  # Track if corpus is fitted
        self.fitted_corpus = None  # Store the corpus used for fitting

    def fit_corpus(self, chunks: List[str]):
        """
        Fits the BM25 model on the provided list of text chunks and saves it locally.
        
        :param chunks: List of text chunks.
        """
        if not chunks:
            raise ValueError("The corpus (chunks) must not be empty.")
        
        # Create and fit the BM25 model.
        self.bm25 = BM25Encoder()
        self.bm25.fit(chunks)
        self.is_fitted = True
        self.fitted_corpus = chunks
        
        # Save the fitted model to a local file.
        with open(self.model_path, "wb") as f:
            pickle.dump(self.bm25, f)

    def load_model(self):
        """
        Loads the BM25 model from the local file if available.
        """
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.bm25 = pickle.load(f)
            self.is_fitted = True
        else:
            self.is_fitted = False

    def generate_sparse_embeddings(self, chunks: List[str]) -> List[Dict]:
        """
        Generates sparse embeddings for each chunk.
        
        :param chunks: List of text chunks.
        :return: List of sparse embeddings for each chunk.
        """
        if not chunks:
            raise ValueError("Chunks list must not be empty.")
        
        # Try to load the model from file if not already fitted.
        if not self.is_fitted:
            self.load_model()
        
        # If still not fitted, then fit the model.
        if not self.is_fitted:
            self.fit_corpus(chunks)
        
        # Generate embeddings for each chunk.
        return [self.bm25.encode_documents(chunk) for chunk in chunks]
    
    def generate_query_embedding(self, query: str):
        """
        Generates a sparse embedding for a query.
        
        :param query: The query string.
        :return: Sparse query embedding.
        """
        # Ensure the model is loaded; if not, try to load from file.
        if not self.is_fitted:
            self.load_model()
        if not self.is_fitted:
            raise RuntimeError("BM25 model is not fitted. Please call fit_corpus with your document corpus first.")
        
        return self.bm25.encode_queries(query)
