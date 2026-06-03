import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingManager:
    """Manages the creation, storage, and retrieval of document embeddings using Sentence Transformers."""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the embedding manager

        Args: 
            model_name (str): The name of the Sentence Transformer model to use for creating embeddings.
        """
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        """Loads the Sentence Transformer model if it hasn't been loaded already."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"Error loading model '{self.model_name}': {e}")
                raise e

    def embed(self, texts: List[str]) -> np.ndarray:
        """Creates embeddings for a list of texts."""
        self._load_model()              # ensure loaded on first real use
        embeddings = self.model.encode(texts)
        return embeddings

    def save_embeddings(self, embeddings, file_path):
        pass

    def load_embeddings(self, file_path):
        pass