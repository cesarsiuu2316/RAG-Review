from sentence_transformers import SentenceTransformer
import numpy as np
import torch

class EmbeddingManager:
    """Manages the creation, storage, and retrieval of document embeddings using Sentence Transformers."""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the embedding manager

        Args: 
            model_name (str): The name of the Sentence Transformer model to use for creating embeddings.
            device (str): The device to use for creating embeddings ('cpu' or 'cuda').
        """
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    def _load_model(self):
        """Loads the Sentence Transformer model if it hasn't been loaded already."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as e:
                print(f"Error loading model '{self.model_name}': {e}")
                raise e

    def embed(self, texts: list[str]) -> np.ndarray:
        """Creates embeddings for a list of texts."""
        self._load_model()              # ensure loaded on first real use
        embeddings = self.model.encode(texts)
        return embeddings