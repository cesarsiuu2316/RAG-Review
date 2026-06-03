
import chromadb
import uuid
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

class VectorStore: 
    """
    Manages storage of document embeddings in Chromadb vectorStore
    """

    def __init__(self, collection_name: str = "document_embeddings", persist_directory: str = "./data/chromadb_data"):
        """
        Initializes the vector store

        Args:
            collection_name (str): The name of the Chromadb collection to use for storing embeddings.
            persist_directory (str): The directory where the Chromadb data will be persisted.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_chromadb()
        
    def _initialize_chromadb(self):
        """
        Initialize ChromaDB client and collection
        """
        try: 
            # Ensure the persistence directory exists and initialize the ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Create or get the collection for storing document embeddings
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for RAG system"}
            )
        except Exception as e: 
            print(f"Error initializing ChromaDB: {e}")
            raise e


    def add_documents_and_embeddings(self, documents: list[Document], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document content
            documents_text.append(doc.page_content)
            
            # Embedding
            embeddings_list.append(embedding.tolist())
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise