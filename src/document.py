# Disable Hugging Face Hub symlink warning, can be safely ignored in this context since 
# we're not using symlinks for model storage and hf token warning
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedding_manager import EmbeddingManager
from vector_store import VectorStore
import json

def loading_txt_json_md_document_from_file(file_path: str) -> Document:
    # Determine the file type based on the file extension for pdf, txt, json, md
    document = None
    loader = None
    if file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
        document = loader.load()    
    elif file_path.endswith('.pdf'):
        loader = PyMuPDFLoader(file_path)
        document = loader.load()    
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        document = [Document(page_content=str(data))]
    elif file_path.endswith('.md'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        document = [Document(page_content=content)]
    else:
        print(f"Unsupported file type: {file_path}")
    return document

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # Returns the length of the text in characters. You can customize this function to count tokens instead if needed.
        separators=["\n\n", "\n", " ", ""]
        # The separators are ordered from the most to least preferred. 
        # The text will be split at the first separator that results in chunks smaller than the specified chunk size.
    )
    return text_splitter.split_documents(documents)

def main():
    # Example usage of loading a txt document
    docs = []
    print("==== Loading documents from files ====")
    docs.extend(loading_txt_json_md_document_from_file("../data/Documentación 2.txt"))
    docs.extend(loading_txt_json_md_document_from_file("../data/Documentación 1.pdf"))
    docs.extend(loading_txt_json_md_document_from_file("../data/Documentación 4.json"))
    docs.extend(loading_txt_json_md_document_from_file("../data/Documentación 3.md"))
    #for doc in docs:
        #print(doc)

    # Split the documents in chunks
    print("==== Splitting documents into chunks ====")
    split_docs = split_documents(docs)
    print(f"Number of split documents: {len(split_docs)}, Original number of documents: {len(docs)}")
    #print(f"First split document: {split_docs[0]}\n")
    #print(f"Second split document: {split_docs[1]}")

    # Initialize embedding manager
    embedding_manager = EmbeddingManager()
    # Create embeddings for the split documents
    texts = [doc.page_content for doc in split_docs]

    print(("==== Creating embeddings for split documents ===="))
    embeddings = embedding_manager.embed(texts)

    # Initialize vector store
    vector_store = VectorStore(
        collection_name="document_embeddings",
        persist_directory="../data/chromadb_data"
    )

    # Add documents and their embeddings to the vector store - this will persist the data to disk
    print("==== Adding documents and embeddings to vector store ====")
    vector_store.add_documents_and_embeddings(split_docs, embeddings)


if __name__ == "__main__":
    main()