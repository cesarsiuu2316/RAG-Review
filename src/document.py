from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        pass
    elif file_path.endswith('.md'):
        pass
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
    docs.extend(loading_txt_json_md_document_from_file("../data/Documentación 2.txt"))
    docs.extend(loading_txt_json_md_document_from_file("../data/Documentación 1.pdf"))
    #for doc in docs:
        #print(doc)

    # Split the documents
    split_docs = split_documents(docs)
    print(f"Number of split documents: {len(split_docs)}, Original number of documents: {len(docs)}")
    print(f"First split document: {split_docs[0]}\n")
    print(f"Second split document: {split_docs[1]}")

if __name__ == "__main__":
    main()