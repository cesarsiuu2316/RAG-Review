from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

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


def main():
    # Example usage of loading a txt document
    txt_document = loading_txt_json_md_document_from_file("../data/Documentación 2.txt")
    print(txt_document)
    pdf_document = loading_txt_json_md_document_from_file("../data/Documentación 1.pdf")
    print(pdf_document)

if __name__ == "__main__":
    main()

# jq schema option for json files jsonloader
# unstructured markdown loader for md files
# text loader for txt files
# use directory loader for loading multiple files in a directory
# difference between lazy load and async load in langchain document loaders
# i think i will use pydpf because pymupdf license forces to open source the code
# load json using json library and then create Document objects manually


# their are other options.

# Use pathlib, os, glob 