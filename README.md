# RAG-Review

Retrieval-Augmented Generation is the process of optimizing the output of an LLM by giving it data from an tursted knowledge base

## Data Ingestion Pipeline

1. Parsing data

Converting data from the original format and chunking (dividing it in groups) in a document structure.

Chunking needs to be done smartly to avoid cutting data mid-sentences or directly connected info.
This will be done depending on the input format type

2. Embeddings

Creating numerical representations for text and storing them in vectors

3. VectorDBs

Storing embeddings vectors in vectorDBs to avoid having to create the embeddings again

## Retrieval Pipelinee

1. User asks question

2. Embeddings are created and sent to the vectorDB

3. Information is gathered using a query retrieval algorithm 

4. Context is given to the LLM + prompt to get an output. Getting data based on algorithms that calculate similarity with the embedding vectors. Similarity search, cosine similarity, etc.


### Parsing Data

Documents
- page_content: all data (str)
- metadata (dict)

Langchain document loaders
- pdfLoader
- csvLoader
- webBaseLoader
- DirectoryLoader