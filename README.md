# RAG-Review

Retrieval-Augmented Generation is the process of optimizing the output of an LLM by giving it data from an tursted knowledge base

## Data Ingestion Pipeline

Tips: 
- Dealing with folders is easier with pathlib, os and glob 
- Use tqdm for checking process and cuda if available. 

1. Parsing data

### Loading documents

Converting data from the original format and chunking (dividing it in groups) in a document structure.

Langchain documents can contain text content in any format (md, txt, etc)
- It is better to use langchain_core, langchain integration libraries or custom implementations
- Langchain_community is being sunset

Document Loading Types
- Async (sequential but with await, to allow doing other stuff) or Sync (sequential, one after the other but the whole process has to finish for other tasks  to execute)
- All at a time / default (all units in memory) or Lazy Load (one unit at a time)

Specific loaders
- pymupdf loader works great but it has a license that forces your project to stay open source
- pypdf reduces dependencies
- other ones in langchain integrations site. 

True parallelism can be achieved by loading many async loads with asyncio.

### Chunking

Chunking needs to be done smartly to avoid cutting data mid-sentences or directly connected info.
This will be done depending on the input format type

Text Splitters
- Langchain integrations documentations provides custom splitters for json, md, code, html
- RecursiveCharacterTextSplitter is good for most scenarios, only fine-tune if necessary. 

2. Embeddings

Creating numerical representations for text and storing them in vectors

Creating an embedding manager to better control how embeddings are made and load the transformer model allows for good management. 
Hugging face transformers are very common.

3. VectorDBs

Storing embeddings vectors in vectorDBs to avoid having to create the embeddings again

ChromaDB 

- Full vector database that allows you to store persistently all indexes, original text, embeddings and metadata with search engine
- Supports metadata filtering
- has built-in add/query APIs
- Heavier dependency and not as fast at large scale.

FAISS 

- Library engine to store only vectors and find the nearest ones to a query vector
- Extremely fast, even for billions of vectors
- Doesn't manage anything else (persistency, storage, or metadata filtering)
- Runs locally

How to choose, briefly: for local development and learning, Chroma or FAISS (you've already got the two best entry points). If you want local-but-more-scalable, Qdrant or LanceDB. If you already use Postgres, pgvector saves you running a second system. If you want zero infrastructure management and don't mind paying, Pinecone.

Self-hosted servers, scale-oriented: Qdrant (rust based with strong filtering), Milvus, Weaviate (also offers hybrid keyword+vector search and a cloud option), pgvector — which is worth special mention: it's an extension that adds vector search to PostgreSQL, so if you already run Postgres, you can store embeddings right alongside your normal relational data, no separate system.

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