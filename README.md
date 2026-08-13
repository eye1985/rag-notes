# RAG notes

Uses LangChain packages.


## Chain Patterns

LCEL (current standard)

```python
prompt | model | parser
```

### Sequential

Step 1 -> Step 2 -> Step 3 etc..

### Parallel 

input -> summarize_chain, keywords_chain -> result


### Passthrough (Keep original)

question -> (pass through) context (from retriever) -> prompt | model

### Debugging

Verbose logging, quick and simple (basic)
Callbacks, attach handlers to intercept and log each step (flexible)
LangSmith, Visual trace viewer, cost tracking, eval tools etc.. (Recommended)


## Text loading and splitters

Text, Document, Lazy, Web, PDF, check langchain community (deprecated) see alternative

## Chunking strategies

### RecursiveCharacterTextSplitter

Split text into chunks with overlap. Too much or less gives less precision or noise. Keep it optimal based on content.

Tip use a chunk size comparison first, check the sizes and pick the optimal size. E.g
E.g 

```python
for size in sizes:
  splitter = RecursiveCharacterTextSplitter(
    chunk_size=size, chunk_overlap=size // 5 # 20% overlap
  )
  chunks = splitter.split_text(SAMPLE_TEXT)
  print(f" Size {size}:" {len(chunks)} chunks)
```

```console
Size 200: 6 chunks
Size 500: 3 chunks (optimal)
Size 1000: 1 chunks
```

#### Overlap

Reason for overlap is for context to be kept. Its a little redundancy, but helps AI with context. 
Always have overlap. 

#### Markdown
Check out MarkdownHeaderTextSplitter when using MD's. This helps give better context to the chunks.

```python

headers_to_consider = [
  ("#", "h1"),
  ("##", "h2"),
  ("###", "h3")
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_consider)
chunks = splitter.split_text(SAMPLE_TEXT)
```

#### Code splitter

Use from_language for code splitter for better results.

```python
splitter = RescursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=500, chunk_overlap=50)
splitter.split_text(PYTHON_CODE)
``` 


### Semantic Chunking

requires API but can chunk based on topics. Best for mixed topics documents


## Embeddings

### Open AI

text-embedding-3-small - General use
text-embedding-3-large - high accuracy

```python
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-emedding-3-small")
embedding = embeddings.embed_query(SAMPLE_TEXT) # Check out embed_document for full documents

len(embedding) #Check dimensions
```

### Free models

```python
from langchain_community.embeddings import HuggingFaceBgeEmeddings # Deprecated, find something else
from dotenv import load_dotenv

load_dotenv()
embeddings = HuggingFaceBgeEmeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # 384 dimensions
```

#### Olama
```python
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = OllamaEmbeddings(model="llama2-7b-embedding-q4_0") # Find more online
```

#### Similarity

```python
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
from ollama import embeddings

def similarity_search():

    # Documents
    docs = [
        "Python is a programming language",
        "JavaScript is used for web development",
        "Machine learning enables AI applications",
        "Deep learning uses neural networks",
        "Cats are popular pets",
    ]

    query = "What programming languages exist?"

    # embed documents and query
    doc_vector = embeddings_model.embed_documents(docs)
    query_vector = embeddings_model.embed_query(query)

    # compute cosine similarities
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in doc_vector]

    # rank documents by similarity
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked by similarity:")
    for doc, score in ranked_docs:
        print(f"  {score:.4f}: {doc}")
```

#### Caching

```python

# classic might be deprecated, check for newer solution
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage from LocalFileStore
import tempfile

with tempfile.TemporaryDirectory() as tempdir:
  store = LocalFileStore(directory=tempdir)

  cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embeddings, # embeddings is your embeddings
    document_embedding_cache=store,
    namespace="store"
  )

  text = "random text"

  #hits API
  vectors1 = cached_embeddings.embed_documents([text])

  #hits cache
  vectors2 = cached_embeddings.embed_documents([text])
```

#### Search types

##### Normal Similarity SearchThe Goal: 

Find the mathematically closest, most relevant matches to the query.The Risk: Redundancy. If your top three matches all say the exact same thing, it will return all three, wasting your LLM's context window.Best For: Strict factual lookups, narrow questions, and high-speed data retrieval (e.g., "What is the employee ID for Jane Doe?").
  
##### MMR (Maximal Marginal Relevance) SearchThe Goal: 

Find highly relevant matches but intentionally force them to be diverse and different from one another.The Risk: Slower speed. It requires a two-step process that uses extra CPU power to calculate diversity.Best For: Broad research, open-ended questions, and text summarization (e.g., "Give me an overview of the company's financial risks").

Normally you can use the AI to determine if the question is of MMR or Similarity

## Multi-Agent Patterns

### Supervisor 
One agent coordinates others - Complex workflows

### Hierarchical 
Multiple levels of supervisors - Large organizations

### Collaborative 
Agents communicate peer-to-peer - Creative tasks

### Sequential 
Agents process in order - Pipelines

### Parallel
Agents work simultaneously - Speed optimization

## Tools

Always return errors rather than raising them. This allows the LLM to produce a natural language feedback.

## Security

### Layer 1: Sanitizer
Regex: block known attack patterns

### Layer 2: PII (personally identifiable information) Detector
Regex: mask personal data before LLM sees it

### Layer 3: LLM Guard
LLM: catch attacks regex cannot see

### Layer 4: Your LLM
The actual work

### Layer 5: Output validator
Catch PII leaks and harmful content on the way OUT


