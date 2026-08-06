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

