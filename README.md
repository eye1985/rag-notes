# RAG notes


## Text loading and splitters

Text, Lazy, Web, PDF, check langchain community (deprecated) see alternative

### Markdown
Check out MarkdownHeaderTextSplitter when using MD's.

```python

headers_to_consider = [
  ("#", "h1"),
  ("##", "h2"),
  ("###", "h3")
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_consider)
chunks = splitter.split_text(SAMPLE_TEXT)
..
..
```

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

### Overlap

Reason for overlap is for context to be kept. Its a little redundancy, but helps AI with context. 
Always have overlap. 


### Semantic Chunking

requires API but can chunk based on topics. Best for mixed topics documents

