# rag-notes


## Text loading

- Text, Lazy, Web, PDF, check langchain community (deprecated) see alternative

## Chunking strategies

- RecursiveCharacterTextSplitter, split text into chunks with overlap. Too much or less gives less precision or noise. Keep it optimal based on content
- Semantic Chunking, requires API but can chunk based on topics. Best for mixed topics documents
