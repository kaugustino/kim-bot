# kim-bot

## Goal

Looking to create an LLM based on all my project file directories. I want the LLM to capture and reference all my past work within a given source directory.

## Initial Research

Retrieval-Augmented Generation, or RAG, is an AI technique that gives LLMs access to an external knowledge base in order to provide more context before generating a response. This eliminates the need to retrain the model and also reduces hallucinations.

I can implement RAG using embeddings, which are dense vectors in high-dimensional space that can represent words, sentences, or documents. I want to create embeddings of my context first, which would be my project file directories. A user's query gets compressed into a vector and can then be compared against the embeddings space of the given context. Then I can retrieve the documents that have the most similarity to the query embedding. The retrieved context will augment the model to generate more accurate responses. 
