# kim-bot

## Goal

I want to create an LLM that can capture and reference everything I've learned in my academic life. *What did I learn in XYZ class?* *What was the main takeaway of that assignment?*

## Initial Research

First idea is to train on top of an existing LLM using my files. This is called **LLM fine-tuning**. However this approach needs a labeled data set to train on. I'm not sure exactly what the labeled data would be for my input files. This results in teaching the LLM to excel at a particular task in a specific style format or tone. It can also inject domain expertise but is not sufficient to reliable introduce new facts.

Second idea is to use another AI technique called **Retrieval-Augmented Generation**, or RAG. RAG gives LLMs access to an external knowledge base in order to provide more context before generating a response. This eliminates the need to retrain the model and reduces hallucinations. It also provides a way to retrieve real-time accurate data instead of data baked into the model during training.

RAG is implemented using embedding models, a type of neural network that converts complex data (text, images, audio) into dense numerical vectors called embeddings. Embeddings capture semantic meaning of relationships and context in high-dimensional space. First, I create embeddings of my project file directories. Second, a user's query gets compressed into a vector and is then compared against the embeddings space of the given context. Finally, I can retrieve the documents that have the most similarity to the query embedding. The retrieved context will augment the model to generate more accurate responses.

And then there are **LLM agents**, which are systems that can interact with APIs and other tools to perform actions. Agents transform simple LLMs into orchestrators. A recent standard for agent interaction with these backend tools is called Model Context Protocol, or MCP. MCP servers act as bridges between LLMs to access and interact with live systems, i.e. real-world tools, APIs, databases, and company data.

Testing out MCP servers sound fun, but I'll save that for another project. This bot will only need a RAG implementation.

## Tools

### Docling

https://docling-project.github.io/docling/concepts/

- Parses multiple document formats, e.g. PDF, DOCX, PPTX, XLSX, HTML, WAV, MP3, VTT, images (PNG, TIFF, JPEG, ...), and more into unified DoclingDocument
- Plug-and-play integrations with LangChain for agentic AI

### Ollama

https://docs.ollama.com
https://docs.ollama.com/capabilities/embeddings
https://github.com/ollama/ollama-python

Ollama is an open-source platform that allows users to run large language models locally. It is like what Docker is to images, Ollama is to LLMs. You can interact with Ollama via the CLI or API. Ollama starts up a server and provides a REST API wrapper that your application can use to access cloud or local LLMs. Ollama is a wrapper that allows the application to avoid configuring direct local runtimes. Although LangChain has the capability to run models locally, Ollama provides an optimized runtime that handles the hardware and ease of use.

### Chroma DB

https://ollama.com/blog/embedding-models
https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide

To implement RAG I have to find similarity between the user query and the external documents to retrieve the context I want used to generate the answer. How do we go about running similarity algorithms? We start by using embeddings. Ollama supports embedding models, which are models trained to generate vector embeddings, which are long array of numbers that represent semantic meaning for a given text. Resulting vector embedding arrays can be stored in a database. This is where vector stores like Chroma DB come in. They are optimized to index and search quickly for similar vectors of a given vector query.

### LangChain

https://docs.langchain.com/oss/python/langchain/overview
https://docs.langchain.com/oss/python/langchain/retrieval

LangChain is an open-source framework with a pre-built agent architecture and integrations used to build agents. LangChain also provides different methods to implement RAG. We are looking for a 2-Step RAG where retrieval always happens before generation. Typically used for documentation bots. High control, fast latency. Simple and predictable. There is also agentic RAG LLM, which uses tools to perform actions. The workflow loops through many iterations of reasoning where tools output responses that get fed back into the next "chain" in the LLM until the final output or iteration limit is reached.

The more I learn about LangChain, the more I see that it is just a wrapper best used for starting out and generic use cases. If you need more customized workflows, you can build out a custom framework yourself. It does provide splitting functionality though.

## Approaches

### Ollama + Chroma DB

1. Generate document embeddings with Ollama
2. Store embeddings into Chroma DB
3. Use Ollama to generate query vector embedding
4. Make vector query with Chroma DB
5. Retrieve relevant documents from Chroma DB query
6. Generate response with Ollama passing in query results into the prompt

### LangChain (2-Step RAG)

https://docs.langchain.com/oss/python/langchain/rag
https://docs.langchain.com/oss/python/langchain/agents#example-of-react-loop
https://docs.langchain.com/oss/python/langchain/knowledge-base

- Has suite of integrations
- Select a chat model, an embeddings model, and a vector store

1. Read data from different sources into LangChain's Document format using document loaders, e.g. Docling
2. Use text splitters to break large Documents into smaller chunks
3. Embed and store splits to be searched over using an embeddings model and vector store, e.g. Ollama and Chroma DB, respectively
4. Given user input, retrieve relevant splits from storage
5. Pass the query along with the retrieved data to the LLM to generate a response

## Updates

- Will not use LangChain. Ran into Docling error:
```
The plugin langchain_docling will not be loaded because Docling is being executed with allow_external_plugins=false.
```
- Would be fun to implement for learning, but seems like overkill

...

- Docling file format error on .c files
```
Input document main.c with format None does not match any allowed format: (dict_keys([<InputFormat.DOCX: 'docx'>, <InputFormat.PPTX: 'pptx'>, <InputFormat.HTML: 'html'>, <InputFormat.IMAGE: 'image'>, <InputFormat.PDF: 'pdf'>, <InputFormat.ASCIIDOC: 'asciidoc'>, <InputFormat.MD: 'md'>, <InputFormat.CSV: 'csv'>, <InputFormat.XLSX: 'xlsx'>, <InputFormat.XML_USPTO: 'xml_uspto'>, <InputFormat.XML_JATS: 'xml_jats'>, <InputFormat.METS_GBS: 'mets_gbs'>, <InputFormat.JSON_DOCLING: 'json_docling'>, <InputFormat.AUDIO: 'audio'>, <InputFormat.VTT: 'vtt'>]))
```

- Tokenization is the process of converting raw text into tokens, which can be words, part of words, or characters; if I don't explicitly define a tokenizer, docling will default to using character based
- In a RAG / retrieval context, for more controlled chunking it is important to make sure that the chunker and embedding model are using the same tokenizer
- https://docling-project.github.io/docling/examples/hybrid_chunking/#basic-usage
- Different embedding models use different tokenizers
- Ollama does not provide a way to get the tokenizer used with the embeddings model, so I pulled the tokenizer information from the same model in HuggingFace and used that for Docling chunking
- `transformers` is a HuggingFace Python library
```
Token indices sequence length is longer than the specified maximum sequence length for this model (520 > 512). Running this sequence through the model will result in indexing errors
```

- I need to fix the `handler.py` code and give it better prompts to narrow down the search space for each of the classes
- I need to organize and seed all my files into Chroma DB
- Organize the content to make it more digestible without too much data processing
- Goal is to have a more seamless conversation that is specific to the coursework provided to the LLM by RAG... Need to ideate
