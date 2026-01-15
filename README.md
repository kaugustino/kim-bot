# kim-bot

## Goal

I want to create an LLM that can capture and reference everything I've learned in my academic life. *What did I learn in XYZ class?* *What was the main takeaway of that assignment?*

## Initial Research

First idea is to train on top of an existing LLM using my files. This is called **LLM fine-tuning**. However this approach needs a labeled data set to train on. I'm not sure exactly what the labeled data would be for my input files. This results in teaching the LLM to excel at a particular task in a specific style format or tone. It can also inject domain expertise but is not sufficient to reliable introduce new facts.

Second idea is to use another AI technique called **Retrieval-Augmented Generation**, or RAG. RAG gives LLMs access to an external knowledge base in order to provide more context before generating a response. This eliminates the need to retrain the model and reduces hallucinations. It also provides a way to retrieve real-time accurate data instead of data baked into the model during training.

RAG is implemented using embedding models, a type of neural network that converts complex data (text, images, audio) into dense numerical vectors called embeddings. Embeddings capture semantic meaning of relationships and context in high-dimensional space. First, I create embeddings of my project file directories. Second, a user's query gets compressed into a vector and is then compared against the embeddings space of the given context. Finally, I can retrieve the documents that have the most similarity to the query embedding. The retrieved context will augment the model to generate more accurate responses.

And then there are **LLM agents**, which are systems that can interact with APIs and other tools to perform actions. Agents transform simple LLMs into orchestrators. A recent standard for agent interaction with these backend tools is called Model Context Protocol, or MCP. MCP servers act as bridges between LLMs to access and interact with live systems, i.e. real-world tools, APIs, databases, and company data.

Testing out MCP servers sound fun, but I'll save that for another project. This bot will only need a RAG implementation.