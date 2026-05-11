import ollama

from kim_bot.config import EMBEDDINGS_MODEL
from kim_bot.loader import get_collection


def is_rag_required(content: str) -> bool:
    # use structure output to determine whether rag needs to be performed
    return False


def create_rag_query(history: list, query: str) -> str:
    prompt = "Using conversation history, make this query as specific as possible to perform a query into a knowledge database."

    # Return response
    return prompt


def extra_knowledge(query: str) -> str:
    collection = get_collection()
    response = ollama.embed(model=EMBEDDINGS_MODEL, input=query)
    results = collection.query(query_embeddings=response["embeddings"], n_results=3)

    data = ""
    for res in results["documents"][0]:
        data += res + "\n"

    prompt = f'Here is the relevant data: {data}\nUse the data to answer the following: "{query}". If there is no data, say you cannot comment without relevant data. Use the relevant data only when it is relevant.'

    return prompt
