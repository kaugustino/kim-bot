import ollama

from config import EMBEDDINGS_MODEL
from loader import get_collection


def generate_response(user_input: str) -> None:
    collection = get_collection()

    response = ollama.embed(model=EMBEDDINGS_MODEL, input=user_input)
    results = collection.query(query_embeddings=response["embeddings"], n_results=3)

    primary_data = results["documents"][0][0]
    additional_data = results["documents"][0][1]
    additional_data2 = results["documents"][0][2]

    output = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": f"This is course material provided from a previous student: {primary_data}. Here is additional info: {additional_data} and {additional_data2}. Use this material to answer the following question: {user_input}",
            }
        ],
        stream=True,
    )

    for chunk in output:
        print(chunk["message"]["content"], end="", flush=True)
