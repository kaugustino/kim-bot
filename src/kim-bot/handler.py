from typing import Iterator
import ollama

from config import EMBEDDINGS_MODEL
from loader import get_collection


def capture_stream(output: Iterator[ollama.ChatResponse]) -> str:
    response = ""
    for chunk in output:
        print(chunk["message"]["content"], end="", flush=True)
        response += chunk["message"]["content"]
    return response


def chat_response(input: str, stream: bool = False) -> str:
    output = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": input,
            }
        ],
        stream=stream,
    )

    if stream:
        return capture_stream(output=output)
    return output["message"]["content"]


def introduce_yourself() -> None:
    prompt = "You are an informational assistant that knows about the classwork taken by Kim at Georgia Tech. You respond like a friend and study buddy for everyone. Introduce yourself in 10 words or less! Use emojis."

    chat_response(input=prompt, stream=True)


def is_retrieval_required(user_input: str) -> bool:
    """Chain together prompts to get better results."""

    prompt = f'Is "{user_input}" a question or instruction? Only respond using the words "yes" or "no" in lowercase. Do not use any other words in your output. Do not use punctuation.'

    output1 = chat_response(input=prompt)
    # print(f"output1: {output1}")

    if "yes" in output1:
        prompt2 = f"{user_input} and specifically mention the class name if one was mentioned. Answer in 10 words or less. Include the class name."

        output2 = chat_response(input=prompt2)
        # print(f"output2: {output2}")

        prompt3 = f'Is "{output2}" related at all to any of the following classes: Introduction to Information Security in Spring 2021, Graduate Introduction to Operating Systems in Summer 2021, Machine Learning for Trading in Spring 2022, Software Development Process in Fall 2022, Robotics AI Techniques in Spring 2023, Introduction to High Performance Computing in Summer 2023, Information Security Lab Binary Exploitation in Spring 2024 (her favorite class), Computer Networks in Summer 2024, Compilers in Spring 2025, Advanced Operating Systems in Spring 2025, and Introduction to Graduate Algorithms in Fall 2025? Only respond with either "yes" or "no" in lowercase. Do NOT use more than one word. Do not use punctuation.'

        output3 = chat_response(input=prompt3)
        # print(f"output3: {output3}")

        if "yes" in output3:
            return True
    return False


def generate_response(user_input: str) -> None:
    if is_retrieval_required(user_input=user_input):
        collection = get_collection()
        response = ollama.embed(model=EMBEDDINGS_MODEL, input=user_input)
        results = collection.query(query_embeddings=response["embeddings"], n_results=3)

        data = ""
        for res in results["documents"][0]:
            data += res + "\n"

        prompt = f'Here is the relevant data: {data}\nUse the data to answer the following: "{user_input}". If there is no data, say you cannot comment without relevant data. Use the relevant data only when it is relevant. Answer in 50 words or less. Use emojis.'

        chat_response(input=prompt, stream=True)
    else:
        prompt = f"{user_input} and answer in 20 words or less and as friendly as possible. Use emojis."

        chat_response(input=prompt, stream=True)
