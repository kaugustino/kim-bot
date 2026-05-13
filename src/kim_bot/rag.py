import json
import ollama

from kim_bot.config import config
from kim_bot.loader import get_collection
from kim_bot.shared import OllamaRole


def create_query(history: list, user_input: str) -> str:
    query = generate_better_query(context=history, user_input=user_input)

    if is_rag_required(context=history, query=query):
        query = extra_knowledge(query=query)

    return query


def generate_better_query(context: list, user_input: str) -> str:
    rewrite_assistant = {
        "role": OllamaRole.system,
        "content": """You are a query rewriting assistant.

        Your task is to convert the user's latest message into a standalone search query that can be used for retrieval from an external knowledge base.

        Use the conversation history for context resolution.

        Rules:
        - Preserve the user's actual intent
        - Replace ambiguous references like "it", "that", "they", or "those"
        - Include important entities and technical terms
        - Do NOT answer the question
        - Do NOT add explanations
        - Keep the rewritten query concise but specific
        - If the latest message is already standalone, return it unchanged""",
    }

    task_prompt = {
        "role": OllamaRole.user,
        "content": f"Conversation History: {context}\nLatest User Message: {user_input}\nStandalone Search Query: ",
    }

    output = ollama.chat(
        model=config["MODEL"],
        messages=[rewrite_assistant, task_prompt],
        stream=False,
    )

    return output["message"]["content"]


def is_rag_required(context: list, query: str) -> bool:
    # Use structure output to determine whether rag needs to be performed
    rag_assistant = {
        "role": OllamaRole.system,
        "content": "You are a library assistant. You determine whether a query needs additional information from an external database to be answered. If you want more information to answer the question, respond YES. If you already know the answer, respond NO.",
    }

    task_prompt = {
        "role": OllamaRole.user,
        "content": f'Do you need to reference an external database to retrieve more information for this question: "{query}"?',
    }

    output = ollama.chat(
        model=config["MODEL"],
        messages=[rag_assistant, task_prompt],
        stream=False,
        # JSON is based on pydantic and model_json_schema()
        format={
            "properties": {
                "answer": {"enum": ["YES", "NO"], "title": "Answer", "type": "string"}
            },
            "required": ["answer"],
            "title": "YesNoResponse",
            "type": "object",
        },
    )

    answer_dict = json.loads(output["message"]["content"])

    return True if answer_dict["answer"] == "YES" else False


def extra_knowledge(query: str) -> str:
    collection = get_collection()
    response = ollama.embed(model=config["EMBEDDINGS_MODEL"], input=query)
    results = collection.query(query_embeddings=response["embeddings"], n_results=1)

    data = ""
    for res in results["documents"][0]:
        data += res + "\n"

    prompt = f'Here is the relevant data: {data}\nUse the data to answer the following: "{query}". If there is no data, say you cannot comment without relevant data. Use the relevant data only when it is relevant.'

    return prompt
