import ollama
from ollama import chat
from loader import collection
from loader import load_external_knowledge_dir


def main():
    load_external_knowledge_dir()

    user_input = "What were the course topics in the class syllabus for Information Security Lab: Reverse Engineering and Exploitation Labs?"

    response = ollama.embed(model="mxbai-embed-large", input=user_input)
    results = collection.query(query_embeddings=response["embeddings"], n_results=1)

    data = results["documents"][0][0]

    output = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": f"Using this data: {data}. Respond to this prompt: {user_input}",
            }
        ],
        stream=True,
    )

    for chunk in output:
        print(chunk["message"]["content"], end="", flush=True)

    # # Ollama will start model if it's not already running
    # stream = chat(
    #     model="mistral",
    #     messages=[{"role": "user", "content": "Why is the sky blue? Answer in 5 words or less."}],
    #     stream=True,
    # )

    # for chunk in stream:
    #     print(chunk["message"]["content"], end="", flush=True)


if __name__ == "__main__":
    main()
