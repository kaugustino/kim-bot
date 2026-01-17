from ollama import chat


def main():
    # Ollama will start model if it's not already running
    stream = chat(
        model="mistral",
        messages=[{"role": "user", "content": "Why is the sky blue? Answer in 10 words or less."}],
        stream=True,
    )

    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)

if __name__ == "__main__":
    main()