from typing import Iterator
import ollama

from kim_bot.config import MODEL
from kim_bot.rag import create_query
from kim_bot.shared import OllamaRole


class Conversation:
    def __init__(self):
        self.model: str = MODEL
        self.system_prompt = "You are a concise informational assistant that knows about all the previous classwork taken by Kim at Georgia Tech. You were her study buddy for these classes: Introduction to Information Security in Spring 2021, Graduate Introduction to Operating Systems in Summer 2021, Machine Learning for Trading in Spring 2022, Software Development Process in Fall 2022, Robotics AI Techniques in Spring 2023, Introduction to High Performance Computing in Summer 2023, Information Security Lab Binary Exploitation in Spring 2024 (her favorite class), Computer Networks in Summer 2024, Compilers in Spring 2025, Advanced Operating Systems in Spring 2025, and Introduction to Graduate Algorithms in Fall 2025. Use emojis when applicable."
        self.conversation_history: list[str] = []

        self.add_history(OllamaRole.system, content=self.system_prompt)

    def add_history(self, role: OllamaRole, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})

    def introduce_yourself(self) -> None:
        self.add_history(
            OllamaRole.user,
            content="Introduce yourself in 15 words or less. Keep it brief. Use emojis.",
        )

        self.chat_response(save=False, stream=True)

        self.conversation_history.pop()

    def capture_stream(self, output: Iterator[ollama.ChatResponse]) -> str:
        response = ""
        for chunk in output:
            print(chunk["message"]["content"], end="", flush=True)
            response += chunk["message"]["content"]
        return response

    def chat_response(self, stream: bool = False, save: bool = True) -> None:
        # When stream = True, returns a generator
        output = ollama.chat(
            model=self.model,
            messages=self.conversation_history,
            stream=stream,
        )

        if stream:
            response = self.capture_stream(output=output)
        else:
            response = output["message"]["content"]

        # No need to save extraneous assistant responses, only relevant context
        if save:
            self.add_history(role=OllamaRole.assistant, content=response)

    def generate_response(self, role: OllamaRole, user_input: str) -> None:
        query = create_query(history=self.conversation_history, user_input=user_input)

        self.add_history(role=str(role), content=query)
        self.chat_response(stream=True)
