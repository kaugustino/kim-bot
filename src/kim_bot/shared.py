from enum import StrEnum


class OllamaRole(StrEnum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"
