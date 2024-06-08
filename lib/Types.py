from enum import Enum
from dataclasses import dataclass


class MessageTypes(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    original: str | dict
    role: MessageTypes = None
    content: str = None
    lengthInTokens: int = 0

    def __post_init__(self):
        if type(self.original) == dict:
            self.role = self.original["role"]
            self.content = self.original["content"]
            self.lengthInTokens = len(self.content.split(" "))
            return

        # If it's a string its a user prompt.
        # if it's not a user prompt ur using it wrong!
        self.role = "user"
        self.content = self.original
        self.original = {"role": self.role, "content": self.content}
