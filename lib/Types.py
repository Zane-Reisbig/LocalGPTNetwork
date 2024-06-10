from enum import Enum
from dataclasses import dataclass
from typing import Callable


class GPT_MessageTokens(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LLAMA_MessageTokens(Enum):
    SYSTEM = "<<SYS>>"
    SYSTEM_CLOSE = "<</SYS>>"

    BACK_AND_FORTH_START = "[INST]"
    BACK_AND_FORTH_END = "[/INST]"

    CHAT_START = "<s>"
    CHAT_END = "</s>"


class ArchType(Enum):
    GPT = GPT_MessageTokens
    LLAMA = LLAMA_MessageTokens


class ChatMessage:
    def __init__(
        self,
        message: str,
        _format: ArchType,
        isStart=False,
        isEnd=False,
        isSystem: bool = False,
    ) -> None:
        self.isStart = isStart
        self.isEnd = isEnd
        self.archFormat = _format
        self.message = message
        self.formatted = self.__formatIt()

    def __formatIt(self):
        stringBuilder = ""
