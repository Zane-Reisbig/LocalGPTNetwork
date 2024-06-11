from enum import Enum
from dataclasses import dataclass
from openai import OpenAI
from typing import Callable

DEFAULT_SYSTEM_PROMPT_LLAMA = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""


class GPT_MessageTokens(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LLAMA_MessageTokens(Enum):
    SYSTEM = "<<SYS>>"
    SYSTEM_CLOSE = "<</SYS>>"

    INSTANCE_START = "[INST]"
    INSTANCE_END = "[/INST]"

    CHAT_START = "<s>"
    CHAT_END = "</s>"
