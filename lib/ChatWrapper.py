from numbers import Number
from openai import OpenAI
from .HistoryKeeper import Keeper
from .Types import MessageTypes, ChatMessage


DEFAULT_SYSTEM_PROMPT_LLAMA = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
"""


class ChatWrapper:
    def __init__(
        self,
        openAIClient: OpenAI,
        model: str,
        memoryTokenLength=2048,
        initalSytemPrompt=DEFAULT_SYSTEM_PROMPT_LLAMA,
    ) -> None:
        self.client = openAIClient
        self.model = model
        self.messageHistory = Keeper()
        self.memoryTokenLength = memoryTokenLength

        self.getHistory = lambda: self.messageHistory.returnNTokens(
            self.memoryTokenLength
        )
        self.messageHistory.addMessage({"role": "system", "content": initalSytemPrompt})

    def generateChat(self, prompt: str, withHistory=True, **kwargs):
        temperature: Number = kwargs.get("temperature", 0)

        messages = None
        if withHistory:
            self.messageHistory.addMessage(prompt)
            messages = self.getHistory()

        res = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.memoryTokenLength,
            temperature=temperature,
        )

        self.messageHistory.addMessage(
            {
                "role": MessageTypes.ASSISTANT.value,
                "content": res.choices[0].message.content,
            }
        )

        return res

    def getAssistantMessages(self, justMessage=True):
        ourContent = [
            item
            for item in self.messageHistory.messageLog
            if item.role == MessageTypes.ASSISTANT.value
        ]

        if justMessage:
            ourContent = [item.content for item in self.messageHistory.messageLog]

        else:
            ourContent = [item.original for item in self.messageHistory.messageLog]

        return ourContent
