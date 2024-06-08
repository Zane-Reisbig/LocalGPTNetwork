from openai import OpenAI
from .HistoryKeeper import Keeper
from .Types import MessageTypes


class ChatWrapper:
    def __init__(
        self, openAIClient: OpenAI, model: str, memoryTokenLength=2048
    ) -> None:
        self.client = openAIClient
        self.model = model
        self.messageHistory = Keeper()
        self.memoryTokenLength = memoryTokenLength

        self.getHistory = lambda: self.messageHistory.returnNTokens(
            self.memoryTokenLength
        )

    def generateChat(self, prompt: str, withHistory=True, **kwargs):
        messages = None
        if withHistory:
            self.messageHistory.addMessage(prompt)
            messages = self.getHistory()

        res = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=self.memoryTokenLength
        )

        self.messageHistory.addMessage(
            {
                "role": MessageTypes.ASSISTANT.value,
                "content": res.choices[0].message.content,
            }
        )

        return res
