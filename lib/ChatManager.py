from .ArchImplementations.LLAMA_Sentence import LLAMA_Sentence, LLAMA_MessageTokens

from openai import OpenAI
from dataclasses import dataclass, field


@dataclass
class ChatManager:
    client: OpenAI
    model: str
    historyTokenLimit: int = 9999
    messageHistory: list = field(default_factory=list)

    def addMessage(self, toAdd):
        self.messageHistory.append(toAdd)

    def getMessage(self, index=-1):
        return self.messageHistory[index]

    def getMessagesUntilTokenLimit(self):
        msgAccum = []
        _sum = 0
        for message in reversed(self.messageHistory):
            msgAccum.append(message)
            _sum += message.tokenLength

            if _sum >= self.historyTokenLimit:
                break

        return reversed(msgAccum)

    def run(
        self, message: LLAMA_Sentence, appendHistory: bool = False, **kwargs
    ) -> str:
        _max_tokens = kwargs.get("max_tokens", 999)
        kwargs.pop("max_tokens", None)

        _frequency_penalty = kwargs.get("frequency_penalty", 1.18)
        kwargs.pop("frequency_penalty", None)

        self.addMessage(message)

        # fmt: off
        _messages = self.getMessagesUntilTokenLimit() if appendHistory else [self.messageHistory[-1], ]
        # fmt: on

        res = self.client.chat.completions.create(
            messages=[
                {
                    "content": content.sentence,
                    "role": "system" if content.isSystem else "user",
                }
                for content in _messages
            ],
            model=self.model,
            frequency_penalty=_frequency_penalty,
            max_tokens=_max_tokens,
            **kwargs,
        )

        message.modelMessage = res.choices[0].message.content

        message.sentence += message.modelMessage
        message.sentence += LLAMA_MessageTokens.CHAT_END.value

        message.tokenLength += len(message.modelMessage)
        message.tokenLength += len(LLAMA_MessageTokens.CHAT_END.value)

        return message.modelMessage
