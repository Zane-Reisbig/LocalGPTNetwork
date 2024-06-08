from .Types import ChatMessage


class Keeper:
    def __init__(self) -> None:
        self.messageLog: ChatMessage = []

    def addMessage(self, message: ChatMessage | str):
        if type(message) != ChatMessage:
            message = ChatMessage(message)

        self.messageLog.append(message)

    def serialize(self):
        return [item.original for item in self.messageLog]

    def returnNTokens(self, tokenAmount=2048):
        total = 0
        messages = []

        message: ChatMessage
        for message in self.messageLog:
            total += message.lengthInTokens

            if total > tokenAmount:
                break

            messages.append(message.original)

        return messages
