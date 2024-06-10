from .Types import ChatMessage


class Keeper:
    def __init__(self) -> None:
        self.chatHistory = []

    def addChat(self, message: ChatMessage):
        # TODO:
        # Do this more
        self.chatHistory.append(message)

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
