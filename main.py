from lib.ChatWrapper import ChatWrapper
from random import randint
from openai import OpenAI

localURL = "http://localhost:9090/v1"
model = "llama-2-7b-chat.Q4_0.gguf"

client = OpenAI(
    api_key="None",
    base_url=localURL,
)

clientWrapper = ChatWrapper(client, model)


def main():
    # Llama 2 hates using its memory, after i even so carefully
    # gave it a message history
    clientWrapper.generateChat("Hello!")
    clientWrapper.generateChat("Say the word 'howdy', thats it. only the word 'howdy'")
    clientWrapper.generateChat("What was the last thing you said?, verbatim")

    [print(item) for item in clientWrapper.getAssistantMessages()]


main()
