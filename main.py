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
    print(clientWrapper.generateChat("Hello!"))
    print(clientWrapper.generateChat("Say the word 'howdy'"))
    print(clientWrapper.generateChat("What was the last thing you said?"))


main()
