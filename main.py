import os

from openai import OpenAI

from lib.ChatManager import ChatManager
from lib.ArchImplementations.LLAMA_Sentence import LLAMA_Sentence


def main():
    models = os.getenv(r"LOCALAPPDATA")
    models = list(os.listdir(models + r"\nomic.ai\GPT4All"))
    models = [item for item in models if item.endswith(".gguf")]

    localURL = "http://localhost:9090/v1"
    creds = OpenAI(api_key="None", base_url=localURL)

    summarizeText = None
    path = os.getcwd() + r"\SavingPrivateRyan.txt"
    with open(path) as file:
        summarizeText = file.read()

    one = LLAMA_Sentence.start()
    one.setPromptContent(
        content=summarizeText + "\n---\n" + "Summarize that text", addNoise=True
    )
    one.end()

    two = LLAMA_Sentence.start()
    two.setPromptContent("Do you know what movie that's from?", addNoise=True)
    two.end()

    chatMan = ChatManager(creds, models[0])
    chatMan.run(one)
    chatMan.run(two, True)


main()
