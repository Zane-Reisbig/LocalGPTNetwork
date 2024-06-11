from openai import OpenAI
from dataclasses import dataclass
from uuid import uuid1, UUID

if __name__ == "__main__":
    from lib.Types.Types import LLAMA_MessageTokens
else:
    from ..Types.Types import LLAMA_MessageTokens


@dataclass
class LLAMA_Sentence:
    sentence: str = ""
    userMessage: str = None
    modelMessage: str = None

    isStarted = False
    isSystem = False

    lastCalled: UUID = None

    __startTrace = uuid1()
    __setIsSystemTrace = uuid1()
    __setIsUserTrace = uuid1()
    __setPromptContentTrace = uuid1()
    __endTrace = uuid1()
    __runTrace = uuid1()

    @staticmethod
    def start(isSystem: bool = False):
        inst = LLAMA_Sentence()

        assert inst.isStarted is False, "Don't call the start function more than once."

        inst.sentence += LLAMA_MessageTokens.CHAT_START.value
        inst.sentence += LLAMA_MessageTokens.INSTANCE_START.value

        inst.isStarted = True
        inst.lastCalled = inst.__startTrace

        if isSystem:
            inst.setIsSystem()
        else:
            inst.setIsUser()

        return inst

    def setIsSystem(self):
        assert (
            self.lastCalled == self.__startTrace
        ), "The 'System Tag' must be used only after the 'start()' function or use the kwarg 'isSystem=True' in the 'start()' function"

        self.sentence += LLAMA_MessageTokens.SYSTEM.value
        self.isSystem = True

        self.lastCalled = self.__setIsSystemTrace

    def setIsUser(self):
        assert (
            self.lastCalled == self.__startTrace
        ), "The 'System Tag' must be used only after the 'start()' function or use the kwarg 'isSystem=True' in the 'start()' function"
        # This doesn't actually do anything. It's only here to keep the style consistent

        self.lastCalled = self.__setIsUserTrace

    def setPromptContent(self, content: str, addNoise: bool = False):
        assert self.lastCalled in [
            self.__startTrace,
            self.__setIsSystemTrace,
            self.__setIsUserTrace,
        ], "The 'Prompt Content' may only be set after the 'start()' or 'setIsSystem()' functions"

        if addNoise:
            self.sentence += "[MSGID]" + str(uuid1()) + "[/MSGID]"

        self.sentence += content
        self.userMessage = content

        self.lastCalled = self.__setPromptContentTrace

    def end(self) -> str | None:
        assert (
            self.lastCalled == self.__setPromptContentTrace and self.isStarted
        ), "Can't end the prompt with no content. You also can't just end the prompt."

        if self.isSystem:
            self.sentence += LLAMA_MessageTokens.SYSTEM_CLOSE.value

        self.sentence += LLAMA_MessageTokens.INSTANCE_END.value
        # self.sentence += "${MODEL_ANSWER}"
        # self.sentence += LLAMA_MessageTokens.CHAT_END

        self.lastCalled = self.__endTrace


if __name__ == "__main__":
    import os

    models = os.getenv(r"LOCALAPPDATA")
    models = list(os.listdir(models + r"\nomic.ai\GPT4All"))
    models = [item for item in models if item.endswith(".gguf")]

    localURL = "http://localhost:9090/v1"
    creds = OpenAI(api_key="None", base_url=localURL)

    summarizeText = None
    path = os.getcwd() + r"\SavingPrivateRyan.txt"
    with open(path) as file:
        summarizeText = file.read()

    x: LLAMA_Sentence = LLAMA_Sentence.start()
    x.setPromptContent(
        content=summarizeText + "\n---\n" + "Summarize that text", addNoise=True
    )
    x.end()
    x.run(model=models[0], client=creds, temperature=0, max_tokens=999)

    print(x.modelMessage)
