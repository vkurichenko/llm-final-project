from langchain_openai import ChatOpenAI as GPT
from langchain_openai import OpenAIEmbeddings as OpenAIEmbeds
from openai import AsyncOpenAI, OpenAI


class NDTOpenAI(OpenAI):
    server_url: str = "https://api.neuraldeep.tech/"

    def __init__(self, api_key, **kwargs):
        super().__init__(api_key=api_key, base_url=self.server_url, **kwargs)


class AsyncNDTOpenAI(AsyncOpenAI):
    server_url: str = "https://api.neuraldeep.tech/"

    def __init__(self, api_key, **kwargs):
        super().__init__(api_key=api_key, base_url=self.server_url, **kwargs)


class ChatOpenAI(GPT):
    """
    Класс ChatOpenAI по аналогии с одноименным классом из библиотеки langchain
    """

    openai_api_key: str = "api_key"

    def __init__(self, course_api_key, **kwargs):
        super().__init__(
            client=NDTOpenAI(api_key=course_api_key).chat.completions,
            async_client=AsyncNDTOpenAI(api_key=course_api_key).chat.completions,
            **kwargs,
        )


class OpenAIEmbeddings(OpenAIEmbeds):
    """
    Класс OpenAIEmbeddings по аналогии с одноименным классом из библиотеки langchain
    """

    openai_api_key: str = "api_key"

    def __init__(self, course_api_key, **kwargs):
        super().__init__(
            client=NDTOpenAI(api_key=course_api_key).embeddings,
            async_client=AsyncNDTOpenAI(api_key=course_api_key).embeddings,
            **kwargs,
        )
