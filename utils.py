import requests
from dotenv import load_dotenv
import os


from langchain.llms.base import BaseLLM
from langchain.schema import Generation, LLMResult
from typing import List, Optional, Dict, Any
import requests

class mws_gpt_alpha(BaseLLM):
    api_key: str
    model: str = "mws-gpt-alpha"
    temperature: float = 0
    max_tokens: int = 50
    top_p: float = 1.0
    frequency_penalty: float = 0
    presence_penalty: float = 0


    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            response = requests.post(url=
                "https://api.gpt.mws.ru/v1/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "frequency_penalty": self.frequency_penalty,
                    "presence_penalty": self.presence_penalty,
                }
            )
            response.raise_for_status()
            result = response.json()
            text = result["choices"][0]["text"]
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "mws_gpt_alpha"


def db_init():
    POSTGRES_USER = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_DB = os.getenv("POSTGRES_DB")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT")

    engine = create_engine(
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
    )
    from models import Base 
    Base.metadata.create_all(bind=engine)
    SessionFactory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    return SessionFactory


# def get_embedding(text):
#     url = "https://api.gpt.mws.ru/v1/embeddings"
#     load_dotenv()
#     # Ваш API‑ключ
#     api_key = os.getenv("MWS_API_KEY")
#
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json"
#     }
#
#     payload = {
#         "model": "bge-m3",
#         "input": text
#     }
#
#     try:
#         response = requests.post(url, headers=headers, json=payload)
#         response.raise_for_status()
#         data = response.json()
#         return data["data"][0]["embedding"]
#
#     except requests.exceptions.HTTPError as errh:
#         print(f"HTTP Error: {errh} — {response.text}")
#     except requests.exceptions.RequestException as err:
#         print(f"Request Error: {err}")
