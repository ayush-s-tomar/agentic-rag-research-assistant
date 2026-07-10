import requests
from langchain_core.embeddings import Embeddings


class HFInferenceEmbeddings(Embeddings):
    def __init__(self, api_token: str, model: str = "BAAI/bge-small-en-v1.5"):
        self.url = f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
        self.headers = {"Authorization": f"Bearer {api_token}"}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(self.url, headers=self.headers, json={"inputs": texts})
        response.raise_for_status()
        return response.json()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]