import requests
from langchain_core.embeddings import Embeddings


class HFInferenceEmbeddings(Embeddings):
    def __init__(self, api_token: str, model: str = "BAAI/bge-small-en-v1.5", timeout: int = 30):
        self.url = f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
        self.headers = {"Authorization": f"Bearer {api_token}"}
        # HF's serverless Inference API cold-starts unloaded models, which can
        # take well over a minute with no response — without an explicit
        # timeout, requests waits indefinitely, which is indistinguishable
        # from a frozen app to the end user. 30s is enough for a cold start
        # in practice; failing loud after that beats hanging forever.
        self.timeout = timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            response = requests.post(
                self.url, headers=self.headers, json={"inputs": texts}, timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"HuggingFace Inference API did not respond within {self.timeout}s. "
                "The model may be cold-starting — try again in a minute."
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"HuggingFace Inference API returned an error: {e}. "
                "Check that HUGGINGFACEHUB_API_TOKEN is valid and has inference permissions."
            )
        return response.json()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]