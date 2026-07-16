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
            # IMPORTANT: response.text carries HF's actual explanation (e.g.
            # "model X is not supported for task feature-extraction",
            # "Invalid credentials", "model is loading", 404 route changes,
            # etc). Without this, every failure looks identical and
            # undiagnosable from the Streamlit error screen. Always surface
            # status code + body, never just the exception repr.
            body = ""
            try:
                body = response.text[:500]
            except Exception:
                pass
            raise RuntimeError(
                f"HuggingFace Inference API returned {response.status_code}: {e}\n"
                f"Response body: {body}\n"
                "Check that HUGGINGFACEHUB_API_TOKEN is valid, has inference "
                "permissions, and that this model is available on the "
                "serverless Inference API for the feature-extraction task."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"HuggingFace Inference API request failed: {e}")

        data = response.json()
        # feature-extraction can return either a flat vector per input, or
        # a nested [tokens][hidden_size] structure per input depending on
        # the model/pipeline version — guard against the shape being wrong
        # rather than returning garbage silently.
        if not isinstance(data, list) or not data:
            raise RuntimeError(
                f"Unexpected response shape from HuggingFace Inference API: {type(data)}. "
                f"Raw response: {str(data)[:500]}"
            )
        return data

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]