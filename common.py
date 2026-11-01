"""Shared set-up for the three steps: settings, and the two Azure OpenAI models.

Sign-in is the same as Ask Barry and langgraph-hello: Microsoft Entra ID via your `az login`, no API key.
Settings come from environment variables or a .env file in this folder:
  AZURE_OPENAI_ENDPOINT          e.g. https://barrybsisk-6154-resource.cognitiveservices.azure.com
  AZURE_OPENAI_CHAT_DEPLOYMENT   default gpt-4.1-mini
  AZURE_OPENAI_EMBED_DEPLOYMENT  default text-embedding-3-small (step 3 only)

LangChain's ChatOpenAI talks to Azure's v1 endpoint like any OpenAI-compatible server. `api_key` accepts a
function, so we pass Entra ID's token provider and it fetches a fresh token whenever one is needed.
"""
import hashlib
import math
import os
import re

from langchain_core.embeddings import Embeddings

COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"


def load_dotenv_if_present(path=".env"):
    """Tiny .env reader (KEY=value lines), so no extra package is needed."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"'))


def v1_base_url(endpoint):
    """Accept either form of the endpoint and return the v1 base URL."""
    endpoint = endpoint.strip().rstrip("/")
    if not endpoint.endswith("/openai/v1"):
        endpoint += "/openai/v1"
    return endpoint + "/"


def online():
    """True when an endpoint is configured (otherwise the steps run offline)."""
    load_dotenv_if_present()
    return bool(os.environ.get("AZURE_OPENAI_ENDPOINT"))


def _token_provider():
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    return get_bearer_token_provider(DefaultAzureCredential(), COGNITIVE_SCOPE)


def azure_chat(temperature=0.2):
    from langchain_openai import ChatOpenAI
    load_dotenv_if_present()
    return ChatOpenAI(
        model=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4.1-mini"),   # on Azure, the deployment name
        base_url=v1_base_url(os.environ["AZURE_OPENAI_ENDPOINT"]),
        api_key=_token_provider(),
        temperature=temperature,
        timeout=30,
        max_retries=2,
    )


def azure_embeddings():
    from langchain_openai import OpenAIEmbeddings
    load_dotenv_if_present()
    return OpenAIEmbeddings(
        model=os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT", "text-embedding-3-small"),
        base_url=v1_base_url(os.environ["AZURE_OPENAI_ENDPOINT"]),
        api_key=_token_provider(),
        check_embedding_ctx_length=False,   # skip the tiktoken pre-check: Azure deployment names aren't model names
        chunk_size=16,      # texts per request: small batches stay under the deployment's per-minute token limit
        max_retries=6,      # a 429 waits for the time Azure asks for (retry-after), then tries again
    )


class KeywordEmbeddings(Embeddings):
    """Offline stand-in for the embedding model: hashed word counts, normalised.

    Texts that share words come out close together, so --offline retrieval still makes some sense.
    It knows nothing about meaning ("deploy" and "release" are unrelated to it), which is exactly
    what a real embedding model adds.
    """

    def __init__(self, size=512):
        self.size = size

    def _vector(self, text):
        v = [0.0] * self.size
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.size] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)
