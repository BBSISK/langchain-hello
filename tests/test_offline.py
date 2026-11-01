"""Offline tests: fake models and fake embeddings, no Azure. Run: python -m pytest -q"""
import sys
from pathlib import Path

from langchain_core.language_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import step1_chain, step2_structured, step3_rag          # noqa: E402
from common import KeywordEmbeddings, v1_base_url                           # noqa: E402


class FakeStructured:
    """Stands in for a chat model: with_structured_output returns a Runnable that always gives `reply`,
    and records what the prompt sent."""

    def __init__(self, reply):
        self.reply, self.seen = reply, []

    def with_structured_output(self, schema):
        def respond(prompt_value):
            self.seen.append(prompt_value.to_string())
            return self.reply
        return RunnableLambda(respond)


def test_endpoint_either_form_becomes_v1():
    assert v1_base_url("https://x.cognitiveservices.azure.com") == "https://x.cognitiveservices.azure.com/openai/v1/"
    assert v1_base_url("https://x.openai.azure.com/openai/v1/") == "https://x.openai.azure.com/openai/v1/"


def test_step1_chain_invoke_batch_and_stream():
    chain = step1_chain.build_chain(FakeListChatModel(responses=["a", "b"]))
    assert chain.invoke({"concept": "stack"}) == "a"
    assert sorted(chain.batch([{"concept": "x"}, {"concept": "y"}])) == ["a", "b"]
    streamed = list(step1_chain.build_chain(FakeListChatModel(responses=["hello"])).stream({"concept": "q"}))
    assert "".join(streamed) == "hello" and len(streamed) > 1          # one piece per character


def test_step2_schema_has_must_and_nice():
    schema = step2_structured.JobAd.model_json_schema()
    assert schema["$defs"]["Requirement"]["properties"]["priority"]["enum"] == ["must", "nice"]


def test_step2_chain_returns_the_object():
    ad = step2_structured.JobAd(title="Dev", requirements=[{"skill": "Java", "priority": "must"}])
    model = FakeStructured(ad)
    out = step2_structured.build_chain(model).invoke({"ad": "Dev. Java required."})
    assert out.requirements[0].skill == "Java" and "Java required" in model.seen[0]


def write(tmp_path, name, text):
    p = tmp_path / name / "README.md"
    p.parent.mkdir()
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_step3_sections_keep_file_and_heading(tmp_path):
    path = write(tmp_path, "dice", "# Dice API\n\n## Architecture\n\nLambda behind API Gateway.\n")
    sections = step3_rag.load_sections([path, str(tmp_path / "missing.md")])
    assert sections[0].metadata == {"source": "dice/README.md", "heading": "Dice API > Architecture"}


def rag(tmp_path, reply):
    path = write(tmp_path, "dice", "# Dice API\n\n## Architecture\n\nLambda behind API Gateway.\n\n## Cost\n\nFree tier.\n")
    store = InMemoryVectorStore.from_documents(step3_rag.load_sections([path]), KeywordEmbeddings())
    model = FakeStructured(reply)
    return step3_rag.build_chain(store.as_retriever(search_kwargs={"k": 2}), model), model


def test_step3_valid_citation_is_kept(tmp_path):
    chain, model = rag(tmp_path, step3_rag.Answer(supported=True, answer="It runs on Lambda.", citations=[1, 1]))
    out = chain.invoke("Where does it run?")
    assert out["answer"] == "It runs on Lambda." and len(out["sources"]) == 1
    assert "[1] dice/README.md" in model.seen[0] and "Question: Where does it run?" in model.seen[0]


def test_step3_citation_outside_the_retrieved_sections_means_no_evidence(tmp_path):
    chain, _ = rag(tmp_path, step3_rag.Answer(supported=True, answer="Made up.", citations=[7]))
    assert chain.invoke("q") == {"answer": step3_rag.NO_EVIDENCE, "sources": []}


def test_step3_unsupported_means_no_evidence(tmp_path):
    chain, _ = rag(tmp_path, step3_rag.Answer(supported=False, answer="", citations=[]))
    assert chain.invoke("q")["answer"] == step3_rag.NO_EVIDENCE


def test_keyword_embeddings_rank_shared_words_first():
    e = KeywordEmbeddings()
    q, near, far = e.embed_query("lambda deploy"), e.embed_query("deploy to lambda"), e.embed_query("free tier cost")
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert dot(q, near) > dot(q, far)
