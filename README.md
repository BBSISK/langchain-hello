# langchain-hello

Three small LangChain examples, built to learn the framework the way people use it in jobs, and to compare it with my hand-written RAG assistant, [Ask Barry](https://github.com/BBSISK/ask-barry). They use the same Azure OpenAI models and the same sign-in (Microsoft Entra ID, no API keys) as Ask Barry.

| Step | File | What it shows |
|---|---|---|
| 1 | `step1_chain.py` | A chain, `prompt \| model \| parser`, and the three ways to run any Runnable: `invoke`, `stream`, `batch` |
| 2 | `step2_structured.py` | Structured output: the model fills a Pydantic class (a job ad split into must-have and nice-to-have requirements) |
| 3 | `step3_rag.py` | Retrieval-augmented generation over my own READMEs: split by heading, embed, retrieve the top 4, answer with numbered citations, then check the citations in code |

## Why LangChain, and where it stops

LangChain gives every piece (prompt, model, parser, retriever) the same interface, so they snap together with `|` and swap easily: change `ChatOpenAI` for another provider's class and the chain doesn't change. Splitting, embedding, a vector store, retrieval and schema-checked output are each one line in step 3; in Ask Barry I wrote each of them myself.

What it doesn't give you is judgement about your own data. The citation check in step 3 (`check_citations`) is my code, the same rule Ask Barry enforces: an answer is kept only if it cites a section that was actually retrieved, otherwise it becomes "I can't find that in these documents." No framework knows which claims need a source.

**LangChain or LangGraph?** A chain runs straight through, start to end. When the workflow needs loops, branches, parallel steps, a pause for a human or saved state, that's LangGraph (I rebuilt Ask Barry's job-ad agent in it, ASK-70).

## Run it

```bash
python3.12 -m venv ~/.venvs/langchain-hello
source ~/.venvs/langchain-hello/bin/activate
pip install -r requirements.txt
python -m pytest -q                 # 9 offline tests, no Azure needed
az login --tenant <your tenant id>
```

A `.env` file in this folder (never committed):

```
AZURE_OPENAI_ENDPOINT=https://<resource>.cognitiveservices.azure.com
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small
```

Your account needs the *Cognitive Services OpenAI User* role on the resource.

```bash
python step1_chain.py "doubly linked list"
python step1_chain.py --batch "stack" "queue" "binary search"
python step2_structured.py
python step3_rag.py "How is the playlist API deployed?"
python step3_rag.py "What colour are the dice?"        # not in the docs: should say it can't find it
python step3_rag.py --doc ../ask_barry/README.md "How does Ask Barry check citations?"   # a bigger file
```

Every step also runs with `--offline`: a fake model (step 1), the JSON schema the model would receive (step 2), or word-count vectors in place of the embedding model (step 3), so you can see the wiring without Azure.

## Limits

- `InMemoryVectorStore` keeps vectors in memory and re-embeds the files on every run. Embeddings go 16 sections per request with retries, because the Azure deployment has a per-minute limit: the first run against the whole Ask Barry README (about 80 sections in one request) hit a 429 rate-limit error. Fine for a few READMEs; a real system uses a search service (Ask Barry uses Azure AI Search with hybrid keyword and vector retrieval).
- **Vector search alone can miss the best section.** Asked how Ask Barry checks citations, step 3 retrieved the README's overview sections and answered about the evaluation's citation test, not the check every answer goes through. The answer was faithful to what it was given; the retrieval was the weak step. Ask Barry uses hybrid retrieval (keywords plus vectors) for this kind of miss; I haven't compared the two on this question.
- **A README is data too.** The first version of this README listed a skill-check question as the example of "should say it can't find it", and step 3 indexed this README, so it found its own example and answered from it. Ask Barry hit the same trap (its docs must never quote its test questions), so the examples now search only the other projects' READMEs, and the example question names no skill.
- These are learning examples with offline unit tests, not an evaluated system. Ask Barry has the golden set, the judges and the regression checks.
