"""
Step 1: a chain. prompt | model | parser

The `|` joins three Runnables into one. Every Runnable has the same methods, so the whole chain does too:
  .invoke(input)     one input, one output
  .stream(input)     the output piece by piece, as the model writes it
  .batch([inputs])   several inputs, sent in parallel

Here: a revision helper that explains a CS210 idea in three bullet points.

Run:  python step1_chain.py "doubly linked list"
      python step1_chain.py --stream "Big O of a nested loop"
      python step1_chain.py --batch "stack" "queue" "binary search"
      python step1_chain.py --offline "stack"          (no Azure: a fake model shows the wiring)
"""
import argparse

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common import azure_chat, online

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You help a student revise for a data structures and algorithms exam in Java. "
               "Explain the idea in exactly three short bullet points, then give its Big O for the "
               "main operations on one line. No code."),
    ("human", "{concept}"),
])


def build_chain(model):
    """prompt -> model -> plain string. The model is passed in so tests can use a fake one."""
    return PROMPT | model | StrOutputParser()


def fake_model():
    from langchain_core.language_models import FakeListChatModel
    return FakeListChatModel(responses=["- (offline) bullet one\n- bullet two\n- bullet three\nBig O: n/a"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("concepts", nargs="+")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    use_azure = online() and not args.offline
    chain = build_chain(azure_chat() if use_azure else fake_model())
    if not use_azure:
        print("(offline: fake model)\n")

    # What the model actually receives: the template filled in.
    print(PROMPT.invoke({"concept": args.concepts[0]}).to_string(), "\n---")

    if args.batch:
        answers = chain.batch([{"concept": c} for c in args.concepts], config={"max_concurrency": 3})
        for concept, answer in zip(args.concepts, answers):
            print(f"\n## {concept}\n{answer}")
    elif args.stream:
        for piece in chain.stream({"concept": args.concepts[0]}):
            print(piece, end="", flush=True)
        print()
    else:
        print(chain.invoke({"concept": args.concepts[0]}))


if __name__ == "__main__":
    main()
