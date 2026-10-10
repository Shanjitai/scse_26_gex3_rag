"""Interactive RAG agent for the university IT policy exercise."""

from rag_tool import RagTool


def main():
    print("University IT Policy RAG Agent")
    print("Using Qwen3-0.6B. Type 'quit' or 'exit' to stop.")

    tool = RagTool()

    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            break

        answer = tool.run(question)
        print("\nAnswer:", answer)


if __name__ == "__main__":
    main()