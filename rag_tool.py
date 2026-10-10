"""Tool wrapper around the RAG pipeline."""

from rag import RAGPipeline


class RagTool:
    name = "university_it_policy_rag"
    description = (
        "Answers questions about university IT policies using the local policy "
        "knowledge base. Use this for IT Service Desk, accounts, VPN, Wi-Fi, "
        "printing, storage, and email questions. It will not invent phone numbers."
    )

    def __init__(self, model_path=None):
        self.rag = RAGPipeline(model_path=model_path)

    def run(self, question: str) -> str:
        return self.rag.answer(question)

    def __call__(self, question: str) -> str:
        return self.run(question)


def get_tool(model_path=None):
    return RagTool(model_path=model_path)