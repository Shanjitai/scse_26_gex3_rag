"""RAG pipeline for SCSE 26 GE3.

Uses the provided load_policy.py knowledge base and Qwen3-0.6B.
"""

import os
import re
import math
from collections import Counter

from policy_loader import load_policy_documents


STOPWORDS = set("""
a an the and or of to in on for with by is are was were be been being it its
this that these those as at from should must can could may might will would
i you your yours we our ours they them their theirs do does did doing have has
had having
""".split())

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s\-().]{5,}\d)(?!\d)")


def tokenize(text: str):
    return [
        t.lower()
        for t in TOKEN_RE.findall(text)
        if t.lower() not in STOPWORDS
    ]


class BM25Retriever:
    """Small dependency-free BM25 retriever for the local policy documents."""

    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokenized = [
            tokenize(
                d.page_content
                + " "
                + d.metadata.get("topic", "")
                + " "
                + d.metadata.get("policy_id", "")
            )
            for d in documents
        ]
        self.doc_len = [len(tokens) for tokens in self.tokenized]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.N = len(documents)
        self.tf = [Counter(tokens) for tokens in self.tokenized]

        self.df = Counter()
        for tokens in self.tokenized:
            for term in set(tokens):
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def _score(self, query: str, idx: int) -> float:
        q_tokens = tokenize(query)
        score = 0.0
        for term in q_tokens:
            f = self.tf[idx].get(term, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (
                1 - self.b + self.b * self.doc_len[idx] / max(self.avgdl, 1e-9)
            )
            score += self._idf(term) * (f * (self.k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 5):
        if not self.documents:
            return []

        scored = [(self._score(query, i), i) for i in range(self.N)]
        scored.sort(key=lambda x: x[0], reverse=True)

        results = [self.documents[i] for s, i in scored[:top_k] if s > 0]
        return results if results else self.documents[:top_k]


class RAGPipeline:
    def __init__(self, model_path=None, top_k=5):
        self.documents = load_policy_documents()
        self.retriever = BM25Retriever(self.documents)
        self.top_k = top_k

        # 优先读环境变量。download_qwen.py 下载到哪里，就把哪里填进来。
        self.model_path = model_path or os.environ.get(
            "QWEN_MODEL_PATH", "Qwen/Qwen3-0.6B"
        )
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def retrieve(self, question: str, top_k=None):
        return self.retriever.search(question, top_k=top_k or self.top_k)

    @staticmethod
    def _format_context(documents):
        parts = []
        for d in documents:
            parts.append(
                f"[{d.metadata.get('policy_id', 'unknown')} | "
                f"{d.metadata.get('topic', 'unknown')}]\n"
                f"{d.page_content.strip()}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _asks_phone(question: str) -> bool:
        q = question.lower()
        return any(
            k in q
            for k in [
                "phone",
                "telephone",
                "phone number",
                "telephone number",
                "contact number",
                "service desk number",
                "call",
            ]
        )

    @staticmethod
    def _has_phone(text: str) -> bool:
        return bool(PHONE_RE.search(text))

    @staticmethod
    def _clean_answer(text: str) -> str:
        text = text.strip()

        # Qwen3 可能输出 thinking 块，去掉。
        text = re.sub(r" thinking.*?", "", text, flags=re.DOTALL).strip()

        for prefix in ["Answer:", "Assistant:", "A:"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        return text

    def answer(self, question: str) -> str:
        docs = self.retrieve(question)
        context = self._format_context(docs)

        # 关键安全规则：如果问电话号码，而检索到的知识库没有电话号码，
        # 不让模型自由发挥，直接说不知道。
        if self._asks_phone(question) and not self._has_phone(context):
            return (
                "I don't know the exact phone number for the University IT Service Desk. "
                "The provided knowledge base does not contain a phone number."
            )

        self._load_model()

        system_prompt = (
            "You are a university IT support assistant. "
            "Answer only using the provided context. Do not use outside knowledge. "
            "If the answer is not present in the context, say that you do not know. "
            "Never invent a phone number, email address, URL, or contact detail. "
            "If the context does not contain an exact phone number, say that you do not know "
            "the phone number and cannot provide one. Keep the answer concise."
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True)
        answer = self._clean_answer(answer)

        # 第二层安全规则：如果模型编造了上下文里没有的电话号码，抑制掉。
        if self._asks_phone(question) and not self._has_phone(context):
            return (
                "I don't know the exact phone number for the University IT Service Desk. "
                "The provided knowledge base does not contain a phone number."
            )

        if self._has_phone(answer) and not self._has_phone(context):
            return "I don't know. The provided knowledge base does not contain a phone number."

        return answer


if __name__ == "__main__":
    rag = RAGPipeline()
    print(
        rag.answer(
            "What exact phone number should I call for the University IT Service Desk?"
        )
    )