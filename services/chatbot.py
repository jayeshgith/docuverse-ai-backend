import os
import re
import json
from openai import OpenAI

client = None
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    client = OpenAI(api_key=api_key)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def chunk_text(text: str) -> list[dict]:
    if not text or len(text) <= CHUNK_SIZE:
        return [{"text": text, "index": 0}]

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end >= len(text):
            chunks.append({"text": text[start:], "index": idx})
            break
        split_at = text.rfind(" ", start, end)
        if split_at > start + CHUNK_SIZE // 2:
            end = split_at
        chunks.append({"text": text[start:end], "index": idx})
        start = end - CHUNK_OVERLAP
        idx += 1
    return chunks


def score_relevance(chunk_text: str, question: str) -> int:
    q_words = set(re.findall(r"\w+", question.lower()))
    c_words = set(re.findall(r"\w+", chunk_text.lower()))
    common = q_words & c_words
    return len(common)


def find_relevant_chunks(chunks: list[dict], question: str, max_chunks: int = 3) -> list[dict]:
    scored = [(score_relevance(c["text"], question), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:max_chunks]]


def build_history_block(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def ask_question(raw_text: str, extracted_data: dict, question: str, history: list[dict] = None) -> str:
    extracted_str = "\n".join(
        f"  {k}: {v}" for k, v in extracted_data.items() if v
    ) if extracted_data else "  (no fields extracted yet)"

    chunks = chunk_text(raw_text)
    relevant = find_relevant_chunks(chunks, question) if len(chunks) > 1 else chunks
    context = "\n\n---\n\n".join(c["text"] for c in relevant)
    history_block = build_history_block(history or [])

    system_prompt = "You are a document assistant. Answer based ONLY on the document text provided. Be concise (1-3 sentences). If the answer is not in the document, say 'I could not find that in your document.'"

    user_prompt = f"Document text:\n{context}\n\nExtracted data:\n{extracted_str}"

    if history_block:
        user_prompt += f"\n\nRecent conversation:\n{history_block}"

    user_prompt += f"\n\nQuestion: {question}"

    if not client:
        return "Chat is unavailable — OpenAI API key not configured."

    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=400,
            timeout=20,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"Sorry, I could not process that. Error: {str(e)}"
