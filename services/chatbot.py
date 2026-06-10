import json
import os
import re
import urllib.request
from openai import OpenAI

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")

openai_client = None
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key:
    openai_client = OpenAI(api_key=openai_api_key)

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
    return len(q_words & c_words)


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


def _search_in_text(raw_text: str, question: str) -> str | None:
    q_lower = question.lower()
    for kw in re.findall(r"\w+", q_lower):
        if len(kw) < 3:
            continue
        for line in raw_text.split("\n"):
            if kw in line.lower():
                stripped = line.strip()
                if stripped:
                    return stripped
    return None


def _build_prompt(raw_text: str, extracted_data: dict, question: str, history: list[dict] | None) -> str:
    extracted_str = "\n".join(
        f"  {k}: {v}" for k, v in extracted_data.items() if v
    ) if extracted_data else "  (no fields extracted yet)"

    chunks = chunk_text(raw_text)
    relevant = find_relevant_chunks(chunks, question) if len(chunks) > 1 else chunks
    context = "\n\n---\n\n".join(c["text"] for c in relevant)
    history_block = build_history_block(history or [])

    prompt = f"""You are a helpful document assistant for DocuVerse. Answer the user's question based ONLY on the document provided below. Be concise (1-3 sentences). If the answer is not in the document, say "I could not find that in your document."

Document text:
{context}

Extracted data:
{extracted_str}"""

    if history_block:
        prompt += f"\n\nRecent conversation:\n{history_block}"

    prompt += f"\n\nUser question: {question}"
    return prompt


def _ask_ollama(prompt: str) -> str | None:
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode()
    try:
        req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except Exception:
        return None


def _ask_openai(prompt: str) -> str | None:
    if not openai_client:
        return None
    try:
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a document assistant for DocuVerse. Answer concisely based ONLY on the provided document."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=300,
            timeout=15,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return None


def ask_question(raw_text: str, extracted_data: dict, question: str, history: list[dict] = None) -> str:
    prompt = _build_prompt(raw_text, extracted_data, question, history)

    answer = _ask_ollama(prompt)
    if answer:
        return answer

    answer = _ask_openai(prompt)
    if answer:
        return answer

    fallback = _search_in_text(raw_text, question)
    if fallback:
        return f"Based on the document: {fallback}"
    return "I could not find that in your document."
