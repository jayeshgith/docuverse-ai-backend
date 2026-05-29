import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"


def ask_question(raw_text: str, extracted_data: dict, question: str) -> str:
    extracted_str = "\n".join(
        f"  {k}: {v}" for k, v in extracted_data.items() if v
    ) if extracted_data else "  (no fields extracted yet)"

    prompt = f"""You are a helpful document assistant for DocuVerse. Answer the user's question based ONLY on the document provided below. Be concise (1-3 sentences). If the answer is not in the document, say "I couldn't find that in your document."

Document text:
{raw_text}

Extracted data:
{extracted_str}

User question: {question}"""

    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode()

    try:
        req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data.get("response", "").strip()
    except Exception as e:
        return f"Sorry, I couldn't process that. Error: {str(e)}"
