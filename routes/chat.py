from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from bson import ObjectId

from routes.auth import get_current_user
from services.chatbot import ask_question
from services.database import get_db

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    question: str


@router.post("/chat/{doc_id}")
async def chat_with_document(doc_id: str, body: ChatRequest, user_email: str = Depends(get_current_user)):
    db = get_db()
    try:
        doc = db.documents.find_one({"_id": ObjectId(doc_id), "user_id": user_email})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    raw_text = doc.get("raw_text", "")
    extracted_data = doc.get("extracted_data", {})

    if not raw_text and not extracted_data:
        return {"answer": "This document has no extracted text yet. It may still be processing."}

    answer = ask_question(raw_text, extracted_data, body.question)
    return {"answer": answer}
