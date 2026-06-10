"""In-memory pub/sub for WebSocket progress events.

Keeps a dict of doc_id -> list of asyncio.Queue for broadcasting
status updates to all connected WebSocket clients for that document.
"""

import asyncio
from datetime import datetime, timezone
from bson import ObjectId
from services.database import get_db

_subscribers: dict[str, list[asyncio.Queue]] = {}


def subscribe(doc_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _subscribers.setdefault(doc_id, []).append(q)
    return q


def unsubscribe(doc_id: str, q: asyncio.Queue):
    subs = _subscribers.get(doc_id, [])
    if q in subs:
        subs.remove(q)
    if not subs:
        _subscribers.pop(doc_id, None)


async def publish(doc_id: str, event: dict):
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    subs = _subscribers.get(doc_id, [])
    for q in subs:
        await q.put(event)
    update_document_status_in_db(doc_id, event)


def publish_sync(doc_id: str, event: dict):
    """Synchronous version for BackgroundTasks workers."""
    event["timestamp"] = datetime.now(timezone.utc).isoformat()
    subs = _subscribers.get(doc_id, [])
    for q in subs:
        try:
            q.put_nowait(event)
        except Exception:
            pass
    update_document_status_in_db(doc_id, event)


def update_document_status_in_db(doc_id: str, event: dict):
    try:
        db = get_db()
        step = event.get("step", "")
        status = event.get("status", "processing")
        payload = event.get("payload", {})

        if status == "completed":
            set_fields = {
                "status": "completed",
                "extracted_data": payload.get("extracted_data", {}),
                "confidence_scores": payload.get("confidence_scores", {}),
                "overall_confidence": payload.get("overall_confidence", 0.0),
                "raw_text": payload.get("raw_text", ""),
                "ocr_words": payload.get("ocr_words", []),
                "error_message": None,
                "updated_at": datetime.now(timezone.utc),
            }
        elif status == "failed":
            set_fields = {
                "status": "failed",
                "error_message": payload.get("error_message", "Extraction failed"),
                "updated_at": datetime.now(timezone.utc),
            }
        else:
            set_fields = {
                "status": "processing",
                "progress_step": step,
                "progress_message": event.get("message", ""),
                "updated_at": datetime.now(timezone.utc),
            }

        db.documents.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": set_fields},
        )
    except Exception as e:
        print(f"[PROGRESS] DB update failed: {e}")
