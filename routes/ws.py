"""WebSocket endpoint for real-time document progress events."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.progress import subscribe, unsubscribe

router = APIRouter()


@router.websocket("/ws/document/{doc_id}")
async def document_ws(websocket: WebSocket, doc_id: str):
    await websocket.accept()
    q = subscribe(doc_id)
    print(f"[WS] Client connected for doc {doc_id}")
    try:
        while True:
            event = await q.get()
            try:
                await websocket.send_json(event)
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe(doc_id, q)
        print(f"[WS] Client disconnected for doc {doc_id}")
