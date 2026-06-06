import os
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone
from bson import ObjectId

from services.database import get_db
from services.ocr import extract_text
from services.ai_extractor import extract_fields
from services.progress import publish

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
MAX_RETRIES = 3
DLQ_COLLECTION = "dead_letter_queue"


def _move_to_dlq(doc_id: str, file_id: str, tenant_id: str, error_message: str):
    try:
        db = get_db()
        doc = db.documents.find_one({"_id": ObjectId(doc_id)})
        if doc:
            dlq_entry = {
                "doc_id": doc_id,
                "file_id": file_id,
                "tenant_id": tenant_id,
                "original_name": doc.get("original_name", ""),
                "error_message": error_message,
                "failed_at": datetime.now(timezone.utc),
            }
            db[DLQ_COLLECTION].insert_one(dlq_entry)
            print(f"[DLQ] Moved doc {doc_id} to dead letter queue")
    except Exception as e:
        print(f"[DLQ] Failed to move to DLQ: {e}")


async def _emit_progress(doc_id: str, step: str, message: str, status: str = "processing", payload: dict = None):
    event = {
        "doc_id": doc_id,
        "step": step,
        "message": message,
        "status": status,
        "payload": payload or {},
    }
    try:
        await publish(doc_id, event)
    except Exception as e:
        print(f"[PROGRESS] Publish error: {e}")


async def process_document_job(ctx, doc_id: str, file_id: str, tenant_id: str = "default"):
    """ARQ worker job: OCR -> AI -> save results, with DLQ on repeated failure."""
    tmp = None
    t_start = time.time()
    raw_text = ""
    extracted_data = {}
    confidence_scores = {}
    overall_confidence = 0.0
    status = "failed"
    error_message = None

    print(f"\n{'='*60}")
    print(f"[START] process_document: doc_id={doc_id}, file_id={file_id}, tenant={tenant_id}")

    await _emit_progress(doc_id, "queued", "Task picked up by worker")

    try:
        db = get_db()
        doc_doc = db.documents.find_one({"_id": ObjectId(doc_id)})
        retry_count = doc_doc.get("retry_count", 0) if doc_doc else 0
    except Exception as e:
        print(f"[WARN] Could not read retry count: {e}")
        retry_count = 0

    try:
        db_mark = get_db()
        db_mark.documents.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"status": "processing", "updated_at": datetime.now(timezone.utc)}}
        )
        print("[INFO] Task started marker saved to DB")
    except Exception as e:
        print(f"[WARN] Could not write start marker: {e}")

    try:
        db = get_db()
        file_doc = db.files.find_one({"_id": ObjectId(file_id)})
        if not file_doc:
            error_message = "File not found in database"
            print(f"[ERROR] {error_message}")
            await _emit_progress(doc_id, "error", error_message, "failed", {"error_message": error_message})
            return
        content = file_doc["data"]
        print(f"[INFO] File loaded: {file_doc.get('filename', '?')}, size={len(content)} bytes")

        ext = Path(file_doc.get("filename", "file.tmp")).suffix or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(content)
            tmp = tmp_file.name

        await _emit_progress(doc_id, "ocr", "Running OCR on document")

        t0 = time.time()
        raw_text = extract_text(tmp)
        t1 = time.time()
        print(f"[TIME] OCR took {t1-t0:.1f}s, text length={len(raw_text)}")

        if not raw_text or len(raw_text.strip()) < 10:
            error_message = "OCR could not extract readable text. The document may be a scanned image."
            print(f"[WARN] OCR returned empty/short text: '{raw_text[:100]}'")
            await _emit_progress(doc_id, "ocr", error_message, "failed", {"error_message": error_message})
            return

        print(f"[INFO] OCR snippet: {raw_text[:300]}")

        await _emit_progress(doc_id, "ai_extraction", "Extracting fields with AI")

        t2 = time.time()
        extracted_data, confidence_scores, overall_confidence = extract_fields(raw_text, tenant_id)
        t3 = time.time()
        print(f"[TIME] Field extraction took {t3-t2:.1f}s, fields={len(extracted_data)}")

        if extracted_data:
            status = "completed"
            print(f"[SUCCESS] Extracted fields: {list(extracted_data.keys())}")
        else:
            status = "failed"
            error_message = "No fields could be extracted from the document text."
            print(f"[WARN] No fields extracted from text of length {len(raw_text)}")

    except Exception as e:
        status = "failed"
        error_message = f"Extraction error: {str(e)}"
        print(f"[ERROR] Exception in process_document: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass

        elapsed = time.time() - t_start
        print(f"[TIME] process_document total: {elapsed:.1f}s, status={status}")

        if status == "failed" and error_message:
            new_retry = retry_count + 1
            if new_retry >= MAX_RETRIES:
                print(f"[DLQ] Doc {doc_id} failed {new_retry} times — moving to DLQ")
                _move_to_dlq(doc_id, file_id, tenant_id, error_message)
                await _emit_progress(doc_id, "dlq", "Moved to dead letter queue after repeated failures", "failed",
                                     {"error_message": error_message})
            else:
                db = get_db()
                db.documents.update_one(
                    {"_id": ObjectId(doc_id)},
                    {"$set": {
                        "status": "failed",
                        "retry_count": new_retry,
                        "error_message": error_message,
                        "updated_at": datetime.now(timezone.utc),
                    }}
                )
        else:
            payload = {
                "extracted_data": extracted_data,
                "confidence_scores": confidence_scores,
                "overall_confidence": overall_confidence,
                "raw_text": raw_text,
            }
            final_status = status
            final_msg = error_message or "Document processed successfully"
            await _emit_progress(doc_id, "done", final_msg, final_status, payload=payload)

        print(f"{'='*60}\n")
