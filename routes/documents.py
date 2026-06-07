import os
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks, Depends
from bson import ObjectId

from pathlib import Path as PathLib
import asyncio
import functools
from services.database import get_db
from services.ocr import extract_text, extract_words_from_image
from services.ai_extractor import extract_fields
from services.redis_pool import get_redis_pool, redis_available
from services.task_queue import process_document_job
from services.progress import publish_sync
from routes.auth import get_current_user, get_current_tenant

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

router = APIRouter(dependencies=[Depends(get_current_user)])

ALLOWED_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}
MAX_SIZE = 10 * 1024 * 1024


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    user_email: str = Depends(get_current_user),
    tenant_id: str = Depends(get_current_tenant),
):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {', '.join(ALLOWED_TYPES.keys())}"
        )

    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

    db = get_db()
    file_doc = {
        "data": content,
        "content_type": file.content_type,
        "filename": file.filename,
        "uploaded_at": datetime.now(timezone.utc),
    }
    file_result = db.files.insert_one(file_doc)
    file_id = str(file_result.inserted_id)
    file_url = f"/files/{file_id}"

    doc = {
        "user_id": user_email,
        "tenant_id": tenant_id.lower(),
        "original_name": file.filename,
        "file_path": file_url,
        "file_type": file.content_type,
        "file_size": len(content),
        "status": "processing",
        "extracted_data": {},
        "confidence_scores": {},
        "overall_confidence": 0.0,
        "raw_text": "",
        "error_message": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }

    result = db.documents.insert_one(doc)
    doc["_id"] = str(result.inserted_id)

    if redis_available():
        try:
            pool = await get_redis_pool()
            if pool:
                await pool.enqueue_job("process_document_job", doc["_id"], file_id, tenant_id.lower())
                print(f"[INFO] Enqueued ARQ job for doc {doc['_id']}")
        except Exception as e:
            print(f"[WARN] ARQ enqueue failed, falling back to BackgroundTasks: {e}")
            if background_tasks:
                background_tasks.add_task(_run_process_document, doc["_id"], file_id, tenant_id.lower())
    elif background_tasks:
        background_tasks.add_task(_run_process_document, doc["_id"], file_id, tenant_id.lower())

    return doc


_MAX_EXTRACTION_SECS = 120


async def _run_process_document(doc_id: str, file_id: str, tenant_id: str = "default"):
    loop = asyncio.get_event_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, process_document, doc_id, file_id, tenant_id),
            timeout=_MAX_EXTRACTION_SECS,
        )
    except asyncio.TimeoutError:
        print(f"[TIMEOUT] process_document({doc_id}) exceeded {_MAX_EXTRACTION_SECS}s")
        try:
            db = get_db()
            db.documents.update_one(
                {"_id": ObjectId(doc_id)},
                {"$set": {"status": "failed", "error_message": f"Extraction timed out after {_MAX_EXTRACTION_SECS}s", "updated_at": datetime.now(timezone.utc)}}
            )
        except Exception:
            pass


def process_document(doc_id: str, file_id: str, tenant_id: str = "default"):
    """Background task: OCR -> AI -> save results, with DLQ on repeated failure."""
    tmp = None
    t_start = time.time()
    raw_text = ""
    extracted_data = {}
    confidence_scores = {}
    overall_confidence = 0.0
    status = "failed"
    error_message = None
    ocr_words = []

    print(f"\n{'='*60}")
    print(f"[START] process_document: doc_id={doc_id}, file_id={file_id}, tenant={tenant_id}")

    publish_sync(doc_id, {"step": "queued", "message": "Task picked up by background thread"})

    db = get_db()
    doc_doc = db.documents.find_one({"_id": ObjectId(doc_id)})
    retry_count = doc_doc.get("retry_count", 0) if doc_doc else 0

    try:
        db.documents.update_one(
            {"_id": ObjectId(doc_id)},
            {"$set": {"status": "processing", "updated_at": datetime.now(timezone.utc)}}
        )
        print("[INFO] Task started marker saved to DB")
    except Exception as e:
        print(f"[WARN] Could not write start marker: {e}")

    try:
        file_doc = db.files.find_one({"_id": ObjectId(file_id)})
        if not file_doc:
            error_message = "File not found in database"
            print(f"[ERROR] {error_message}")
            publish_sync(doc_id, {"step": "error", "message": error_message, "status": "failed", "payload": {"error_message": error_message}})
            return
        content = file_doc["data"]
        print(f"[INFO] File loaded: {file_doc.get('filename', '?')}, size={len(content)} bytes")

        ext = Path(file_doc.get("filename", "file.tmp")).suffix or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(content)
            tmp = tmp_file.name

        publish_sync(doc_id, {"step": "ocr", "message": "Running OCR on document"})

        t0 = time.time()
        raw_text = extract_text(tmp)
        t1 = time.time()
        print(f"[TIME] OCR took {t1-t0:.1f}s, text length={len(raw_text)}")

        ext = Path(file_doc.get("filename", "")).suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".webp"):
            try:
                ocr_words = extract_words_from_image(tmp)
                print(f"[INFO] Extracted {len(ocr_words)} word boxes from image")
            except Exception as we:
                print(f"[WARN] Could not extract word boxes: {we}")

        if not raw_text or len(raw_text.strip()) < 10:
            error_message = "OCR could not extract readable text. The document may be a scanned image — ensure Tesseract OCR is installed."
            print(f"[WARN] OCR returned empty/short text: '{raw_text[:100]}'")
            publish_sync(doc_id, {"step": "ocr", "message": error_message, "status": "failed", "payload": {"error_message": error_message}})
            return

        print(f"[INFO] OCR snippet: {raw_text[:300]}")

        publish_sync(doc_id, {"step": "ai_extraction", "message": "Extracting fields with AI"})

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
                publish_sync(doc_id, {"step": "dlq", "message": "Moved to dead letter queue after repeated failures", "status": "failed", "payload": {"error_message": error_message}})
            else:
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
                "ocr_words": ocr_words,
            }
            final_msg = error_message or "Document processed successfully"
            publish_sync(doc_id, {"step": "done", "message": final_msg, "status": status, "payload": payload})

        print(f"{'='*60}\n")


@router.get("/documents/stats")
async def get_document_stats(
    user_email: str = Depends(get_current_user),
    tenant_id: str = Depends(get_current_tenant),
):
    db = get_db()
    query = {"user_id": user_email, "tenant_id": tenant_id.lower()}

    status_pipeline = [
        {"$match": query},
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]
    status_cursor = db.documents.aggregate(status_pipeline)

    status_counts = {"completed": 0, "processing": 0, "failed": 0, "queued": 0}
    total_docs = 0
    for item in status_cursor:
        status_name = item["_id"] or "processing"
        status_counts[status_name] = item["count"]
        total_docs += item["count"]

    type_pipeline = [
        {"$match": query},
        {"$group": {"_id": "$extracted_data.document_type", "count": {"$sum": 1}}}
    ]
    type_cursor = db.documents.aggregate(type_pipeline)

    type_counts = {}
    for item in type_cursor:
        doc_type = item["_id"]
        if not doc_type:
            doc_type = "Unclassified"
        else:
            doc_type = str(doc_type).title()
        type_counts[doc_type] = type_counts.get(doc_type, 0) + item["count"]

    avg_conf_pipeline = [
        {"$match": {**query, "status": "completed", "overall_confidence": {"$gt": 0}}},
        {"$group": {"_id": None, "avg_confidence": {"$avg": "$overall_confidence"}}}
    ]
    avg_conf_cursor = list(db.documents.aggregate(avg_conf_pipeline))
    avg_confidence = round(avg_conf_cursor[0]["avg_confidence"] * 100, 1) if avg_conf_cursor else 0.0

    time_pipeline = [
        {"$match": query},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
            "count": {"$sum": 1}
        }},
        {"$sort": {"_id": 1}},
        {"$limit": 7}
    ]
    time_cursor = db.documents.aggregate(time_pipeline)
    volume_over_time = []
    for item in time_cursor:
        volume_over_time.append({
            "date": item["_id"],
            "uploads": item["count"]
        })
    if not volume_over_time:
        volume_over_time.append({"date": "No Data", "uploads": 0})

    return {
        "totalDocs": total_docs,
        "statusCounts": status_counts,
        "typeCounts": type_counts,
        "averageConfidence": avg_confidence,
        "volumeOverTime": volume_over_time
    }


@router.get("/documents")
async def list_documents(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
    search: Optional[str] = Query(None),
    year: Optional[int] = Query(None, ge=2000, le=2100),
    month: Optional[int] = Query(None, ge=1, le=12),
    day: Optional[int] = Query(None, ge=1, le=31),
    user_email: str = Depends(get_current_user),
    tenant_id: str = Depends(get_current_tenant),
):
    db = get_db()
    query = {"user_id": user_email, "tenant_id": tenant_id.lower()}

    if search:
        query["original_name"] = {"$regex": search, "$options": "i"}

    if year is not None:
        if month is not None:
            if day is not None:
                start_date = datetime(year, month, day, tzinfo=timezone.utc)
                end_date = start_date + timedelta(days=1)
            else:
                start_date = datetime(year, month, 1, tzinfo=timezone.utc)
                if month == 12:
                    end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                else:
                    end_date = datetime(year, month + 1, 1, tzinfo=timezone.utc)
        else:
            start_date = datetime(year, 1, 1, tzinfo=timezone.utc)
            end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

        query["created_at"] = {"$gte": start_date, "$lt": end_date}

    total = db.documents.count_documents(query)
    total_pages = max(1, (total + limit - 1) // limit)

    cursor = (
        db.documents.find(query)
        .sort("created_at", -1)
        .skip((page - 1) * limit)
        .limit(limit)
    )

    docs = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        docs.append(doc)

    return {
        "documents": docs,
        "total": total,
        "page": page,
        "totalPages": total_pages,
    }


@router.get("/documents/{doc_id}")
async def get_document(
    doc_id: str,
    user_email: str = Depends(get_current_user),
    tenant_id: str = Depends(get_current_tenant),
):
    db = get_db()
    try:
        doc = db.documents.find_one({"_id": ObjectId(doc_id), "user_id": user_email, "tenant_id": tenant_id.lower()})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc["_id"] = str(doc["_id"])
    return doc


@router.put("/documents/{doc_id}")
async def update_document(
    doc_id: str,
    data: dict,
    user_email: str = Depends(get_current_user),
    tenant_id: str = Depends(get_current_tenant),
):
    db = get_db()
    try:
        obj_id = ObjectId(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    existing = db.documents.find_one({"_id": obj_id, "user_id": user_email, "tenant_id": tenant_id.lower()})
    if not existing:
        raise HTTPException(status_code=404, detail="Document not found")

    update_fields = {}
    if "extracted_data" in data:
        update_fields["extracted_data"] = data["extracted_data"]
    if "status" in data:
        update_fields["status"] = data["status"]

    update_fields["updated_at"] = datetime.now(timezone.utc)

    db.documents.update_one({"_id": obj_id}, {"$set": update_fields})

    updated = db.documents.find_one({"_id": obj_id})
    updated["_id"] = str(updated["_id"])
    return updated


@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    user_email: str = Depends(get_current_user),
    tenant_id: str = Depends(get_current_tenant),
):
    db = get_db()
    try:
        obj_id = ObjectId(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    doc = db.documents.find_one({"_id": obj_id, "user_id": user_email, "tenant_id": tenant_id.lower()})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = doc.get("file_path", "")
    if file_path.startswith("/files/"):
        try:
            db.files.delete_one({"_id": ObjectId(file_path.split("/files/")[1])})
        except Exception:
            pass

    db.documents.delete_one({"_id": obj_id})

    return {"message": "Document deleted successfully"}


@router.post("/upload/bulk")
async def upload_bulk_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
    user_email: str = Depends(get_current_user),
    tenant_id: str = Depends(get_current_tenant),
):
    results = []
    for file in files:
        if file.content_type not in ALLOWED_TYPES:
            results.append({"filename": file.filename, "status": "rejected", "error": "Unsupported file type"})
            continue
        content = await file.read()
        if len(content) > MAX_SIZE:
            results.append({"filename": file.filename, "status": "rejected", "error": "File too large (max 10MB)"})
            continue

        db = get_db()
        file_doc = {
            "data": content,
            "content_type": file.content_type,
            "filename": file.filename,
            "uploaded_at": datetime.now(timezone.utc),
        }
        file_result = db.files.insert_one(file_doc)
        file_id = str(file_result.inserted_id)

        doc = {
            "user_id": user_email,
            "tenant_id": tenant_id.lower(),
            "original_name": file.filename,
            "file_path": f"/files/{file_id}",
            "file_type": file.content_type,
            "file_size": len(content),
            "status": "queued",
            "extracted_data": {},
            "confidence_scores": {},
            "overall_confidence": 0.0,
            "raw_text": "",
            "error_message": None,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        doc_result = db.documents.insert_one(doc)
        doc_id = str(doc_result.inserted_id)

        if redis_available():
            try:
                pool = await get_redis_pool()
                if pool:
                    await pool.enqueue_job("process_document_job", doc_id, file_id, tenant_id.lower())
            except Exception:
                if background_tasks:
                    background_tasks.add_task(_run_process_document, doc_id, file_id, tenant_id.lower())
        elif background_tasks:
            background_tasks.add_task(_run_process_document, doc_id, file_id, tenant_id.lower())

        results.append({"filename": file.filename, "doc_id": doc_id, "status": "queued"})

    return {"results": results, "total": len(results)}
