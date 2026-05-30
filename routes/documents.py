import os
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks, Depends
from bson import ObjectId

from services.database import get_db
from services.ocr import extract_text
from services.ai_extractor import extract_fields
from services.redis_pool import get_redis_pool, redis_available
from services.task_queue import process_document_job
from routes.auth import get_current_user, get_current_tenant

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
                background_tasks.add_task(process_document, doc["_id"], file_id, tenant_id.lower())
    elif background_tasks:
        background_tasks.add_task(process_document, doc["_id"], file_id, tenant_id.lower())

    return doc


def process_document(doc_id: str, file_id: str, tenant_id: str = "default"):
    """Background task: OCR -> Regex -> (optional) OpenAI -> save results."""
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
            return
        content = file_doc["data"]
        print(f"[INFO] File loaded: {file_doc.get('filename', '?')}, size={len(content)} bytes")

        ext = Path(file_doc.get("filename", "file.tmp")).suffix or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(content)
            tmp = tmp_file.name

        t0 = time.time()
        raw_text = extract_text(tmp)
        t1 = time.time()
        print(f"[TIME] OCR took {t1-t0:.1f}s, text length={len(raw_text)}")

        if not raw_text or len(raw_text.strip()) < 10:
            error_message = "OCR could not extract readable text. The document may be a scanned image — ensure Tesseract OCR is installed."
            print(f"[WARN] OCR returned empty/short text: '{raw_text[:100]}'")
            return

        print(f"[INFO] OCR snippet: {raw_text[:300]}")

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
        try:
            db = get_db()
            db.documents.update_one(
                {"_id": ObjectId(doc_id)},
                {"$set": {
                    "status": status,
                    "extracted_data": extracted_data,
                    "confidence_scores": confidence_scores,
                    "overall_confidence": overall_confidence,
                    "raw_text": raw_text,
                    "error_message": error_message,
                    "updated_at": datetime.now(timezone.utc),
                }}
            )
            print(f"[INFO] Document status saved: {status}")
        except Exception as db_err:
            print(f"[ERROR] CRITICAL: Failed to save document status: {db_err}")

        print(f"{'='*60}\n")


@router.get("/documents")
async def list_documents(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
    user_email: str = Depends(get_current_user),
    tenant_id: str = Depends(get_current_tenant),
):
    db = get_db()
    query = {"user_id": user_email, "tenant_id": tenant_id.lower()}
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
                    background_tasks.add_task(process_document, doc_id, file_id, tenant_id.lower())
        elif background_tasks:
            background_tasks.add_task(process_document, doc_id, file_id, tenant_id.lower())

        results.append({"filename": file.filename, "doc_id": doc_id, "status": "queued"})

    return {"results": results, "total": len(results)}
