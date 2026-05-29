import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks, Depends
from bson import ObjectId

from services.database import get_db
from services.ocr import extract_text
from services.ai_extractor import extract_fields
from routes.auth import get_current_user

router = APIRouter(dependencies=[Depends(get_current_user)])

ALLOWED_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}
MAX_SIZE = 10 * 1024 * 1024


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None, user_email: str = Depends(get_current_user)):
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

    if background_tasks:
        background_tasks.add_task(process_document, doc["_id"], file_id)

    return doc


def process_document(doc_id: str, file_id: str):
    tmp = None
    try:
        db = get_db()
        file_doc = db.files.find_one({"_id": ObjectId(file_id)})
        if not file_doc:
            raise FileNotFoundError("File not found in database")
        content = file_doc["data"]

        ext = Path(file_doc.get("filename", "file.tmp")).suffix or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(content)
            tmp = tmp_file.name

        raw_text = extract_text(tmp)
        
        doc = db.documents.find_one({"_id": ObjectId(doc_id)})
        tenant_id = doc.get("user_id", "default") if doc else "default"
        
        extracted_data, confidence_scores, overall_confidence = extract_fields(raw_text, tenant_id)
        status = "completed" if extracted_data else "failed"
        error_message = None
    except FileNotFoundError:
        raw_text = ""
        extracted_data = {}
        confidence_scores = {}
        overall_confidence = 0.0
        status = "failed"
        error_message = "Tesseract OCR not found. Install Tesseract or set TESSERACT_CMD env var."
    except Exception as e:
        raw_text = ""
        extracted_data = {}
        confidence_scores = {}
        overall_confidence = 0.0
        status = "failed"
        error_message = f"Extraction error: {str(e)}"
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)

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
    except Exception:
        pass


@router.get("/documents")
async def list_documents(page: int = Query(1, ge=1), limit: int = Query(10, ge=1, le=50), user_email: str = Depends(get_current_user)):
    db = get_db()
    query = {"user_id": user_email}
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
async def get_document(doc_id: str, user_email: str = Depends(get_current_user)):
    db = get_db()
    try:
        doc = db.documents.find_one({"_id": ObjectId(doc_id), "user_id": user_email})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc["_id"] = str(doc["_id"])
    return doc


@router.put("/documents/{doc_id}")
async def update_document(doc_id: str, data: dict, user_email: str = Depends(get_current_user)):
    db = get_db()
    try:
        obj_id = ObjectId(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    existing = db.documents.find_one({"_id": obj_id, "user_id": user_email})
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
async def delete_document(doc_id: str, user_email: str = Depends(get_current_user)):
    db = get_db()
    try:
        obj_id = ObjectId(doc_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid document ID")

    doc = db.documents.find_one({"_id": obj_id, "user_id": user_email})
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
