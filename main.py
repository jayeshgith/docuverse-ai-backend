import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from bson import ObjectId

from services.database import get_db

load_dotenv()

from routes.documents import router as documents_router
from routes.auth import router as auth_router
from routes.chat import router as chat_router
from routes.admin import router as admin_router


def seed_default_configs():
    try:
        db = get_db()
        existing = db.document_configs.count_documents({"tenant_id": "default"})
        if existing > 0:
            return
        defaults = [
            {
                "document_type": "passport",
                "display_name": "Passport",
                "tenant_id": "default",
                "confidence_threshold": 0.78,
                "fields": [
                    {"key": "document_type", "description": "Document type label", "regex_pattern": None, "is_required": True},
                    {"key": "passport_number", "description": "Passport number (1 letter + 7 digits)", "regex_pattern": r"(?:passport\s*(?:no|number|#|\.)?\s*[:\-]\s*)([A-Z]\s*[0-9]\s*[0-9]\s*[0-9]\s*[0-9]\s*[0-9]\s*[0-9]\s*[0-9])", "is_required": True},
                    {"key": "name", "description": "Full name of the passport holder", "regex_pattern": r"(?:name|full name|given name|surname|applicant name)\s*[:\-]\s*([A-Za-z\s\.'\-]+?)(?:\n|$|\||email|\d{2}|[0-9])", "is_required": True},
                    {"key": "dob", "description": "Date of birth", "regex_pattern": r"(?:dob|date\s*of\s*birth|birth\s*date|d\.o\.b)\s*[:\-]\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})", "is_required": True},
                    {"key": "nationality", "description": "Nationality", "regex_pattern": r"(?:nationality|citizenship)\s*[:\-]\s*([A-Za-z\s]+?)(?:\n|$|\d)", "is_required": False},
                    {"key": "gender", "description": "Gender (M/F)", "regex_pattern": r"(?:gender|sex)\s*[:\-]\s*(M|F|Male|Female|MALE|FEMALE)", "is_required": False},
                    {"key": "issue_date", "description": "Date of issue", "regex_pattern": r"(?:issue date|date of issue|issued on|date of issuance)\s*[:\-]\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})", "is_required": False},
                    {"key": "expiry_date", "description": "Expiry date", "regex_pattern": r"(?:expiry date|date of expiry|valid until|expiration date)\s*[:\-]\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})", "is_required": False},
                    {"key": "place_of_birth", "description": "Place of birth", "regex_pattern": r"(?:place\s*of\s*birth|pob|birth\s*place)\s*[:\-]\s*([A-Za-z\s\.'\-]+?)(?:\n|$)", "is_required": False},
                    {"key": "place_of_issue", "description": "Place of issue", "regex_pattern": r"(?:place\s*of\s*issue|poi|issue\s*place)\s*[:\-]\s*([A-Za-z\s\.'\-]+?)(?:\n|$)", "is_required": False},
                    {"key": "address", "description": "Residential address", "regex_pattern": r"(?:address|residence|permanent address)\s*[:\-]\s*([\w\s,\.\-/#]+?)(?:\n{2,}|$)", "is_required": False},
                ],
            },
            {
                "document_type": "pan_card",
                "display_name": "PAN Card",
                "tenant_id": "default",
                "confidence_threshold": 0.78,
                "fields": [
                    {"key": "document_type", "description": "Document type label", "regex_pattern": None, "is_required": True},
                    {"key": "pan_number", "description": "PAN number (5 letters + 4 digits + 1 letter)", "regex_pattern": r"(?:pan\s*(?:no|number|#|\.|:)?\s*[:\-]\s*)?([A-Z]\s*[A-Z]\s*[A-Z]\s*[A-Z]\s*[A-Z]\s*\d\s*\d\s*\d\s*\d\s*[A-Z])", "is_required": True},
                    {"key": "name", "description": "Cardholder name", "regex_pattern": r"(?:name|full name|given name|surname|applicant name|candidate name|holder name)\s*[:\-]\s*([A-Za-z\s\.'\-]+?)(?:\n|$|\||email|\d{2}|[0-9])", "is_required": True},
                    {"key": "father_name", "description": "Father's name", "regex_pattern": r"(?:father|father's name|father name|fathers name)\s*[:\-]\s*([A-Za-z\s\.'\-]+?)(?:\n|$)", "is_required": False},
                    {"key": "dob", "description": "Date of birth", "regex_pattern": r"(?:dob|date\s*of\s*birth|birth\s*date|d\.o\.b)\s*[:\-]\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})", "is_required": True},
                ],
            },
            {
                "document_type": "aadhaar_card",
                "display_name": "Aadhaar Card",
                "tenant_id": "default",
                "confidence_threshold": 0.78,
                "fields": [
                    {"key": "document_type", "description": "Document type label", "regex_pattern": None, "is_required": True},
                    {"key": "aadhaar_number", "description": "12-digit Aadhaar number", "regex_pattern": r"(\d{4}\s?\d{4}\s?\d{4})", "is_required": True},
                    {"key": "name", "description": "Full name of the Aadhaar holder", "regex_pattern": r"(?:name|full name|applicant|holder)\s*[:.\-]?\s*([A-Za-z\s.'-]+?)(?:\n|$)", "is_required": True},
                    {"key": "dob", "description": "Date of birth", "regex_pattern": r"(?:dob|date\s*of\s*birth|birth\s*date|d\.o\.b)\s*[:\-]\s*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})", "is_required": False},
                    {"key": "gender", "description": "Gender (Male/Female)", "regex_pattern": r"(?:gender|sex)\s*[:\-]\s*(M|F|Male|Female|MALE|FEMALE)", "is_required": False},
                    {"key": "address", "description": "Residential address", "regex_pattern": r"(?:address|residence|permanent address)\s*[:\-]\s*([\w\s,\.\-/#]+?)(?:\n{2,}|$)", "is_required": False},
                    {"key": "mobile_number", "description": "Mobile phone number", "regex_pattern": r"(?:mobile|phone|contact|mobile\s*number|phone\s*number)\s*[:\-]\s*(\+?\d[\d\s\-()]{7,15})", "is_required": False},
                ],
            },
            {
                "document_type": "invoice",
                "display_name": "Invoice",
                "tenant_id": "default",
                "confidence_threshold": 0.78,
                "fields": [
                    {"key": "document_type", "description": "Document type label", "regex_pattern": None, "is_required": True},
                    {"key": "invoice_number", "description": "Invoice number", "regex_pattern": r"(?:invoice\s*(?:no|number|#|\.)?\s*[:\-]\s*)([A-Z0-9\-/]+)", "is_required": True},
                    {"key": "vendor", "description": "Vendor/supplier name", "regex_pattern": r"(?:vendor|seller|supplier|store|shop|company|merchant)\s*[:\-]\s*([A-Za-z0-9\s\.'\-&]+?)(?:\n|$)", "is_required": True},
                    {"key": "date", "description": "Invoice date", "regex_pattern": r"(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})", "is_required": False},
                    {"key": "total_amount", "description": "Total amount due", "regex_pattern": r"(?:total|grand total|total amount|amount due|net amount|balance due|sum)\s*[:\-]?\s*[₹$]?\s*([\d,]+\.\d{2})", "is_required": True},
                    {"key": "name", "description": "Customer name", "regex_pattern": r"(?:name|full name|given name|bill to)\s*[:\-]\s*([A-Za-z\s\.'\-]+?)(?:\n|$)", "is_required": False},
                ],
            },
            {
                "document_type": "bill",
                "display_name": "Bill / Receipt",
                "tenant_id": "default",
                "confidence_threshold": 0.78,
                "fields": [
                    {"key": "document_type", "description": "Document type label", "regex_pattern": None, "is_required": True},
                    {"key": "bill_number", "description": "Bill number", "regex_pattern": r"(?:bill\s*(?:no|number|#|\.)?\s*[:\-]\s*)([A-Z0-9\-/]+)", "is_required": True},
                    {"key": "vendor", "description": "Vendor/store name", "regex_pattern": r"(?:vendor|seller|supplier|store|shop|company|merchant|billed to|bill from)\s*[:\-]\s*([A-Za-z0-9\s\.'\-&]+?)(?:\n|$)", "is_required": False},
                    {"key": "date", "description": "Bill date", "regex_pattern": r"(\d{2}[/\-\.]\d{2}[/\-\.]\d{4})", "is_required": False},
                    {"key": "total_amount", "description": "Total amount", "regex_pattern": r"(?:total|grand total|total amount|amount due|net amount|balance due|sum)\s*[:\-]?\s*[₹$]?\s*([\d,]+\.\d{2})", "is_required": True},
                    {"key": "name", "description": "Customer name", "regex_pattern": r"(?:name|full name|given name)\s*[:\-]\s*([A-Za-z\s\.'\-]+?)(?:\n|$)", "is_required": False},
                ],
            },
            {
                "document_type": "resume",
                "display_name": "Resume / CV",
                "tenant_id": "default",
                "confidence_threshold": 0.78,
                "fields": [
                    {"key": "document_type", "description": "Document type label", "regex_pattern": None, "is_required": True},
                    {"key": "name", "description": "Candidate full name", "regex_pattern": r"(?:name|full name|given name|surname)\s*[:\-]\s*([A-Za-z\s\.'\-]+?)(?:\n|$)", "is_required": True},
                    {"key": "email", "description": "Email address", "regex_pattern": r"([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})", "is_required": False},
                    {"key": "phone", "description": "Phone number", "regex_pattern": r"((?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})", "is_required": False},
                    {"key": "skills", "description": "Skills and competencies", "regex_pattern": None, "is_required": False},
                    {"key": "education", "description": "Education history", "regex_pattern": None, "is_required": False},
                    {"key": "experience_summary", "description": "Work experience summary", "regex_pattern": None, "is_required": False},
                ],
            },
        ]
        for cfg in defaults:
            db.document_configs.update_one(
                {"document_type": cfg["document_type"], "tenant_id": "default"},
                {"$set": cfg},
                upsert=True,
            )
        print(f"[INFO] Seeded {len(defaults)} default document configs")
    except Exception as e:
        print(f"[WARN] Could not seed default configs: {e}")


app = FastAPI(
    title="DocuVerse — AI Document Extraction API",
    version="1.0.0",
    description="AI-powered document extraction system supporting passports, PAN cards, Aadhaar cards, invoices, and more.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router, prefix="/api")
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(admin_router)

seed_default_configs()


@app.on_event("startup")
async def startup():
    from services.redis_pool import get_redis_pool, redis_available
    if redis_available():
        try:
            await get_redis_pool()
            print("[INFO] Redis pool created")
        except Exception as e:
            print(f"[WARN] Redis connection failed: {e}")
            print("[WARN] Background tasks will use BackgroundTasks fallback instead of ARQ")


@app.on_event("shutdown")
async def shutdown():
    from services.redis_pool import close_redis_pool
    await close_redis_pool()


@app.get("/files/{file_id}")
async def serve_file(file_id: str):
    try:
        obj_id = ObjectId(file_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file ID")
    db = get_db()
    file_doc = db.files.find_one({"_id": obj_id})
    if not file_doc:
        raise HTTPException(status_code=404, detail="File not found")
    content_type = file_doc.get("content_type", "application/octet-stream")
    filename = file_doc.get("filename", "file")
    data = file_doc["data"]
    return Response(
        content=data,
        media_type=content_type,
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Content-Length": str(len(data)),
            "Accept-Ranges": "bytes",
        },
    )


@app.get("/uploads/{file_path:path}")
async def serve_upload_legacy(file_path: str):
    full_path = Path(__file__).parent / "uploads" / file_path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    from fastapi.responses import FileResponse
    return FileResponse(str(full_path), headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "*",
    })


@app.get("/api/health")
async def health_check():
    from services.ocr import tesseract_available, tesseract_cmd
    return {
        "status": "ok",
        "service": "DocuVerse API",
        "tesseract_installed": tesseract_available,
        "tesseract_path": tesseract_cmd if os.path.exists(tesseract_cmd) else "not found",
    }
