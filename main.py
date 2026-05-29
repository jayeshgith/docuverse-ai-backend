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
