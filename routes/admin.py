from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from services.database import get_db
from routes.auth import get_current_user, get_current_tenant

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(get_current_user)])


class DocumentFieldSchema(BaseModel):
    key: str
    description: str
    regex_pattern: Optional[str] = None
    is_required: bool = True


class DocumentConfigSchema(BaseModel):
    document_type: str
    display_name: str
    fields: List[DocumentFieldSchema]
    confidence_threshold: float = 0.78


@router.get("/document-configs")
async def list_document_configs(
    tenant_id: str = Depends(get_current_tenant),
    user_email: str = Depends(get_current_user),
):
    db = get_db()
    cursor = db.document_configs.find({"tenant_id": tenant_id.lower()})
    configs = []
    for c in cursor:
        c["_id"] = str(c["_id"])
        configs.append(c)
    return configs


@router.get("/document-configs/{doc_type}")
async def get_document_config(
    doc_type: str,
    tenant_id: str = Depends(get_current_tenant),
    user_email: str = Depends(get_current_user),
):
    db = get_db()
    cfg = db.document_configs.find_one({
        "document_type": doc_type.lower(),
        "tenant_id": tenant_id.lower()
    })
    if not cfg:
        raise HTTPException(status_code=404, detail="Configuration not found")
    cfg["_id"] = str(cfg["_id"])
    return cfg


@router.post("/document-configs")
async def save_document_config(
    body: DocumentConfigSchema,
    tenant_id: str = Depends(get_current_tenant),
    user_email: str = Depends(get_current_user),
):
    db = get_db()

    existing = db.document_configs.find_one({
        "document_type": body.document_type.lower(),
        "tenant_id": tenant_id.lower()
    })

    config_data = {
        "document_type": body.document_type.lower(),
        "display_name": body.display_name,
        "fields": [f.dict() for f in body.fields],
        "tenant_id": tenant_id.lower(),
        "confidence_threshold": body.confidence_threshold,
    }

    if existing:
        db.document_configs.update_one(
            {"_id": existing["_id"]},
            {"$set": config_data}
        )
        return {"message": f"Updated config for {body.document_type}", "config": config_data}

    db.document_configs.insert_one(config_data)
    if "_id" in config_data:
        del config_data["_id"]
    return {"message": f"Created config for {body.document_type}", "config": config_data}


@router.put("/document-configs/{doc_type}")
async def update_document_config(
    doc_type: str,
    body: DocumentConfigSchema,
    tenant_id: str = Depends(get_current_tenant),
    user_email: str = Depends(get_current_user),
):
    db = get_db()

    existing = db.document_configs.find_one({
        "document_type": doc_type.lower(),
        "tenant_id": tenant_id.lower()
    })
    if not existing:
        raise HTTPException(status_code=404, detail="Configuration not found")

    config_data = {
        "document_type": body.document_type.lower(),
        "display_name": body.display_name,
        "fields": [f.dict() for f in body.fields],
        "tenant_id": tenant_id.lower(),
        "confidence_threshold": body.confidence_threshold,
    }

    db.document_configs.update_one(
        {"_id": existing["_id"]},
        {"$set": config_data}
    )
    config_data["_id"] = str(existing["_id"])
    return {"message": f"Updated config for {body.document_type}", "config": config_data}


@router.delete("/document-configs/{doc_type}")
async def delete_document_config(
    doc_type: str,
    tenant_id: str = Depends(get_current_tenant),
    user_email: str = Depends(get_current_user),
):
    db = get_db()
    result = db.document_configs.delete_one({
        "document_type": doc_type.lower(),
        "tenant_id": tenant_id.lower()
    })
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Configuration not found")
    return {"message": f"Deleted configuration for {doc_type}"}
