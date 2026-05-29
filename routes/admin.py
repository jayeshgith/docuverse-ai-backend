from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from services.database import get_db
from routes.auth import get_current_user

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
    tenant_id: str = "default"


@router.post("/document-configs")
async def save_document_config(body: DocumentConfigSchema, user_email: str = Depends(get_current_user)):
    db = get_db()
    
    # Check if a config already exists for this document_type and tenant_id
    existing = db.document_configs.find_one({
        "document_type": body.document_type.lower(),
        "tenant_id": body.tenant_id.lower()
    })
    
    config_data = {
        "document_type": body.document_type.lower(),
        "display_name": body.display_name,
        "fields": [f.dict() for f in body.fields],
        "tenant_id": body.tenant_id.lower()
    }
    
    if existing:
        db.document_configs.update_one(
            {"_id": existing["_id"]},
            {"$set": config_data}
        )
        return {"message": f"Updated config for {body.document_type}", "config": config_data}
    else:
        db.document_configs.insert_one(config_data)
        # Remove MongoDB object ID before returning
        if "_id" in config_data:
            del config_data["_id"]
        return {"message": f"Created config for {body.document_type}", "config": config_data}


@router.get("/document-configs")
async def list_document_configs(tenant_id: str = "default", user_email: str = Depends(get_current_user)):
    db = get_db()
    cursor = db.document_configs.find({"tenant_id": tenant_id.lower()})
    configs = []
    for c in cursor:
        c["_id"] = str(c["_id"])
        configs.append(c)
    return configs


@router.delete("/document-configs/{doc_type}")
async def delete_document_config(doc_type: str, tenant_id: str = "default", user_email: str = Depends(get_current_user)):
    db = get_db()
    result = db.document_configs.delete_one({
        "document_type": doc_type.lower(),
        "tenant_id": tenant_id.lower()
    })
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Configuration not found")
    return {"message": f"Deleted configuration for {doc_type}"}
