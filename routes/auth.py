import secrets
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import BackgroundTasks
from pydantic import BaseModel

from services.auth_service import hash_password, verify_password, create_access_token, decode_access_token
from services.database import get_db
from services.email_service import send_reset_email

router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)


class SignupRequest(BaseModel):
    email: str
    name: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class ProfileUpdateRequest(BaseModel):
    name: str | None = None
    email: str | None = None
    avatar_url: str | None = None

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    password: str


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload["sub"]


def get_current_admin(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return payload["sub"]


def get_current_tenant(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload.get("tenant_id", "default")


@router.post("/signup")
async def signup(body: SignupRequest):
    db = get_db()
    existing = db.users.find_one({"email": body.email.lower()})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    tenant_id = body.email.lower().replace("@", "_at_").replace(".", "_dot_")

    is_first = db.users.count_documents({}) == 0
    role = "admin" if is_first else "user"

    user = {
        "email": body.email.lower(),
        "name": body.name,
        "hashed_password": hash_password(body.password),
        "tenant_id": tenant_id,
        "role": role,
    }
    db.users.insert_one(user)
    token = create_access_token({"sub": body.email.lower(), "tenant_id": tenant_id, "role": role})
    return {"token": token, "user": {"email": user["email"], "name": user["name"], "role": role}, "tenant_id": tenant_id}


@router.post("/login")
async def login(body: LoginRequest):
    db = get_db()
    user = db.users.find_one({"email": body.email.lower()})
    if not user or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    tenant_id = user.get("tenant_id", "default")
    role = user.get("role", "user")
    token = create_access_token({"sub": user["email"], "tenant_id": tenant_id, "role": role})
    return {"token": token, "user": {"email": user["email"], "name": user["name"], "role": role}, "tenant_id": tenant_id}


@router.get("/me")
async def get_me(email: str = Depends(get_current_user)):
    db = get_db()
    user = db.users.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"email": user["email"], "name": user["name"], "role": user.get("role", "user"), "tenant_id": user.get("tenant_id", "default")}


@router.put("/profile")
async def update_profile(body: ProfileUpdateRequest, email: str = Depends(get_current_user)):
    db = get_db()
    user = db.users.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    update = {}
    if body.name is not None:
        update["name"] = body.name.strip()
    if body.avatar_url is not None:
        update["avatar_url"] = body.avatar_url

    if update:
        db.users.update_one({"email": email}, {"$set": update})

    updated = db.users.find_one({"email": email})
    token = create_access_token({
        "sub": updated["email"],
        "tenant_id": updated.get("tenant_id", "default"),
        "role": updated.get("role", "user"),
    })
    return {
        "token": token,
        "user": {
            "email": updated["email"],
            "name": updated.get("name", ""),
            "role": updated.get("role", "user"),
            "tenant_id": updated.get("tenant_id", "default"),
            "avatar_url": updated.get("avatar_url", ""),
        },
    }


@router.post("/forgot-password")
async def forgot_password(body: ForgotPasswordRequest, bg: BackgroundTasks):
    db = get_db()
    user = db.users.find_one({"email": body.email.lower()})
    token = secrets.token_urlsafe(32)
    expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    if user:
        db.users.update_one(
            {"_id": user["_id"]},
            {"$set": {"reset_token": token, "reset_token_expires": expires}},
        )
    bg.add_task(send_reset_email, body.email, token)
    return {"message": "If an account exists, a reset link has been sent"}


@router.post("/reset-password")
async def reset_password(body: ResetPasswordRequest):
    db = get_db()
    user = db.users.find_one({"reset_token": body.token})
    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    expires = user.get("reset_token_expires")
    if expires and datetime.now(timezone.utc) > datetime.fromisoformat(expires):
        raise HTTPException(status_code=400, detail="Token expired")
    db.users.update_one(
        {"_id": user["_id"]},
        {
            "$set": {"hashed_password": hash_password(body.password)},
            "$unset": {"reset_token": "", "reset_token_expires": ""},
        },
    )
    return {"message": "Password reset successful"}
