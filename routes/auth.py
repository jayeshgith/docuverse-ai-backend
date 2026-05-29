import secrets
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import BackgroundTasks
from pydantic import BaseModel, EmailStr

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


@router.post("/signup")
async def signup(body: SignupRequest):
    db = get_db()
    existing = db.users.find_one({"email": body.email.lower()})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = {
        "email": body.email.lower(),
        "name": body.name,
        "hashed_password": hash_password(body.password),
    }
    db.users.insert_one(user)
    token = create_access_token({"sub": body.email.lower()})
    return {"token": token, "user": {"email": user["email"], "name": user["name"]}}


@router.post("/login")
async def login(body: LoginRequest):
    db = get_db()
    user = db.users.find_one({"email": body.email.lower()})
    if not user or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": user["email"]})
    return {"token": token, "user": {"email": user["email"], "name": user["name"]}}


@router.get("/me")
async def get_me(email: str = Depends(get_current_user)):
    db = get_db()
    user = db.users.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"email": user["email"], "name": user["name"]}


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
