from pydantic import BaseModel, EmailStr
from typing import Optional

class UserInDB(BaseModel):
    email: str
    name: str
    hashed_password: str
    reset_token: Optional[str] = None
    reset_token_expires: Optional[str] = None

class UserResponse(BaseModel):
    email: str
    name: str
