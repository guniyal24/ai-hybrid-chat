import os
from datetime import datetime, timedelta
from typing import Optional
from jose import jwt, JWTError
import bcrypt
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def verify_password(plain_password: str, hashed_password: str) -> bool:
    pw_bytes = plain_password.encode("utf-8")
    
    if len(pw_bytes) > 72:
        pw_bytes = pw_bytes[:72]
    return bcrypt.checkpw(pw_bytes, hashed_password.encode("utf-8"))

def get_password_hash(password: str) -> str:

    pw_bytes = password.encode("utf-8")
    
    if len(pw_bytes) > 72:
        pw_bytes = pw_bytes[:72]
    
    salt = bcrypt.gensalt()
    hashed_bytes = bcrypt.hashpw(pw_bytes, salt)

    return hashed_bytes.decode("utf-8")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

