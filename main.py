import logging
import uuid
import redis
from fastapi import FastAPI, Header, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from jose import JWTError, jwt

from hybrid_chat import HybridRAG
import auth_utils

app = FastAPI(title="VietBot AI Backend (Secured)", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system = HybridRAG()


try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=1)
    redis_client.ping() 
    logging.info("✅ Redis connected for Rate Limiting.")
except (redis.ConnectionError, redis.TimeoutError):
    logging.warning("⚠️ Redis is DOWN. Rate Limiting is DISABLED (App will still work).")
    redis_client = None


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

# --- Pydantic Models ---
class UserRegister(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = []

class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    rating: int 
    comment: Optional[str] = None


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, auth_utils.SECRET_KEY, algorithms=[auth_utils.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = rag_system.get_user(email)
    if user is None:
        raise credentials_exception
    return user

async def check_rate_limit(user: dict = Depends(get_current_user)):
    """
    Rate Limiter: 10 requests per minute per user.
    Skipped if Redis is down.
    """
    if redis_client is None:
        return user 
    
    try:
        user_email = user['email']
        key = f"rate_limit:{user_email}"
        
        request_count = redis_client.incr(key)
        
        if request_count == 1:
            redis_client.expire(key, 60)
            
        if request_count > 10:
            raise HTTPException(status_code=429, detail="Rate limit exceeded (10 req/min).")
    except Exception as e:
        logging.error(f"Redis failed during request: {e}")
    
    return user



@app.post("/api/register")
async def register(user: UserRegister):
    if rag_system.get_user(user.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pw = auth_utils.get_password_hash(user.password)
    rag_system.create_user(user.email, hashed_pw)
    return {"message": "User created successfully"}

@app.post("/api/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = rag_system.get_user(form_data.username)
    if not user or not auth_utils.verify_password(form_data.password, user['password_hash']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = auth_utils.create_access_token(data={"sub": user['email']})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/api/chat")
async def chat_endpoint(
    request: ChatRequest, 
    x_session_id: Optional[str] = Header(None),
    current_user: dict = Depends(check_rate_limit) 
):
    session_id = x_session_id or str(uuid.uuid4())
    logging.info(f"User: {current_user['email']} | Session: {session_id} | Query: {request.query}")

    def stream_response():
        yield from rag_system.get_answer(request.query, session_id, request.history)

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"X-Session-ID": session_id}
    )

@app.post("/api/feedback")
async def feedback_endpoint(
    request: FeedbackRequest, 
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    def save_feedback(sid, q, r, c):
        query = """
        MATCH (s:Session {id: $sid})-[:HAS_LOG]->(l:Log {query: $q})
        SET l.rating = $r, l.comment = $c
        """
        try:
            with rag_system.neo4j_driver.session() as session:
                session.run(query, sid=sid, q=q, r=r, c=c)
        except Exception as e:
            logging.error(f"Feedback error: {e}")

    background_tasks.add_task(
        save_feedback, request.session_id, request.query, request.rating, request.comment
    )
    return {"status": "received"}

@app.on_event("shutdown")
def shutdown_event():
    rag_system.close()