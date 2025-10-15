# main.py
import logging
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from hybrid_chat import HybridRAG

# --- App Initialization ---
app = FastAPI(
    title="AI Hybrid Chat API",
    description="API for the hybrid RAG chatbot using Pinecone, Neo4j, and OpenAI.",
    version="1.0.0"
)
rag_system = HybridRAG()

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    query: str
    

# --- API Endpoints ---
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Receives a user's query and returns a streaming response from the RAG system.
    """
    try:
        return StreamingResponse(
            rag_system.get_answer(request.query),
            media_type="text/event-stream"
        )
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        return {"error": "An error occurred while processing your request."}

@app.on_event("shutdown")
def shutdown_event():
    """Gracefully close the Neo4j driver on shutdown."""
    logging.info("Closing Neo4j driver...")
    rag_system.close()