# 🌏 AI Hybrid Travel Chatbot for Vietnam

An intelligent, full-stack travel assistant powered by advanced AI and a hybrid RAG (Retrieval-Augmented Generation) architecture. This chatbot combines semantic search with graph database traversal to deliver accurate, context-aware travel recommendations for Vietnam in real-time.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![React](https://img.shields.io/badge/React-18+-61DAFB)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- **🔍 Hybrid RAG Architecture**: Combines Pinecone's semantic search (understanding *what* users ask) with Neo4j's graph traversal (understanding *how* information connects) for superior context retrieval
- **⚡ Real-Time Streaming**: AI responses stream token-by-token for an interactive, conversational experience
- **💾 Smart Caching**: Redis-powered persistent caching drastically reduces API costs and improves response times for repeated queries
- **🎨 Modern Web Interface**: Clean, responsive React UI with full conversation history
- **🏗️ Production-Ready Backend**: Async FastAPI server with structured logging, comprehensive error handling, and environment-based configuration
- **🔄 Decoupled Architecture**: Independent frontend and backend services for maximum flexibility and scalability

## 🏛️ Architecture

```
┌─────────────────┐
│   React UI      │  ← User Interface
│  (Port 5173)    │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│  FastAPI Server │  ← Backend API
│  (Port 8000)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│      RAG Pipeline           │
│  ┌──────────────────────┐   │
│  │  Pinecone (Vectors)  │   │  ← Semantic Search
│  └──────────────────────┘   │
│  ┌──────────────────────┐   │
│  │   Neo4j (Graph DB)   │   │  ← Structured Context
│  └──────────────────────┘   │
│  ┌──────────────────────┐   │
│  │   Redis (Cache)      │   │  ← Embedding Cache
│  └──────────────────────┘   │
└─────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   OpenAI API    │  ← LLM & Embeddings
└─────────────────┘
```

**Query Flow:**
1. User sends message from React UI
2. FastAPI backend receives request
3. RAG pipeline retrieves context from Pinecone (semantic) and Neo4j (structured)
4. OpenAI generates streaming response with enriched context
5. Response streams back to user in real-time

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:

- **Python 3.11+**
- **Node.js 16+** and npm
- **Docker** and Docker Compose
- **API Keys:**
  - [OpenAI API Key](https://platform.openai.com/)
  - [Pinecone API Key](https://www.pinecone.io/)

### 1️⃣ Backend Setup

#### Clone and Navigate
```bash
git clone <your-repository-url>
cd Ai-Hybrid-Chat
```

#### Create Virtual Environment
```bash
# macOS/Linux
python3.11 -m venv venv
source venv/bin/activate

# Windows
py -3.11 -m venv venv
venv\Scripts\activate
```

#### Configure Environment Variables
Create a `.env` file in the root directory:

```ini
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key-here

# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX_NAME=vietnam-travel
PINECONE_VECTOR_DIM=3072

# Neo4j Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-password
NEO4J_DATABASE=ai-chat-db

# Redis Configuration (optional - uses defaults)
REDIS_HOST=localhost
REDIS_PORT=6379
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Start Database Services
```bash
# Start Neo4j with APOC plugin
docker run -d \
  --name my-neo4j-db \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-secure-password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:latest

# Start Redis cache
docker run -d \
  --name my-redis-cache \
  -p 6379:6379 \
  redis:latest
```

**Note:** Ensure the `NEO4J_PASSWORD` in your `.env` matches the Docker command.

#### Load Data into Databases
```bash
# 1. Populate Neo4j graph database
python3.11 load_to_neo4j.py

# 2. Upload embeddings to Pinecone
python3.11 pinecone_upload.py
```

#### Start Backend Server
```bash
uvicorn main:app --reload
```

Backend now running at **http://127.0.0.1:8000** 🎉

API documentation available at **http://127.0.0.1:8000/docs**

### 2️⃣ Frontend Setup

#### Navigate to Frontend Directory
Open a **new terminal** and run:

```bash
cd chat-frontend
```

#### Install Dependencies
```bash
npm install
```

#### Start Development Server
```bash
npm run dev
```

Frontend now running at **http://localhost:5173** 🎉

Your browser should automatically open the chat interface!

## 📁 Project Structure

```
Ai-Hybrid-Chat/
├── chat-frontend/              # React frontend application
│   ├── src/
│   │   ├── App.jsx            # Main chat component
│   │   ├── App.css            # UI styling
│   │   └── main.jsx           # React entry point
│   ├── package.json
│   └── vite.config.js
│
├── .env                        # Environment variables (DO NOT COMMIT)
├── .env.example               # Template for environment variables
├── .gitignore
├── requirements.txt           # Python dependencies
│
├── vietnam_travel_dataset.json # Source travel data
│
├── load_to_neo4j.py           # Neo4j data loader script
├── pinecone_upload.py         # Pinecone embedding uploader
├── hybrid_chat.py             # Core RAG pipeline logic
├── main.py                    # FastAPI backend server
│
└── README.md                  # You are here!
```

## 🛠️ Tech Stack

### Backend
- **Python 3.11** - Core language
- **FastAPI** - Modern async web framework
- **Uvicorn** - Lightning-fast ASGI server
- **Pydantic** - Data validation and settings management

### Frontend
- **React 18** - UI library
- **Vite** - Next-generation build tool
- **JavaScript (ES6+)** - Modern web development

### Databases
- **Neo4j** - Graph database for relational travel data
- **Pinecone** - Vector database for semantic similarity search
- **Redis** - In-memory cache for embeddings

### AI Services
- **OpenAI GPT-4o-mini** - Language model for response generation
- **OpenAI text-embedding-3-large** - High-quality text embedding
