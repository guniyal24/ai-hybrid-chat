AI Hybrid Travel Chatbot 🚀
This repository contains the code for a full-stack, AI-powered travel assistant for Vietnam. It leverages a sophisticated Retrieval-Augmented Generation (RAG) architecture with a hybrid data backend, combining the strengths of semantic search and graph databases to provide intelligent, fact-based answers. The application features a real-time, interactive web interface.

## Features
Hybrid RAG Architecture: Combines Pinecone for fast semantic search (understanding the what) with Neo4j for structured graph traversal (understanding the how).

Real-Time Streaming: AI responses are streamed token-by-token to the user, providing an interactive, real-time experience.

Persistent Caching: Uses Redis to cache expensive OpenAI embedding calls, significantly improving speed and reducing API costs on repeated queries.

Interactive Web UI: A clean, modern chat interface built with React that includes conversation history.

Decoupled Full-Stack Design: A robust Python backend API built with FastAPI serves the AI logic, while a standalone JavaScript (React) frontend handles the user experience.

Production-Ready Backend: The backend is built with asynchronous capabilities, loads all secrets from a .env file, and includes structured logging and error handling.

## Architecture
The application is built on a decoupled, client-server architecture.

Frontend (React): A standalone web application that runs in the user's browser. It captures user input and communicates with the backend via HTTP requests.

Backend (FastAPI): A Python API server that orchestrates the entire RAG pipeline. It has no user interface and only exposes data endpoints.

Query Flow:

[User @ React UI] -> [HTTP Request] -> [FastAPI Backend]
                                             |
                                             V
                                     [RAG Pipeline]
                                     /            \
                       [Pinecone: Semantic Search]  [Neo4j: Graph Context]
                                     \            /
                                             |
                                             V
                                      [OpenAI LLM] -> [Streaming Response] -> [User @ React UI]
## Getting Started
Follow these steps to set up and run the project locally.

### Prerequisites
Python 3.11

Node.js and npm

Docker (for Redis databases)

Accounts:

OpenAI API Key

Pinecone API Key

### 1. Backend Setup 🐍
Clone the Repository and navigate to the project's root directory.

Create a Virtual Environment:

macOS/Linux: python3.11 -m venv venv

Windows: py -3.11 -m venv venv

Activate the Virtual Environment:

macOS/Linux: source venv/bin/activate

Windows: venv\Scripts\activate

Create Your .env File: Create a file named .env in the root directory. Copy the content from .env.example (if provided) or use the template below and fill in your credentials.

Ini, TOML

# .env file
OPENAI_API_KEY="sk-..."
PINECONE_API_KEY="..."
PINECONE_INDEX_NAME="vietnam-travel"
PINECONE_VECTOR_DIM=3072
NEO4J_URI="neo4j://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="your-neo4j-password"
NEO4J_DATABASE="ai-chat-db"
Install Python Dependencies:

Bash

pip install -r requirements.txt
Start Databases with Docker: Run these commands in your terminal to start Neo4j (with the APOC plugin) and Redis containers.

Bash

# Start Neo4j with APOC
docker run -d --name my-neo4j-db -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/your-neo4j-password -e NEO4J_PLUGINS='["apoc"]' neo4j:latest

# Start Redis
docker run -d --name my-redis-cache -p 6379:6379 redis
Note: Make sure the NEO4J_PASSWORD in your .env file matches the one you set in the docker command.

Load the Data: Run the two loading scripts one after the other.

Bash

# 1. Load data into Neo4j
python3.11 load_to_neo4j.py

# 2. Load data into Pinecone
python3.11 pinecone_upload.py
Start the Backend Server:

Bash

uvicorn main:app --reload
Your backend is now running at http://127.0.0.1:8000.

### 2. Frontend Setup ⚛️
Open a NEW Terminal Window.

Navigate to the Frontend Directory:

Bash

cd chat-frontend
Install JavaScript Dependencies:

Bash

npm install
Start the Frontend Server:

Bash

npm run dev
Your browser will automatically open to the chat application, typically at http://localhost:5173.

## Project Structure
.
├── Ai-Hybrid-Chat/
│   ├── chat-frontend/         # React Frontend Application
│   │   ├── src/
│   │   │   ├── App.jsx        # Main chat component
│   │   │   └── App.css        # Styling
│   │   └── package.json
│   │
│   ├── .env                   # All secrets and configuration
│   ├── .gitignore
│   ├── requirements.txt       # Python dependencies
│   ├── vietnam_travel_dataset.json # Source data
│   │
│   ├── load_to_neo4j.py       # Script to load data into Neo4j
│   ├── pinecone_upload.py     # Script to load data into Pinecone
│   ├── hybrid_chat.py         # Core RAG logic class
│   └── main.py                # FastAPI backend server
│
└── README.md
## Key Technologies Used
Backend: Python, FastAPI, Uvicorn

Frontend: JavaScript, React, Vite

Databases:

Neo4j: Graph database for storing structured, relational data.

Pinecone: Vector database for high-speed semantic search.

Redis: In-memory database for persistent caching.

AI Services:

OpenAI: For language models (gpt-4o-mini) and text embeddings (text-embedding-3-large).
