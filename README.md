# 🇻🇳 VietBot - AI-Powered Vietnam Travel Assistant

<div align="center">

**An advanced AI travel assistant specialized for Vietnam tourism, powered by Hybrid RAG architecture**

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-008CC1?style=for-the-badge&logo=neo4j&logoColor=white)](https://neo4j.com/)
[![Pinecone](https://img.shields.io/badge/Pinecone-000000?style=for-the-badge&logo=pinecone&logoColor=white)](https://www.pinecone.io/)

</div>

---

## 🌟 Overview

VietBot is not your average travel chatbot. Unlike standard chatbots that rely solely on text generation, VietBot uses a **Hybrid RAG (Retrieval-Augmented Generation)** architecture that combines:

- 🔍 **Fuzzy semantic understanding** via Vector Databases
- 🕸️ **Structured factual connections** via Graph Databases
- 🧠 **Intelligent reasoning** via GPT-4o

This powerful combination enables VietBot to provide **accurate, context-aware travel itineraries and recommendations** for Vietnam tourism.

---

## 🏗️ System Architecture

VietBot follows a **microservices-lite pattern** with clear separation between Client (Frontend) and Server (Backend).

### High-Level Data Flow

```
User Query
    ↓
┌─────────────────────────────────────────────────┐
│  1. Auth Layer (JWT + Redis Rate Limits)        │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  2. Router Agent (Classify Intent)              │
│     • "General Chat" vs "Travel Search"         │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  3. Hybrid RAG Engine                           │
│     ┌──────────────────┬─────────────────────┐ │
│     │ Vector Search    │ Graph Search        │ │
│     │ (Pinecone)       │ (Neo4j)             │ │
│     │ • Find similar   │ • Find connected    │ │
│     │   places         │   facts             │ │
│     └──────────────────┴─────────────────────┘ │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  4. Re-Ranking (FlashRank)                      │
│     • Filter top results for relevance          │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  5. Reasoning Engine (GPT-4o)                   │
│     • Synthesize data into logical itinerary    │
└─────────────────────────────────────────────────┘
    ↓
Response → Streamed back to Frontend
```

---

## 🔑 Key Architectural Features

### 1️⃣ Hybrid RAG (Graph + Vector)

**The Problem**: Vector search can be imprecise for specific facts. For example, asking for "hotels in Da Nang" might return a hotel in a similar city just because the description matches.

**The Solution**:
- Query **Pinecone** for semantic match ("luxury beach resort")
- Cross-reference with **Neo4j** to ensure location relationships are factually correct
- This ensures both semantic relevance AND factual accuracy

### 2️⃣ Re-Ranking Pipeline

**Process**: 
1. Retrieve **20 candidates** from Pinecone (Broad Search)
2. Re-rank using **FlashRank** 
3. Keep top **5** results (Precision)

**Why?** Vector search relies on cosine similarity, which isn't always contextually perfect. The re-ranker acts as a "second opinion" to double-check relevance before sending data to GPT, saving token costs.

### 3️⃣ Fail-Safe Rate Limiting

**Feature**: Users are limited to **10 requests/minute**

**Resilience**: Implemented with a `try-except` block around Redis. If Redis crashes:
- The app **fails open** (disables rate limiting)
- Chat keeps working rather than crashing the entire backend

### 4️⃣ Secure Authentication

**Flow**: 
```
User Register → Password Hashed (Bcrypt) → User Login → JWT Token Issued
```

**Protection**: 
- Every chat request requires a valid Bearer Token header
- Fixed 72-byte Bcrypt limitation by migrating from `passlib` to raw `bcrypt`
- Enforces security without compromising functionality

---

## 💻 Technology Stack

### Backend (Python & FastAPI)

| Library | Purpose | Justification |
|---------|---------|---------------|
| **fastapi** | Web Framework | High performance, native async support for AI streaming |
| **uvicorn** | ASGI Server | Lightning-fast server implementation to run FastAPI |
| **pydantic** | Data Validation | Ensures data integrity (e.g., valid email format) before it hits logic |
| **python-jose** | JWT Security | Handles encoding/decoding JSON Web Tokens for secure stateless auth |
| **bcrypt** | Password Hashing | Industry standard for hashing passwords. *Note: Migrated from passlib to raw bcrypt to fix a 72-byte limit crash* |
| **redis** | Caching & Rate Limiting | In-memory store used to prevent API abuse and cache embeddings for speed |

### The AI Engine (The "Brain")

| Component | Model/Library | Justification |
|-----------|--------------|---------------|
| **LLM Orchestrator** | openai (GPT-4o) | GPT-4o-mini for fast routing, GPT-4o for complex reasoning (itinerary planning) |
| **Embeddings** | text-embedding-3-large | OpenAI's latest model. "Large" chosen for better semantic nuance in travel queries |
| **Vector DB** | pinecone-client | Serverless vector database. Chosen for speed and ease of filtering metadata |
| **Graph DB** | neo4j | Stores relationships (City → HAS_HOTEL → Hotel). Vectors find "similar" things; Graphs find "connected" things |
| **Re-Ranker** | flashrank | Lightweight, local re-ranking library. Instead of calling a paid API (like Cohere), runs a tiny BERT model locally to filter results essentially for free |

### Frontend (React & Vite)

| Library | Purpose | Justification |
|---------|---------|---------------|
| **vite** | Build Tool | Significantly faster startup and hot-reload times compared to Create-React-App |
| **react-markdown** | Text Rendering | Allows the bot to output Bold, Lists, and Itineraries cleanly instead of raw text |
| **App.css** | Styling | Custom CSS implementing **Glassmorphism** (frosted glass) and gradients for a modern aesthetic |

---

## 🚀 Features

✅ **Hybrid RAG Architecture** - Combines vector and graph databases for accurate, contextually relevant responses  
✅ **Intelligent Intent Classification** - Automatically routes between general chat and travel search  
✅ **Multi-Stage Re-Ranking** - Ensures only the most relevant results reach the LLM  
✅ **Real-Time Streaming** - Get responses as they're generated, not after  
✅ **Rate Limiting & Abuse Prevention** - Redis-backed protection against API abuse  
✅ **Secure JWT Authentication** - Industry-standard token-based authentication  
✅ **Modern UI with Glassmorphism** - Beautiful frosted glass effects and gradients  
✅ **Fail-Safe Error Handling** - Graceful degradation when services are unavailable  

---

## 📦 Installation & Setup

### Prerequisites

- Python 3.9+
- Node.js 16+
- Redis Server
- Neo4j Database
- OpenAI API Key
- Pinecone API Key

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/vietbot.git
cd vietbot/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
REDIS_HOST=localhost
REDIS_PORT=6379
JWT_SECRET_KEY=your_secret_key
EOF

# Run the server
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Create .env file
echo "VITE_API_URL=http://localhost:8000" > .env

# Run development server
npm run dev
```

### Database Setup

**Neo4j Graph Database:**
```cypher
// Create sample hotel node
CREATE (h:Hotel {
  name: "Luxury Beach Resort",
  city: "Da Nang",
  rating: 4.5,
  price_range: "$$$"
})

// Create city node
CREATE (c:City {name: "Da Nang"})

// Create relationship
CREATE (c)-[:HAS_HOTEL]->(h)
```

**Pinecone Vector Database:**
```python
# Initialize and populate with embeddings
# See backend/scripts/populate_pinecone.py
```

---

## 📖 Usage

1. **Register/Login**: Create an account or login to receive your JWT token
2. **Start Chatting**: Ask questions like:
   - "What are the best beaches in Da Nang?"
   - "Plan a 3-day itinerary for Hanoi"
   - "Find luxury hotels in Ho Chi Minh City"
3. **Get Recommendations**: VietBot will search both vector and graph databases to provide accurate, contextual recommendations
4. **Enjoy Streaming Responses**: Watch as the AI generates your personalized travel plan in real-time

---

## 🔐 API Endpoints

### Authentication
- `POST /register` - Register new user
- `POST /login` - Login and receive JWT token

### Chat
- `POST /chat` - Send message (requires Bearer token)
- `GET /chat/history` - Retrieve chat history

### Health
- `GET /health` - Check API status

---

## 🎯 Future Enhancements

- [ ] Multi-language support (Vietnamese, English, French)
- [ ] Image recognition for landmark identification
- [ ] Integration with booking APIs
- [ ] User preference learning and personalization
- [ ] Mobile app (React Native)
- [ ] Voice interface support

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

---

## 🙏 Acknowledgments

- OpenAI for GPT-4o and embeddings
- Pinecone for vector database services
- Neo4j for graph database technology
- FastAPI community for excellent documentation
- All contributors and testers

---

<div align="center">

**Made with ❤️ for Vietnam Tourism**

⭐ Star this repo if you find it helpful!

</div>
