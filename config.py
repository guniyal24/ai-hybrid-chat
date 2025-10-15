# config.py
import os
from dotenv import load_dotenv

# Load variables from the .env file into the environment
load_dotenv()

# -- Neo4j Database Credentials --
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# -- OpenAI API Key --
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -- Pinecone API Key & Settings --
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# -- Model Configuration for MAXIMUM QUALITY --
# This setup is for the highest performance.
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o"

# Vector dimension for text-embedding-3-large MUST be 3072
PINECONE_VECTOR_DIM = 3072

# --- Sanity Checks ---
# Ensures the script fails if a required key is missing.
if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY, PINECONE_API_KEY]):
    raise ValueError("One or more required environment variables are not set. Please check your .env file.")