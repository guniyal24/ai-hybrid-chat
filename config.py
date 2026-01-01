# config.py
import os
from dotenv import load_dotenv

load_dotenv()


NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL = "gpt-4o"

PINECONE_VECTOR_DIM = 3072


if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY, PINECONE_API_KEY]):
    raise ValueError("One or more required environment variables are not set. Please check your .env file.")