import os
import json
import logging
import time
from tqdm import tqdm
from openai import OpenAI, APIError
from pinecone import Pinecone, ServerlessSpec, PineconeException
from dotenv import load_dotenv
import httpx
# --- Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PineconeUploader:
    """Handles creating an index and uploading vectorized data to Pinecone."""

    def __init__(self, data_file="vietnam_travel_dataset.json"):
        self.data_file = data_file
        self.batch_size = 32
        self.embed_model = "text-embedding-3-large"

        try:
            # Initialize clients from environment variables
            http_client = httpx.Client(trust_env=False)
            self.openai_client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    http_client=http_client)
            self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            
            self.index_name = os.getenv("PINECONE_INDEX_NAME")
            self.vector_dim = int(os.getenv("PINECONE_VECTOR_DIM", 3072))
            
            self._ensure_index_exists()
            self.pinecone_index = self.pinecone_client.Index(self.index_name)
            logging.info("Clients initialized and connected to Pinecone index successfully.")
            
        except (APIError, PineconeException, TypeError) as e:
            logging.error(f"Failed to initialize clients: {e}")
            raise

    def _ensure_index_exists(self):
        """Checks if the Pinecone index exists and creates it if not."""
        existing_indexes = self.pinecone_client.list_indexes().names()
        if self.index_name not in existing_indexes:
            logging.warning(f"Index '{self.index_name}' not found. Creating a new one.")
            try:
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=self.vector_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                # Wait a moment for the index to be ready
                time.sleep(5)
                logging.info(f"Index '{self.index_name}' created successfully.")
            except PineconeException as e:
                logging.error(f"Failed to create Pinecone index: {e}")
                raise
        else:
            logging.info(f"Index '{self.index_name}' already exists.")

    def _get_embeddings(self, texts):
        """Generate embeddings for a list of texts."""
        try:
            resp = self.openai_client.embeddings.create(model=self.embed_model, input=texts)
            return [data.embedding for data in resp.data]
        except APIError as e:
            logging.error(f"OpenAI API error during embedding: {e}")
            return []

    def _prepare_data(self):
        """Loads data from JSON and prepares it for Pinecone."""
        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                nodes = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Could not read or parse data file {self.data_file}: {e}")
            return []

        items_to_upload = []
        for node in nodes:
            semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
            if not semantic_text.strip():
                continue
            
            metadata = {
                "id": node.get("id"),
                "type": node.get("type"),
                "name": node.get("name"),
                "city": node.get("city", node.get("region", "")),
                "tags": node.get("tags", [])
            }
            items_to_upload.append((node["id"], semantic_text, metadata))
        return items_to_upload

    def run(self):
        """Executes the full upload pipeline."""
        items = self._prepare_data()
        if not items:
            logging.warning("No valid items to upload. Exiting.")
            return

        logging.info(f"Preparing to upsert {len(items)} items to Pinecone...")
        
        def chunked(iterable, n):
            for i in range(0, len(iterable), n):
                yield iterable[i:i+n]

        for batch in tqdm(list(chunked(items, self.batch_size)), desc="Uploading batches"):
            ids = [item[0] for item in batch]
            texts = [item[1] for item in batch]
            metadatas = [item[2] for item in batch]

            embeddings = self._get_embeddings(texts)
            if not embeddings:
                logging.error(f"Skipping batch starting with ID '{ids[0]}' due to embedding failure.")
                continue

            vectors_to_upsert = [
                {"id": _id, "values": emb, "metadata": meta}
                for _id, emb, meta in zip(ids, embeddings, metadatas)
            ]
            
            try:
                self.pinecone_index.upsert(vectors_to_upsert)
            except PineconeException as e:
                logging.error(f"Failed to upsert batch starting with ID '{ids[0]}': {e}")
        
        logging.info("All items processed successfully.")

if __name__ == "__main__":
    try:
        uploader = PineconeUploader()
        uploader.run()
    except Exception as e:
        logging.critical(f"A critical error occurred: {e}")