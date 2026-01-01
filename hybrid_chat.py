import os
import logging
import json
from typing import List, Dict, Any, Generator
from openai import OpenAI
from pinecone import Pinecone
from neo4j import GraphDatabase
from dotenv import load_dotenv
import httpx
import redis
from flashrank import Ranker, RerankRequest 

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridRAG:
    def __init__(self):
        try:
            http_client = httpx.Client(trust_env=False)
            self.openai_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                http_client=http_client
            )
            self.embed_model = "text-embedding-3-large"
            self.router_model = "gpt-4o-mini"
            self.chat_model = "gpt-4o-mini"
            self.reasoning_model = "gpt-4o"

            self.ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2", cache_dir="./opt")
            logging.info("FlashRank Re-ranker initialized.")

            # --- Pinecone ---
            self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            self.index_name = os.getenv("PINECONE_INDEX_NAME")
            self._ensure_pinecone_index()
            self.pinecone_index = self.pinecone_client.Index(self.index_name)

            # --- Neo4j ---
            self.neo4j_driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI"),
                auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
                database=os.getenv("NEO4J_DATABASE", "neo4j") 
            )
            self.neo4j_driver.verify_connectivity()

            # --- Redis (Optional / Fail-Safe) ---
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=1)
                self.redis_client.ping()
                logging.info("Redis connection successful.")
            except (redis.ConnectionError, redis.TimeoutError):
                logging.warning("Redis down. Caching DISABLED.")
                self.redis_client = None

        except Exception as e:
            logging.critical(f"Init failed: {e}")
            raise

    def close(self):
        if self.neo4j_driver: self.neo4j_driver.close()

    def _ensure_pinecone_index(self):
        if self.index_name not in self.pinecone_client.list_indexes().names():
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=int(os.getenv("PINECONE_VECTOR_DIM", 3072)),
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )

    def create_user(self, email: str, password_hash: str):
        """Creates a new User node in Neo4j."""
        cypher = """
        MERGE (u:User {email: $email})
        ON CREATE SET u.password_hash = $ph, u.created_at = datetime()
        RETURN u
        """
        try:
            with self.neo4j_driver.session() as session:
                session.run(cypher, email=email, ph=password_hash)
            logging.info(f"User created: {email}")
        except Exception as e:
            logging.error(f"Failed to create user: {e}")
            raise

    def get_user(self, email: str) -> Dict[str, Any]:
        """Fetches a user and their password hash."""
        cypher = "MATCH (u:User {email: $email}) RETURN u.email as email, u.password_hash as password_hash"
        with self.neo4j_driver.session() as session:
            result = session.run(cypher, email=email).single()
            return result.data() if result else None

    def embed_text(self, text: str) -> List[float]:
        if not self.redis_client:
            return self.openai_client.embeddings.create(model=self.embed_model, input=[text]).data[0].embedding
        
        cache_key = f"embedding:{text}"
        if self.redis_client.get(cache_key):
            return json.loads(self.redis_client.get(cache_key))
        
        resp = self.openai_client.embeddings.create(model=self.embed_model, input=[text])
        emb = resp.data[0].embedding
        self.redis_client.setex(cache_key, 86400, json.dumps(emb))
        return emb

    def _classify_intent(self, query: str) -> str:
        try:
            resp = self.openai_client.chat.completions.create(
                model=self.router_model,
                messages=[
                    {"role": "system", "content": "Classify: 'general_chat' or 'travel_search'. Output label only."},
                    {"role": "user", "content": query}
                ], temperature=0.0, max_tokens=10
            )
            return resp.choices[0].message.content.strip().lower()
        except: return "travel_search"


    def log_interaction(self, session_id: str, query: str, response: str, retrieved_ids: List[str]):
        cypher = """
        MERGE (s:Session {id: $session_id})
        CREATE (l:Log {id: randomUUID(), query: $query, response: $response, timestamp: datetime()})
        CREATE (s)-[:HAS_LOG]->(l)
        WITH l UNWIND $node_ids AS nid
        MATCH (e:Entity {id: nid}) MERGE (l)-[:USED_CONTEXT]->(e)
        """
        try:
            with self.neo4j_driver.session() as session:
                session.run(cypher, session_id=session_id, query=query, response=response, node_ids=retrieved_ids)
        except Exception as e: logging.error(f"Log failed: {e}")

    def pinecone_query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        vec = self.embed_text(query_text)
        if not vec: return []
        try:
            return self.pinecone_index.query(vector=vec, top_k=top_k, include_metadata=True)["matches"]
        except: return []

    def fetch_graph_context(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        if not node_ids: return []
        cypher = """
        UNWIND $node_ids AS nid MATCH (n:Entity {id: nid})-[r]-(m:Entity)
        RETURN nid AS source_id, type(r) AS rel, m.id AS target_id, m.name AS name, 
               COALESCE(left(m.description, 200), "") AS desc LIMIT 25
        """
        with self.neo4j_driver.session() as session:
            return [record.data() for record in session.run(cypher, node_ids=node_ids)]

    def _get_search_summary(self, query: str, pinecone: list, graph: list) -> str:
        if not pinecone and not graph: return "No data found."
        context = "### Matches:\n"
        for m in pinecone:
            meta = m.get('metadata') or m.get('meta', {})
            context += f"- {meta.get('name')} ({meta.get('type')}): {meta.get('city', '')}\n"
            
        context += "\n### Connections:\n" + "\n".join([f"- {f['source_id']} --[{f['rel']}]--> {f['name']} ({f['desc']})" for f in graph])
        
        resp = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize data for user query."},
                {"role": "user", "content": f"Query: {query}\nData:\n{context}"}
            ]
        )
        return resp.choices[0].message.content

    def _handle_general_chat(self, query: str) -> Generator[str, None, None]:
        stream = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "system", "content": "You are a polite travel assistant."}, {"role": "user", "content": query}],
            stream=True
        )
        for chunk in stream: yield chunk.choices[0].delta.content or ""

    def _handle_travel_search(self, query: str, session_id: str, history: list) -> Generator[str, None, None]:
        candidates = self.pinecone_query(query, top_k=20)
        passages = [
            {"id": c["id"], "text": c["metadata"].get("semantic_text", "")[:500], "meta": c["metadata"]} 
            for c in candidates
        ]
        
        final_matches = []
        if passages:
            req = RerankRequest(query=query, passages=passages)
            final_matches = self.ranker.rerank(req)[:5]
            logging.info(f"Re-ranking: {len(candidates)} -> {len(final_matches)}")

        match_ids = [m["id"] for m in final_matches]
        graph_facts = self.fetch_graph_context(match_ids)
        
        summary = self._get_search_summary(query, final_matches, graph_facts)
        
        clean_history = []
        for msg in (history or []):
            role = "assistant" if msg.get("role") == "bot" else msg.get("role", "user")
            clean_history.append({"role": role, "content": msg.get("content", "")})
        
        msgs = [{"role": "system", "content": "You are VietBot. Answer using context. Think in <thinking> tags."}] + \
               clean_history + \
               [{"role": "user", "content": f"Context: {summary}\nQuery: {query}"}]
        
        stream = self.openai_client.chat.completions.create(
            model=self.reasoning_model, messages=msgs, stream=True
        )
        
        full_response = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            full_response += content
            yield content
            
        self.log_interaction(session_id, query, full_response, match_ids)

    def get_answer(self, query: str, session_id: str, history: list = None) -> Generator[str, None, None]:
        if "general" in self._classify_intent(query):
            yield from self._handle_general_chat(query)
        else:
            yield from self._handle_travel_search(query, session_id, history or [])

if __name__ == "__main__":
    rag = HybridRAG()
    print("Bot Ready. Type exit.")
    while True:
        q = input("> ")
        if q == "exit": break
        for token in rag.get_answer(q, "console-test"): print(token, end="", flush=True)
        print()