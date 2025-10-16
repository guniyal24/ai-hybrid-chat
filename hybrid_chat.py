import os
import logging
from typing import List, Dict, Any , Generator
from openai import OpenAI, APIError
from pinecone import Pinecone, PineconeException
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from dotenv import load_dotenv
import httpx
import redis
import json

# --- 1. Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HybridRAG:
    """A class for a hybrid RAG system using OpenAI, Pinecone, and Neo4j."""

    def __init__(self):
        try:
            # OpenAI
            http_client = httpx.Client(trust_env=False)
            self.openai_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                http_client=http_client
            )
            self.embed_model = "text-embedding-3-large"
            self.chat_model = "gpt-4o-mini"

            # Pinecone
            self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            self.index_name = os.getenv("PINECONE_INDEX_NAME")
            self._ensure_pinecone_index()
            self.pinecone_index = self.pinecone_client.Index(self.index_name)

            # Neo4j
            self.neo4j_driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI"),
                auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
                database=os.getenv("NEO4J_DATABASE")
            )
            self.neo4j_driver.verify_connectivity()

            # ---Connect to Redis ---
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_client.ping()
            logging.info("Redis connection successful.")

            logging.info("All clients initialized successfully.")

        except redis.exceptions.ConnectionError as e:
            logging.error(f"Could not connect to Redis: {e}. Caching will be disabled.")
            self.redis_client = None
        except (APIError, PineconeException, Neo4jError, TypeError) as e:
            logging.error(f"Failed to initialize clients: {e}")
            raise

    def _ensure_pinecone_index(self):
        """Checks if the Pinecone index exists and creates it if not."""
        if self.index_name not in self.pinecone_client.list_indexes().names():
            logging.warning(f"Index '{self.index_name}' not found. Creating a new one.")
            self.pinecone_client.create_index(
                name=self.index_name,
                # MODIFIED: Default value updated to match the large model
                dimension=int(os.getenv("PINECONE_VECTOR_DIM", 3072)),
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
            logging.info(f"Index '{self.index_name}' created successfully.")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a text string.
        First, checks Redis cache. If not found, calls API and stores result in Redis.
        """
        # Fallback to no caching if Redis is not available
        if not self.redis_client:
            resp = self.openai_client.embeddings.create(model=self.embed_model, input=[text])
            return resp.data[0].embedding

        cache_key = f"embedding:{text}"
        try:
            # 1. Check the cache first
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                logging.info("Embedding cache HIT from Redis.")
                return json.loads(cached_result)

            # 2. If not in cache (a "miss"), call the API
            logging.info("Embedding cache MISS. Calling OpenAI API.")
            resp = self.openai_client.embeddings.create(model=self.embed_model, input=[text])
            embedding = resp.data[0].embedding

            # 3. Store the new result in Redis for next time (expires in 24 hours)
            self.redis_client.setex(cache_key, 86400, json.dumps(embedding))
            return embedding
        except APIError as e:
            logging.error(f"OpenAI API error during embedding: {e}")
            return []
        except redis.exceptions.RedisError as e:
            logging.error(f"Redis error during caching: {e}. Falling back to API call without caching.")
            resp = self.openai_client.embeddings.create(model=self.embed_model, input=[text])
            return resp.data[0].embedding
        
    def pinecone_query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Query Pinecone index."""
        try:
            vec = self.embed_text(query_text)
            if not vec:
                return []
            res = self.pinecone_index.query(
                vector=vec, top_k=top_k, include_metadata=True
            )
            logging.info(f"Pinecone found {len(res['matches'])} matches.")
            return res["matches"]
        except PineconeException as e:
            logging.error(f"Pinecone query failed: {e}")
            return []

    def fetch_graph_context(self, node_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch neighboring nodes from Neo4j in a single efficient query."""
        if not node_ids:
            return []
        cypher_query = """
        UNWIND $node_ids AS nid
        MATCH (n:Entity {id: nid})-[r]-(m:Entity)
        RETURN
          nid AS source_id,
          type(r) AS relation,
          m.id AS target_id,
          m.name AS target_name,
          COALESCE(left(m.description, 300), "No description available.") AS target_desc,
          labels(m) AS target_labels
        LIMIT 20
        """
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(cypher_query, node_ids=node_ids)
                facts = [record.data() for record in result]
                logging.info(f"Neo4j found {len(facts)} graph facts.")
                return facts
        except Neo4jError as e:
            logging.error(f"Neo4j query failed: {e}")
            return []

    def _get_search_summary(self, query: str, pinecone_matches: list, graph_facts: list) -> str:
        """Summarizes retrieved context using an LLM call."""
        if not pinecone_matches and not graph_facts:
            return "No relevant information was found in the knowledge base."

        # Combine the retrieved info into a single text block
        summary_context = "### Pinecone Semantic Search Results:\n"
        for match in pinecone_matches:
            meta = match.get('metadata', {})
            summary_context += f"- ID: {meta.get('id', 'N/A')}, Name: {meta.get('name', 'N/A')}, Type: {meta.get('type', 'N/A')}\n"
        
        summary_context += "\n### Neo4j Graph Facts:\n"
        for fact in graph_facts:
            summary_context += f"- The entity `{fact['source_id']}` has a `{fact['relation']}` relation with `{fact['target_name']}` (`{fact['target_id']}`).\n"
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a highly skilled summarization assistant. Your task is to synthesize the provided search results into a concise, single paragraph. Focus only on the information that is most relevant to answering the user's query."},
                    {"role": "user", "content": f"Please summarize the following information in the context of this user query: '{query}'\n\n### Search Results:\n{summary_context}"}
                ],
                temperature=0.0
            )
            summary = response.choices[0].message.content
            logging.info(f"Generated search summary: {summary}")
            return summary
        except APIError as e:
            logging.error(f"Failed to generate search summary: {e}")
            return "Error: Could not generate a summary of the search results."

    def build_prompt_with_summary(self, user_query: str, summary: str, history: list) -> list:
        """Builds the final prompt using the pre-generated summary."""
        system_content = system_content = system_content = """You are 'VietBot', a meticulous and expert AI travel planner for Vietnam. Your task is to follow a strict reasoning process to provide the most accurate and helpful answer possible, based **exclusively** on the provided context.

### Step 1: Internal Thought Process

First, you must reason through the user's request by completing the following steps inside `<thinking>` tags. This is your private workspace.

<thinking>
**1. User's Goal:** [Identify the user's primary intent in one clear sentence.]
**2. Key Information & Constraints:** [Extract all key entities, locations, durations, preferences (e.g., 'romantic'), or other constraints from the user's query.]
**3. Context Analysis:** [Analyze the provided 'CONTEXT SUMMARY'. List the specific pieces of information that directly address the user's goal and constraints.]
**4. Sufficiency Check:** [Based on the analysis, explicitly state whether the context is sufficient to fully answer the query. If not, identify exactly what information is missing.]
**5. Plan:** [Based on the available information, outline a clear, step-by-step plan for constructing the final answer.]
</thinking>

### Step 2: Final Answer to the User

After your thought process, provide the final answer to the user inside `<answer>` tags.

### Rules for the Final Answer:
-   The answer must be based **only** on your 'Plan' from the `<thinking>` block.
-   If the context was insufficient, state clearly what you can answer and what information you couldn't find. **Do not make up information.**
-   Do not mention your thought process or the context summary in the final answer. Speak directly to the user.
-   Format the answer for clarity using Markdown (lists, bold text).
-   Cite sources by including the node ID in parentheses, like `Hoi An (town_hoi_an)`.
-   Maintain a friendly, expert tone.
"""

        context_str = f"## CONTEXT SUMMARY\n\n{summary}"
        user_content = f"{context_str}\n\n## QUERY\n\n{user_query}"

        # Combine system message, conversation history, and the final user query with context
        full_prompt = [{"role": "system", "content": system_content}]
        if history:
            full_prompt.extend(history)
        full_prompt.append({"role": "user", "content": user_content})
        
        return full_prompt
    
    def get_answer(self, query: str, history: List[Dict[str, str]] = None) -> Generator[str, None, None]:
        """
        Main method to get an answer, with print statements for debugging each step.
        """
        history = history or []
        
        # --- STEP 1: RETRIEVE from Pinecone ---
        matches = self.pinecone_query(query, top_k=5)
        match_ids = [m["id"] for m in matches]
        
        print("\n\n--- [STEP 1] IDs from Pinecone ---")
        print(match_ids)
        print("----------------------------------\n")

        # --- STEP 2: RETRIEVE from Neo4j ---
        graph_facts = self.fetch_graph_context(match_ids)
        
        print("\n--- [STEP 2] Context from Neo4j Graph ---")
        print(json.dumps(graph_facts, indent=2))
        print("-----------------------------------------\n")

        # --- STEP 3: SUMMARIZE the context ---
        summary = self._get_search_summary(query, matches, graph_facts)
        
        print("\n--- [STEP 3] Generated Search Summary ---")
        print(summary)
        print("-----------------------------------------\n")

        # --- STEP 4: BUILD the final prompt ---
        prompt_messages = self.build_prompt_with_summary(query, summary, history)
        
        print("\n--- [STEP 4] Final Prompt to LLM ---")
        print(json.dumps(prompt_messages, indent=2))
        print("------------------------------------\n")

        # --- STEP 5: GENERATE the final answer ---
        print("\n--- [STEP 5] Streaming Assistant Answer ---\n")
        stream = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=prompt_messages,
            stream=True
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            yield content


def main():
    """Run an interactive chat session."""
    try:
        rag_system = HybridRAG()
        print("🚀 Hybrid Travel Assistant is ready! Type 'exit' to quit.")
        while True:
            query = input("\nEnter your travel question: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                break
            print("\n=== Assistant Answer ===\n")
            for chunk in rag_system.get_answer(query):
                print(chunk, end="", flush=True)
            print("\n\n========================\n")

        rag_system.close()
    except Exception as e:
        logging.critical(f"A critical error occurred: {e}")


if __name__ == "__main__":
    main()