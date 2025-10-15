import os
import logging
from typing import List, Dict, Any
from openai import OpenAI, APIError
from pinecone import Pinecone, PineconeException
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from dotenv import load_dotenv
import httpx

# --- 1. Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class HybridRAG:
    """A class for a hybrid RAG system using OpenAI, Pinecone, and Neo4j."""

    def __init__(self):
        # --- 2. Securely Initialize Clients ---
        try:
            # OpenAI
            http_client = httpx.Client(trust_env=False)
            self.openai_client = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    http_client=http_client)
            # MODIFIED: Changed to match the screenshot
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
                auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")) , database = os.getenv("NEO4J_DATABASE"))

            self.neo4j_driver.verify_connectivity()
            logging.info("All clients initialized successfully.")

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
        """Generate embedding for a text string."""
        try:
            resp = self.openai_client.embeddings.create(model=self.embed_model, input=[text])
            return resp.data[0].embedding
        except APIError as e:
            logging.error(f"OpenAI API error during embedding: {e}")
            return []

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

    def build_enhanced_prompt(self, user_query: str, pinecone_matches: list, graph_facts: list) -> list:
        """Build an enhanced, structured chat prompt."""
        # --- 3. Enhanced Prompt ---
        system_content = """You are an expert travel assistant for Vietnam.
Your goal is to provide helpful, concise, and fact-based answers.
- Use the **CONTEXT** provided below to answer the user's **QUERY**.
- The context includes semantic matches from a vector database and factual relationships from a graph database.
- **Synthesize information from both sources** to create a complete answer.
- **Cite your sources** by mentioning the node IDs (e.g., `city_hanoi`) in parentheses after the name of a place.
- If the context does not contain enough information to answer, state that you cannot answer fully and explain what information is missing. **Do not make up information.**
- Format your response using Markdown for readability. Use lists or bold text where appropriate."""

        context_str = "## CONTEXT\n\n"
        if pinecone_matches:
            context_str += "### Top Semantic Matches:\n"
            for m in pinecone_matches:
                meta = m.get('metadata', {})
                context_str += f"- **{meta.get('name', 'N/A')}** (`{m['id']}`): A {meta.get('type', 'N/A')} in {meta.get('city', 'N/A')}.\n"
        else:
            context_str += "No semantic matches found.\n"

        if graph_facts:
            context_str += "\n### Related Factual Connections:\n"
            for f in graph_facts:
                context_str += f"- The entity `{f['source_id']}` has a relation `{f['relation']}` with **{f.get('target_name', 'N/A')}** (`{f['target_id']}`), which is described as: {f['target_desc']}\n"
        else:
            context_str += "\nNo factual connections found.\n"

        user_content = f"{context_str}\n\n## QUERY\n\n{user_query}"

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]

    def get_answer(self, query: str) -> str:
        """Main method to get an answer from the hybrid RAG system."""
        matches = self.pinecone_query(query, top_k=3)
        match_ids = [m["id"] for m in matches]
        graph_facts = self.fetch_graph_context(match_ids)
        prompt = self.build_enhanced_prompt(query, matches, graph_facts)

        try:
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=prompt,
                temperature=0.1,
                max_tokens=800
            )
            return response.choices[0].message.content
        except APIError as e:
            logging.error(f"OpenAI API error during chat completion: {e}")
            return "I'm sorry, but I encountered an error while trying to generate a response. Please try again later."

    def close(self):
        """Close the Neo4j driver connection."""
        self.neo4j_driver.close()
        logging.info("Neo4j driver closed.")


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

            answer = rag_system.get_answer(query)
            print("\n=== Assistant Answer ===\n")
            print(answer)
            print("\n========================\n")

        rag_system.close()
    except Exception as e:
        logging.critical(f"A critical error occurred: {e}")


if __name__ == "__main__":
    main()