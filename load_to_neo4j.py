import os
import json
import logging
from tqdm import tqdm
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from dotenv import load_dotenv

# --- Basic Setup ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Neo4jLoader:
    """Handles loading graph data from JSON into Neo4j using efficient batching."""

    def __init__(self, data_file="vietnam_travel_dataset.json"):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE")
        self.database = database
        self.data_file = data_file
        self.batch_size = 500

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password), database=database)
            self.driver.verify_connectivity()
            logging.info(f"Neo4j connection to database '{self.database}' successful.")
        except (Neo4jError, ValueError) as e:
            logging.error(f"Could not connect to Neo4j: {e}")
            raise

    def close(self):
        """Closes the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logging.info("Neo4j connection closed.")

    def _load_json_data(self):
        """Loads and returns the data from the source JSON file."""
        try:
            with open(self.data_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Could not read or parse data file {self.data_file}: {e}")
            return []

    def _run_write_transaction(self, query, params=None):
        """Executes a write transaction against the specified database."""
        with self.driver.session(database=self.database) as session:
            try:
                session.run(query, params)
            except Neo4jError as e:
                logging.error(f"Cypher query failed: {e}\nQuery: {query}")

    def create_constraints(self):
        """Ensures an `id` uniqueness constraint exists for `Entity` nodes."""
        logging.info("Applying uniqueness constraint for :Entity(id)...")
        query = "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE"
        self._run_write_transaction(query)

    def batch_upsert_nodes(self, nodes):
        """Upserts nodes into Neo4j in batches using UNWIND."""
        logging.info(f"Upserting {len(nodes)} nodes in batches...")
        query = """
        UNWIND $batch as node_data
        MERGE (n:Entity {id: node_data.id})
        SET n += apoc.map.removeKey(node_data.props, 'id')
        WITH n, node_data
        CALL apoc.create.addLabels(n, [node_data.type]) YIELD node
        RETURN count(node)
        """
        for i in tqdm(range(0, len(nodes), self.batch_size), desc="Upserting Nodes"):
            batch = nodes[i:i + self.batch_size]
            batch_data = [
                {
                    "id": node["id"],
                    "type": node.get("type", "Unknown"),
                    "props": {k: v for k, v in node.items() if k != "connections"}
                } for node in batch
            ]
            self._run_write_transaction(query, params={"batch": batch_data})

    def batch_create_relationships(self, nodes):
        """Creates relationships in batches from the node connection data."""
        logging.info("Creating relationships in batches...")
        all_relationships = []
        for node in nodes:
            source_id = node["id"]
            for conn in node.get("connections", []):
                if "target" in conn and "relation" in conn:
                    all_relationships.append({
                        "source": source_id,
                        "target": conn["target"],
                        "type": conn["relation"]
                    })

        query = """
        UNWIND $batch as rel_data
        MATCH (a:Entity {id: rel_data.source})
        MATCH (b:Entity {id: rel_data.target})
        CALL apoc.create.relationship(a, rel_data.type, {}, b) YIELD rel
        RETURN count(rel)
        """
        for i in tqdm(range(0, len(all_relationships), self.batch_size), desc="Creating Relationships"):
            batch = all_relationships[i:i + self.batch_size]
            self._run_write_transaction(query, params={"batch": batch})

    def run(self):
        """Executes the full data loading pipeline."""
        self.create_constraints()
        nodes_data = self._load_json_data()
        if not nodes_data:
            logging.warning("No data found to load. Exiting.")
            return

        self.batch_upsert_nodes(nodes_data)
        self.batch_create_relationships(nodes_data)
        logging.info("Data loading process completed successfully.")


if __name__ == "__main__":
    try:
        loader = Neo4jLoader()
        loader.run()
    except Exception as e:
        logging.critical(f"An unhandled error occurred: {e}")
    finally:
        if 'loader' in locals() and loader:
            loader.close()