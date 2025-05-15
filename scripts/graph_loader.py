from neo4j import GraphDatabase
import networkx as nx

NEO4J_URI = "neo4j+s://7ffe6d75.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "7l2PonEcEPWgdWYsD_Q8i4bfBTwAKTFjhGe2oD0f_Dw"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def build_attack_graph_from_neo4j():
    query = """
    MATCH (source)-[r]->(target)
    RETURN source.name AS source, target.name AS target, type(r) AS relation, r.risk_score AS risk
    """
    with driver.session() as session:
        result = session.run(query)
        G = nx.DiGraph()
        for record in result:
            G.add_edge(record['source'], record['target'], weight=record.get('risk', 1.0), label=record['relation'])
    return G
