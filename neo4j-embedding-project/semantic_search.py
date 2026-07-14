from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer


URI = "bolt://localhost:7687"
DATABASE = "bbc-news"


driver = GraphDatabase.driver(URI)


# Load same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def search_articles(question, limit=5):

    # Create query embedding
    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()


    cypher = """
    CALL db.index.vector.queryNodes(
        'article_embeddings',
        $limit,
        $embedding
    )
    YIELD node, score

    RETURN
        node.title AS title,
        node.category AS category,
        score
    ORDER BY score DESC
    """


    with driver.session(database=DATABASE) as session:

        result = session.run(
            cypher,
            limit=limit,
            embedding=query_embedding
        )

        return list(result)


question = input("\nAsk something: ")

results = search_articles(question)


print("\nTop matching articles:\n")

for r in results:
    print("--------------------")
    print("Title:", r["title"])
    print("Category:", r["category"])
    print("Score:", round(r["score"], 4))