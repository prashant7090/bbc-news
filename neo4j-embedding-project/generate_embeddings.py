from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# -----------------------------
# Neo4j Configuration
# -----------------------------

URI = "bolt://localhost:7687"
DATABASE = "bbc-news"

driver = GraphDatabase.driver(URI)


# -----------------------------
# Load embedding model
# -----------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded")


# -----------------------------
# Fetch articles
# -----------------------------

def get_articles():

    query = """
    MATCH (a:Article)
    RETURN elementId(a) AS id,
           a.title AS title,
           a.content AS content
    """

    with driver.session(database=DATABASE) as session:
        result = session.run(query)

        return [
            {
                "id": record["id"],
                "title": record["title"],
                "content": record["content"]
            }
            for record in result
        ]


# -----------------------------
# Save embedding
# -----------------------------

def save_embedding(article_id, embedding):

    query = """
    MATCH (a:Article)
    WHERE elementId(a) = $id
    SET a.embedding = $embedding
    """

    with driver.session(database=DATABASE) as session:
        session.run(
            query,
            id=article_id,
            embedding=embedding
        )


# -----------------------------
# Main process
# -----------------------------

try:

    articles = get_articles()

    print(f"Found {len(articles)} articles")


    for article in tqdm(articles):

        # Combine title + content
        text = f"""
        {article['title']}

        {article['content']}
        """

        # Generate vector
        embedding = model.encode(
            text,
            normalize_embeddings=True
        )

        # Convert numpy array to list
        embedding = embedding.tolist()


        # Store in Neo4j
        save_embedding(
            article["id"],
            embedding
        )


    print("Embedding generation completed")


finally:

    driver.close()