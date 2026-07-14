from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = """
Ad sales boost Time Warner profit

Quarterly profits at US media giant TimeWarner jumped 76%.
"""

embedding = model.encode(text)

print("Embedding size:", len(embedding))
print("First 5 values:", embedding[:5])