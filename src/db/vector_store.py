from sentence_transformers import SentenceTransformer
from src.db.redis_client import redis_client
import json
import numpy as np

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")
def store_transcript_chunks(video_id: str, chunks: list[str]):
    """
    Convert transcript chunks into embeddings and store in Redis
    """
    embeddings = model.encode(chunks)

    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        key = f"video:{video_id}:chunk:{i}"

        data = {
            "text": chunk,
            "embedding": vector.tolist()
        }

        redis_client.set(key, json.dumps(data))
        def search_similar_chunks(query: str, video_id: str, top_k: int = 3):
    """
    Find most relevant transcript chunks using cosine similarity
    """

    query_embedding = model.encode([query])[0]

    keys = redis_client.keys(f"video:{video_id}:chunk:*")

    results = []

    for key in keys:
        data = json.loads(redis_client.get(key))
        vector = np.array(data["embedding"])

        # cosine similarity
        similarity = np.dot(query_embedding, vector) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(vector)
        )

        results.append((similarity, data["text"]))

    results.sort(reverse=True)
    return [text for _, text in results[:top_k]]