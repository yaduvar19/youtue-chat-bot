from src.db.vector_store import search_similar_chunks
import ollama

def generate_answer(question: str, video_id: str) -> str:
    """
    Full RAG pipeline:
    1. Retrieve transcript chunks from Redis
    2. Send context + question to LLM
    """

    # 1️⃣ Retrieve relevant transcript chunks
    chunks = search_similar_chunks(question, video_id)

    context = "\n\n".join(chunks)

    # 2️⃣ Create grounded prompt
    prompt = f"""
You are a helpful YouTube assistant.

Answer the question ONLY using the context below.
If the answer is not in the context, say:
"I couldn't find this in the video."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    # 3️⃣ Call local LLM
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]