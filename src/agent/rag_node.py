from langsmith import traceable
from src.services.rag_service import generate_answer
from langchain_core.messages import HumanMessage

@traceable(name="rag_node")
async def rag_node(state):
    """
    Uses Redis vector DB + Ollama to answer questions
    based on stored transcript embeddings.
    """

    last_message = state["messages"][-1]

    if not isinstance(last_message, HumanMessage):
        return state

    # we expect message format:
    # "ask: VIDEO_ID | question"
    text = last_message.content

    if "|" not in text:
        return state

    video_id, question = text.split("|", 1)
    video_id = video_id.replace("ask:", "").strip()
    question = question.strip()

    answer = generate_answer(question, video_id)

    return {
        **state,
        "rag_answer": answer
    }