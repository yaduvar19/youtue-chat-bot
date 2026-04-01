# src/queue/tasks.py
from redis import Redis
from rq import Queue
import os

from ..agent.graph import agent
from ..agent.state import AgentState
from langchain_core.messages import HumanMessage


redis_conn = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
task_queue = Queue("youtube_chatbot", connection=redis_conn)


async def process_message(user_message: str, thread_id: str) -> dict:
    """Process a user message through the agent."""
    
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_message)],
        "intent": None,
        "guardrail_passed": True,
        "guardrail_message": None,
        "videos": [],
        "transcript": None,
        "channel": None,
        "error": None,
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    result = await agent.ainvoke(initial_state, config)
    
    # Extract the last AI message
    last_message = result["messages"][-1]
    
    return {
        "response": last_message.content,
        "intent": result.get("intent"),
        "thread_id": thread_id,
    }


def enqueue_message(user_message: str, thread_id: str) -> str:
    """Add a message to the processing queue."""
    job = task_queue.enqueue(
        process_message,
        user_message,
        thread_id,
        job_timeout=120,
        result_ttl=3600,
    )
    return job.id
