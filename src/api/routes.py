# src/api/routes.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from redis import Redis
from rq import Queue
from rq.job import Job
import os
import uuid
import asyncio

from ..agent.graph import agent
from ..agent.state import AgentState
from ..queue.tasks import enqueue_message, process_message
from langchain_core.messages import HumanMessage


app = FastAPI(title="YouTube Chatbot API")
redis_conn = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    intent: str | None
    thread_id: str


class QueuedResponse(BaseModel):
    job_id: str
    thread_id: str
    status: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Synchronous chat endpoint - processes immediately."""
    thread_id = request.thread_id or str(uuid.uuid4())
    
    result = await process_message(request.message, thread_id)
    
    return ChatResponse(
        response=result["response"],
        intent=result["intent"],
        thread_id=thread_id,
    )


@app.post("/chat/async", response_model=QueuedResponse)
async def chat_async(request: ChatRequest):
    """Asynchronous chat endpoint - queues for processing."""
    thread_id = request.thread_id or str(uuid.uuid4())
    
    job_id = enqueue_message(request.message, thread_id)
    
    return QueuedResponse(
        job_id=job_id,
        thread_id=thread_id,
        status="queued",
    )


@app.get("/chat/status/{job_id}")
async def get_job_status(job_id: str):
    """Check the status of a queued chat job."""
    try:
        job = Job.fetch(job_id, connection=redis_conn)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.is_finished:
        return {
            "status": "completed",
            "result": job.result,
        }
    elif job.is_failed:
        return {
            "status": "failed",
            "error": str(job.exc_info),
        }
    else:
        return {
            "status": job.get_status(),
        }


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint - returns tokens as they're generated."""
    thread_id = request.thread_id or str(uuid.uuid4())
    
    initial_state: AgentState = {
        "messages": [HumanMessage(content=request.message)],
        "intent": None,
        "guardrail_passed": True,
        "guardrail_message": None,
        "videos": [],
        "transcript": None,
        "channel": None,
        "error": None,
    }
    
    config = {"configurable": {"thread_id": thread_id}}
    
    async def generate():
        async for event in agent.astream_events(initial_state, config, version="v2"):
            if event["event"] == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "healthy"}
