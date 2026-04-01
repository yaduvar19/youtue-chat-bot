# src/queue/worker.py
from rq import Worker, Queue
from redis import Redis
import os

from dotenv import load_dotenv
load_dotenv()


def run_worker():
    """Start the RQ worker for processing chatbot tasks."""
    redis_conn = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    queues = [Queue("youtube_chatbot", connection=redis_conn)]
    
    worker = Worker(queues, connection=redis_conn)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    run_worker()
