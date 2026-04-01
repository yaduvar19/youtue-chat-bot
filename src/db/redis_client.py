import os
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True  
)

def check_redis_connection():
    """Ping Redis to ensure connection works"""
    try:
        redis_client.ping()
        print("✅ Redis connected successfully")
    except Exception as e:
        print("❌ Redis connection failed:", e)