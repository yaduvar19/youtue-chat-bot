# src/tools/youtube_search.py
from googleapiclient.discovery import build
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import os
from src.db.redis_client import redis_client
import json
class SearchParams(BaseModel):
    query: str = Field(description="Search query for YouTube videos")
    max_results: int = Field(default=5, description="Maximum number of results")


@tool
def search_youtube_videos(query: str, max_results: int = 5) -> list[dict]:
    """
    Search YouTube videos with Redis caching.
    """

    # 1️⃣ Create cache key
    cache_key = f"youtube_search:{query}:{max_results}"

    # 2️⃣ Check Redis cache first
    cached_data = redis_client.get(cache_key)
    if cached_data:
        print("⚡ Returning cached YouTube results")
        return json.loads(cached_data)

    # 3️⃣ If not cached → call YouTube API
    youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

    request = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=max_results
    )

    response = request.execute()

    videos = []
    for item in response.get("items", []):
        snippet = item["snippet"]
        videos.append({
            "video_id": item["id"]["videoId"],
            "title": snippet["title"],
            "channel": snippet["channelTitle"],
            "description": snippet["description"][:200],
            "thumbnail_url": snippet["thumbnails"]["medium"]["url"],
            "published_at": snippet["publishedAt"]
        })

    # 4️⃣ Save to Redis (cache for 1 hour)
    redis_client.setex(cache_key, 3600, json.dumps(videos))

    return videos
