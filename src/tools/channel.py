# src/tools/channel.py
from googleapiclient.discovery import build
from langchain_core.tools import tool
import os
import re


def extract_channel_identifier(input_str: str) -> tuple[str, str]:
    """Extract channel ID or username from URL or direct input."""
    # Channel ID pattern
    if re.match(r'^UC[\w-]{22}$', input_str):
        return "id", input_str
    
    # URL patterns
    patterns = [
        (r'youtube\.com/channel/(UC[\w-]{22})', "id"),
        (r'youtube\.com/@([\w-]+)', "handle"),
        (r'youtube\.com/c/([\w-]+)', "custom"),
        (r'youtube\.com/user/([\w-]+)', "username"),
    ]
    
    for pattern, id_type in patterns:
        match = re.search(pattern, input_str)
        if match:
            return id_type, match.group(1)
    
    # Assume it's a channel name/handle
    return "handle", input_str.lstrip("@")


@tool
def get_channel_info(channel_identifier: str) -> dict:
    """
    Get information about a YouTube channel.
    Accepts channel URL, ID, or @handle.
    """
    youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
    
    id_type, value = extract_channel_identifier(channel_identifier)
    
    try:
        if id_type == "id":
            request = youtube.channels().list(part="snippet,statistics", id=value)
        elif id_type == "handle":
            request = youtube.channels().list(part="snippet,statistics", forHandle=value)
        else:
            request = youtube.search().list(part="snippet", type="channel", q=value, maxResults=1)
            search_response = request.execute()
            if search_response.get("items"):
                channel_id = search_response["items"][0]["snippet"]["channelId"]
                request = youtube.channels().list(part="snippet,statistics", id=channel_id)
            else:
                return {"error": f"Channel not found: {channel_identifier}"}
        
        response = request.execute()
        
        if not response.get("items"):
            return {"error": f"Channel not found: {channel_identifier}"}
        
        channel = response["items"][0]
        snippet = channel["snippet"]
        stats = channel.get("statistics", {})
        
        return {
            "channel_id": channel["id"],
            "name": snippet["title"],
            "description": snippet["description"][:500],
            "subscriber_count": int(stats.get("subscriberCount", 0)),
            "video_count": int(stats.get("videoCount", 0)),
            "thumbnail_url": snippet["thumbnails"]["medium"]["url"]
        }
        
    except Exception as e:
        return {"error": str(e)}
