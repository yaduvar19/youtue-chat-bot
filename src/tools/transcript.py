# src/tools/transcript.py
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import re


def extract_video_id(url_or_id: str) -> str:
    """Extract video ID from URL or return as-is if already an ID."""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return url_or_id


@tool
def get_video_transcript(video_url_or_id: str, summarize: bool = True) -> dict:
    """
    Get the transcript of a YouTube video and optionally summarize it.
    Accepts either a video URL or video ID.
    """
    video_id = extract_video_id(video_url_or_id)
    
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry["text"] for entry in transcript_list])
        
        result = {
            "video_id": video_id,
            "text": full_text[:5000],  # Truncate for context limits
            "summary": None
        }
        
        if summarize and len(full_text) > 500:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            summary_prompt = f"""Summarize this YouTube video transcript in 3-5 bullet points:

{full_text[:8000]}

Provide a concise summary highlighting the main topics and key takeaways."""
            
            summary_response = llm.invoke(summary_prompt)
            result["summary"] = summary_response.content
        
        return result
        
    except Exception as e:
        return {
            "video_id": video_id,
            "text": "",
            "summary": None,
            "error": str(e)
        }
