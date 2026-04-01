# src/agent/state.py
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class YouTubeVideo(BaseModel):
    video_id: str
    title: str
    channel: str
    description: str
    thumbnail_url: str
    published_at: str


class ChannelInfo(BaseModel):
    channel_id: str
    name: str
    description: str
    subscriber_count: int
    video_count: int


class TranscriptResult(BaseModel):
    video_id: str
    text: str
    summary: str | None = None


class AgentState(TypedDict):
    """Shared state flowing through the graph."""
    messages: Annotated[list, add_messages]
    intent: Literal["search", "transcript", "channel", "chat", "blocked"] | None
    guardrail_passed: bool
    guardrail_message: str | None
    videos: list[YouTubeVideo]
    transcript: TranscriptResult | None
    channel: ChannelInfo | None
    error: str | None
