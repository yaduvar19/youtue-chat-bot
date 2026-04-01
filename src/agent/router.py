# src/agent/router.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Literal


class IntentClassification(BaseModel):
    intent: Literal["search", "transcript", "channel", "chat"] = Field(
        description="The detected user intent"
    )
    confidence: float = Field(description="Confidence score 0-1")
    extracted_query: str = Field(description="The extracted search query or identifier")


INTENT_PROMPT = """Classify the user's intent for a YouTube assistant chatbot.

Intents:
- search: User wants to find/search for YouTube videos
- transcript: User wants to get or summarize a video transcript (needs video URL/ID)
- channel: User wants information about a YouTube channel
- chat: General conversation or questions about YouTube that don't fit above

User message: {message}

Classify the intent and extract any relevant query or identifier."""


async def classify_intent(message: str) -> IntentClassification:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(IntentClassification)
    
    response = await structured_llm.ainvoke(
        INTENT_PROMPT.format(message=message)
    )
    return response
