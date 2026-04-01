# src/agent/nodes.py
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langsmith import traceable

from .state import AgentState
from .router import classify_intent
from ..guardrails.rails import guardrails_manager
from ..tools.youtube_search import search_youtube_videos
from ..tools.transcript import get_video_transcript
from ..tools.channel import get_channel_info


SYSTEM_PROMPT = """You are a helpful YouTube assistant. You help users:
- Search for videos on any topic
- Get and summarize video transcripts
- Find information about YouTube channels

Be concise, helpful, and engaging. When presenting search results, format them clearly.
When summarizing transcripts, highlight key points."""


@traceable(name="router_node")
async def router_node(state: AgentState) -> AgentState:
    """Classify user intent and route to appropriate handler."""
    last_message = state["messages"][-1]
    
    if isinstance(last_message, HumanMessage):
        intent_result = await classify_intent(last_message.content)
        return {
            **state,
            "intent": intent_result.intent,
        }
    
    return state


@traceable(name="guardrails_node")
async def guardrails_node(state: AgentState) -> AgentState:
    """Check user input against guardrails."""
    last_message = state["messages"][-1]
    
    if isinstance(last_message, HumanMessage):
        passed, blocked_message = await guardrails_manager.check_input(
            last_message.content
        )
        
        if not passed:
            return {
                **state,
                "guardrail_passed": False,
                "guardrail_message": blocked_message,
                "intent": "blocked"
            }
    
    return {**state, "guardrail_passed": True}


@traceable(name="search_node")
async def search_node(state: AgentState) -> AgentState:
    """Execute YouTube video search."""
    last_message = state["messages"][-1]
    query = last_message.content
    
    try:
        # Extract search terms from the message
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        extraction = await llm.ainvoke(
            f"Extract the YouTube search query from this message. Return only the search terms:\n\n{query}"
        )
        search_query = extraction.content.strip()
        
        videos = search_youtube_videos.invoke({"query": search_query, "max_results": 5})
        
        return {**state, "videos": videos, "error": None}
        
    except Exception as e:
        return {**state, "error": str(e)}


@traceable(name="transcript_node")
async def transcript_node(state: AgentState) -> AgentState:
    """Fetch and summarize video transcript."""
    last_message = state["messages"][-1]
    
    try:
        result = get_video_transcript.invoke({
            "video_url_or_id": last_message.content,
            "summarize": True
        })
        
        return {**state, "transcript": result, "error": None}
        
    except Exception as e:
        return {**state, "error": str(e)}


@traceable(name="channel_node")
async def channel_node(state: AgentState) -> AgentState:
    """Fetch channel information."""
    last_message = state["messages"][-1]
    
    try:
        result = get_channel_info.invoke({"channel_identifier": last_message.content})
        
        return {**state, "channel": result, "error": None}
        
    except Exception as e:
        return {**state, "error": str(e)}


@traceable(name="response_node")
async def response_node(state: AgentState) -> AgentState:
    """Generate final response based on gathered data."""
    
    # Handle blocked requests
    if state.get("intent") == "blocked":
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content=state.get("guardrail_message", "I can't help with that request."))
            ]
        }
    
    # Handle errors
    if state.get("error"):
        return {
            **state,
            "messages": state["messages"] + [
                AIMessage(content=f"I encountered an issue: {state['error']}. Please try again.")
            ]
        }
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    # Build context based on what data we have
    context_parts = []
    
    if state.get("videos"):
        videos_text = "\n".join([
            f"- **{v['title']}** by {v['channel']} (ID: {v['video_id']})"
            for v in state["videos"]
        ])
        context_parts.append(f"Search Results:\n{videos_text}")
    
    if state.get("transcript"):
        t = state["transcript"]
        if t.get("summary"):
            context_parts.append(f"Transcript Summary:\n{t['summary']}")
        elif t.get("text"):
            context_parts.append(f"Transcript excerpt:\n{t['text'][:1000]}...")
    
    if state.get("channel"):
        c = state["channel"]
        if not c.get("error"):
            context_parts.append(
                f"Channel Info:\n- Name: {c['name']}\n- Subscribers: {c['subscriber_count']:,}\n- Videos: {c['video_count']}"
            )
    
    context = "\n\n".join(context_parts) if context_parts else "No specific data gathered."
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
        HumanMessage(content=f"[Context for response]\n{context}")
    ]
    
    response = await llm.ainvoke(messages)
    
    # Validate output through guardrails
    passed, final_content = await guardrails_manager.check_output(response.content)
    
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=final_content)]
    }
