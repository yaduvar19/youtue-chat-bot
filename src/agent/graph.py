# src/agent/graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .rag_node import rag_node
from .state import AgentState
from .nodes import (
    router_node,
    guardrails_node,
    search_node,
    transcript_node,
    channel_node,
    response_node,
)


def route_by_intent(state: AgentState) -> str:
    """Route to appropriate node based on detected intent."""
    if not state.get("guardrail_passed", True):
        return "response"
    
    intent = state.get("intent")
    
    routing = {
        "search": "search",
        "transcript": "transcript",
        "channel": "channel",
        "chat": "response",
        "rag": "rag",
        "blocked": "response",
    }
    
    return routing.get(intent, "response")


def build_graph() -> StateGraph:
    """Construct the YouTube chatbot graph."""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("guardrails", guardrails_node)
    workflow.add_node("search", search_node)
    workflow.add_node("transcript", transcript_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("channel", channel_node)
    workflow.add_node("response", response_node)
    
    # Define edges
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "guardrails")
    workflow.add_edge("rag", "response")
    # Conditional routing after guardrails
    workflow.add_conditional_edges(
        "guardrails",
        route_by_intent,
        {
            "search": "search",
            "transcript": "transcript",
            "channel": "channel",
            "response": "response",
        }
    )
    
    # All tool nodes lead to response
    workflow.add_edge("search", "response")
    workflow.add_edge("transcript", "response")
    workflow.add_edge("channel", "response")
    workflow.add_edge("response", END)
    
    return workflow


def create_agent():
    """Create the compiled agent with memory."""
    workflow = build_graph()
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


# Singleton agent instance
agent = create_agent()
