"""
agents/graph.py
Builds and compiles the Self-RAG + CRAG LangGraph state machine.
"""
from langgraph.graph import END, StateGraph
from core.state import GraphState
from agents.nodes import (
    retrieve, grade_documents, web_search_node,
    generate, transform_query,
    decide_to_generate, grade_generation_quality,
)


def build_graph():
    """Construct and compile the 7-layer Self-RAG + CRAG graph."""
    workflow = StateGraph(GraphState)

    # Register nodes
    workflow.add_node("retrieve",        retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search",      web_search_node)
    workflow.add_node("generate",        generate)
    workflow.add_node("transform_query", transform_query)

    # Entry point
    workflow.set_entry_point("retrieve")

    # Fixed edges
    workflow.add_edge("retrieve",       "grade_documents")
    workflow.add_edge("web_search",     "generate")
    workflow.add_edge("transform_query","retrieve")

    # Conditional edges
    workflow.add_conditional_edges(
        "grade_documents", decide_to_generate,
        {
            "generate":        "generate",
            "web_search":      "web_search",
            "transform_query": "transform_query",
        }
    )
    workflow.add_conditional_edges(
        "generate", grade_generation_quality,
        {
            "useful":        END,
            "not_useful":    "transform_query",
            "not_supported": "generate",
        }
    )

    return workflow.compile()


# Singleton — compiled once on import
app = build_graph()
print("Self-RAG + CRAG graph compiled")
print("Nodes:", ["retrieve", "grade_documents", "web_search", "generate", "transform_query"])
