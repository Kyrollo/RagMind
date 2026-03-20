"""
core/state.py
LangGraph TypedDict state shared across all nodes.
"""
from typing import TypedDict, List


class GraphState(TypedDict):
    question:          str          # current (possibly rewritten) question
    original_question: str          # user's raw question — never overwritten
    generation:        str          # LLM answer
    documents:         List[object] # retrieved + filtered documents
    retry_count:       int          # rewrite attempts counter
    used_web:          bool         # whether web search was used this turn
    steps:             List[str]    # reasoning trace for the UI
    history:           str          # formatted chat history injected into prompts
