"""
agent.py - Bounded ReAct orchestration for conversational paper research.

The agent coordinates session memory, Groq decisions, and the approved tools.
It never executes arbitrary model-provided code: actions are validated against
TOOLS before a function is called.
"""

import inspect
import json
import os
from typing import Any

import generator
import memory
from tools import TOOLS


_MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "5"))
_MAX_JSON_ATTEMPTS = 3

_AGENT_SYSTEM_PROMPT = """You are the reasoning controller for a research paper assistant.

Use the conversation history and tool observations to answer the user's
question. Choose a tool when you need information. Use local paper search for
questions about indexed papers, get_paper_summary for a specific paper,
list_papers when the user asks what is available, and web_search for recent or
out-of-corpus research.

If the user's question has multiple distinct parts (e.g. it asks about
several papers, compares two concepts, or asks two separate things in one
message), do not search for all of it with one broad query. Issue one
focused, single-topic search_papers call per part across separate
iterations, review each observation, and only give a final_answer once you
have evidence for every part of the question.

Respond with ONLY one valid JSON object in one of these shapes:
{"action": "search_papers", "query": "focused search query"}
{"action": "search_papers", "query": "focused search query", "filter_source": "paper.pdf"}
{"action": "get_paper_summary", "filename": "paper.pdf"}
{"action": "list_papers"}
{"action": "web_search", "query": "research topic"}
{"action": "final_answer", "answer": "complete answer for the user"}

Do not invent tool names. Do not include markdown fences or reasoning outside
the JSON object. Give a final_answer only when the available evidence is
sufficient. Ground factual claims in tool observations and be transparent when
the tools do not contain the answer."""


def _build_prompt(
    question: str,
    history: str,
    observations: list[dict[str, Any]],
    retry_message: str | None = None,
) -> str:
    """Build the text prompt for one ReAct decision."""
    sections = []
    if history:
        sections.append(f"Conversation history:\n{history}")
    sections.append(f"Current user question:\n{question}")
    if observations:
        sections.append(
            "Tool observations:\n"
            + "\n".join(json.dumps(item, ensure_ascii=True) for item in observations)
        )
    if retry_message:
        sections.append(retry_message)
    return "\n\n".join(sections)


def _parse_decision(raw: str) -> dict[str, Any] | None:
    """Return a valid JSON object or None when the model output is malformed."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _request_decision(
    question: str,
    history: str,
    observations: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    """Request a decision, retry malformed JSON twice, then return raw text."""
    raw = ""
    retry_message = None
    for attempt in range(_MAX_JSON_ATTEMPTS):
        raw = generator.generate_agent_decision(
            _AGENT_SYSTEM_PROMPT,
            _build_prompt(question, history, observations, retry_message),
        )
        decision = _parse_decision(raw)
        if decision is not None:
            return decision, raw
        retry_message = (
            "Your previous response was not a valid JSON object. Retry with only "
            "one JSON object matching the required action shapes."
        )
    return None, raw


def _validate_action(decision: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Validate an action and its keyword arguments before dispatch."""
    action = decision.get("action")
    if not isinstance(action, str) or action == "final_answer":
        raise ValueError("Decision is not a tool action.")
    if action not in TOOLS:
        raise ValueError(f"Unknown tool action: {action}")

    arguments = {key: value for key, value in decision.items() if key != "action"}
    signature = inspect.signature(TOOLS[action])
    accepted = set(signature.parameters)
    unexpected = set(arguments) - accepted
    if unexpected:
        raise ValueError(f"Unexpected arguments for {action}: {sorted(unexpected)}")
    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty and name not in arguments
    ]
    if missing:
        raise ValueError(f"Missing arguments for {action}: {missing}")
    return action, arguments


def _collect_sources(value: Any, sources: set[str]) -> None:
    """Collect paper source names from serializable tool results."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"source", "filename"} and isinstance(item, str):
                sources.add(item)
            else:
                _collect_sources(item, sources)
    elif isinstance(value, list):
        for item in value:
            _collect_sources(item, sources)


def _fallback_answer(raw: str) -> str:
    """Turn repeated malformed model output into a user-visible answer."""
    return raw.strip() or "I could not produce a valid response for that question."


def run_agent(
    question: str,
    session_id: str,
    filter_source: str | None = None,
) -> dict[str, Any]:
    """Run the bounded ReAct loop for one question.

    Returns an answer, unique source filenames, and the number of iterations.
    The completed user and assistant messages are added to session memory once.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")

    history = memory.format_history(session_id)
    observations: list[dict[str, Any]] = []
    sources: set[str] = set()
    answer = ""
    iterations = 0

    for iterations in range(1, _MAX_AGENT_ITERATIONS + 1):
        decision, raw = _request_decision(question, history, observations)
        if decision is None:
            answer = _fallback_answer(raw)
            break

        action = decision.get("action")
        if action == "final_answer":
            candidate = decision.get("answer")
            answer = candidate.strip() if isinstance(candidate, str) else ""
            if not answer:
                answer = "The model returned an empty final answer."
            break

        try:
            tool_name, arguments = _validate_action(decision)
            if tool_name == "search_papers" and filter_source and "filter_source" not in arguments:
                arguments["filter_source"] = filter_source
            result = TOOLS[tool_name](**arguments)
            _collect_sources(result, sources)
            observations.append({"tool": tool_name, "result": result})
        except Exception as exc:
            observations.append({"tool_error": str(exc)})
    else:
        answer = "I reached the agent's tool-use limit before completing the answer."

    if not answer:
        answer = "I could not complete the request."

    memory.add_turn(session_id, "user", question)
    memory.add_turn(session_id, "assistant", answer)
    return {
        "answer": answer,
        "sources": sorted(sources),
        "iterations": iterations,
    }
