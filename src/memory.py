"""
memory.py — Rolling Conversation Memory

Responsibility:
    Holds the last MEMORY_WINDOW_SIZE turns of conversation for each active
    session, so agent.py's ReAct loop can inject prior turns into the prompt
    on every call and produce continuity across a multi-turn conversation,
    even though each POST /query request is otherwise stateless.

    This module owns no LLM calls, no HTTP, no disk I/O — it is pure
    in-process state management, keyed by session_id.

Why an in-memory dict instead of a database?
    Per CLAUDE.md this is explicitly session-scoped, single-process, single-
    demo-user memory ("resets on refresh... no cross-session persistence").
    A module-level dict keyed by session_id is the simplest thing that
    satisfies that spec. It would need to become a shared store (SQLite,
    Redis) only if the server ever ran as multiple processes/instances or
    needed to survive a restart — neither is a v2 requirement.

Why a deque(maxlen=...) per session instead of trimming a list?
    A rolling window that drops the oldest turn once full is exactly what
    collections.deque(maxlen=N) does natively — appending past the limit
    silently evicts from the opposite end with no manual slicing logic.
"""

import os
from collections import deque


_MEMORY_WINDOW_SIZE = int(os.getenv("MEMORY_WINDOW_SIZE", "8"))

# session_id -> deque of {"role": "user" | "assistant", "content": str}
# maxlen is turns, and one turn is one message here (not one user+assistant
# pair), so the window holds MEMORY_WINDOW_SIZE * 2 messages — that's what
# "last 8 turns" means per CLAUDE.md (8 user + 8 assistant messages).
_sessions: dict[str, deque] = {}


def _get_session(session_id: str) -> deque:
    """
    Return the deque for session_id, creating an empty one if this is the
    first time this session has been seen.
    """
    if session_id not in _sessions:
        _sessions[session_id] = deque(maxlen=_MEMORY_WINDOW_SIZE * 2)
    return _sessions[session_id]


def add_turn(session_id: str, role: str, content: str) -> None:
    """
    Append one message to a session's rolling history.

    Args:
        session_id: the browser-generated UUID identifying this conversation
        role:       "user" or "assistant"
        content:    the message text

    Once the session's deque is full, appending drops the oldest message
    automatically (deque(maxlen=...) behavior) — no explicit trimming needed.
    """
    _get_session(session_id).append({"role": role, "content": content})


def get_history(session_id: str) -> list[dict]:
    """
    Return this session's turns, oldest first, as plain dicts.

    Returns an empty list for a session_id that has never sent a message —
    callers don't need to special-case a brand-new session.
    """
    return list(_sessions.get(session_id, []))


def format_history(session_id: str) -> str:
    """
    Render a session's history as a text block for injection into the
    agent's prompt, ahead of the current user message.

    Returns an empty string for a session with no prior turns, so agent.py
    can always splice this in without checking for the empty case itself.

    Example output:
        User: What does the LoRA paper say about rank?
        Assistant: LoRA uses a low-rank decomposition...
    """
    turns = get_history(session_id)
    if not turns:
        return ""

    label = {"user": "User", "assistant": "Assistant"}
    return "\n".join(f"{label[turn['role']]}: {turn['content']}" for turn in turns)


def clear_session(session_id: str) -> bool:
    """
    Discard a session's entire history.

    Backs the DELETE /session/{session_id} endpoint from CLAUDE.md.

    Returns:
        True if the session existed and was cleared, False if session_id
        was never seen (still a valid, non-error outcome for the caller).
    """
    return _sessions.pop(session_id, None) is not None
