"""
generator.py — Groq LLM Calls (Query Decomposition + Answer Generation)

Responsibility:
    The only module in the pipeline that talks to the LLM (OpenAI GPT-OSS 120B
    via Groq). It is used twice per query, for two different purposes:

    LLM call #1 — decompose_query()
        Takes the user's raw question and asks the LLM to break it into
        2-4 sharper sub-queries. This runs BEFORE retrieval — its output
        feeds embedder.py + retriever.py.

    LLM call #2 — generate_answer()
        Takes an already-assembled prompt (system instructions + retrieved
        chunks + question, built by pipeline.py) and asks the LLM to
        produce the final grounded answer. This runs AFTER retrieval.

    generator.py owns the decomposition prompt entirely (call #1 is fully
    self-contained: question in, sub-queries out — no other module is
    involved in building that prompt). For call #2, prompt assembly is
    pipeline.py's job per the architecture in CLAUDE.md — generator.py just
    relays the finished system/user prompt to Groq and returns the raw
    text. This keeps generator.py as a thin, swappable Groq transport
    layer plus the one piece of prompt logic (decomposition) that has
    nowhere else to live.

Why two temperatures?
    Decomposition (temp 0) needs to be deterministic and structured — it's
    closer to a parsing task than a creative one, and JSON output must be
    reliable. Generation (temp 0.7) benefits from some variability to
    produce natural, well-phrased answers while still being grounded by
    the retrieved context.
"""

import json
import os
from groq import Groq
from dotenv import load_dotenv


load_dotenv()
_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

_DECOMPOSE_SYSTEM_PROMPT = """You are a query decomposition assistant for a research paper search system.

Given a user's question, break it into 2 to 4 focused sub-queries that together \
cover everything the original question is asking. Each sub-query should target \
one distinct piece of information so a semantic search can retrieve precisely \
relevant passages for it.

If the question is already narrow and single-focus, return it as a single \
sub-query — do not invent unnecessary splits.

Respond with ONLY a JSON object in this exact shape, no other text:
{"sub_queries": ["sub-query 1", "sub-query 2"]}"""


def _call_groq(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.7,
    json_mode: bool = False,
) -> str:
    """
    Send a single generation request to Groq.

    Shared low-level transport used by both decompose_query() and
    generate_answer(). Isolating the API call here means the rest of
    generator.py (and the rest of the pipeline) never touches the Groq
    API shape directly.

    Args:
        prompt:      the user/instruction text to send
        system:      optional system prompt
        temperature: sampling temperature (0 = deterministic, higher = more varied)
        json_mode:   if True, asks Groq to constrain output to valid JSON
                     (used for decomposition, which must be machine-parseable)

    Returns:
        the model's raw text response (already stripped)

    Raises:
        RuntimeError: if Groq rejects the request or is unavailable.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        completion = _groq_client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"} if json_mode else None,
        )
    except Exception as e:
        raise RuntimeError(f"Groq request failed for model '{_GROQ_MODEL}': {e}") from e

    return completion.choices[0].message.content.strip()


def decompose_query(question: str) -> list[str]:
    """
    LLM call #1: break a user question into 2-4 sub-queries.

    Calls Groq with temperature 0 (deterministic) and JSON-constrained
    output, per CLAUDE.md. Parses the response and validates its shape.
    If Groq is unavailable, returns malformed JSON, or returns an empty/
    invalid sub_queries list, falls back to treating the original question
    as the only sub-query — decomposition failing should never block the
    rest of the pipeline from running.

    Args:
        question: the user's raw question, exactly as submitted

    Returns:
        list of 1-4 sub-query strings. Length 1 means either the question
        was already narrow enough that the LLM chose not to split it, or
        decomposition failed and this is the fallback.

    Example:
        decompose_query("How does LoRA compare to full fine-tuning, and "
                         "how does it relate to attention layers?")
        → ["How does LoRA compare to full fine-tuning in memory and performance?",
           "How is LoRA applied to attention layers in transformers?"]
    """
    try:
        raw = _call_groq(
            prompt=f"Question: {question}",
            system=_DECOMPOSE_SYSTEM_PROMPT,
            temperature=0,
            json_mode=True,
        )
        parsed = json.loads(raw)
        sub_queries = parsed["sub_queries"]

        if (
            isinstance(sub_queries, list)
            and 1 <= len(sub_queries) <= 4
            and all(isinstance(q, str) and q.strip() for q in sub_queries)
        ):
            return sub_queries

        print(f"  Decomposition returned an unexpected shape, falling back: {parsed}")
    except RuntimeError as e:
        print(f"  Decomposition failed ({e}), falling back to original question.")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"  Could not parse decomposition response ({e}), falling back to original question.")

    return [question]


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    """
    LLM call #2: generate the final answer from an already-assembled prompt.

    pipeline.py builds system_prompt (instructions to answer only from the
    provided context, cite the source paper per claim, and explicitly say
    when the context doesn't contain the answer) and user_prompt (the
    retrieved chunks plus the user's question) per CLAUDE.md. This function
    just relays them to Groq at temperature 0.7 and returns the raw
    answer text — it has no opinion on how the context was formatted.

    Args:
        system_prompt: grounding/citation instructions for the model
        user_prompt:   retrieved context + the user's question, formatted
                        and ready to send as-is

    Returns:
        the model's generated answer text

    Raises:
        RuntimeError: if Groq is unavailable or errors — unlike
        decompose_query(), there is no safe fallback for generation, so
        this propagates up to pipeline.py / the FastAPI layer to surface
        as an error response.
    """
    return _call_groq(
        prompt=user_prompt,
        system=system_prompt,
        temperature=0.7,
        json_mode=False,
    )


def generate_agent_decision(system_prompt: str, user_prompt: str) -> str:
    """
    Generate one structured decision for the ReAct agent.

    The agent prompt must instruct Groq to return either a tool action or a
    final answer as JSON. The raw JSON text is returned so agent.py owns
    parsing, validation, retries, and tool dispatch.
    """
    return _call_groq(
        prompt=user_prompt,
        system=system_prompt,
        temperature=0,
        json_mode=True,
    )
