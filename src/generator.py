"""
generator.py — LLM Transport (Query Decomposition + Answer Generation)

Responsibility:
    The only module in the pipeline that talks to the LLM. It is used for
    several distinct purposes: decompose_query() (v1's sub-query
    decomposition), generate_answer() (v1's grounded answer), and the v2
    additions generate_paper_summary() and generate_agent_decision() (the
    ReAct loop's decision calls).

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
    relays the finished system/user prompt to the LLM and returns the raw
    text. This keeps generator.py as a thin, swappable LLM transport layer
    plus the one piece of prompt logic (decomposition) that has nowhere
    else to live.

Provider modularity:
    Groq, Cerebras, and OpenRouter (the three registered below) all expose
    an OpenAI-compatible `/chat/completions` endpoint, so one `openai.OpenAI`
    client pointed at a different `base_url` + API key + model covers all
    of them — no per-provider SDK needed. LLM_PROVIDER (env var, default
    "groq") selects which entry in _PROVIDERS is active; each entry names
    its own API-key and model env vars so switching providers is just
    setting LLM_PROVIDER plus that provider's key in .env, no code change.
    Non-Groq model IDs and free-tier terms are more likely to drift than
    Groq's (which CLAUDE.md pins deliberately) — check the provider's own
    docs/console if a default model here stops being available.

    Not every provider/model handles response_format={"type":"json_object"}
    (used for decomposition and agent decisions) equally reliably — this is
    a real property of the chosen model, not something this transport layer
    can paper over. If a non-Groq provider's JSON-mode output looks
    consistently worse than Groq's, that's a model-quality tradeoff to
    weigh, not a bug here.

Why two temperatures?
    Decomposition (temp 0) needs to be deterministic and structured — it's
    closer to a parsing task than a creative one, and JSON output must be
    reliable. Generation (temp 0.7) benefits from some variability to
    produce natural, well-phrased answers while still being grounded by
    the retrieved context.
"""

import json
import os
import requests
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

# Registered OpenAI-compatible free-tier providers. Add an entry here (and
# document its env vars in .env.example) to support another one — nothing
# else in this file needs to change, since the chat-completions call shape
# is identical across all of them.
_PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-120b",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "model_env": "CEREBRAS_MODEL",
        "default_model": "llama-3.3-70b",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
    },
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
if LLM_PROVIDER not in _PROVIDERS:
    raise RuntimeError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Supported: {', '.join(_PROVIDERS)}"
    )

_provider = _PROVIDERS[LLM_PROVIDER]
_MODEL_NAME = os.getenv(_provider["model_env"], _provider["default_model"])
_API_KEY = os.getenv(_provider["api_key_env"])
_BASE_URL = _provider["base_url"]
_client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)

_DECOMPOSE_SYSTEM_PROMPT = """You are a query decomposition assistant for a research paper search system.

Given a user's question, break it into 2 to 4 focused sub-queries that together \
cover everything the original question is asking. Each sub-query should target \
one distinct piece of information so a semantic search can retrieve precisely \
relevant passages for it.

If the question is already narrow and single-focus, return it as a single \
sub-query — do not invent unnecessary splits.

Respond with ONLY a JSON object in this exact shape, no other text:
{"sub_queries": ["sub-query 1", "sub-query 2"]}"""


def _call_llm(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.7,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> str:
    """
    Send a single generation request to the active LLM provider.

    Shared low-level transport used by every LLM call in this module
    (decompose_query(), generate_answer(), generate_paper_summary(),
    generate_agent_decision()). Isolating the API call here means the rest
    of generator.py (and the rest of the pipeline) never touches the
    provider's request/response shape directly — and, per the module
    docstring's provider-modularity note, that shape is identical across
    every registered provider, so this function needs no per-provider
    branching.

    Args:
        prompt:      the user/instruction text to send
        system:      optional system prompt
        temperature: sampling temperature (0 = deterministic, higher = more varied)
        json_mode:   if True, asks the model to constrain output to valid
                     JSON (used for decomposition and agent decisions,
                     which must be machine-parseable)
        max_tokens:  optional cap on completion length. Passed through as-is
                     (None means no cap, matching the API default) — useful
                     for calls where the input is already close to the
                     account's tokens-per-minute limit and an unexpectedly
                     long completion could push the request over it (see
                     generate_paper_summary()).

    Returns:
        the model's raw text response (already stripped)

    Raises:
        RuntimeError: if the provider rejects the request or is unavailable.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        completion = _client.chat.completions.create(
            model=_MODEL_NAME,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"} if json_mode else None,
            max_tokens=max_tokens,
        )
        content = completion.choices[0].message.content
    except Exception as e:
        raise RuntimeError(
            f"LLM request failed (provider '{LLM_PROVIDER}', model '{_MODEL_NAME}'): {e}"
        ) from e

    if content is None:
        # Some models/providers return a null content field instead of text
        # (seen with cohere/north-mini-code:free on OpenRouter) — e.g. a
        # tool-call-only response, a content filter, or a provider quirk.
        # Treat it the same as any other failed generation rather than
        # leaking a raw AttributeError from .strip() on None: every caller
        # already knows how to handle RuntimeError (agent.py retries it,
        # decompose_query() falls back), so raising here keeps that
        # handling uniform instead of adding a new failure shape callers
        # don't expect.
        raise RuntimeError(
            f"LLM returned empty content (provider '{LLM_PROVIDER}', model '{_MODEL_NAME}')"
        )

    return content.strip()


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
        raw = _call_llm(
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
    just relays them to the LLM at temperature 0.7 and returns the raw
    answer text — it has no opinion on how the context was formatted.

    Args:
        system_prompt: grounding/citation instructions for the model
        user_prompt:   retrieved context + the user's question, formatted
                        and ready to send as-is

    Returns:
        the model's generated answer text

    Raises:
        RuntimeError: if the LLM provider is unavailable or errors — unlike
        decompose_query(), there is no safe fallback for generation, so
        this propagates up to pipeline.py / the FastAPI layer to surface
        as an error response.
    """
    return _call_llm(
        prompt=user_prompt,
        system=system_prompt,
        temperature=0.7,
        json_mode=False,
    )


_SUMMARY_SYSTEM_PROMPT = """You are summarizing an indexed research paper for a paper-search assistant.

Write a concise 4-6 sentence summary covering the paper's core contribution,
its method or approach at a high level, and its key result or finding. Plain
prose only — no headers, no markdown, no bullet points. Do not mention that
this is a summary or refer to yourself; just describe the paper."""

# Calibrated against Groq's on-demand tier, which caps requests at 8000
# tokens/minute (TPM) for this model — a full paper can run 10k-70k+
# tokens, well over that (measured ~1.7 tokens/word on a 42k-word paper via
# this same chunker.extract_text_by_page path). 3000 words keeps the input
# around ~5100 tokens; combined with the capped completion below (max ~400
# tokens) and the system prompt, the whole request stays comfortably under
# 8000 even on dense papers. Applied regardless of LLM_PROVIDER — it's a
# conservative budget that should stay safe on other providers' free tiers
# too, not something that needs per-provider tuning. Front-loaded content
# (abstract, intro, and usually a chunk of the method) covers what a
# summary needs most of the time — this is a real quality tradeoff for
# later sections, not truncation for its own sake.
_SUMMARY_INPUT_WORD_LIMIT = 3000
_SUMMARY_MAX_OUTPUT_TOKENS = 400


def generate_paper_summary(filename: str, paper_text: str) -> str:
    """
    Summarize one full paper for the get_paper_summary tool.

    Called once per newly-indexed PDF by index.py (temperature 0.3 — low
    enough to stay factual and consistent, with a little room since this is
    prose generation rather than structured parsing). The result is stored
    verbatim as a "type": "summary" chunk (vectorstore.add_summary) and
    served back as-is by tools.get_paper_summary — no retrieval or further
    generation involved at answer time.

    Args:
        filename:   source filename, included in the prompt so the model
                    can ground the summary in a concrete document
        paper_text: the paper's full extracted text — truncated to
                    _SUMMARY_INPUT_WORD_LIMIT words before sending (see
                    that constant's comment for why)

    Returns:
        the model's summary text (stripped)

    Raises:
        RuntimeError: if the LLM provider is unavailable or errors — unlike
        decompose_query(), a failed summary has no safe fallback; index.py
        surfaces this as a failure for that one paper rather than silently
        storing an empty summary.
    """
    words = paper_text.split()
    truncated_text = " ".join(words[:_SUMMARY_INPUT_WORD_LIMIT])

    return _call_llm(
        prompt=f"Paper filename: {filename}\n\n{truncated_text}",
        system=_SUMMARY_SYSTEM_PROMPT,
        temperature=0.3,
        json_mode=False,
        max_tokens=_SUMMARY_MAX_OUTPUT_TOKENS,
    )


def generate_agent_decision(system_prompt: str, user_prompt: str) -> str:
    """
    Generate one structured decision for the ReAct agent.

    The agent prompt must instruct the LLM to return either a tool action
    or a final answer as JSON. The raw JSON text is returned so agent.py
    owns parsing, validation, retries, and tool dispatch.
    """
    return _call_llm(
        prompt=user_prompt,
        system=system_prompt,
        temperature=0,
        json_mode=True,
    )


def get_active_provider_info() -> dict:
    """
    Report which provider/model the ReAct agent is currently configured to use.

    Backs GET /health so the frontend can display the real, live provider
    and model instead of a hardcoded string — necessary now that
    LLM_PROVIDER is configurable at runtime via .env (see "LLM Provider
    Modularity" in CLAUDE.md): a UI string baked in at write time would go
    stale the moment someone switches providers without touching the
    frontend.

    Returns:
        {"provider": str, "model": str}
    """
    return {"provider": LLM_PROVIDER, "model": _MODEL_NAME}


def check_llm_reachable(timeout: float = 5.0) -> bool:
    """
    Lightweight reachability check for the active LLM provider.

    Backs GET /health (api/main.py). Hits the provider's OpenAI-compatible
    /models listing endpoint directly via requests rather than going
    through the `openai` client — that endpoint is auth-gated but free (it
    lists available models rather than running a completion), so this
    costs no generation tokens against the account's rate limits. Kept
    here rather than in api/main.py so the HTTP layer doesn't need to know
    any provider's URL/key shape — api/main.py just calls this.

    Returns:
        True if the provider responds with 200, False on any failure
        (network error, bad key, timeout, non-200 status) — never raises,
        since a health check that itself throws defeats the purpose.
    """
    try:
        response = requests.get(
            f"{_BASE_URL}/models",
            headers={"Authorization": f"Bearer {_API_KEY}"},
            timeout=timeout,
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False
