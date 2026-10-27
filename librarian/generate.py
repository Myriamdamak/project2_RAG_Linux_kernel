"""Answer generation with per-claim citations.

The model only sees the retrieved chunks, numbered [1]..[k]. It must cite a
chunk number after every sentence that states a fact, and must answer exactly
"I don't know" when the chunks do not contain the answer. We then *check*
the citations in code (``parse_citations``): numbers that do not exist are
removed and reported, so a hallucinated "[9]" never reaches the user as a link.

We call the OpenAI Responses API with ``requests`` rather than the SDK: one
function, no extra dependency, and every field of the request is visible.
"""
from __future__ import annotations

import os
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .bm25 import tokenize
from .config import settings

IDK = "I don't know"

SYSTEM_PROMPT = f"""You are The Librarian, a question-answering assistant for the Linux kernel documentation (the Documentation/ tree of the kernel source).

You will receive numbered source excerpts and a question. Rules:
1. Answer ONLY from the excerpts. Do not use outside knowledge, even if you are confident.
2. After every sentence that states a fact, cite the excerpt(s) that support it with bracketed numbers, e.g. "The default is 60 [2]." or "[1][3]". Cite only excerpts that actually say it.
3. If the excerpts do not contain the answer, reply with exactly: {IDK}
   Then, on a new line, one sentence saying what is missing. Do not guess.
4. If the excerpts answer only part of the question, answer that part with citations and say plainly which part is not covered.
5. Be concise: usually 1-5 sentences. Keep exact names of files, sysctls, config options, commands and functions as written in the excerpts. Use a short code block for commands.
6. Questions refer to the kernel version stated in the excerpts' header; do not speculate about other versions."""

REWRITE_PROMPT = """Rewrite the user's question into 1-3 short search queries for a keyword+semantic search over the Linux kernel Documentation tree.
- Expand vague or very short questions with the likely kernel terms (sysctl names, subsystem names, config options).
- Split multi-part questions into one query per part.
Return only the queries, one per line, no numbering, no commentary."""


_BRACKET_NUM = re.compile(r"\[(\d+)\]")


def format_sources(chunks) -> str:
    """Number the excerpts [1]..[k]. Documents contain their own bracketed
    numbers (footnotes like "fstests[2]", array sizes), which the model or our
    citation parser could mistake for citations, so they become (2) here."""
    parts = []
    for n, c in enumerate(chunks, 1):
        when = f", last changed {c.date}" if c.date else ""
        text = _BRACKET_NUM.sub(r"(\1)", c.text)
        parts.append(f"[{n}] {c.path} — {c.section} (kernel {c.version}{when})\n{text}")
    return "\n\n".join(parts)


_CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`(])")


def parse_citations(answer: str, n_sources: int) -> dict:
    """Validate citation numbers and measure citation coverage.

    Returns the cleaned answer, which source numbers were cited, invalid
    numbers that were stripped, and the share of factual sentences that carry
    at least one citation (a cheap, offline proxy for faithfulness).
    """
    invalid: set[int] = set()

    def fix(m):
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        ok = [x for x in nums if 1 <= x <= n_sources]
        invalid.update(x for x in nums if not 1 <= x <= n_sources)
        return "".join(f"[{x}]" for x in ok)

    cleaned = _CITE.sub(fix, answer)
    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", cleaned)})
    body = cleaned.split("```")
    prose = " ".join(body[::2])                     # ignore code blocks
    sentences = [s for s in _SENT.split(prose.strip()) if len(s.split()) >= 4]
    with_cite = sum(1 for s in sentences if re.search(r"\[\d+\]", s))
    return {
        "answer": cleaned.strip(),
        "cited": cited,
        "invalid_citations": sorted(invalid),
        "citation_coverage": (with_cite / len(sentences)) if sentences else 0.0,
        "idk": cleaned.strip().lower().startswith(IDK.lower()),
    }


# --------------------------------------------------------------------------
class OpenAIGenerator:
    """OpenAI Responses API over REST; no additional SDK dependency."""

    def __init__(self, model: str):
        self.model = model
        self.key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY is not set (or use LIB_GENERATOR=extractive)")
        self.base = settings.openai_base_url.rstrip("/")
        self.session = requests.Session()
        # Retry explicit transient failures, but not ambiguous read timeouts.
        retry = Retry(total=3, connect=3, read=0, status=3, backoff_factor=1,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset({"POST"}), raise_on_status=False)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))

    def _call(self, model, system, user, max_tokens, json_mode=False):
        body = {
            "model": model,
            "instructions": system,
            "input": user,
            "max_output_tokens": max_tokens,
            "store": False,
        }
        if settings.reasoning_effort:
            body["reasoning"] = {"effort": settings.reasoning_effort}
        if json_mode:
            # Responses JSON mode validates input messages, not just instructions.
            body["input"] = "Return your evaluation as JSON.\n\n" + user
            body["text"] = {"format": {"type": "json_object"}}
        r = self.session.post(f"{self.base}/responses", timeout=60,
                              headers={"Authorization": f"Bearer {self.key}",
                                       "Content-Type": "application/json"}, json=body)
        if r.status_code != 200:
            # Report safe machine-readable fields, never raw bodies or prompts.
            detail = ""
            try:
                error = r.json().get("error") or {}
                fields = [f"{name}={error[name]}" for name in ("type", "code", "param")
                          if isinstance(error.get(name), str)
                          and re.fullmatch(r"[A-Za-z0-9_.\[\]-]{1,100}", error[name])
                          and self.key not in error[name]]
                if fields:
                    detail = " (" + ", ".join(fields) + ")"
            except (ValueError, AttributeError, TypeError):
                pass
            raise RuntimeError(f"OpenAI API HTTP {r.status_code}{detail}")
        data = r.json()
        status = data.get("status")
        if status != "completed":
            reason = (data.get("incomplete_details") or {}).get("reason", status)
            raise RuntimeError(f"OpenAI response was not completed ({reason})")
        parts = [part for item in data.get("output", []) if item.get("type") == "message"
                 for part in item.get("content", [])]
        if any(p.get("type") == "refusal" for p in parts):
            raise RuntimeError("OpenAI declined the request")
        text = "".join(p.get("text", "") for p in parts if p.get("type") == "output_text")
        if not text.strip():
            raise RuntimeError("OpenAI returned no answer text")
        return text, normalise_usage(data.get("usage") or {})

    def answer(self, question: str, chunks) -> tuple[str, dict]:
        user = f"<sources>\n{format_sources(chunks)}\n</sources>\n\nQuestion: {question}"
        return self._call(self.model, SYSTEM_PROMPT, user, settings.max_answer_tokens)

    def rewrite(self, question: str) -> tuple[list[str], dict]:
        text, usage = self._call(settings.rewrite_model, REWRITE_PROMPT, question, settings.max_rewrite_tokens)
        qs = [q.strip(" -•\t") for q in text.splitlines() if q.strip()]
        return qs[:3], usage


def normalise_usage(u: dict) -> dict:
    """Separate cached input and reasoning output without double-counting."""
    cached = (u.get("input_tokens_details") or {}).get("cached_tokens", 0)
    reasoning = (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
    return {
        "input_tokens": max(0, u.get("input_tokens", 0) - cached),
        "cached_input_tokens": cached,
        "output_tokens": max(0, u.get("output_tokens", 0) - reasoning),
        "thinking_tokens": reasoning,
    }


class ExtractiveGenerator:
    """Offline baseline: no LLM. Picks the sentences of the retrieved chunks
    that share the most terms with the question and cites them. It cannot
    synthesise, but it cannot hallucinate either, which makes it a useful
    floor for the generation eval and lets the service run without a key."""
    model = "extractive"

    def answer(self, question: str, chunks) -> tuple[str, dict]:
        q = set(tokenize(question))
        scored = []
        for n, c in enumerate(chunks[:4], 1):
            for s in re.split(r"(?<=[.!?])\s+|\n\n", c.text):
                s = _BRACKET_NUM.sub(r"(\1)", " ".join(s.split()))
                if len(s.split()) < 5:
                    continue
                overlap = len(q & set(tokenize(s))) / (len(q) or 1)
                scored.append((overlap - 0.02 * n, s, n))
        scored.sort(reverse=True)
        if not scored or scored[0][0] < 0.25:
            return f"{IDK}\nNo retrieved passage matches the question closely enough.", {}
        picked = scored[:2]
        return " ".join(f"{s.rstrip('.')}. [{n}]" for _, s, n in picked), {}

    def rewrite(self, question):
        return [], {}


def get_generator(spec: str | None = None):
    spec = spec or settings.generator
    if spec.startswith("openai:"):
        return OpenAIGenerator(spec.split(":", 1)[1])
    if spec == "extractive":
        return ExtractiveGenerator()
    raise ValueError(f"unknown generator {spec!r}")
