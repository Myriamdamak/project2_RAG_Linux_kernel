"""Split documents into chunks.

Two strategies, so the choice can be measured rather than assumed:

``section`` (default)
    Never crosses a section boundary. Inside a section, text is split into
    *blocks* (paragraphs, whole literal/code blocks, whole tables, whole list
    items) and blocks are packed greedily up to ``chunk_tokens``. Overlap is
    whole blocks from the end of the previous chunk, up to ``chunk_overlap``
    tokens. Each chunk carries a breadcrumb header ("doc title > section") that
    is embedded and indexed with it, so a chunk that only says "Default: 60"
    still knows it is about vm.swappiness.

``fixed``
    The textbook baseline: sliding windows of ``chunk_tokens`` words over the
    whole document text, ignoring structure. Kept only for the ablation.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from .config import settings
from .corpus import Document


@dataclass
class Chunk:
    id: str
    path: str
    title: str
    section: str              # "A > B > C"
    anchor: str
    doc_type: str
    subsystem: str
    authors: list[str]
    version: str
    date: str
    url: str
    text: str                 # body only, shown to users and the LLM
    header: str               # breadcrumb, prepended for indexing

    @property
    def index_text(self) -> str:
        return f"{self.header}\n{self.text}"

    def to_dict(self):
        return asdict(self)


def approx_tokens(text: str) -> int:
    # ~4 characters per token for English technical prose. Good enough for
    # packing; real token counts for cost come from the API's usage field.
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------
def split_blocks(text: str) -> list[str]:
    """Paragraph-level blocks that must not be cut in the middle.

    * a paragraph ending in ``::`` is glued to the indented literal block that
      follows (the command and its explanation stay together);
    * indented continuation (code, list item bodies, tables) stays with the
      block it belongs to.
    """
    raw = re.split(r"\n\s*\n", text)
    blocks: list[str] = []
    for para in raw:
        if not para.strip():
            continue
        is_indented = para[:1].isspace()
        if blocks and (is_indented or blocks[-1].rstrip().endswith("::")):
            blocks[-1] = blocks[-1] + "\n\n" + para
        else:
            blocks.append(para)
    return blocks


def _split_oversized(block: str, limit: int) -> list[str]:
    """A single block bigger than the limit (huge table, long code listing)
    is cut at line boundaries. This is the one place structure is broken."""
    out, cur, size = [], [], 0
    for line in block.splitlines():
        t = approx_tokens(line) + 1
        if cur and size + t > limit:
            out.append("\n".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += t
    if cur:
        out.append("\n".join(cur))
    return out


def _make_chunk(doc: Document, heading_path: list[str], anchor: str, body: str, n: int) -> Chunk:
    section = " > ".join(heading_path)
    cid = hashlib.sha1(f"{doc.path}|{section}|{n}|{body[:64]}".encode()).hexdigest()[:12]
    # the breadcrumb starts with the doc title unless the section path already does
    header = section if heading_path[:1] == [doc.title] else f"{doc.title} > {section}"
    return Chunk(id=cid, path=doc.path, title=doc.title, section=section,
                 anchor=anchor, doc_type=doc.doc_type, subsystem=doc.subsystem,
                 authors=doc.authors, version=doc.version, date=doc.date, url=doc.url,
                 text=body.strip(), header=f"[{doc.path}] {header}")


def chunk_section_aware(doc: Document, size: int, overlap: int, min_words: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    n = 0
    for sec in doc.sections:
        blocks: list[str] = []
        for b in split_blocks(sec.text):
            blocks.extend(_split_oversized(b, size) if approx_tokens(b) > size else [b])
        cur: list[str] = []
        cur_tok = 0
        for b in blocks:
            bt = approx_tokens(b)
            if cur and cur_tok + bt > size:
                chunks.append(_make_chunk(doc, sec.heading_path, sec.anchor, "\n\n".join(cur), n)); n += 1
                # overlap: carry trailing whole blocks while they fit
                carry, ct = [], 0
                for prev in reversed(cur):
                    pt = approx_tokens(prev)
                    if ct + pt > overlap:
                        break
                    carry.insert(0, prev); ct += pt
                cur, cur_tok = carry, ct
            cur.append(b)
            cur_tok += bt
        if cur:
            chunks.append(_make_chunk(doc, sec.heading_path, sec.anchor, "\n\n".join(cur), n)); n += 1
    return [c for c in chunks if len(c.text.split()) >= min_words]


def chunk_fixed(doc: Document, size: int, overlap: int, min_words: int) -> list[Chunk]:
    words = "\n\n".join(s.text for s in doc.sections).split()
    # size is in tokens; ~0.75 words per token
    w, o = max(20, int(size * 0.75)), int(overlap * 0.75)
    chunks, n, i = [], 0, 0
    while i < len(words):
        body = " ".join(words[i:i + w])
        chunks.append(_make_chunk(doc, [doc.title], "", body, n)); n += 1
        if i + w >= len(words):
            break
        i += w - o
    return [c for c in chunks if len(c.text.split()) >= min_words]


def chunk_documents(docs: list[Document], strategy: str | None = None,
                    size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    strategy = strategy or settings.chunker
    size = size or settings.chunk_tokens
    overlap = settings.chunk_overlap if overlap is None else overlap
    fn = chunk_section_aware if strategy == "section" else chunk_fixed
    out: list[Chunk] = []
    for d in docs:
        out.extend(fn(d, size, overlap, settings.min_chunk_words))
    # ids must be unique for citations; disambiguate rare collisions
    seen: dict[str, int] = {}
    for c in out:
        if c.id in seen:
            seen[c.id] += 1
            c.id = f"{c.id}{seen[c.id]}"
        else:
            seen[c.id] = 0
    return out
