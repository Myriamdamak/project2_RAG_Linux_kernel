"""Load the kernel Documentation tree into structured documents.

Three source shapes, each parsed differently:

* ``.rst``  reStructuredText. We recover the heading hierarchy so every chunk
  knows which section it came from ("Title > Section > Subsection").
* ``.txt``  legacy plain text. Headings are detected with the same underline
  rule when present, otherwise the file is one section.
* ``ABI/``  sysfs ABI descriptions (no extension). Each ``What:`` entry is its
  own section, because users ask about one sysfs file at a time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import KERNEL_TAG, settings

UNDERLINE_CHARS = set("=-~^*+#\"'`:.,_")
_ADORN = re.compile(r"^([=\-~^*+#\"'`:.,_])\1{2,}\s*$")


@dataclass
class Section:
    heading_path: list[str]
    text: str
    anchor: str = ""          # slug of the deepest heading, for links


@dataclass
class Document:
    path: str                 # relative to Documentation/, e.g. admin-guide/sysctl/vm.rst
    title: str
    doc_type: str             # rst | txt | abi
    subsystem: str            # first path component, e.g. admin-guide
    authors: list[str] = field(default_factory=list)
    version: str = KERNEL_TAG
    date: str = ""            # last git change, "YYYY-MM-DD" or "<2021-01-01"
    sections: list[Section] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://github.com/torvalds/linux/blob/{self.version}/Documentation/{self.path}"


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# --------------------------------------------------------------------------
# reStructuredText / text
# --------------------------------------------------------------------------
_DROP_DIRECTIVES = ("toctree", "only", "raw", "kernel-figure", "image", "figure",
                    "contents", "sectnum", "include", "literalinclude")
_AUTHOR_FIELD = re.compile(r"^:(?:Authors?|Author\(s\)|Maintainers?):\s*(.*)$", re.I)
_AUTHOR_LINE = re.compile(r"^(?:Authors?|Written by|Copyright \(c\)[\d ,\-]*)[: ]\s*(.+)$", re.I)


def _clean_rst(lines: list[str]) -> list[str]:
    """Drop markup that carries no answerable content.

    We keep ``code-block``/literal blocks (users ask about commands) and drop
    toctrees, images and SPDX headers. ``kernel-doc::`` directives are dropped
    too: they pull API docs out of C source comments at build time, and that
    text is not in this tree (a known failure case, see design.md).
    """
    out: list[str] = []
    skip_indent = -1               # >=0 while inside a dropped directive
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if skip_indent >= 0:
            # a dropped directive's body is indented deeper than the
            # directive itself; it ends at the first line that is not
            if stripped and indent <= skip_indent:
                skip_indent = -1
            else:
                continue
        if stripped.startswith(".. SPDX") or re.match(r"^\.\. _[^:]+:\s*$", stripped):
            continue
        m = re.match(r"^\.\.\s+([\w-]+)::", stripped)
        if m and (m.group(1) in _DROP_DIRECTIVES or m.group(1).startswith("kernel-")):
            skip_indent = indent
            continue
        if stripped.startswith(".. |") or stripped == "..":   # substitutions / comments
            continue
        out.append(line.rstrip())
    return out


def _is_heading(lines: list[str], i: int) -> tuple[bool, int, str]:
    """Return (is_heading, lines_consumed, style) for a heading starting at i.

    RST headings are a text line with an adornment underline at least as long,
    optionally with a matching overline. The style (char + overline?) defines
    the level, in order of first appearance within the document.
    """
    n = len(lines)
    # overline + title + underline
    if i + 2 < n and _ADORN.match(lines[i]) and lines[i + 1].strip() and _ADORN.match(lines[i + 2]):
        if lines[i][0] == lines[i + 2][0]:
            return True, 3, "o" + lines[i][0]
    # title + underline
    if (i + 1 < n and lines[i].strip() and not lines[i][:1].isspace()
            and not _ADORN.match(lines[i]) and _ADORN.match(lines[i + 1])
            and len(lines[i + 1].rstrip()) >= len(lines[i].rstrip()) - 1):
        return True, 2, "u" + lines[i + 1][0]
    return False, 0, ""


def parse_rst(text: str, path: str, doc_type: str) -> Document:
    lines = text.splitlines()
    authors = []
    for line in lines[:80]:
        m = _AUTHOR_FIELD.match(line.strip()) or _AUTHOR_LINE.match(line.strip())
        if m and m.group(1).strip():
            authors.append(re.sub(r"\s*<[^>]+>", "", m.group(1)).strip(" -"))
    lines = _clean_rst(lines)

    styles: list[str] = []          # heading style -> level by first appearance
    stack: list[str] = []
    sections: list[Section] = []
    buf: list[str] = []
    title = ""

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(list(stack) or [title or path], body,
                                    slugify(stack[-1]) if stack else ""))
        buf.clear()

    i = 0
    while i < len(lines):
        ok, used, style = _is_heading(lines, i)
        if ok:
            heading = lines[i + 1 if used == 3 else i].strip()
            flush()
            if style not in styles:
                styles.append(style)
            level = styles.index(style)
            stack = stack[:level] + [heading]
            if not title:
                title = heading
            i += used
            continue
        if _ADORN.match(lines[i]) and len(lines[i].strip()) > 20:   # transition line
            i += 1
            continue
        buf.append(lines[i])
        i += 1
    flush()

    if not title:
        title = Path(path).stem.replace("-", " ").replace("_", " ")
    return Document(path=path, title=title, doc_type=doc_type,
                    subsystem=path.split("/")[0] if "/" in path else "core",
                    authors=authors[:5], sections=sections)


# --------------------------------------------------------------------------
# ABI files
# --------------------------------------------------------------------------
_ABI_KEY = re.compile(r"^(What|Date|Contact|Description|Users|KernelVersion):\s*(.*)$")


def parse_abi(text: str, path: str) -> Document:
    """Each entry starts with one or more ``What:`` lines."""
    entries: list[dict] = []
    cur: dict | None = None
    key = None
    for line in text.splitlines():
        m = _ABI_KEY.match(line)
        if m:
            k, v = m.groups()
            if k == "What" and (cur is None or key != "What"):
                cur = {"What": [], "Description": [], "Date": "", "Contact": "",
                       "Users": [], "KernelVersion": ""}
                entries.append(cur)
            if cur is None:
                continue
            key = k
            if k in ("What", "Description", "Users"):
                cur[k].append(v) if v else None
            else:
                cur[k] = v
        elif cur is not None and key in ("Description", "Users", "What"):
            cur[key].append(line.strip())
    doc = Document(path=path, title=f"ABI: {Path(path).name}", doc_type="abi",
                   subsystem="ABI")
    for e in entries:
        whats = [w for w in e["What"] if w]
        if not whats:
            continue
        desc = "\n".join(e["Description"]).strip()
        meta = []
        if e["Date"]:
            meta.append(f"Date: {e['Date']}")
        if e["KernelVersion"]:
            meta.append(f"KernelVersion: {e['KernelVersion']}")
        if e["Contact"]:
            meta.append(f"Contact: {re.sub(r'<[^>]+>', '', e['Contact']).strip()}")
        body = "What: " + "\nWhat: ".join(whats) + "\n" + "\n".join(meta) + "\n\n" + desc
        doc.sections.append(Section([doc.title, whats[0]], body.strip(), slugify(whats[0])))
    return doc


# --------------------------------------------------------------------------
# kernel-parameters.txt
# --------------------------------------------------------------------------
_PARAM = re.compile(r"^\t([A-Za-z0-9_.\-\[\]<>/]+=?)(?:\s|$)")


def parse_kernel_params(text: str, path: str) -> Document:
    """The most-asked file in the tree is one 8,700-line list of boot
    parameters. Generic chunking cut entries in half and lost the parameter
    name, so each entry (a line at one-tab indent starting with ``name=`` or
    ``name [FLAGS]``) becomes its own section headed by the parameter name."""
    doc = Document(path=path, title="The kernel's command-line parameters",
                   doc_type="txt", subsystem="admin-guide")
    name, buf = "Introduction and flag abbreviations", []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            doc.sections.append(Section([doc.title, name], body, slugify(name)))

    for line in text.splitlines():
        m = _PARAM.match(line)
        first = line.split()[0] if line.strip() else ""
        if m and ("=" in first or "[" in line):
            flush()
            name, buf = m.group(1).rstrip("="), []
        buf.append(line.rstrip())
    flush()
    return doc


# --------------------------------------------------------------------------
def iter_files(root: Path):
    excluded = set(settings.excluded)
    for p in sorted(root.rglob("*")):
        # The corpus checkout can contain Unix symlinks that Windows cannot
        # resolve (e.g. Documentation/Changes); they are not source documents.
        if p.is_symlink():
            continue
        try:
            if not p.is_file():
                continue
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1920:
                continue  # Linux link represented as an unreadable Windows reparse point
            raise
        rel = p.relative_to(root).as_posix()
        if rel.split("/")[0] in excluded or p.name.startswith("."):
            continue
        if rel.startswith("ABI/") and "/" in rel[4:] and p.suffix == "" and not rel.endswith("README"):
            yield rel, "abi", p
        elif p.suffix == ".rst":
            yield rel, "rst", p
        elif p.suffix == ".txt":
            yield rel, "txt", p


def load_dates(root: Path) -> dict[str, str]:
    """{path: last-changed date} written by scripts/fetch_corpus.sh."""
    import json
    p = root.parent.parent / "dates.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_corpus(root: str | Path | None = None) -> list[Document]:
    root = Path(root or settings.corpus_dir)
    if not root.exists():
        raise SystemExit(f"Corpus not found at {root}. Run: make corpus")
    dates = load_dates(root)
    docs = []
    for rel, kind, p in iter_files(root):
        text = p.read_text(encoding="utf-8", errors="replace")
        if kind == "abi":
            doc = parse_abi(text, rel)
        elif rel == "admin-guide/kernel-parameters.txt":
            doc = parse_kernel_params(text, rel)
        else:
            doc = parse_rst(text, rel, kind)
        doc.date = dates.get(rel, "")
        if doc.sections:
            docs.append(doc)
    return docs
