"""Split a regulation article into retrieval-sized chunks.

! There was no chunking before this. The source holds exactly one row per
(regulation, chapter, article) — verified: zero article groups have more than
one row, and chunk_id is 1 everywhere — and ingest copied those rows 1:1. So an
article of 32,000 characters became a single 32,000-character "chunk".

That breaks three things at once:

  retrieval   one vector averaging an entire annex points nowhere, which is a
              standing suspect for the dense arm sitting at 0.0% Recall@5
  citation    "the whole LAMPIRAN" sends a human to a haystack, and the
              verifiable locator is the product (CLAUDE.md §5.5)
  the gate    lexical evidence is measured against retrieved bodies, so chunk
              size moves the floor the domain gate is tuned to

Splitting follows the document's own structure rather than a character count:
Indonesian regulations mark ayat as "- (1)" and lettered items as "- a.", and
those boundaries are where a human would cut too. Character limits only apply
once structure runs out.

! Provenance is preserved, never invented. Every chunk keeps its regulation,
chapter and article, and carries its position within the article. A chunk is
addressable as "Pasal 12, part 2 of 4" — still a real locator.
"""

from __future__ import annotations

import re

# An article at or under this stays whole: one clause, one chunk, one citation.
TARGET = 800
# Packing may overshoot to here rather than emit a stranded tail.
HARD_MAX = 1200
# Below this a trailing fragment is merged into its neighbour rather than left
# stranded — the text is still part of the law.
MIN = 80
# Indexability is about whether a chunk says anything, ✗ whether it is long.
# "Pengurusan Perusahaan dilakukan oleh Direksi." is 44 characters and a
# complete, citable rule; a length floor alone would throw it away.
MIN_INDEX_CHARS = 25
MIN_INDEX_WORDS = 5

# Ayat: "- (1)", "(2)", at a boundary. Lettered items: "- a.", "b.".
AYAT = re.compile(r"(?=(?:^|\s)-?\s*\(\d{1,2}\)\s)")
ITEM = re.compile(r"(?=(?:^|\s)-\s*[a-z]\.\s)")
# A sentence end followed by something that starts one: keeps "Rp1.000.000,00"
# and "Pasal 12." from being treated as sentence boundaries.
SENTENCE = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")


def normalize(text: str) -> str:
    """Collapse whitespace runs without touching the text itself."""
    return " ".join(text.split())


def _split_by(pattern: re.Pattern[str], unit: str) -> list[str]:
    parts = [p.strip() for p in pattern.split(unit)]
    return [p for p in parts if p]


def _hard_wrap(unit: str, limit: int) -> list[str]:
    """Last resort: break on whitespace at `limit`, never mid-word."""
    words, out, cur = unit.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > limit:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        out.append(cur)
    return out


def _atoms(text: str) -> list[str]:
    """Break text down until every piece is at most HARD_MAX, structure first."""
    out: list[str] = []
    for ayat in _split_by(AYAT, text) or [text]:
        if len(ayat) <= HARD_MAX:
            out.append(ayat)
            continue
        for item in _split_by(ITEM, ayat) or [ayat]:
            if len(item) <= HARD_MAX:
                out.append(item)
                continue
            for sent in _split_by(SENTENCE, item) or [item]:
                if len(sent) <= HARD_MAX:
                    out.append(sent)
                else:
                    out.extend(_hard_wrap(sent, TARGET))
    return out


def split_article(text: str) -> list[str]:
    """Chunks for one article, in order. Never returns empty for real text.

    An article that already fits is returned untouched — most of the corpus —
    so this is a no-op for the 45% that were always the right size.
    """
    body = normalize(text or "")
    if not body:
        return []
    if len(body) <= HARD_MAX:
        return [body]

    chunks: list[str] = []
    cur = ""
    for atom in _atoms(body):
        if not cur:
            cur = atom
        elif len(cur) + 1 + len(atom) <= TARGET:
            cur = f"{cur} {atom}"
        else:
            chunks.append(cur)
            cur = atom
    if cur:
        chunks.append(cur)

    # ! A stranded tail is worse than a slightly oversized chunk: it is the
    # fragment that would later be excluded, taking real legal text with it.
    if len(chunks) > 1 and len(chunks[-1]) < MIN:
        tail = chunks.pop()
        chunks[-1] = f"{chunks[-1]} {tail}"
    return chunks


def is_indexable(body: str) -> bool:
    """Whether a chunk can carry its own weight as a retrieval unit.

    ! Being unindexable is not the same as being discarded. The row is still
    stored and still readable through `read_chunk`; it simply does not compete
    in search, where it would only ever be noise.
    """
    if len(body) < MIN_INDEX_CHARS:
        return False
    # A few real words, not a few characters — "Ayat", "...", "pada" are
    # artefacts of the source extraction, not clauses.
    return len(re.findall(r"\w\w+", body)) >= MIN_INDEX_WORDS
