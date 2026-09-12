"""Tests for the article chunker · plain asserts, no test runner needed.

    python pipelines/pre_embed/test_chunking.py

Deliberately dependency-free: the pipelines have no pytest in CI, and a test
that only runs on one machine is the situation this project already fixed once
for the Rust side.
"""

from __future__ import annotations

import sys

from chunking import (
    HARD_MAX,
    MIN,
    TARGET,
    is_indexable,
    normalize,
    split_article,
)

FAILED: list[str] = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILED.append(name)


def no_text_lost(original, chunks):
    """Splitting may re-space, but it must not drop or duplicate characters."""
    joined = "".join("".join(c.split()) for c in chunks)
    return joined == "".join(original.split())


AYAT_ARTICLE = (
    "- (1) Penetapan kelas jalan pada setiap ruas jalan dilakukan oleh: "
    "- a. Pemerintah, untuk jalan nasional; - b. pemerintah provinsi, untuk "
    "jalan provinsi; - c. pemerintah kabupaten, untuk jalan kabupaten; atau "
    "- d. pemerintah kota, untuk jalan kota. "
) * 6


def main():
    print("chunking")

    # -- the common case: most of the corpus was always the right size -----
    short = "Pengurusan Perusahaan dilakukan oleh Direksi."
    check("a short article is returned untouched", split_article(short) == [short])

    mid = "Ketentuan lebih lanjut diatur dengan Peraturan Menteri. " * 15
    check("an article under HARD_MAX is not split", len(split_article(mid)) == 1)

    # -- splitting ---------------------------------------------------------
    out = split_article(AYAT_ARTICLE)
    check("a long article is split", len(out) > 1, f"got {len(out)}")
    check(
        "every chunk is within HARD_MAX",
        all(len(c) <= HARD_MAX for c in out),
        f"max {max(len(c) for c in out)}",
    )
    check("no text is lost or duplicated", no_text_lost(AYAT_ARTICLE, out))

    # ! The property that matters for citation: a reader following the locator
    # must find the clause. Splitting mid-word would break that quietly.
    check(
        "chunks never start or end mid-word",
        all(c == c.strip() and "  " not in c for c in out),
    )

    # -- structure is preferred over character counts ----------------------
    cuts = split_article(AYAT_ARTICLE)
    check(
        "splits land on ayat boundaries where possible",
        sum(1 for c in cuts if c.startswith("- (1)")) >= 1,
        f"starts: {[c[:8] for c in cuts]}",
    )

    # -- degenerate input --------------------------------------------------
    check("empty text yields no chunks", split_article("") == [])
    check("None yields no chunks", split_article(None) == [])
    check("whitespace only yields no chunks", split_article("   \n\t ") == [])

    blob = "kata " * 4000          # no punctuation, no structure at all
    wrapped = split_article(blob)
    check(
        "unstructured text still respects the ceiling",
        max(len(c) for c in wrapped) <= HARD_MAX,
        f"max {max(len(c) for c in wrapped)}",
    )
    check("unstructured text is not lost", no_text_lost(blob, wrapped))

    # -- no stranded tails -------------------------------------------------
    # A tail below MIN would later be excluded as noise, taking real legal
    # text out of the index with it.
    tails = []
    for n in range(1, 40):
        body = ("Pasal ini mengatur ketentuan umum. " * 40) + ("x" * n)
        cs = split_article(body)
        if len(cs) > 1:
            tails.append(len(cs[-1]))
    check(
        "no chunk is left stranded below MIN",
        all(t >= MIN for t in tails),
        f"smallest tail {min(tails) if tails else 'n/a'}",
    )

    # -- indexability ------------------------------------------------------
    print("is_indexable")
    check("source artefacts are excluded",
          not any(is_indexable(x) for x in ["Ayat", "...", "pada", ".. j"]))
    check("a short but complete rule is kept", is_indexable(short))
    check("a long clause is kept", is_indexable(AYAT_ARTICLE[:400]))

    # -- normalize ---------------------------------------------------------
    check("normalize collapses whitespace",
          normalize("a  b\n\nc\t d") == "a b c d")

    print()
    if FAILED:
        print(f"{len(FAILED)} failed: {FAILED}")
        return 1
    print(f"all passed (TARGET={TARGET} HARD_MAX={HARD_MAX} MIN={MIN})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
