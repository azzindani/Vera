"""Byte-aware batching for the embedding endpoint.

! A fixed batch size is not safe on this corpus. Chunk length spans 3 chars to
32,767, so 32 documents can be 2 KB or 1 MB depending on where you are in the
table. TEI rejects payloads over 2 MB with HTTP 413, and a fixed size means the
run dies partway through -- after burning GPU hours.

Budget by serialized bytes AND count, whichever binds first.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")

# TEI's default payload_limit is 2,000,000 bytes. Stay well under: JSON
# escaping inflates non-ASCII, and Indonesian legal text is full of it.
MAX_BATCH_BYTES = 1_200_000
MAX_BATCH_ITEMS = 32


def batched(
    items: Sequence[T],
    text_of=lambda x: x,
    max_bytes: int = MAX_BATCH_BYTES,
    max_items: int = MAX_BATCH_ITEMS,
) -> Iterator[list[T]]:
    """Yield batches bounded by both payload size and item count."""
    batch: list[T] = []
    size = 0
    for item in items:
        n = len(text_of(item).encode("utf-8"))
        # A single oversized document still has to go out on its own; TEI will
        # reject it loudly rather than silently truncating, which is correct.
        if batch and (size + n > max_bytes or len(batch) >= max_items):
            yield batch
            batch, size = [], 0
        batch.append(item)
        size += n
    if batch:
        yield batch
