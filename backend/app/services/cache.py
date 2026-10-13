"""A small in-process cache for results that are expensive to recompute."""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable

import pandas as pd


def frame_fingerprint(frame: pd.DataFrame) -> str:
    """A digest that changes whenever any date or value in the frame changes."""
    hashes = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()


class ResultCache[T]:
    """Least-recently-used cache keyed by a data fingerprint.

    Keying on the data itself means there is nothing to invalidate: an upload
    that changes a dataset simply produces a different key. Computation happens
    under a lock, so concurrent requests for the same result do the work once
    (and a small host is never asked to train several models at the same time).
    """

    def __init__(self, max_entries: int = 64) -> None:
        self._entries: OrderedDict[str, T] = OrderedDict()
        self._max_entries = max_entries
        self._lock = threading.Lock()

    def get_or_compute(self, key: str, compute: Callable[[], T]) -> T:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key]
            value = compute()
            self._entries[key] = value
            if len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
