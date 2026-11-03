"""Initialization shares a single HDF5 context."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import h5py
import pytest

from yoki5.base import Store


def test_initialize_opens_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Store.open
    calls = []

    @contextmanager
    def tracked(self: Store, mode: str | None = None) -> Iterator[h5py.File]:
        calls.append(mode)
        with original(self, mode) as h5:
            yield h5

    monkeypatch.setattr(Store, "open", tracked)
    store = Store(tmp_path / "groups.h5", groups=["one", "two"], attributes={"VERSION": "1.0"})
    assert len(calls) == 1
    assert store.HDF5_GROUPS == ["one", "two"]
    assert store.HDF5_ATTRIBUTES["VERSION"] == "1.0"
    calls.clear()
    reopened = Store(store.path, mode="r")
    assert len(calls) == 1
    assert reopened.HDF5_GROUPS == store.HDF5_GROUPS
