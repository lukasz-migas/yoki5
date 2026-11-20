"""Stamp files with the id of the analysis step that wrote them.

The step id comes from a context variable (set with `step_context`, for work in the same process) or the
`PROVENANCE_STEP_ID` environment variable (for subprocesses). When neither is set nothing is stamped.
"""

from __future__ import annotations

import json
import os
import typing as ty
from contextlib import contextmanager
from contextvars import ContextVar

if ty.TYPE_CHECKING:
    import h5py

STEP_ID_ENV = "PROVENANCE_STEP_ID"
STEP_IDS_ATTR = "provenance_step_ids"

_CURRENT_STEP_ID: ContextVar[str | None] = ContextVar("provenance_step_id", default=None)


def current_step_id() -> str | None:
    """Return the id of the step currently running, if any."""
    return _CURRENT_STEP_ID.get() or os.environ.get(STEP_ID_ENV) or None


@contextmanager
def step_context(step_id: str) -> ty.Generator[None, None, None]:
    """Attribute writes made inside the block to `step_id`."""
    token = _CURRENT_STEP_ID.set(step_id)
    try:
        yield
    finally:
        _CURRENT_STEP_ID.reset(token)


def read_step_ids(h5: h5py.Group) -> list[str]:
    """Return the ids of the steps that wrote to a file, oldest first."""
    raw = h5.attrs.get(STEP_IDS_ATTR)
    if raw is None:
        return []
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        ids = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(step_id) for step_id in ids] if isinstance(ids, list) else []


def stamp(h5: h5py.Group, step_id: str) -> None:
    """Add `step_id` to the steps that wrote to a file, keeping order and skipping duplicates."""
    ids = read_step_ids(h5)
    if step_id not in ids:
        h5.attrs[STEP_IDS_ATTR] = json.dumps([*ids, step_id])
