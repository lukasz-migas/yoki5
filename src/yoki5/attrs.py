"""Attributes."""

from __future__ import annotations

import typing as ty
from contextlib import suppress

if ty.TYPE_CHECKING:
    from yoki5.base import Store


class Attributes:
    """Attributes"""

    def __init__(self, parent: Store):
        self.parent = parent

    def __getitem__(self, item: str) -> ty.Any:
        """Get attribute."""
        with self.parent.open() as h5:
            return h5.attrs[item]

    def __contains__(self, key: str) -> bool:
        """Check attribute."""
        with self.parent.open() as h5:
            return key in h5.attrs

    def __setitem__(self, key: str, value: ty.Any) -> None:
        """Set attribute."""
        self.parent.check_can_write()
        with self.parent.open() as h5:
            h5.attrs[key] = value
            h5.flush()

    def to_dict(self) -> dict[str, ty.Any]:
        """Return all attributes."""
        attrs = {}
        with self.parent.open() as h5:
            keys = list(h5.attrs.keys())
            for key in keys:
                attrs[key] = h5.attrs[key]
        return attrs

    def get(self, key: str, default: ty.Any = None) -> ty.Any:
        """Get attribute if it exists."""
        with suppress(KeyError, PermissionError):
            return self[key]
        return default
