"""Attributes."""
from contextlib import suppress


class Attributes:
    """Attributes"""

    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, item):
        with self.parent.open() as h5:
            return h5.attrs[item]

    def __setitem__(self, key, value):
        if self.parent.mode == "r":
            raise ValueError("Cannot save data in the `r` mode")
        with self.parent.open() as h5:
            h5.attrs[key] = value
            h5.flush()

    def to_dict(self):
        """Return all attributes."""
        attrs = {}
        with self.parent.open() as h5:
            keys = list(h5.attrs.keys())
            for key in keys:
                attrs[key] = h5.attrs[key]
        return attrs

    def get(self, key, default=None):
        """Get attribute if it exists."""
        with suppress(KeyError):
            return self[key]
        return default
