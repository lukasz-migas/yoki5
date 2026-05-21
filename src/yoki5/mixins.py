"""Mixins."""

from __future__ import annotations

import typing as ty

import numpy as np

from yoki5.types import H5Protocol


class ColorMixin(H5Protocol):
    """Mixin exposing a color array stored in the metadata group."""

    COLOR_NAME_KEY = "Metadata"

    @property
    def color(self) -> np.ndarray:
        """Return the stored RGB/RGBA color array."""
        return ty.cast(np.ndarray, self.get_array(self.COLOR_NAME_KEY, "color"))

    @color.setter
    def color(self, value: np.ndarray) -> None:
        """Store the RGB/RGBA color array."""
        self.check_can_write()
        self.set_array(self.COLOR_NAME_KEY, "color", value)


class DisplayMixin(H5Protocol):
    """Mixin exposing human-readable metadata attributes."""

    DISPLAY_NAME_KEY = "Metadata"

    @property
    def name(self) -> str:
        """Return the stored name, or an empty string if unavailable."""
        if self.has_group(self.DISPLAY_NAME_KEY):
            return str(self.get_attr(self.DISPLAY_NAME_KEY, "name", ""))
        return ""

    @name.setter
    def name(self, value: str) -> None:
        """Store the name metadata attribute."""
        self.check_can_write()
        self.set_attr(self.DISPLAY_NAME_KEY, "name", value)

    @property
    def display_name(self) -> str:
        """Return the display name, or an empty string if unavailable."""
        if self.has_group(self.DISPLAY_NAME_KEY):
            return str(self.get_attr(self.DISPLAY_NAME_KEY, "display_name", ""))
        return ""

    @display_name.setter
    def display_name(self, value: str) -> None:
        """Store the display name metadata attribute."""
        self.check_can_write()
        self.set_attr(self.DISPLAY_NAME_KEY, "display_name", value)

    @property
    def about(self) -> str:
        """Return the free-text description, or an empty string if unavailable."""
        return str(self.get_attr(self.DISPLAY_NAME_KEY, "about", ""))

    @about.setter
    def about(self, value: str) -> None:
        """Store the free-text description metadata attribute."""
        self.check_can_write()
        self.set_attr(self.DISPLAY_NAME_KEY, "about", value)
