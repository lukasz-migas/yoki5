"""Mixins."""

from __future__ import annotations

import numpy as np

from yoki5.types import H5Protocol


class ColorMixin(H5Protocol):
    """Display mixin class."""

    COLOR_NAME_KEY = "Metadata"

    @property
    def color(self) -> np.ndarray:
        """Retrieve alternative registration name based on the image path."""
        return self.get_array(self.COLOR_NAME_KEY, "color")

    @color.setter
    def color(self, value: np.ndarray):
        self.check_can_write()
        self.set_array(self.COLOR_NAME_KEY, "color", value)


class DisplayMixin(H5Protocol):
    """Display mixin class."""

    DISPLAY_NAME_KEY = "Metadata"

    @property
    def name(self) -> str:
        """Retrieve alternative registration name based on the image path."""
        if self.has_group(self.DISPLAY_NAME_KEY):
            return self.get_attr(self.DISPLAY_NAME_KEY, "name", "")
        return ""

    @name.setter
    def name(self, value: str):
        self.check_can_write()
        self.set_attr(self.DISPLAY_NAME_KEY, "name", value)

    @property
    def display_name(self) -> str:
        """Retrieve alternative registration name based on the image path."""
        if self.has_group(self.DISPLAY_NAME_KEY):
            return self.get_attr(self.DISPLAY_NAME_KEY, "display_name", "")
        return ""

    @display_name.setter
    def display_name(self, value: str):
        self.check_can_write()
        self.set_attr(self.DISPLAY_NAME_KEY, "display_name", value)

    @property
    def about(self) -> str:
        """Retrieve alternative registration name based on the image path."""
        return self.get_attr(self.DISPLAY_NAME_KEY, "about")

    @about.setter
    def about(self, value: str):
        self.check_can_write()
        self.set_attr(self.DISPLAY_NAME_KEY, "about", value)
