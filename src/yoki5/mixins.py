"""Mixins."""
import typing as ty

import numpy as np
import typing_extensions as tye


class H5Protocol(tye.Protocol):
    """Mixin class"""

    def can_write(self) -> bool:
        """Check whether we can write."""
        ...

    def check_can_write(self):
        """Check whether we can write."""
        ...

    def has_array(self, *args, **kwargs) -> bool:
        """Get array."""
        ...

    def set_array(self, *args, **kwargs):
        """Get array."""
        ...

    def get_array(self, *args, **kwargs) -> np.ndarray:
        """Get array."""
        ...

    def get_arrays(self, *args, **kwargs) -> ty.Iterable[np.ndarray]:
        """Get array."""
        ...

    def set_attr(self, *args, **kwargs):
        """Get attribute."""
        ...

    def get_attr(self, *args, **kwargs) -> ty.Any:
        """Get array."""
        ...

    def get_dataset_attrs(self, *args, **kwargs) -> ty.Any:
        """Get array."""
        ...

    def has_group(self, *args, **kwargs) -> ty.Any:
        """Get array."""
        ...


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
