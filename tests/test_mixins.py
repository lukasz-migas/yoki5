"""Test mixin classes."""

import numpy as np
from yoki5.base import Store
from yoki5.mixins import ColorMixin, DisplayMixin


class ColorStore(Store, ColorMixin):
    """Test mixin"""

    def __init__(self, path):
        super().__init__(path=path, groups=["Metadata"])


def test_color(tmp_path):
    """Test color."""
    store = ColorStore(tmp_path / "test.h5")
    color = np.random.randint(0, 255, 3)
    store.color = color
    assert np.allclose(store.color, color)


class DisplayStore(Store, DisplayMixin):
    """Test mixin"""

    def __init__(self, path):
        super().__init__(path=path, groups=["Metadata"])


def test_display(tmp_path):
    """Test color."""
    store = DisplayStore(tmp_path / "test.h5")
    assert not store.display_name
    store.display_name = "Test Display"
    assert store.display_name == "Test Display"
    assert not store.about
    store.about = "Test About"
    assert store.about == "Test About"
