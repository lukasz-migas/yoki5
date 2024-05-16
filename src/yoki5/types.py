"""Types."""

import typing as ty
from pathlib import Path

import numpy as np

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class H5Protocol(Protocol):
    """Mixin class."""

    path: Path

    def open(self, *args, **kwargs):
        """Open dataset."""
        ...

    def get_unique_name(self, *args, **kwargs):
        """Get unique name by incrementing.."""
        ...

    def has_data(self, *args, **kwargs):
        """Get unique name by incrementing.."""
        ...

    def add_data_to_dataset(self, *args, **kwargs):
        """Add data to group.."""
        ...

    def get_group_names(self, *args, **kwargs):
        """Get group names."""
        ...

    def get_df(self, *args, **kwargs):
        """Get dataframe."""
        ...

    def get_dataset_data(self, *args, **kwargs):
        """Get dataset data."""
        ...

    def get_dataset_data_attrs(self, *args, **kwargs):
        """Get dataset data."""
        ...

    def can_write(self) -> bool:
        """Check whether we can write."""
        ...

    def enable_write(self):
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


class H5MultiProtocol(Protocol):
    """Multi-protocol class."""

    _objs: ty.Dict[str, H5Protocol]

    def can_write(self) -> bool:
        """Check whether we can write."""
        ...

    def _get_any_obj(self) -> ty.Any:
        """Get object."""
        ...
