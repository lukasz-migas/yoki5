"""Types."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

if ty.TYPE_CHECKING:
    import pandas as pd
    from scipy.sparse import spmatrix


class H5Protocol(Protocol):
    """Mixin class."""

    path: Path

    def open(self, mode: str | None = None) -> ty.Generator[ty.Any, None, None]:
        """Open dataset."""
        ...

    def has_any_data(self, group: str) -> bool:
        """Get unique name by incrementing."""
        ...

    def add_data_to_group(
        self,
        group: str,
        data: dict | spmatrix,
        attributes: dict | None = None,
        dtype: ty.Any = None,
        as_sparse: bool = False,
        **kwargs: ty.Any,
    ) -> None:
        """Get group names."""
        ...

    def get_df(self, group: str) -> pd.DataFrame:
        """Get dataframe."""
        ...

    def get_group_data(self, group: str) -> tuple[dict, list[str], list[str]]:
        """Get dataset data."""
        ...

    def get_group_data_and_attrs(self, group: str) -> tuple[dict, dict]:
        """Get dataset data."""
        ...

    def can_write(self) -> bool:
        """Check whether we can write."""
        ...

    def enable_write(self) -> ty.Generator[ty.Any, None, None]:
        """Check whether we can write."""
        ...

    def check_can_write(self, msg: str) -> bool:
        """Check whether we can write."""
        ...

    def has_array(self, group: str, name: str) -> bool:
        """Get array."""
        ...

    def set_array(self, group: str, name: str, array: np.ndarray, dtype: ty.Any = None, **kwargs: ty.Any) -> None:
        """Get array."""
        ...

    def get_array(self, group: str, name: str) -> np.ndarray:
        """Get array."""
        ...

    def get_arrays(self, group: str, *names: str) -> list[np.ndarray]:
        """Get array."""
        ...

    def set_attr(self, group: str, attr: str, value: str | int | float | bool) -> None:
        """Get attribute."""
        ...

    def get_attr(self, group: str, attr: str, default: ty.Any = None) -> ty.Any:
        """Get array."""
        ...

    def get_group_attrs(self, *args, **kwargs) -> ty.Any:
        """Get array."""
        ...

    def has_group(self, *groups: str) -> bool:
        """Get array."""
        ...


class H5MultiProtocol(Protocol):
    """Multi-protocol class."""

    _objs: dict[str, H5Protocol]

    def can_write(self) -> bool:
        """Check whether we can write."""
        ...

    def _get_any_obj(self) -> ty.Any:
        """Get object."""
        ...
