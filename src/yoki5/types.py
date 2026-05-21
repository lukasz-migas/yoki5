"""Types."""

from __future__ import annotations

import typing as ty
from pathlib import Path
from typing import Protocol

import numpy as np

if ty.TYPE_CHECKING:
    import pandas as pd
    from scipy.sparse import spmatrix


class H5Protocol(Protocol):
    """Protocol implemented by HDF5-backed store classes."""

    path: Path

    def open(self, mode: str | None = None) -> ty.ContextManager[ty.Any]:
        """Open the backing HDF5 file."""
        ...

    def has_any_data(self, group: str) -> bool:
        """Return whether a group exists and contains at least one item."""
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
        """Write arrays or sparse matrix data into a group."""
        ...

    def get_df(self, group: str) -> pd.DataFrame:
        """Read a dataframe from a group."""
        ...

    def get_group_data(self, group: str) -> tuple[dict, list[str], list[str]]:
        """Read group datasets and return data with full and short child names."""
        ...

    def get_group_data_and_attrs(self, group: str) -> tuple[dict, dict]:
        """Read group datasets and attributes separately."""
        ...

    def can_write(self) -> bool:
        """Check whether we can write."""
        ...

    def enable_write(self) -> ty.ContextManager[ty.Any]:
        """Temporarily switch the store to append mode."""
        ...

    def check_can_write(self, msg: str = ...) -> bool:
        """Raise if the store is not currently writable."""
        ...

    def has_array(self, group: str, name: str) -> bool:
        """Return whether a named dataset exists in a group."""
        ...

    def set_array(self, group: str, name: str, array: np.ndarray, dtype: ty.Any = None, **kwargs: ty.Any) -> None:
        """Write an array dataset into a group."""
        ...

    def get_array(self, group: str, name: str) -> ty.Any:
        """Read a scalar or array dataset from a group."""
        ...

    def get_arrays(self, group: str, *names: str) -> list[ty.Any]:
        """Read multiple scalar or array datasets from a group."""
        ...

    def set_attr(self, group: str | None, attr: str, value: ty.Any) -> None:
        """Set an attribute on a group or the root file."""
        ...

    def get_attr(self, group: str | None, attr: str, default: ty.Any = None) -> ty.Any:
        """Read an attribute from a group or the root file."""
        ...

    def get_group_attrs(self, group: str) -> dict:
        """Get group attributes."""
        ...

    def has_group(self, *groups: str) -> bool:
        """Return whether all groups exist."""
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
