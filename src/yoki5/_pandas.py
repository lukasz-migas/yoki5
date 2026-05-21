"""Pandas DataFrame utilities for Yoki5."""

from __future__ import annotations

import typing as ty

import numpy as np

if ty.TYPE_CHECKING:
    import pandas as _pd_types  # noqa: ICN001

try:
    import pandas as _pd  # noqa: ICN001
except ImportError:
    _pd = None
    HAS_PANDAS = False
else:
    HAS_PANDAS = True

pd: ty.Any = _pd

__all__ = ["HAS_PANDAS", "buffer_to_df", "check_pandas", "df_to_buffer", "df_to_dict", "dict_to_df", "pd"]


def df_to_buffer(df: _pd_types.DataFrame) -> np.ndarray:
    """Serialize a dataframe to a uint8 pickle buffer."""
    import pickle

    check_pandas()
    data = pickle.dumps(df.to_dict())
    return np.frombuffer(data, dtype=np.uint8)


def buffer_to_df(buffer: np.ndarray) -> _pd_types.DataFrame:
    """Deserialize a dataframe from a uint8 pickle buffer."""
    import pickle

    check_pandas()
    pandas_module = ty.cast(ty.Any, _pd)
    data = pickle.loads(buffer.tobytes())  # noqa: S301 - yoki5 files are trusted local HDF5 stores.
    return pandas_module.DataFrame.from_dict(data)


def df_to_dict(df: _pd_types.DataFrame) -> dict[str, np.ndarray]:
    """Convert pandas dataframe to dict with arrays."""
    check_pandas()
    return {
        "columns": df.columns.to_numpy(dtype="S"),
        "index": df.index.to_numpy(),
        "data": df.to_numpy(),
        "dtypes": df.dtypes.to_numpy().astype("S"),
    }


def dict_to_df(data: dict[str, np.ndarray]) -> _pd_types.DataFrame:
    """Convert dict of arrays to a pandas dataframe."""
    check_pandas()
    pandas_module = ty.cast(ty.Any, _pd)
    columns = data["columns"].astype("str")

    df = pandas_module.DataFrame(
        columns=columns,
        index=data["index"],
        data=data["data"],
    )
    for col, dtype in zip(columns, data["dtypes"].astype("str")):
        df[col] = df[col].astype(dtype)
    return df


def check_pandas() -> None:
    """Raise when pandas is unavailable."""
    if not HAS_PANDAS:
        raise ImportError("Pandas is not installed. Please install pandas to use this module.")
