"""Pandas DataFrame utilities for Yoki5."""

from __future__ import annotations

import numpy as np

try:
    import pandas as pd

    HAS_PANDAS = True

    def df_to_buffer(df: pd.DataFrame) -> np.ndarray:
        """Turn pandas dataframe into a buffer."""
        import pickle

        data = pickle.dumps(df.to_dict())
        return np.frombuffer(data, dtype=np.uint8)

    def buffer_to_df(buffer: np.ndarray) -> pd.DataFrame:
        """Turn buffer into pandas dataframe."""
        import pickle

        import pandas as pd

        data = pickle.loads(buffer.tobytes())
        return pd.DataFrame.from_dict(data)

    def df_to_dict(df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Convert pandas dataframe to dict with arrays."""
        return {
            "columns": df.columns.to_numpy(dtype="S"),
            "index": df.index.to_numpy(),
            "data": df.to_numpy(),
            "dtypes": df.dtypes.to_numpy().astype("S"),
        }

    def dict_to_df(data: dict[str, np.ndarray]) -> pd.DataFrame:
        """Convert dict to pandas dataframe."""
        import pandas as pd

        columns = data["columns"].astype("str")

        df = pd.DataFrame(
            columns=columns,
            index=data["index"],
            data=data["data"],
        )
        for col, dtype in zip(columns, data["dtypes"].astype("str")):
            df[col] = df[col].astype(dtype)
        return df
except ImportError:
    HAS_PANDAS = False

    class pd:
        DataFrame = None
        Series = None

        def __getattr__(self, item):
            check_pandas()

    def df_to_buffer(df: pd.DataFrame) -> np.ndarray:
        """Turn pandas dataframe into a buffer."""
        check_pandas()

    def buffer_to_df(buffer: np.ndarray) -> pd.DataFrame:
        """Turn buffer into pandas dataframe."""
        check_pandas()

    def df_to_dict(df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Convert pandas dataframe to dict with arrays."""
        check_pandas()

    def dict_to_df(data: dict[str, np.ndarray]) -> pd.DataFrame:
        """Convert dict to pandas dataframe."""
        check_pandas()


def check_pandas() -> None:
    """Check if pandas is installed."""
    if not HAS_PANDAS:
        raise ImportError("Pandas is not installed. Please install pandas to use this module.")
