"""Utilities for yoki5."""
import hashlib
import typing as ty
import uuid
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from natsort import natsorted

if ty.TYPE_CHECKING:
    import pandas as pd

TIME_FORMAT = "%d/%m/%Y-%H:%M:%S:%f"


def encode_str_array(array: np.ndarray) -> np.ndarray:
    """Encode array of U strings as S strings."""
    if np.issubdtype(array.dtype, np.dtype("S")):
        return array
    out = []
    for row in array:
        out.append([v.encode() for v in row])
    return np.asarray(out)


def decode_str_array(array: np.ndarray) -> np.ndarray:
    """Decode array of S strings to U strings."""
    if np.issubdtype(array.dtype, np.dtype("U")):
        return array
    out = []
    for row in array:
        out.append([v.decode() for v in row])
    return np.asarray(out)


def resize_by_append_1d(h5, key: str, array: np.ndarray):
    """Resize 2D array along specified dimension."""
    h5[key].resize(h5[key].shape[0] + array.shape[0], axis=0)
    h5[key][-array.shape[0] :] = array


def resize_by_insert_1d(h5, key: str, array: np.ndarray, indices: ty.List[int]):
    """Resize 2D array along specified dimension."""
    h5[key].resize(h5[key].shape[0] + len(indices), axis=0)
    h5[key][-array.shape[0] :] = array[indices]


def resize_by_append_2d(h5, key: str, array: np.ndarray, axis: int):
    """Resize 2D array along specified dimension."""
    if axis > 1:
        raise ValueError("Cannot resize 2d array with index larger than 1.")
    h5[key].resize(h5[key].shape[axis] + array.shape[axis], axis=axis)
    if axis == 0:
        h5[key][-array.shape[axis] :, :] = array
    else:
        h5[key][:, -array.shape[axis] :] = array


def resize_by_insert_2d(h5, key: str, array: np.ndarray, axis: int, indices: ty.List[int]):
    """Resize 2D array along specified dimension."""
    h5[key].resize(h5[key].shape[axis] + len(indices), axis=axis)
    for i, _index in enumerate(indices):
        h5[key][:, -len(indices) + i] = array


def df_to_buffer(df: "pd.DataFrame") -> np.ndarray:
    """Turn pandas dataframe into a buffer."""
    import pickle

    data = pickle.dumps(df.to_dict())
    data = np.frombuffer(data, dtype=np.uint8)
    return data


def buffer_to_df(buffer: np.ndarray) -> "pd.DataFrame":
    """Turn buffer into pandas dataframe."""
    import pickle

    import pandas as pd

    data = pickle.loads(buffer.tobytes())
    return pd.DataFrame.from_dict(data)


def df_to_dict(df: "pd.DataFrame") -> ty.Dict[str, np.ndarray]:
    """Convert pandas dataframe to dict with arrays."""
    return {
        "columns": df.columns.to_numpy(dtype="S"),
        "index": df.index.to_numpy(),
        "data": df.to_numpy(),
        "dtypes": df.dtypes.to_numpy().astype("S"),
    }


def dict_to_df(data: ty.Dict[str, np.ndarray]) -> "pd.DataFrame":
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


def get_unique_str():
    """Gives random, unique name."""
    return str(uuid.uuid4().hex)


def get_short_hash(n: int = 0) -> str:
    """Get short hash."""
    value = get_unique_str()
    return value[0:n] if n else value


def hash_obj(data: ty.Union[ty.Iterable, ty.List, ty.Dict, ty.Tuple, str, int, float]) -> str:
    """Hash python object."""
    hash_id = hashlib.md5()
    hash_id.update(repr(data).encode("utf-8"))
    return hash_id.hexdigest()


def hash_iterable(iterable, n: int = 0) -> str:
    """Hash iterable object."""
    hash_id = hash_obj(natsorted(iterable))
    return hash_id[0:n] if n else hash_id


def check_base_attributes(attrs: ty.Dict):
    """Check attributes for missing keys."""
    if "unique_id" not in attrs:
        attrs["unique_id"] = get_short_hash()
    if "date_created" not in attrs:
        attrs["date_created"] = datetime.now().strftime(TIME_FORMAT)
    if "date_edited" not in attrs:
        attrs["date_edited"] = datetime.now().strftime(TIME_FORMAT)


def check_data_keys(data: ty.Dict, keys: ty.List[str]):
    """Check whether all keys have been defined in the data."""
    for key in keys:
        if key not in data:
            return False
    return True


def prettify_names(names: ty.List[str]) -> ty.List[str]:
    """Prettify names by removing slashes."""
    if not isinstance(names, Iterable):
        raise ValueError("Cannot prettify list")
    return [_name.split("/")[-1] for _name in names]


def parse_from_attribute(attribute):
    """Parse attribute from cache."""
    if isinstance(attribute, str) and attribute == "__NONE__":
        attribute = None
    return attribute


def parse_to_attribute(attribute):
    """Parse attribute to cache."""
    if attribute is None:
        attribute = "__NONE__"
    return attribute


def check_read_mode(mode: str):
    """Check file opening mode."""
    if mode not in ["r", "a"]:
        raise ValueError(
            "Incorrect opening mode - Please use either `r` or `a` mode to open this file to avoid overwriting"
            " existing data."
        )


def find_case_insensitive(key: str, available_options: ty.List[str]):
    """Find the closest match."""
    _available = [_key.lower() for _key in available_options]
    try:
        index = _available.index(key.lower())
    except IndexError:
        raise KeyError("Could not retrieve item.")
    return available_options[index]


def get_unique_id(path: PathLike) -> str:
    """Get unique ID from path."""
    import h5py

    with h5py.File(path, mode="r", rdcc_nbytes=1024 * 1024 * 4) as f_ptr:
        unique_id = f_ptr.attrs.get("unique_id")
    return unique_id


def display_name_contains(klass, filelist: ty.Iterable[PathLike], contains: str, get_first: bool = False):
    """Return list or item which has specified display name."""
    assert hasattr(klass, "display_name"), "Class object is missing 'display_name' attribute."
    _filelist = []
    for file in filelist:
        obj = klass(file)
        if get_first:
            if contains == obj.display_name:
                _filelist.append(file)
        else:
            if contains in obj.display_name:
                _filelist.append(file)
    if get_first and _filelist:
        return _filelist[0]
    return _filelist


def name_contains(
    filelist: ty.Iterable[PathLike], contains: str, get_first: bool = False, base_dir: ty.Optional[PathLike] = None
) -> ty.Union[Path, ty.List[Path]]:
    """Return list of items which contain specified string."""
    from pathlib import Path

    if contains is None:
        contains = ""
    contains = str(contains)
    if "*" in contains and base_dir:
        # make sure contains has HDF5 extension
        if not contains.endswith(".h5"):
            end = "*.h5"
            if contains.endswith("*"):
                end = end[0:-1]
            contains += end

        filelist = list(Path(base_dir).glob(contains))
        if get_first and filelist:
            return filelist[0]
        return filelist

    # if contains and Path(contains).exists() and get_first:
    #     return contains
    _filelist = []
    for file in filelist:
        if contains in str(file):
            _filelist.append(file)
    if get_first and _filelist:
        return _filelist[0]
    return _filelist


def get_object_path(path_or_tag: PathLike, func: ty.Callable, kind: str) -> Path:
    """Return path or check whether path with tag exists."""
    if path_or_tag is None or not Path(path_or_tag).exists():
        filelist: ty.List[Path] = func(path_or_tag)
        if not filelist:
            raise ValueError(f"List of '{kind}' was empty. Input={path_or_tag}")
        elif len(filelist) > 1:
            # if by any chance the selected paths end with the specified tag, let's pick it
            for path in filelist:
                if path.stem.endswith(path_or_tag):
                    return path
            filelist_str = "\n".join(map(str, filelist))
            raise ValueError(f"List of '{kind}' had more than one entry. Input={path_or_tag}. Entries=\n{filelist_str}")
        path_or_tag = filelist[0]
    path = Path(path_or_tag)
    if not path.exists():
        raise ValueError(f"The specified {kind} does not exist.")
    return path


def optimize_chunks_along_axis(
    axis: int,
    *,
    array: ty.Optional[np.ndarray] = None,
    shape: ty.Optional[ty.Tuple[int, ...]] = None,
    dtype=None,
    max_size: int = 1e6,
    auto: bool = True,
) -> ty.Optional[ty.Tuple[int, ...]]:
    """Optimize chunk size along specified axis"""
    if array is not None:
        dtype, shape = array.dtype, array.shape
    elif shape is None or dtype is None:
        raise ValueError("You must specify either an array or `shape` and `dtype`")
    assert len(shape) == 2, "Only supporting 2d arrays at the moment."
    assert axis <= 1, "Only supporting 2d arrays at the moment, use -1, 0 or 1 in the `axis` argument"
    assert hasattr(dtype, "itemsize"), "Data type must have the attribute 'itemsize'"
    item_size = np.dtype(dtype).itemsize

    if max_size == 0:
        return None

    n = 0
    if auto:
        max_n = shape[1] if axis == 0 else shape[0]
        while (n * item_size * shape[axis]) <= max_size and n < max_n:
            n += 1
    if n < 1:
        n = 1
    return (shape[0], n) if axis == 0 else (n, shape[1])