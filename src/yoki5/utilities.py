"""Utilities for yoki5."""
import typing as ty
from collections.abc import Iterable
import uuid
import hashlib
from datetime import datetime
from natsort import natsorted

TIME_FORMAT = "%d/%m/%Y-%H:%M:%S:%f"


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
    """Check whether all keys have been defined in the data"""
    for key in keys:
        if key not in data:
            return False
    return True


def prettify_names(names: ty.List[str]) -> ty.List[str]:
    """Prettify names by removing slashes"""
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


def repack(path_from: str, path_to: str):
    """Repack existing file to a new file.

    This is really only useful when trying to reduce size of an existing dataset that had some elements previously
    deleted. Unfortunately, deletion of data from h5 document does not result in reduction of its size since that
    data had already been allocated on the filesystem. The only way to deal with this is to rewrite the entire file.
    """
    # if path_from == path_to:
    #     raise ValueError("The `path_from` and `path_to` cannot be the same.")
    # if not os.path.exists(path_from):
    #     raise ValueError("The `path_from` file must exist")
    # if os.path.exists(path_to):
    #     raise ValueError("The `path_to` file must not exist")
    #
    # # get object
    # obj = DataStore(path_from, [], {})
