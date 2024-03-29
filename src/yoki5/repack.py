"""Repack existing file to a new file."""
from pathlib import Path

import h5py
import hdf5plugin
from koyo.typing import PathLike


def repack(path_from: PathLike, path_to: PathLike):
    """Repack existing file to a new file.

    This is really only useful when trying to reduce size of an existing dataset that had some elements previously
    deleted. Unfortunately, deletion of data from h5 document does not result in reduction of its size since that
    data had already been allocated on the filesystem. The only way to deal with this is to rewrite the entire file.
    """
    if path_from == path_to:
        raise ValueError("The `path_from` and `path_to` cannot be the same.")
    path_from = Path(path_from)
    assert path_from.is_file(), "The `path_from` must be a file"
    if not path_from.exists():
        raise ValueError("The `path_from` file must exist")

    path_to = Path(path_to)
    if path_to.exists():
        raise ValueError("The `path_to` file must not exist")

    with h5py.File(path_from, "r") as from_f:
        with h5py.File(path_to, "w") as to_f:
            # this will copy all groups/arrays
            for key in from_f.keys():
                from_f.copy(key, to_f)
            # copy attributes
            for key in from_f.attrs.keys():
                to_f.attrs[key] = from_f.attrs[key]


def repack_and_replace(path_from: PathLike, path_to: PathLike):
    """Repack an existing file to a new file and replace the original file with the new one."""
    path_from = Path(path_from)
    path_to = Path(path_to)
    repack(path_from, path_to)
    # remove file from disk
    path_from.unlink()
    # rename new file to old file
    path_to.rename(path_from)
