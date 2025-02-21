"""Wrapper around multiple stores."""

import typing as ty
from contextlib import contextmanager
from pathlib import Path

from koyo.secret import hash_iterable
from natsort import natsorted

if ty.TYPE_CHECKING:
    from yoki5.base import Store


class MultiStore:
    """Base class for multi-dataset wrappers."""

    def __init__(self, mode: str = "a"):
        self._objs: ty.Dict[str, Store] = {}
        self.mode = mode
        self._validate()

    def __repr__(self) -> str:
        """Return a string representation of the."""
        return f"{self.__class__.__name__}<present={len(self._objs)}>"

    @property
    def n_objects(self) -> int:
        """Return number of objects."""
        return len(self._objs)

    def _validate(self) -> None:
        """Validate data."""

    def _get_any(self) -> ty.Any:
        """Get any available peaks object."""
        return next(iter(self._objs.keys()))

    def _get_any_obj(self):
        return self._objs[self._get_any()]

    def _get_obj(self, name: str):
        return self._objs[name]

    def _obj_iter(self, sort: bool = False):
        names = list(self._objs.keys())
        if sort:
            names = natsorted(names)
        for name in names:
            yield self._objs[name]

    def name_obj_iter(self, sort: bool = False):
        """Iterator."""
        names = list(self._objs.keys())
        if sort:
            names = natsorted(names)
        for name in names:
            yield name, self._objs[name]

    def _path_iter(self, sort: bool = False) -> ty.Iterable[Path]:
        """Iterator of object paths."""
        for obj in self._obj_iter(sort):
            yield Path(obj.path)

    def can_write(self) -> bool:
        """Checks whether data can be written."""
        return self.mode in ["a", "w"]

    @contextmanager
    def enable_write(self):
        """Temporarily enable writing."""
        mode = self.mode
        self.mode = "a"
        yield self
        self.mode = mode

    def close(self):
        """Close handle."""
        for obj in self._objs.values():
            obj.close()

    @property
    def unique_id(self) -> str:
        """Return unique ID."""
        paths = [path.name for path in self._path_iter()]
        return hash_iterable(paths)
