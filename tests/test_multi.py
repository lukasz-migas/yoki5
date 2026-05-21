"""Test multi-store wrapper helpers."""

from pathlib import Path

import pytest

from yoki5.base import Store
from yoki5.multi import MultiStore


class DemoMultiStore(MultiStore):
    """Concrete multi-store for tests."""

    def __init__(self, stores: dict[str, Store], mode: str = "a"):
        self._input_stores = stores
        super().__init__(mode=mode)

    def _validate(self) -> None:
        self._objs = dict(self._input_stores)
        self._paths = {name: Path(store.path) for name, store in self._objs.items()}


def test_multi_store_helpers(tmp_path):
    store_b = Store(tmp_path / "b.h5")
    store_a = Store(tmp_path / "a.h5")
    multi = DemoMultiStore({"b": store_b, "a": store_a}, mode="r")

    assert multi.n_objects == 2
    assert not multi.can_write()
    assert multi._get_obj("a") is store_a
    assert multi._get_first_obj() is store_b
    assert list(multi._path_iter(sort=True)) == [Path(store_a.path), Path(store_b.path)]
    assert [name for name, _ in multi.name_obj_iter(sort=True)] == ["a", "b"]
    assert multi.unique_id

    with multi.enable_write():
        assert multi.can_write()
    assert not multi.can_write()


def test_empty_multi_store_rejects_any_object_access():
    multi = DemoMultiStore({})

    with pytest.raises(StopIteration):
        multi._get_any()
