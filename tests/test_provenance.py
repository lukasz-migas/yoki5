"""Test stamping files with provenance step ids."""

import h5py
import numpy as np
import pytest

from yoki5.base import Store
from yoki5.provenance import STEP_ID_ENV, STEP_IDS_ATTR, current_step_id, read_step_ids, stamp, step_context


def _step_ids(path) -> list[str]:
    with h5py.File(path, "r") as h5:
        return read_step_ids(h5)


def test_no_step_means_no_stamp(tmp_path, monkeypatch):
    monkeypatch.delenv(STEP_ID_ENV, raising=False)
    path = tmp_path / "test.h5"
    Store(path).add_data_to_group("group", {"array": np.arange(3)})
    with h5py.File(path, "r") as h5:
        assert STEP_IDS_ATTR not in h5.attrs


def test_env_var_stamps_new_file_and_writes(tmp_path, monkeypatch):
    path = tmp_path / "test.h5"
    monkeypatch.setenv(STEP_ID_ENV, "step-1")
    store = Store(path)
    assert _step_ids(path) == ["step-1"]  # creating a file is a write

    store.add_data_to_group("group", {"array": np.arange(3)})
    assert _step_ids(path) == ["step-1"]  # no duplicates

    monkeypatch.setenv(STEP_ID_ENV, "step-2")
    Store(path).set_attr(None, "attr", 1)
    assert _step_ids(path) == ["step-1", "step-2"]


def test_reading_does_not_stamp(tmp_path, monkeypatch):
    path = tmp_path / "test.h5"
    monkeypatch.delenv(STEP_ID_ENV, raising=False)
    Store(path).add_data_to_group("group", {"array": np.arange(3)})

    monkeypatch.setenv(STEP_ID_ENV, "reader")
    store = Store(path)  # opened in "a" mode, as stores usually are
    store.get_array("group", "array")
    assert _step_ids(path) == []


def test_context_variable_wins_over_env_var(tmp_path, monkeypatch):
    path = tmp_path / "test.h5"
    monkeypatch.setenv(STEP_ID_ENV, "from-env")
    with step_context("from-context"):
        assert current_step_id() == "from-context"
        Store(path)
    assert current_step_id() == "from-env"
    assert _step_ids(path) == ["from-context"]


def test_check_can_write_inside_open_block(tmp_path, monkeypatch):
    path = tmp_path / "test.h5"
    monkeypatch.delenv(STEP_ID_ENV, raising=False)
    store = Store(path)

    monkeypatch.setenv(STEP_ID_ENV, "nested")
    with store.open() as h5:
        store.check_can_write()
        h5["direct"] = np.arange(3)
    assert _step_ids(path) == ["nested"]


@pytest.mark.parametrize("raw", ["not json", '{"a": 1}'])
def test_read_step_ids_ignores_bad_values(tmp_path, raw):
    with h5py.File(tmp_path / "test.h5", "w") as h5:
        h5.attrs[STEP_IDS_ATTR] = raw
        assert read_step_ids(h5) == []
        stamp(h5, "step")
        assert read_step_ids(h5) == ["step"]
