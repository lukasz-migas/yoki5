"""Test storage utils"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from yoki5._pandas import HAS_PANDAS, buffer_to_df, df_to_buffer, df_to_dict, dict_to_df
from yoki5.utilities import (
    check_read_mode,
    decode_str_array,
    display_name_contains,
    encode_str_array,
    find_case_insensitive,
    get_object_path,
    name_contains,
    optimize_chunks_along_axis,
    parse_from_attribute,
    parse_to_attribute,
    prettify_names,
    resize_by_insert_2d,
)


@pytest.mark.parametrize(
    "values, expected",
    ((["Norm/Test", "Norm/Test2", "Norm/Test3"], ["Test", "Test2", "Test3"]), (["Test", "Test2"], ["Test", "Test2"])),
)
def test_prettify_names(values, expected):
    result = prettify_names(values)
    assert len(result) == len(expected)
    for _r, _e in zip(result, expected):
        assert _r == _e


def test_prettify_names_rejects_string():
    with pytest.raises(ValueError):
        prettify_names("Norm/Test")


@pytest.mark.parametrize("mode", ("a", "r"))
def test_check_read_mode(mode):
    check_read_mode(mode)


@pytest.mark.parametrize("mode", ("w", "w+"))
def test_check_read_mode_raise(mode):
    with pytest.raises(ValueError):
        check_read_mode(mode)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_encode_str_array(encoding):
    vals = np.asarray(["Test 1", "Test 2", "Test 3"])
    encoded = encode_str_array(vals, encoding=encoding)
    assert isinstance(encoded, np.ndarray)
    decoded = decode_str_array(encoded, encoding=encoding)
    np.testing.assert_array_equal(vals, decoded)

    vals = ["Test 1", "Test 2", "Test 3"]
    encoded = encode_str_array(vals, encoding=encoding)
    assert isinstance(encoded, np.ndarray)
    decoded = decode_str_array(encoded, encoding=encoding)
    np.testing.assert_array_equal(vals, decoded)


@pytest.mark.skipif(not HAS_PANDAS, reason="Pandas not installed")
def test_df():
    import pandas as pd

    df = pd.DataFrame.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
    buffer = df_to_buffer(df)
    assert isinstance(buffer, np.ndarray)
    result = buffer_to_df(buffer)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(df)


@pytest.mark.skipif(not HAS_PANDAS, reason="Pandas not installed")
def test_df_as_dict():
    import pandas as pd

    df = pd.DataFrame.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
    data = df_to_dict(df)
    assert isinstance(data, dict)
    result = dict_to_df(data)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(df)


def test_parse_attribute():
    assert parse_to_attribute("Test") == "Test"
    assert parse_to_attribute(None) == "__NONE__"
    assert parse_to_attribute(1) == 1

    assert parse_from_attribute(parse_to_attribute("Test")) == "Test"
    assert parse_from_attribute(parse_to_attribute(None)) is None


def test_find_case_insensitive():
    options = ["Test", "Test2", "Test3"]
    assert find_case_insensitive("test", options) == "Test"
    assert find_case_insensitive("test2", options) == "Test2"


def test_find_case_insensitive_missing():
    with pytest.raises(KeyError):
        find_case_insensitive("missing", ["Test"])


class FakeDisplayStore:
    def __init__(self, path):
        self.path = Path(path)

    @property
    def display_name(self) -> str:
        return self.path.stem.replace("_", " ")


def test_display_name_contains(tmp_path):
    file_a = tmp_path / "Alpha_One.h5"
    file_b = tmp_path / "Beta_Two.h5"
    file_a.touch()
    file_b.touch()

    result = display_name_contains(FakeDisplayStore, [file_a, file_b], "Alpha")
    assert result == [file_a]
    assert display_name_contains(FakeDisplayStore, [file_a, file_b], "Beta Two", first=True) == file_b


def test_name_contains(tmp_path):
    file_a = tmp_path / "alpha_one.h5"
    file_b = tmp_path / "beta_two.h5"
    file_a.touch()
    file_b.touch()

    result = name_contains([file_a, file_b], "alpha", filename_only=True)
    assert result == [file_a]
    assert name_contains([file_a, file_b], "beta_two", filename_only=True, exact=True, first=True) == file_b
    assert name_contains([], str(file_a), first=True) == file_a
    assert name_contains([], "alpha*", base_dir=tmp_path) == [file_a]


def test_get_object_path(tmp_path):
    file_a = tmp_path / "alpha_one.h5"
    file_b = tmp_path / "beta_two.h5"
    file_a.touch()
    file_b.touch()

    def finder(tag):
        return [file_a, file_b] if tag == "two" else [file_a]

    assert get_object_path(file_a, finder, "dataset") == file_a
    assert get_object_path("two", finder, "dataset") == file_b

    with pytest.raises(ValueError, match="empty"):
        get_object_path("missing", lambda _tag: [], "dataset")


def test_optimize_chunks_along_axis():
    array = np.ones((10, 20), dtype=np.float32)
    assert optimize_chunks_along_axis(0, array=array, max_size=0) is None
    assert optimize_chunks_along_axis(-1, array=array) == optimize_chunks_along_axis(1, array=array)

    with pytest.raises(ValueError):
        optimize_chunks_along_axis(2, array=array)

    with pytest.raises(ValueError):
        optimize_chunks_along_axis(0, shape=(10,), dtype=np.float32)


def test_resize_by_insert_2d(tmp_path):
    path = tmp_path / "resize.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("group")
        group.create_dataset("data", data=np.array([[1, 2], [3, 4]]), maxshape=(None, None))
        source = np.array([[10, 20, 30], [40, 50, 60]])

        resize_by_insert_2d(group, "data", source, axis=1, indices=[0, 2])

        np.testing.assert_array_equal(group["data"][:], np.array([[1, 2, 10, 30], [3, 4, 40, 60]]))
