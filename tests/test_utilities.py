"""Test storage utils"""

import numpy as np
import pandas as pd
import pytest

from yoki5._pandas import HAS_PANDAS, buffer_to_df, df_to_buffer, df_to_dict, dict_to_df
from yoki5.utilities import (
    check_read_mode,
    decode_str_array,
    encode_str_array,
    find_case_insensitive,
    parse_from_attribute,
    parse_to_attribute,
    prettify_names,
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
    df = pd.DataFrame.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
    buffer = df_to_buffer(df)
    assert isinstance(buffer, np.ndarray)
    result = buffer_to_df(buffer)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(df)


@pytest.mark.skipif(not HAS_PANDAS, reason="Pandas not installed")
def test_df_as_dict():
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
