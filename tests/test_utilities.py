"""Test storage utils"""
import pytest
from yoki5.utilities import check_read_mode, prettify_names


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
