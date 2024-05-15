"""Test store class."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from yoki5.base import Store


def test_store_init(tmp_path):
    path = tmp_path / "test.h5"
    store = Store(path)
    assert store.path == str(path), "Path should be the same"
    assert store.store_name == "test", "Store name should be test"
    assert store.unique_id, "Unique id should be None"
    assert store.can_write(), "Should not be able to write"
    store.check_can_write()

    store = Store(path, mode="r")
    assert not store.can_write(), "Should not be able to write"
    with pytest.raises(OSError):
        store.check_can_write()
    with store.enable_write():
        assert store.can_write(), "Should be able to write"


def test_store_api(tmp_path):
    path = tmp_path / "test.h5"
    store = Store(path, groups=["group1", "group2", "group3"], attributes={"attr1": 1, "attr2": "two"})
    # check attrs
    assert store.attrs["attr1"] == 1
    assert store.attrs["attr2"] == "two"
    store.attrs["attr3"] = "three"
    assert store.attrs["attr3"] == "three"

    # check groups
    group_names = store.get_group_names()
    assert len(group_names) == 3
    assert "group1" in group_names
    assert "group2" in group_names
    store.add_data_to_dataset("group1", {"data": [1, 2, 3]}, attributes={"attr": "value"})
    data, attrs = store.get_dataset_data_attrs("group1")
    assert "data" in data
    assert "attr" in attrs
    attrs = store.get_dataset_attrs("group1")
    assert "attr" in attrs
    data, _, _ = store.get_dataset_data("group1")
    assert "data" in data
    _, group_names = store.get_dataset_names("group1")
    assert "data" in group_names
    assert len(group_names) == 1

    # add/get df
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    store.add_df("group2", df)
    df = store.get_df("group2")
    _, group_names = store.get_dataset_names("group2")
    assert "table" in group_names
    assert len(group_names) == 1
    assert isinstance(df, pd.DataFrame)


@pytest.mark.parametrize("fmt", ["csr", "csc", "coo"])
def test_store_sparse(tmp_path, fmt):
    path = tmp_path / "test.h5"
    store = Store(path, groups=["group1"])
    # add/get sparse array
    matrix = csr_matrix(np.eye(3)).asformat(fmt)
    store.add_data_to_dataset("group1", matrix, as_sparse=True)
    matrix2 = store.get_sparse_array("group1")
    assert matrix2.shape == matrix.shape
    assert np.allclose(matrix2.toarray(), matrix.toarray())
    assert matrix.format == matrix2.format