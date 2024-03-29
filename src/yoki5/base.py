"""HDF5 store."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from loguru import logger
from natsort import natsorted
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, issparse

from yoki5.attrs import Attributes
from yoki5.types import PathLike
from yoki5.utilities import (
    TIME_FORMAT,
    check_base_attributes,
    check_data_keys,
    parse_from_attribute,
    parse_to_attribute,
)

# Local globals
RECOGNIZED_MODES = ["a", "r", "r+", "w"]


class Store:
    """Base data store."""

    HDF5_GROUPS: list[str]
    HDF5_ATTRIBUTES: dict[str, str]
    VERSION: str

    def __init__(self, path: PathLike, groups: list, attributes: dict, *, mode="a", init: bool = True):
        self.path = str(path)
        self.mode = mode
        self.attrs = Attributes(self)

        if init:
            self.initialize_dataset(groups, attributes)

    def __repr__(self):
        """Represent ClassName(name='object_name')."""
        return f"{self.__class__.__name__}<path={self.path}>"

    def __str__(self):
        """Return the object name."""
        return self.path

    def __getitem__(self, item: str) -> tuple[dict, list, list] | np.ndarray:
        try:
            return self.get_dataset_data(item)
        except ValueError:
            with self.open() as h5:
                return h5[item][:]

    def __contains__(self, item):
        with self.open() as h5:
            if item in h5:
                return True
        return False

    def __delitem__(self, key):
        with self.open() as h5:
            del h5[key]

    @property
    def store_name(self):
        """Return short name of the storage object."""
        return Path(self.path).stem

    @property
    def unique_id(self) -> str | None:
        """Return short name of the storage object."""
        return self.attrs.get("unique_id")

    @staticmethod
    def parse_key(key: str, name: str):
        """Parse key."""
        if not name.startswith(f"{key}/"):
            name = f"{key}/{name}"
        return name

    def update_date_edited(self):
        """Update edited time."""
        self.check_can_write()
        with self.open() as h5:
            self._update_date_edited(h5)

    @staticmethod
    def _update_date_edited(h5):
        h5["date_edited"] = datetime.now().strftime(TIME_FORMAT)

    def can_write(self) -> bool:
        """Checks whether data can be written."""
        return self.mode in ["a", "w"]

    def check_can_write(self, msg: str = "Cannot write data to file. Try re-opening in append ('a') mode."):
        """Raises `OSError` if cannot write."""
        if not self.can_write():
            raise OSError(msg + f" Current mode: {self.mode}")
        return True

    @contextmanager
    def enable_write(self):
        """Temporarily enable writing."""
        mode = self.mode
        self.mode = "a"
        yield self
        self.mode = mode

    def initialize_dataset(self, groups: list, attributes: dict):
        """Safely initialize storage."""
        if self.can_write():
            with self.open() as h5:
                self._initilize_dataset(h5, groups, attributes)
                self._flush(h5)
        self.HDF5_GROUPS = self.keys()
        self.HDF5_ATTRIBUTES = self.attrs.to_dict()
        self.VERSION = self.HDF5_ATTRIBUTES.get("VERSION", "N/A")

    def get_document_names(self):
        """Get list of names in the document."""
        with self.open() as h5:
            return list(h5.keys())

    def get_group_names(self, group_name: str, include_group: bool = False) -> list[str]:
        """Get list of group names."""
        with self.open() as h5:
            names = self._get_group_names(h5, group_name, include_group)
        return names

    @staticmethod
    def _get_group_names(h5: h5py.Group, group_name: str, include_group: bool = False) -> list[str]:
        """Get list of groups."""
        names = list(h5[group_name].keys())
        if include_group:
            names = [f"{group_name}/{name}" for name in names]
        return names

    def check_missing(self, group_names):
        """Check for missing keys."""
        present_names = self.get_document_names()
        group_names = list(set(group_names) - set(present_names))
        return group_names

    @staticmethod
    def get_short_names(full_names: list):
        """Get short names."""
        short_names = []
        for name in full_names:
            short_names.append(name.split("/")[-1])

        return short_names

    @contextmanager
    def open(self, mode: str | None = None):
        """Safely open storage."""
        if mode is None:
            mode = self.mode
        try:
            f_ptr = h5py.File(self.path, mode=mode, rdcc_nbytes=1024 * 1024 * 4)
        except FileExistsError as err:
            raise err
        try:
            yield f_ptr
        finally:
            f_ptr.close()

    def close(self):
        """Safely close file."""
        self.flush()

    def flush(self):
        """Flush h5 data."""
        with self.open() as h5:
            self._flush(h5)

    @staticmethod
    def _flush(h5):
        """Flush h5 data."""
        h5.flush()

    def keys(self):
        """Return list of h5 keys."""
        with self.open("r") as h5:
            names = list(h5.keys())
        return names

    def reset_group(self, group_name: str):
        """Reset group."""
        self.check_can_write()
        del self[group_name]
        with self.open() as h5:
            self._add_group(h5, group_name)

    def has_groups(self, group_keys: list[str]) -> bool:
        """Check whether object has groups."""
        with self.open("r") as h5:
            for group in group_keys:
                if group not in h5:
                    return False
        return True

    def has_group(self, group: str) -> bool:
        """Check whether object has groups."""
        with self.open("r") as h5:
            return group in h5

    def has_attr(self, group: str | None, attr: str):
        """Check whether specified group has an attribute."""
        with self.open("r") as h5:
            if group:
                group = self._get_group(h5, group)
                return attr in group.attrs
            return attr in h5.attrs

    def has_attrs(self, attrs_keys: list[str]) -> bool:
        """Check whether object has attributes."""
        with self.open("r") as h5:
            for group in attrs_keys:
                if group not in h5.attrs:
                    return False
        return True

    def has_array(self, dataset_name: str, key: str) -> bool:
        """Check whether array is present in the store."""
        with self.open("r") as h5:
            try:
                group = self._get_group(h5, dataset_name)
                return key in group
            except KeyError:
                return False

    def has_data(self, dataset_name: str) -> int:
        """Check whether there is data in specific dataset/group."""
        with self.open("r") as h5:
            try:
                return len(self._get_group(h5, dataset_name)) != 0
            except KeyError:
                return 0

    def has_keys(self, dataset_name: str, data_keys: list[str] | None = None, attrs_keys: list[str] | None = None) -> bool:
        """Checks whether dataset contains specified `data` and/or` attrs` keys.

        Parameters
        ----------
        dataset_name : str
            name of the dataset or group which should be investigated
        data_keys : List[str], optional
            list of data keys which should be searched for
        attrs_keys : List[str], optional
            list of attribute keys which should be searched for

        Returns
        -------
        any_missing: bool
            if there are any data or attribute keys not present in the dataset return ``False``
        """
        if data_keys is None:
            data_keys = []
        if attrs_keys is None:
            attrs_keys = []
        data_miss, attrs_miss = [], []
        with self.open("r") as h5:
            group = self._get_group(h5, dataset_name)
            for _key in data_keys:
                if _key not in group:
                    data_miss.append(_key)
            for _key in attrs_keys:
                if _key not in group.attrs:
                    attrs_miss.append(_key)
        return not data_miss and not attrs_miss

    def get_dataset_data(self, dataset_name: str):
        """Safely retrieve storage data."""
        with self.open("r") as h5:
            data = self._get_group(h5, dataset_name)
            output, full_names, short_names = self._get_dataset_data(data)
        return output, full_names, short_names

    def get_array(self, dataset_name: str, key: str) -> np.ndarray:
        """Safely retrieve 1 array."""
        with self.open("r") as h5:
            group = self._get_group(h5, dataset_name)
            return group[key][:]

    def get_arrays(self, dataset_name: str, *keys: str) -> list[np.ndarray]:
        """Safely retrieve multiple arrays."""
        with self.open("r") as h5:
            group = self._get_group(h5, dataset_name)
            return [group[key][:] for key in keys]

    def set_array(self, dataset_name: str, key: str, array: np.ndarray, dtype=None, **kwargs):
        """Set array for particular key."""
        self.check_can_write()
        with self.open() as h5:
            group = self._add_group(h5, dataset_name, get=True)
            self._add_data_to_group(group, key, array, dtype=dtype, **kwargs)
            self._flush(h5)

    def rename_array(self, old_name: str, new_name: str, dataset_name: str | None = None):
        """Rename object."""
        self.check_can_write()
        with self.open() as h5:
            if dataset_name:
                old_name = f"{dataset_name}/{old_name}"
                new_name = f"{dataset_name}/{new_name}"
            h5.move(old_name, new_name)

    def remove_array(self, dataset_name: str, key: str):
        """Remove an array from store."""
        self.check_can_write()
        with self.open() as h5:
            group = self._get_group(h5, dataset_name)
            del group[key]

    def _remove_array(self, h5, dataset_name: str, key: str):
        """Remove an array from store."""
        group = self._get_group(h5, dataset_name)
        del group[key]

    def get_data(
        self, dataset_name: str, data_keys: list[str] | None = None, attrs_keys: list[str] | None = None
    ) -> tuple[dict, dict]:
        """Get data for a particular dataset."""
        if data_keys is None:
            data_keys = []
        if attrs_keys is None:
            attrs_keys = []
        with self.open("r") as h5:
            group = self._get_group(h5, dataset_name)
            data = {key: group[key][:] for key in data_keys}
            attrs = {key: parse_from_attribute(group.attrs[key]) for key in attrs_keys}
        return data, attrs

    def get_attr(self, dataset_name, attr: str, default=None):
        """Safely retrieve 1 attribute."""
        with self.open("r") as h5:
            group = self._get_group(h5, dataset_name)
            value = parse_from_attribute(group.attrs.get(attr, default))
            return value

    def set_attr(self, dataset_name, attr: str, value: str | int | float | bool):
        """Set attribute value."""
        with self.open(self.mode) as h5:
            group = self._get_group(h5, dataset_name)
            group.attrs[attr] = parse_to_attribute(value)
            self._flush(h5)

    def get_attrs(self, dataset_name: str, attrs: list[str]):
        """Safely retrieve attributes."""
        if isinstance(attrs, str):
            attrs = [attrs]

        with self.open("r") as h5:
            attrs_out = self._get_attrs(h5, dataset_name, attrs)
        return attrs_out

    def _get_attrs(self, h5, group_name: str, attrs: list[str]):
        group = self._get_group(h5, group_name)
        _attrs = [parse_from_attribute(group.attrs.get(item)) for item in attrs]
        return _attrs

    def _get_dataset_data(self, data: h5py.Group | h5py.Dataset):
        """Retrieve storage data."""
        output = {}
        full_names = []

        # check if the data object is a group
        if isinstance(data, h5py.Group):
            # iterate over each chunk
            for group_name, data_chunk in data.items():
                if isinstance(data_chunk, h5py.Group):
                    output[group_name] = self._get_group_data(data_chunk)
                    full_names.append(data_chunk.name)
                # check if the object is a storage
                elif isinstance(data_chunk, h5py.Dataset):
                    output = self._get_group_data(data)
                    if data.name not in full_names:
                        full_names.append(data.name)
        else:
            raise ValueError("Expected a 'Group' object only")

        # generate list of short names
        short_names = self.get_short_names(full_names)
        return output, full_names, short_names

    def _get_dataset_data_attrs(self, data: h5py.Group | h5py.Dataset) -> tuple[dict, dict]:
        """Retrieve storage data."""
        _data, _attrs = {}, {}

        # check if the data object is a group
        if isinstance(data, h5py.Group):
            # iterate over each chunk
            i = 0
            for i, (group_name, data_chunk) in enumerate(data.items()):
                # check if the object is a group
                if isinstance(data_chunk, h5py.Group):
                    _data[group_name], _attrs[group_name] = self._get_group_data_attrs(data_chunk)
                # check if the object is a dataset
                elif isinstance(data_chunk, h5py.Dataset):
                    _data, _attrs = self._get_group_data_attrs(data)
            # also check whether group has any items
            if i == 0:
                _data, _attrs = self._get_group_data_attrs(data)
        else:
            raise ValueError("Expected a 'Group' object only")
        return _data, _attrs

    def _get_dataset_attrs(self, data: h5py.Group | h5py.Dataset) -> dict:
        """Retrieve storage data."""
        _attrs = {}

        # check if the data object is a group
        if isinstance(data, h5py.Group):
            # iterate over each chunk
            for group_name, data_chunk in data.items():
                if isinstance(data_chunk, h5py.Group):
                    _attrs[group_name] = self._get_group_attrs(data_chunk)
                # check if the object is a storage
                elif isinstance(data_chunk, h5py.Dataset):
                    _attrs = self._get_group_attrs(data)
        else:
            raise ValueError("Expected a 'Group' object only")
        return _attrs

    def get_dataset_data_attrs(self, dataset_name: str) -> tuple[dict, dict]:
        """Safely retrieve storage data."""
        with self.open() as h5:
            group = self._get_group(h5, dataset_name)
            _data, _attrs = self._get_dataset_data_attrs(group)
        return _data, _attrs

    def get_dataset_attrs(self, dataset_name: str) -> dict:
        """Safely retrieve all attributes in particular dataset."""
        with self.open() as h5:
            group = self._get_group(h5, dataset_name)
            _attrs = self._get_dataset_attrs(group)
        return _attrs

    def get_dataset_names(self, group_name: str, sort: bool = False):
        """Get groups names.

        Parameters
        ----------
        group_name : str
        sort : bool
        """
        full_names = []
        with self.open("r") as h5:
            group = self._add_group(h5, group_name, get=True)
            for dataset in group.values():
                full_names.append(dataset.name)
        if sort:
            full_names = natsorted(full_names)

        # generate list of short names
        short_names = self.get_short_names(full_names)
        return full_names, short_names

    def add_attribute(self, attribute_name, attribute_value):
        """Safely add single attribute to dataset."""
        with self.open() as h5:
            self._add_attribute(h5, attribute_name, attribute_value)

    def add_attributes(self, *args):
        """Safely add attributes to storage."""
        with self.open() as h5:
            self._add_attributes(h5, *args)

    def add_attributes_to_dataset(self, dataset_name: str, attributes: dict):
        """Add attributes to dataset."""
        with self.open() as h5:
            dataset = self._add_group(h5, dataset_name, get=True)
            self._add_attributes(dataset, attributes)

    def add_df(self, dataset_name: str, df, **kwargs):
        """Add dataframe to storage."""
        with self.open() as h5:
            self._add_df(h5, dataset_name, df)

    def _add_df(self, h5, dataset_name: str, df, **kwargs):
        """Add dataframe to storage."""
        import pickle

        group = self._add_group(h5, dataset_name, get=True)
        array = pickle.dumps(df.to_dict())
        array = np.frombuffer(array, dtype=np.uint8)
        self._add_data_to_group(group, "table", array, dtype=array.dtype, **kwargs)

    def get_df(self, dataset_name: str):
        """Get dataframe from storage."""
        import pickle

        import pandas as pd

        array = self.get_array(dataset_name, "table")
        return pd.DataFrame.from_dict(pickle.loads(array.tobytes()))

    def add_data_to_dataset(self, group_name, data, attributes=None, dtype=None, as_sparse: bool = False, **kwargs):
        """Safely add data to storage."""
        with self.open() as h5:
            if as_sparse or issparse(data):
                self._add_sparse_data_to_dataset(h5, group_name, data, attributes, dtype, **kwargs)
            else:
                self._add_data_to_dataset(h5, group_name, data, attributes, dtype, **kwargs)

    def get_unique_name(self, group_name: str, dataset_name: str, n_fill=4, join=False):
        """Safely get unique name."""
        _group_name = group_name
        if "/" in group_name:
            _group_name = group_name.split("/")[0]
        if _group_name not in self.HDF5_GROUPS:
            raise ValueError("In order to get unique name, the searched group must exist!")

        with self.open() as h5:
            group_obj = self._add_group(h5, group_name, flush=False, get=True)
            dataset_name = self._get_unique_name(group_obj, dataset_name, n_fill)

        if join:
            return group_name + "/" + dataset_name
        return dataset_name

    def get_sparse_array(self, group_name: str) -> csc_matrix | csr_matrix | coo_matrix:
        """Get sparse array from the dataset."""
        data, _, _ = self.get_dataset_data(group_name)
        if "format" not in data:
            raise ValueError("Could not parse sparse dataset!")
        fmt = data["format"]
        assert fmt in ["csc", "csr", "coo"], f"Cannot interpret specified format: {fmt}"
        if fmt == "csc":
            assert check_data_keys(data, ["data", "indices", "indptr", "shape"])
            return csc_matrix((data["data"], data["indices"], data["indptr"]), shape=data["shape"])
        elif fmt == "csr":
            assert check_data_keys(data, ["data", "indices", "indptr", "shape"])
            return csr_matrix((data["data"], data["indices"], data["indptr"]), shape=data["shape"])
        elif fmt == "coo":
            assert check_data_keys(data, ["data", "row", "col", "shape"])
            return coo_matrix((data["data"], data["row"], data["col"]), shape=data["shape"])

    @staticmethod
    def unpack_sparse_array(array) -> tuple[dict, dict]:
        """Unpack sparse array."""
        if not issparse(array):
            return array, {}

        # CSR/CSC matrices have common attributes
        if array.format in ["csr", "csc"]:
            data = {
                "format": array.format, "shape": array.shape, "data": array.data, "indices": array.indices, "indptr": array.indptr
            }
        elif array.format == "coo":
            data = {"format": array.format, "shape": array.shape, "data": array.data, "col": array.col, "row": array.row}
        else:
            raise ValueError("Cannot serialise this sparse format")
        return data, {"format": array.format, "shape": array.shape, "is_sparse": True}

    def _initilize_dataset(self, h5, groups: list[str] | None = None, attributes: dict | None = None):
        """Initilize storage."""
        if groups is None:
            groups = []
        if attributes is None:
            attributes = {}
        check_base_attributes(attributes)
        # check whether any attribute/group needs to be added to the dataset
        groups, attributes = self._pre_initialize_dataset(h5, groups, attributes)
        for attr_key, attr_value in attributes.items():
            self._add_attribute(h5, attr_key, attr_value)
        for group_name in groups:
            self._add_group(h5, group_name)

    @staticmethod
    def _pre_initialize_dataset(h5, groups: list[str] | None = None, attributes: dict | None = None) -> tuple[list[str], dict]:
        """Check whether dataset needs initialization."""
        needs_group, needs_attributes = [], {}
        for key in groups:
            if key not in h5:
                needs_group.append(key)
        for key, value in attributes.items():
            if key not in h5.attrs:
                needs_attributes[key] = value
        return needs_group, needs_attributes

    def _add_attributes(self, dataset, attributes: dict):
        if attributes is None:
            attributes = {}

        if not isinstance(attributes, dict):
            raise ValueError("'Attributes' must be a dictionary with key:value pairs!")

        # add attributes to the group
        for attribute in attributes:
            self._add_attribute(dataset, attribute, attributes[attribute])

    def _add_data_to_dataset(self, h5, group_name, data, attributes=None, dtype=None, **kwargs):
        if "/" not in group_name and group_name not in self.HDF5_GROUPS:
            logger.warning(f"Group {group_name} not in {self.HDF5_GROUPS}...")

        if attributes is None:
            attributes = {}

        group = self._add_group(h5, group_name, get=True)
        # add attributes to the group
        for attribute in attributes:
            self._add_attribute(group, attribute, attributes[attribute])

        # add data to the group
        for value_key, value_data in data.items():
            self._add_data_to_group(group, value_key, value_data, dtype=dtype, **kwargs)
        self._flush(h5)

    def _add_sparse_data_to_dataset(
        self,
        h5,
        group_name,
        data,
        attributes=None,
        dtype=None,
        **kwargs,
    ):
        """Add sparse data to the dataset.

        Parameters
        ----------
        group_name :
        data :
        attributes :
        """
        if not isinstance(attributes, dict):
            attributes = {}

        # unpack sparse array
        data, data_attributes = self.unpack_sparse_array(data)
        attributes.update(data_attributes)
        self._add_data_to_dataset(h5, group_name, data, attributes=attributes, dtype=dtype, **kwargs)

    @staticmethod
    def _add_attribute(hdf_obj, attribute_name, attribute_value):
        try:
            hdf_obj.attrs[attribute_name] = parse_to_attribute(attribute_value)
        except TypeError:
            raise TypeError(
                f"Object dtype {type(attribute_value)} does not have native HDF5 equivalent. (key={attribute_name};"
                f" value={attribute_value})"
            )

    @staticmethod
    def _add_group(hdf, group_name: str, flush: bool = True, get: bool = False):
        try:
            group = hdf[group_name]
        except KeyError:
            group = hdf.create_group(group_name)

            if flush:
                hdf.flush()

        if get:
            return group

    @staticmethod
    def _get_group(h5, group_name: str):
        """Get group."""
        return h5[group_name]

    @staticmethod
    def _add_data_to_group(
        group_obj,
        dataset_name,
        data,
        dtype,
        chunks=None,
        maxshape=None,
        compression=None,
        compression_opts=None,
        shape=None,
    ):
        """Add data to group."""
        replaced_dataset = False

        if dtype is None:
            if hasattr(data, "dtype"):
                dtype = data.dtype
        if shape is None:
            if hasattr(data, "shape"):
                shape = data.shape

        if dataset_name in list(group_obj.keys()):
            if group_obj[dataset_name].dtype == dtype:
                try:
                    group_obj[dataset_name][:] = data
                    replaced_dataset = True
                except TypeError:
                    del group_obj[dataset_name]
            else:
                del group_obj[dataset_name]

        if not replaced_dataset:
            group_obj.create_dataset(
                dataset_name,
                data=data,
                dtype=dtype,
                compression=compression,
                chunks=chunks,
                maxshape=maxshape,
                compression_opts=compression_opts,
                shape=shape,
            )

    @staticmethod
    def _get_group_data(dataset):
        output = {}
        for obj_name in dataset:
            try:
                output[obj_name] = dataset[obj_name][()]
            except TypeError:
                logger.error(f"Failed to load data '{obj_name}'")

        for obj_name in dataset.attrs:
            try:
                output[obj_name] = parse_from_attribute(dataset.attrs[obj_name])
            except TypeError:
                logger.error(f"Failed to load attribute '{obj_name}'")
        return output

    @staticmethod
    def _get_group_attrs(group: h5py.Group):
        output = {}
        for obj_name in group.attrs:
            try:
                output[obj_name] = parse_from_attribute(group.attrs[obj_name])
            except TypeError:
                logger.error(f"Failed to load attribute '{obj_name}'")
        return output

    @staticmethod
    def _get_group_data_attrs(group: h5py.Group):
        data, attrs = {}, {}
        for obj_name in group:
            try:
                data[obj_name] = group[obj_name][()]
            except TypeError:
                logger.error(f"Failed to load data '{obj_name}'")

        for obj_name in group.attrs:
            try:
                attrs[obj_name] = parse_from_attribute(group.attrs[obj_name])
            except TypeError:
                logger.error(f"Failed to load attribute '{obj_name}'")
        return data, attrs

    @staticmethod
    def _get_unique_name(group_obj, dataset_name: str, n_fill: int):
        """Get unique name."""
        n = 0
        while dataset_name + " #" + "%d".zfill(n_fill) % n in group_obj:
            n += 1
        return dataset_name + " #" + "%d".zfill(n_fill) % n

    @staticmethod
    def _remove_group(h5, group_name: str, flush: bool = True):
        """Remove group."""
        try:
            del h5[group_name]
        except KeyError:
            pass

        if flush:
            h5.flush()
