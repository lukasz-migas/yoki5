"""HDF5 store."""

from __future__ import annotations

import typing as ty
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from koyo.typing import PathLike
from loguru import logger
from natsort import natsorted
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, issparse, spmatrix

from yoki5.attrs import Attributes
from yoki5.utilities import (
    TIME_FORMAT,
    check_base_attributes,
    check_data_keys,
    parse_from_attribute,
    parse_to_attribute,
)

if ty.TYPE_CHECKING:
    import pandas as pd

# Local globals
RECOGNIZED_MODES = ["a", "r", "r+", "w"]
WRITABLE_MODES = ["a", "w"]


class Store:
    """Base data store."""

    HDF5_GROUPS: list[str]
    HDF5_ATTRIBUTES: dict[str, str]
    VERSION: str

    def __init__(
        self, path: PathLike, groups: list | None = None, attributes: dict | None = None, *, mode="a", init: bool = True
    ):
        self.path: str = str(path)
        self.mode = mode
        self.attrs = Attributes(self)

        if init:
            self.initialize_dataset(groups, attributes)

    def __repr__(self) -> str:
        """Represent ClassName(name='object_name')."""
        return f"{self.__class__.__name__}<path={self.path}>"

    def __str__(self) -> str:
        """Return the object name."""
        return f"{self.__class__.__name__}<path={self.path}>"

    def __getitem__(self, item: str) -> tuple[dict, list, list] | np.ndarray:
        """Get data from the store."""
        try:
            return self.get_dataset_data(item)
        except ValueError:
            with self.open() as h5:
                return h5[item][:]

    def __contains__(self, item: str) -> bool:
        """Check whether item is in the store."""
        with self.open() as h5:
            if item in h5:
                return True
        return False

    def __delitem__(self, key: str) -> None:
        """Delete item from the store."""
        with self.open() as h5:
            del h5[key]

    @property
    def store_name(self) -> str:
        """Return short name of the storage object."""
        return Path(self.path).stem

    @property
    def unique_id(self) -> str | None:
        """Return short name of the storage object."""
        return self.attrs.get("unique_id")

    @staticmethod
    def parse_key(group: str, name: str) -> str:
        """Parse key."""
        if not name.startswith(f"{group}/"):
            name = f"{group}/{name}"
        return name

    def update_date_edited(self) -> None:
        """Update edited time."""
        self.check_can_write()
        with self.open() as h5:
            self._update_date_edited(h5)

    @staticmethod
    def _update_date_edited(h5) -> None:
        h5["date_edited"] = datetime.now().strftime(TIME_FORMAT)

    def can_write(self) -> bool:
        """Checks whether data can be written."""
        return self.mode in WRITABLE_MODES

    def check_can_write(self, msg: str = "Cannot write data to file. Try re-opening in append ('a') mode."):
        """Raises `OSError` if cannot write."""
        if not self.can_write():
            raise OSError(msg + f" Current mode: {self.mode}")
        return True

    @contextmanager
    def enable_write(self) -> ty.Generator[Store, None, None]:
        """Temporarily enable writing."""
        mode = self.mode
        self.mode = "a"
        yield self
        self.mode = mode

    def initialize_dataset(self, groups: list | None = None, attributes: dict | None = None) -> None:
        """Safely initialize storage."""
        if self.can_write():
            with self.open() as h5:
                self._initialize_dataset(h5, groups, attributes)
                self._flush(h5)
        self.HDF5_GROUPS = self.keys()
        self.HDF5_ATTRIBUTES = self.attrs.to_dict()
        self.VERSION = self.HDF5_ATTRIBUTES.get("VERSION", "N/A")

    def get_document_names(self) -> list[str]:
        """Get list of names in the document."""
        with self.open() as h5:
            return list(h5.keys())

    def get_group_names(self, group: str | None = None, include_group: bool = False) -> list[str]:
        """Get list of group names."""
        with self.open() as h5:
            names = self._get_group_names(h5, group, include_group)
        return names

    @staticmethod
    def _get_group_names(h5: h5py.Group, group: str | None = None, include_group: bool = False) -> list[str]:
        """Get list of groups.

        Parameters
        ----------
        h5 - Group object
        group_name - name of the group to get the names from
        include_group - include group name
        """
        if not group:
            names = list(h5.keys())
        else:
            names = list(h5[group].keys())
        if include_group:
            names = [f"{group}/{name}" for name in names]
        return names

    def check_missing(self, groups: list[str]) -> list[str]:
        """Check for missing keys."""
        present_names = self.get_document_names()
        groups = list(set(groups) - set(present_names))
        return groups

    @staticmethod
    def get_short_names(full_names: list[str]) -> list[str]:
        """Get short names."""
        short_names = []
        for name in full_names:
            short_names.append(name.split("/")[-1])
        return short_names

    @contextmanager
    def open(self, mode: str | None = None) -> ty.Generator[h5py.File, None, None]:
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

    def close(self) -> None:
        """Safely close file."""
        self.flush()

    def flush(self) -> None:
        """Flush h5 data."""
        with self.open() as h5:
            self._flush(h5)

    @staticmethod
    def _flush(h5: h5py.File) -> None:
        """Flush h5 data."""
        h5.flush()

    def keys(self) -> list[str]:
        """Return list of h5 keys."""
        with self.open("r") as h5:
            names = list(h5.keys())
        return names

    def reset_group(self, group: str) -> None:
        """Reset group."""
        self.check_can_write()
        del self[group]
        with self.open() as h5:
            self._add_group(h5, group)

    def has_groups(self, groups: list[str]) -> bool:
        """Check whether object has groups."""
        with self.open("r") as h5:
            for group in groups:
                if group not in h5:
                    return False
        return True

    def has_group(self, group: str) -> bool:
        """Check whether object has groups."""
        with self.open("r") as h5:
            return group in h5

    def has_attr(self, group: str | None, attr: str):
        """Check whether specified group has an attribute."""
        if group:
            with self.open("r") as h5:
                group_obj = self._get_group(h5, group)
                return attr in group_obj.attrs
        return attr in self.attrs

    def has_attrs(self, attrs: list[str]) -> bool:
        """Check whether object has attributes."""
        for group in attrs:
            if group not in self.attrs:
                return False
        return True

    def has_array(self, group: str, name: str) -> bool:
        """Check whether array is present in the store."""
        with self.open("r") as h5:
            try:
                group_obj = self._get_group(h5, group)
                return name in group_obj
            except KeyError:
                return False

    def has_data(self, group: str) -> int:
        """Check whether there is data in specific dataset/group."""
        with self.open("r") as h5:
            try:
                return len(self._get_group(h5, group)) != 0
            except KeyError:
                return 0

    def has_keys(self, group: str, data_keys: list[str] | None = None, attrs_keys: list[str] | None = None) -> bool:
        """Checks whether dataset contains specified `data` and/or` attrs` keys.

        Parameters
        ----------
        group : str
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
            group_obj = self._get_group(h5, group)
            for _key in data_keys:
                if _key not in group_obj:
                    data_miss.append(_key)
            for _key in attrs_keys:
                if _key not in group_obj.attrs:
                    attrs_miss.append(_key)
        return not data_miss and not attrs_miss

    def get_dataset_data(self, group: str) -> tuple[dict, list[str], list[str]]:
        """Safely retrieve storage data."""
        with self.open("r") as h5:
            group_obj = self._get_group(h5, group)
            output, full_names, short_names = self._get_dataset_data(group_obj)
        return output, full_names, short_names

    def get_array(self, group: str, name: str) -> np.ndarray:
        """Safely retrieve 1 array."""
        with self.open("r") as h5:
            group_obj = self._get_group(h5, group)
            return group_obj[name][:]

    def get_arrays(self, group: str, *names: str) -> list[np.ndarray]:
        """Safely retrieve multiple arrays."""
        with self.open("r") as h5:
            group_obj = self._get_group(h5, group)
            return [group_obj[name][:] for name in names]

    def set_array(self, group: str, name: str, array: np.ndarray, dtype: ty.Any = None, **kwargs: ty.Any) -> None:
        """Set array for particular key."""
        self.check_can_write()
        with self.open() as h5:
            group_obj = self._add_group(h5, group, get=True)
            self._add_data_to_group(group_obj, name, array, dtype=dtype, **kwargs)
            self._flush(h5)

    def rename_array(self, old_name: str, new_name: str, group: str | None = None) -> None:
        """Rename object."""
        self.check_can_write()
        with self.open() as h5:
            if group:
                old_name = f"{group}/{old_name}"
                new_name = f"{group}/{new_name}"
            h5.move(old_name, new_name)

    def remove_array(self, group: str, name: str) -> None:
        """Remove an array from store."""
        self.check_can_write()
        with self.open() as h5:
            group_obj = self._get_group(h5, group)
            del group_obj[name]

    def _remove_array(self, h5: h5py.Group, group: str, name: str) -> None:
        """Remove an array from store."""
        group_obj = self._get_group(h5, group)
        del group_obj[name]

    def get_data(
        self, group: str, data_keys: list[str] | None = None, attrs_keys: list[str] | None = None
    ) -> tuple[dict, dict]:
        """Get data for a particular dataset."""
        if data_keys is None:
            data_keys = []
        if attrs_keys is None:
            attrs_keys = []
        with self.open("r") as h5:
            group_obj = self._get_group(h5, group)
            data = {key: group_obj[key][:] for key in data_keys}
            attrs = {key: parse_from_attribute(group_obj.attrs[key]) for key in attrs_keys}
        return data, attrs

    def get_attr(self, name: str, attr: str, default: ty.Any = None) -> None:
        """Safely retrieve 1 attribute."""
        with self.open("r") as h5:
            group_obj = self._get_group(h5, name)
            value = parse_from_attribute(group_obj.attrs.get(attr, default))
            return value

    def set_attr(self, name: str, attr: str, value: str | int | float | bool):
        """Set attribute value."""
        with self.open(self.mode) as h5:
            group_obj = self._get_group(h5, name)
            group_obj.attrs[attr] = parse_to_attribute(value)
            self._flush(h5)

    def get_attrs(self, name: str, attrs: list[str]) -> dict[str, ty.Any]:
        """Safely retrieve attributes."""
        if isinstance(attrs, str):
            attrs = [attrs]

        with self.open("r") as h5:
            attrs_out = self._get_attrs(h5, name, attrs)
        return attrs_out

    def _get_attrs(self, h5, name: str, attrs: list[str]) -> dict[str, ty.Any]:
        group_obj = self._get_group(h5, name)
        _attrs = [parse_from_attribute(group_obj.attrs.get(item)) for item in attrs]
        return _attrs

    def _get_dataset_data(self, group_or_dataset: h5py.Group | h5py.Dataset) -> tuple[dict, list[str], list[str]]:
        """Retrieve storage data."""
        output = {}
        full_names = []

        # check if the data object is a group
        if isinstance(group_or_dataset, h5py.Group):
            # iterate over each chunk
            for group, data_chunk in group_or_dataset.items():
                if isinstance(data_chunk, h5py.Group):
                    output[group] = self._get_group_data(data_chunk)
                    full_names.append(data_chunk.name)
                # check if the object is a storage
                elif isinstance(data_chunk, h5py.Dataset):
                    output = self._get_group_data(group_or_dataset)
                    if group_or_dataset.name not in full_names:
                        full_names.append(group_or_dataset.name)
        else:
            raise ValueError("Expected a 'Group' object only")

        # generate list of short names
        short_names = self.get_short_names(full_names)
        return output, full_names, short_names

    def _get_dataset_data_attrs(self, group_or_dataset: h5py.Group | h5py.Dataset) -> tuple[dict, dict]:
        """Retrieve storage data."""
        _data, _attrs = {}, {}

        # check if the data object is a group
        if isinstance(group_or_dataset, h5py.Group):
            # iterate over each chunk
            i = 0
            for i, (group, data_chunk) in enumerate(group_or_dataset.items()):
                # check if the object is a group
                if isinstance(data_chunk, h5py.Group):
                    _data[group], _attrs[group] = self._get_group_data_attrs(data_chunk)
                # check if the object is a dataset
                elif isinstance(data_chunk, h5py.Dataset):
                    _data, _attrs = self._get_group_data_attrs(group_or_dataset)
            # also check whether group has any items
            if i == 0:
                _data, _attrs = self._get_group_data_attrs(group_or_dataset)
        else:
            raise ValueError("Expected a 'Group' object only")
        return _data, _attrs

    def _get_dataset_attrs(self, group_or_dataset: h5py.Group | h5py.Dataset) -> dict:
        """Retrieve storage data."""
        _attrs = {}

        # check if the data object is a group
        if isinstance(group_or_dataset, h5py.Group):
            # iterate over each chunk
            for group, data_chunk in group_or_dataset.items():
                if isinstance(data_chunk, h5py.Group):
                    _attrs[group] = self._get_group_attrs(data_chunk)
                # check if the object is a storage
                elif isinstance(data_chunk, h5py.Dataset):
                    _attrs = self._get_group_attrs(group_or_dataset)
        else:
            raise ValueError("Expected a 'Group' object only")
        return _attrs

    def get_dataset_data_attrs(self, name: str) -> tuple[dict, dict]:
        """Safely retrieve storage data."""
        with self.open() as h5:
            group_obj = self._get_group(h5, name)
            _data, _attrs = self._get_dataset_data_attrs(group_obj)
        return _data, _attrs

    def get_dataset_attrs(self, name: str) -> dict:
        """Safely retrieve all attributes in particular dataset."""
        with self.open() as h5:
            group_obj = self._get_group(h5, name)
            _attrs = self._get_dataset_attrs(group_obj)
        return _attrs

    def get_dataset_names(self, name: str, sort: bool = False) -> tuple[list[str], list[str]]:
        """Get groups names.

        Parameters
        ----------
        name : str
        sort : bool
        """
        full_names = []
        with self.open("r") as h5:
            group_obj = self._add_group(h5, name, get=True)
            for dataset in group_obj.values():
                full_names.append(dataset.name)
        if sort:
            full_names = natsorted(full_names)

        # generate list of short names
        short_names = self.get_short_names(full_names)
        return full_names, short_names

    def add_attribute(self, attr: str, value: ty.Any) -> None:
        """Safely add single attribute to dataset."""
        with self.open() as h5:
            self._add_attribute(h5, attr, value)

    def add_attributes(self, *args):
        """Safely add attributes to storage."""
        with self.open() as h5:
            self._add_attributes(h5, *args)

    def add_attributes_to_dataset(self, name: str, attributes: dict) -> None:
        """Add attributes to dataset."""
        with self.open() as h5:
            group_obj = self._add_group(h5, name, get=True)
            self._add_attributes(group_obj, attributes)

    def add_df(self, name: str, df: pd.DataFrame, **_kwargs: ty.Any) -> None:
        """Add dataframe to storage."""
        with self.open() as h5:
            self._add_df(h5, name, df)

    def _add_df(self, h5, name: str, df: pd.DataFrame, **kwargs: ty.Any):
        """Add dataframe to storage."""
        import pickle

        group_obj = self._add_group(h5, name, get=True)
        array = pickle.dumps(df.to_dict())
        array = np.frombuffer(array, dtype=np.uint8)
        self._add_data_to_group(group_obj, "table", array, dtype=array.dtype, **kwargs)

    def get_df(self, name: str) -> pd.DataFrame:
        """Get dataframe from storage."""
        import pickle

        import pandas as pd

        array = self.get_array(name, "table")
        return pd.DataFrame.from_dict(pickle.loads(array.tobytes()))

    def add_data_to_dataset(
        self,
        name: str,
        data: dict | spmatrix,
        attributes: dict | None = None,
        dtype: ty.Any = None,
        as_sparse: bool = False,
        **kwargs: ty.Any,
    ) -> None:
        """Safely add data to storage."""
        with self.open() as h5:
            if as_sparse or issparse(data):
                self._add_sparse_data_to_dataset(h5, name, data, attributes, dtype, **kwargs)
            else:
                self._add_data_to_dataset(h5, name, data, attributes, dtype, **kwargs)

    def get_unique_name(self, group: str, name: str, n_fill: int = 4, join: bool = False) -> str:
        """Safely get unique name."""
        _group = group
        if "/" in group:
            _group = group.split("/")[0]
        if _group not in self.HDF5_GROUPS:
            raise ValueError("In order to get unique name, the searched group must exist!")

        with self.open() as h5:
            group_obj = self._add_group(h5, group, flush=False, get=True)
            name = self._get_unique_name(group_obj, name, n_fill)
        if join:
            return group + "/" + name
        return name

    def get_sparse_array(self, name: str) -> csc_matrix | csr_matrix | coo_matrix:
        """Get sparse array from the dataset."""
        data, _, _ = self.get_dataset_data(name)
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
            return coo_matrix((data["data"], (data["row"], data["col"])), shape=data["shape"])

    @staticmethod
    def unpack_sparse_array(array: csc_matrix | csr_matrix | coo_matrix | np.ndarray) -> tuple[dict, dict]:
        """Unpack sparse array."""
        if not issparse(array):
            return array, {}

        # CSR/CSC matrices have common attributes
        if array.format in ["csr", "csc"]:
            data = {
                "format": array.format,
                "shape": array.shape,
                "data": array.data,
                "indices": array.indices,
                "indptr": array.indptr,
            }
        elif array.format == "coo":
            data = {
                "format": array.format,
                "shape": array.shape,
                "data": array.data,
                "col": array.col,
                "row": array.row,
            }
        else:
            raise ValueError("Cannot serialise this sparse format")
        return data, {"format": array.format, "shape": array.shape, "is_sparse": True}

    def _initialize_dataset(self, h5, groups: list[str] | None = None, attributes: dict | None = None):
        """Initialize storage."""
        if groups is None:
            groups = []
        if attributes is None:
            attributes = {}
        check_base_attributes(attributes)
        # check whether any attribute/group needs to be added to the dataset
        groups, attributes = self._pre_initialize_dataset(h5, groups, attributes)
        for attr_key, attr_value in attributes.items():
            self._add_attribute(h5, attr_key, attr_value)
        for group in groups:
            self._add_group(h5, group)

    @staticmethod
    def _pre_initialize_dataset(
        h5, groups: list[str] | None = None, attributes: dict | None = None
    ) -> tuple[list[str], dict]:
        """Check whether dataset needs initialization."""
        needs_group, needs_attributes = [], {}
        for key in groups:
            if key not in h5:
                needs_group.append(key)
        for key, value in attributes.items():
            if key not in h5.attrs:
                needs_attributes[key] = value
        return needs_group, needs_attributes

    def _add_attributes(self, name: str, attributes: dict) -> None:
        if attributes is None:
            attributes = {}

        if not isinstance(attributes, dict):
            raise ValueError("'Attributes' must be a dictionary with key:value pairs!")

        # add attributes to the group
        for attribute in attributes:
            self._add_attribute(name, attribute, attributes[attribute])

    def _add_data_to_dataset(
        self,
        h5: h5py.Group,
        name: str,
        data: dict,
        attributes: dict | None = None,
        dtype: ty.Any = None,
        **kwargs: ty.Any,
    ):
        if "/" not in name and name not in self.HDF5_GROUPS:
            logger.warning(f"Group {name} not in {self.HDF5_GROUPS}...")

        if attributes is None:
            attributes = {}

        group_obj = self._add_group(h5, name, get=True)
        # add attributes to the group
        for attribute in attributes:
            self._add_attribute(group_obj, attribute, attributes[attribute])

        # add data to the group
        for value_key, value_data in data.items():
            self._add_data_to_group(group_obj, value_key, value_data, dtype=dtype, **kwargs)
        self._flush(h5)

    def _add_sparse_data_to_dataset(
        self,
        h5: h5py.Group,
        name: str,
        data: dict,
        attributes: dict | None = None,
        dtype: ty.Any = None,
        **kwargs: ty.Any,
    ):
        """Add sparse data to the dataset."""
        if not isinstance(attributes, dict):
            attributes = {}

        # unpack sparse array
        data, data_attributes = self.unpack_sparse_array(data)
        attributes.update(data_attributes)
        self._add_data_to_dataset(h5, name, data, attributes=attributes, dtype=dtype, **kwargs)

    @staticmethod
    def _add_attribute(h5: h5py.Group, attr: str, value: ty.Any) -> None:
        try:
            h5.attrs[attr] = parse_to_attribute(value)
        except TypeError:
            raise TypeError(
                f"Object dtype {type(value)} does not have native HDF5 equivalent. (key={attr};" f" value={value})"
            )

    @staticmethod
    def _add_group(hdf, name: str, flush: bool = True, get: bool = False) -> h5py.Group | None:
        try:
            group_obj = hdf[name]
        except KeyError:
            group_obj = hdf.create_group(name)

            if flush:
                hdf.flush()

        if get:
            return group_obj

    @staticmethod
    def _get_group(h5: h5py.Group, name: str) -> h5py.Group:
        """Get group."""
        return h5[name]

    @staticmethod
    def _add_data_to_group(
        h5: h5py.Group,
        name: str,
        data: dict,
        dtype: ty.Any,
        chunks: tuple | None = None,
        maxshape: tuple | None = None,
        compression: dict | None = None,
        compression_opts: dict | None = None,
        shape: tuple | None = None,
    ):
        """Add data to group."""
        replaced_dataset = False

        if dtype is None:
            if hasattr(data, "dtype"):
                dtype = data.dtype
        if shape is None:
            if hasattr(data, "shape"):
                shape = data.shape

        if name in list(h5.keys()):
            if h5[name].dtype == dtype:
                try:
                    h5[name][:] = data
                    replaced_dataset = True
                except TypeError:
                    del h5[name]
            else:
                del h5[name]

        if not replaced_dataset:
            h5.create_dataset(
                name,
                data=data,
                dtype=dtype,
                compression=compression,
                chunks=chunks,
                maxshape=maxshape,
                compression_opts=compression_opts,
                shape=shape,
            )

    @staticmethod
    def _get_group_data(group: h5py.Group) -> dict:
        output = {}
        for obj_name in group:
            try:
                output[obj_name] = group[obj_name][()]
            except TypeError:
                logger.error(f"Failed to load data '{obj_name}'")

        for obj_name in group.attrs:
            try:
                output[obj_name] = parse_from_attribute(group.attrs[obj_name])
            except TypeError:
                logger.error(f"Failed to load attribute '{obj_name}'")
        return output

    @staticmethod
    def _get_group_attrs(group: h5py.Group) -> dict:
        output = {}
        for obj_name in group.attrs:
            try:
                output[obj_name] = parse_from_attribute(group.attrs[obj_name])
            except TypeError:
                logger.error(f"Failed to load attribute '{obj_name}'")
        return output

    @staticmethod
    def _get_group_data_attrs(group: h5py.Group) -> tuple[dict, dict]:
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
    def _get_unique_name(group: h5py.Group, name: str, n_fill: int) -> str:
        """Get unique name."""
        n = 0
        while name + " #" + "%d".zfill(n_fill) % n in group:
            n += 1
        return name + " #" + "%d".zfill(n_fill) % n

    @staticmethod
    def _remove_group(h5: h5py.File, name: str, flush: bool = True) -> None:
        """Remove group."""
        try:
            del h5[name]
        except KeyError:
            pass
        if flush:
            h5.flush()
