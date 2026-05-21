"""HDF5 store."""

from __future__ import annotations

import typing as ty
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from koyo.typing import PathLike
from loguru import logger
from natsort import natsorted
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, issparse, spmatrix

from yoki5._pandas import HAS_PANDAS, check_pandas, pd
from yoki5.attrs import Attributes
from yoki5.utilities import (
    TIME_FORMAT,
    check_base_attributes,
    check_data_keys,
    parse_from_attribute,
    parse_to_attribute,
)

# Local globals
RECOGNIZED_MODES = ["a", "r", "r+", "w"]
WRITABLE_MODES = ["a", "w"]


class Store:
    """HDF5-backed data store with helpers for groups, arrays, attributes, and sparse matrices."""

    HDF5_GROUPS: list[str]
    HDF5_ATTRIBUTES: dict[str, str]
    VERSION: str

    def __init__(
        self,
        path: PathLike,
        groups: list | None = None,
        attributes: dict | None = None,
        *,
        mode: str = "a",
        init: bool = True,
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
            return self.get_group_data(item)
        except ValueError:
            with self.open() as h5:
                return h5[item][:]  # type: ignore[no-any-return]

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

    @staticmethod
    def _create_group_id(*groups: str) -> str:
        """Create group id."""
        return "/".join(groups)

    @property
    def store_name(self) -> str:
        """Return short name of the storage object."""
        return Path(self.path).stem

    @property
    def unique_id(self) -> str | None:
        """Return short name of the storage object."""
        return self.attrs.get("unique_id")  # type: ignore[no-any-return]

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
    def _update_date_edited(h5: h5py.Group) -> None:
        h5["date_edited"] = datetime.now().strftime(TIME_FORMAT)

    def can_write(self) -> bool:
        """Checks whether data can be written."""
        return self.mode in WRITABLE_MODES

    def check_can_write(self, msg: str = "Cannot write data to file. Try re-opening in append ('a') mode.") -> bool:
        """Raises `OSError` if cannot write."""
        if not self.can_write():
            raise OSError(msg + f" Current mode: {self.mode}")
        return True

    @contextmanager
    def enable_write(self) -> ty.Generator[Store, None, None]:
        """Temporarily enable writing."""
        mode = self.mode
        self.mode = "a"
        try:
            yield self
        finally:
            self.mode = mode

    def initialize_dataset(self, groups: list | None = None, attributes: dict | None = None) -> None:
        """Safely initialize storage."""
        if self.can_write():
            with self.open() as h5:
                self._initialize_dataset(h5, groups, attributes)
                self._flush(h5)
        elif not Path(self.path).exists():
            self.HDF5_GROUPS = []
            self.HDF5_ATTRIBUTES = {}
            self.VERSION = "N/A"
            return
        self.HDF5_GROUPS = self.keys()
        self.HDF5_ATTRIBUTES = self.attrs.to_dict()
        self.VERSION = self.HDF5_ATTRIBUTES.get("VERSION", "N/A")

    def get_group_names(self, group: str | None = None, include_group: bool = False) -> list[str]:
        """Return child names at the root or within a group."""
        with self.open() as h5:
            return self._get_group_names(h5, group, include_group)

    @staticmethod
    def _get_group_names(h5: h5py.Group, group: str | None = None, include_group: bool = False) -> list[str]:
        """Get list of groups."""
        names = list(h5.keys()) if not group else list(h5[group].keys())
        if include_group:
            names = [f"{group}/{name}" for name in names]
        return names

    def check_missing(self, *groups: str) -> list[str]:
        """Return requested root-level groups that are missing."""
        present_names = self.keys()
        return list(set(groups) - set(present_names))

    @staticmethod
    def get_short_names(full_names: list[str]) -> list[str]:
        """Return the final path component for each HDF5 object name."""
        short_names = []
        for name in full_names:
            short_names.append(name.split("/")[-1])
        return short_names

    @contextmanager
    def open(self, mode: str | None = None) -> ty.Generator[h5py.File, None, None]:
        """Open the HDF5 file and close it when the context exits."""
        if mode is None:
            mode = self.mode
        f_ptr = h5py.File(self.path, mode=mode, rdcc_nbytes=1024 * 1024 * 4)
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
        """Return root-level HDF5 object names."""
        with self.open("r") as h5:
            return list(h5.keys())

    def has_group(self, *groups: str) -> bool:
        """Check whether object has groups."""
        with self.open("r") as h5:
            for group in groups:
                if group not in h5:
                    return False
        return True

    def reset_group(self, group: str) -> None:
        """Reset group."""
        self.check_can_write()
        del self[group]
        with self.open() as h5:
            self._add_group(h5, group)

    def has_attr(self, *attrs: str, group: str | None = None) -> bool:
        """Check whether object has attributes."""
        if group:
            with self.open("r") as h5:
                group_obj = self._get_group(h5, group)
                attrs_obj = group_obj.attrs
                return all(attr in attrs_obj for attr in attrs)
        else:
            attrs_obj = self.attrs
            return all(attr in attrs_obj for attr in attrs)

    def has_array(self, group: str, name: str) -> bool:
        """Check whether array is present in the store."""
        with self.open("r") as h5:
            try:
                group_obj = self._get_group(h5, group)
            except KeyError:
                return False
            else:
                return name in group_obj

    def has_any_data(self, group: str) -> bool:
        """Check whether there is data in specific dataset/group."""
        with self.open("r") as h5:
            try:
                return len(self._get_group(h5, group)) != 0
            except KeyError:
                return False

    def has_keys(self, group: str, names: list[str] | None = None, attrs: list[str] | None = None) -> bool:
        """Return whether a group contains all requested dataset and attribute keys."""
        if names is None:
            names = []
        if attrs is None:
            attrs = []
        data_miss, attrs_miss = [], []
        with self.open("r") as h5:
            try:
                group_obj = self._get_group(h5, group)
            except KeyError:
                return False
            for name in names:
                if name not in group_obj:
                    data_miss.append(name)
            for attr in attrs:
                if attr not in group_obj.attrs:
                    attrs_miss.append(attr)
        return not data_miss and not attrs_miss

    def get_group_data(self, group: str, inner: bool = True) -> tuple[dict, list[str], list[str]]:
        """Read data and attributes from a group, optionally including nested group contents."""
        with self.open("r") as h5:
            group_obj = self._get_group(h5, group)
            output, full_names, short_names = self._get_group_or_dataset_data(group_obj, inner=inner)
        return output, full_names, short_names

    def get_array(self, group: str, name: str) -> ty.Any:
        """Read one scalar or array dataset from a group."""
        with self.open("r") as h5:
            group_obj = self._get_group(h5, group)
            return self._read_dataset(group_obj[name])

    def get_names(self, group: str) -> list[str]:
        """Get list of names."""
        with self.open("r") as h5:
            group_obj = self._get_group(h5, group)
            return list(group_obj.keys())

    def get_arrays(self, group: str, *names: str) -> list[ty.Any]:
        """Read multiple scalar or array datasets from a group."""
        with self.open("r") as h5:
            group_obj = self._get_group(h5, group)
            return [self._read_dataset(group_obj[name]) for name in names]

    def remove_group(self, group: str) -> None:
        """Remove group from store."""
        self.check_can_write()
        with self.open() as h5:
            self._remove_group(h5, group)

    def set_array(self, group: str, name: str, array: np.ndarray, dtype: ty.Any = None, **kwargs: ty.Any) -> None:
        """Set array for particular key."""
        self.check_can_write()
        with self.open() as h5:
            group_obj = self._add_group(h5, group)
            self._add_array_to_group(group_obj, name, array, dtype=dtype, **kwargs)
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

    def remove_arrays(self, group: str, *names: str) -> None:
        """Remove arrays from the store within the same group."""
        self.check_can_write()
        with self.open() as h5:
            group_obj = self._get_group(h5, group)
            for name in names:
                del group_obj[name]

    def get_group_data_and_attrs_for_keys(
        self,
        group: str,
        names: list[str] | None = None,
        attrs: list[str] | None = None,
    ) -> tuple[dict, dict]:
        """Read selected datasets and attributes from a group."""
        if names is None:
            names = []
        if attrs is None:
            attrs = []
        with self.open("r") as h5:
            group_obj = self._get_group(h5, group)
            data_ = {name: self._read_dataset(group_obj[name]) for name in names}
            attrs_ = {attr: parse_from_attribute(group_obj.attrs.get(attr, None)) for attr in attrs}
        return data_, attrs_

    def set_attr(self, group: str | None, attr: str, value: ty.Any) -> None:
        """Set an attribute on a group or the root file."""
        self.check_can_write()
        with self.open(self.mode) as h5:
            group_obj = self._add_group(h5, group)
            self._add_attribute_to_group(group_obj, attr, value)
            self._flush(h5)

    def get_attr(self, group: str | None, attr: str, default: ty.Any = None) -> ty.Any:
        """Read one attribute from a group or the root file."""
        with self.open("r") as h5:
            group_obj = self._get_group(h5, group)
            return parse_from_attribute(group_obj.attrs.get(attr, default))

    def get_attrs(self, group: str | None, *attrs: str) -> dict[str, ty.Any]:
        """Read selected attributes from a group or the root file."""
        with self.open("r") as h5:
            return self._get_attrs(h5, group, *attrs)

    def _get_attrs(self, h5: h5py.Group, group: str | None, *attrs: str) -> dict[str, ty.Any]:
        group_obj = self._get_group(h5, group)
        return {item: parse_from_attribute(group_obj.attrs.get(item)) for item in attrs}

    def _get_group_or_dataset_data(
        self,
        group_or_dataset: h5py.Group | h5py.Dataset,
        inner: bool = True,
    ) -> tuple[dict, list[str], list[str]]:
        """Retrieve storage data."""
        if not isinstance(group_or_dataset, h5py.Group):
            raise TypeError("Expected a 'Group' object only")

        output = self._get_group_attrs(group_or_dataset)
        full_names = []
        for group, group_obj in group_or_dataset.items():
            full_names.append(group_obj.name)
            if isinstance(group_obj, h5py.Group):
                if inner:
                    output[group] = self._get_group_data_with_attrs(group_obj)
            elif isinstance(group_obj, h5py.Dataset):
                output[group] = self._read_dataset(group_obj)

        # generate list of short names
        short_names = self.get_short_names(full_names)
        return output, full_names, short_names

    def get_group_attrs(self, group: str) -> dict:
        """Read all attributes from a group, including nested group attributes."""
        with self.open("r") as h5:
            group_obj = self._get_group(h5, group)
            return self._get_group_or_dataset_attrs(group_obj)

    def _get_group_or_dataset_attrs(self, group_or_dataset: h5py.Group | h5py.Dataset) -> dict:
        """Retrieve storage data."""
        if not isinstance(group_or_dataset, h5py.Group):
            raise TypeError("Expected a 'Group' object only")

        _attrs = self._get_group_attrs(group_or_dataset)
        for group, data_chunk in group_or_dataset.items():
            if isinstance(data_chunk, h5py.Group):
                _attrs[group] = self._get_group_or_dataset_attrs(data_chunk)
        return _attrs

    def get_group_data_and_attrs(self, group: str) -> tuple[dict, dict]:
        """Read datasets and attributes from a group into separate nested dictionaries."""
        with self.open("r") as h5:
            group_obj = self._get_group(h5, group)
            _data, _attrs = self._get_group_or_dataset_data_and_attrs(group_obj)
        return _data, _attrs

    def _get_group_or_dataset_data_and_attrs(self, group_or_dataset: h5py.Group | h5py.Dataset) -> tuple[dict, dict]:
        """Retrieve storage data."""
        if not isinstance(group_or_dataset, h5py.Group):
            raise TypeError("Expected a 'Group' object only")

        _data, _attrs = {}, self._get_group_attrs(group_or_dataset)
        for group, data_chunk in group_or_dataset.items():
            if isinstance(data_chunk, h5py.Group):
                _data[group], _attrs[group] = self._get_group_or_dataset_data_and_attrs(data_chunk)
            elif isinstance(data_chunk, h5py.Dataset):
                _data[group] = self._read_dataset(data_chunk)
        return _data, _attrs

    def get_names_for_group(self, group: str, sort: bool = False) -> tuple[list[str], list[str]]:
        """Return full and short child object names for a group."""
        full_names = []
        with self.open("r") as h5:
            group_obj = self._get_group(h5, group)
            for dataset in group_obj.values():
                full_names.append(dataset.name)
        if sort:
            full_names = natsorted(full_names)

        # generate list of short names
        short_names = self.get_short_names(full_names)
        return full_names, short_names

    def add_attribute(self, **kwargs: ty.Any) -> None:
        """Safely add attributes to storage."""
        self.check_can_write()
        with self.open() as h5:
            self._add_attributes_to_group(h5, kwargs)

    def add_attributes_to_group(self, group: str, **kwargs: ty.Any) -> None:
        """Add attributes to dataset."""
        self.check_can_write()
        with self.open() as h5:
            group_obj = self._add_group(h5, group)
            self._add_attributes_to_group(group_obj, kwargs)

    def _add_attributes_to_group(self, h5: h5py.Group, attributes: dict) -> None:
        if attributes is None:
            attributes = {}
        if not isinstance(attributes, dict):
            raise TypeError("'Attributes' must be a dictionary with key:value pairs!")

        # add attributes to the group
        for attribute in attributes:
            self._add_attribute_to_group(h5, attribute, attributes[attribute])

    def add_arrays_to_group(
        self,
        group: str,
        dtype: ty.Any = None,
        chunks: tuple | None = None,
        maxshape: tuple | None = None,
        compression: str | int | None = None,
        compression_opts: dict | None = None,
        **arrays: ty.Any,
    ) -> None:
        """Add arrays to the group."""
        self.check_can_write()
        with self.open() as h5:
            group_obj = self._add_group(h5, group)
            for key, value in arrays.items():
                self._add_array_to_group(
                    group_obj,
                    key,
                    value,
                    dtype=dtype,
                    chunks=chunks,
                    maxshape=maxshape,
                    compression=compression,
                    compression_opts=compression_opts,
                )

    def add_data_to_group(
        self,
        group: str,
        data: dict | spmatrix,
        attributes: dict | None = None,
        dtype: ty.Any = None,
        as_sparse: bool = False,
        **kwargs: ty.Any,
    ) -> None:
        """Safely add data to storage."""
        self.check_can_write()
        with self.open() as h5:
            if as_sparse or issparse(data):
                self._add_sparse_data_to_group(h5, group, data, attributes, dtype, **kwargs)
            else:
                self._add_data_to_group(h5, group, data, attributes, dtype, **kwargs)

    def get_sparse_array(self, group: str) -> csc_matrix | csr_matrix | coo_matrix:
        """Get sparse array from the dataset."""
        data, _, _ = self.get_group_data(group)
        if "format" not in data:
            raise ValueError("Could not parse sparse dataset!")
        fmt = data["format"]
        if isinstance(fmt, bytes):
            fmt = fmt.decode()
        shape = tuple(data["shape"]) if "shape" in data else None
        if fmt not in ["csc", "csr", "coo"]:
            raise ValueError(f"Cannot interpret specified sparse format: {fmt}")
        if fmt == "csc":
            self._check_sparse_data_keys(data, ["data", "indices", "indptr", "shape"], group)
            return csc_matrix((data["data"], data["indices"], data["indptr"]), shape=shape)
        if fmt == "csr":
            self._check_sparse_data_keys(data, ["data", "indices", "indptr", "shape"], group)
            return csr_matrix((data["data"], data["indices"], data["indptr"]), shape=shape)
        if fmt == "coo":
            self._check_sparse_data_keys(data, ["data", "row", "col", "shape"], group)
            return coo_matrix((data["data"], (data["row"], data["col"])), shape=shape)
        raise ValueError("Could not interpret specified")

    @staticmethod
    def _check_sparse_data_keys(data: dict, keys: list[str], group: str) -> None:
        """Validate sparse matrix component keys."""
        if not check_data_keys(data, keys):
            missing = sorted(set(keys) - set(data))
            raise ValueError(f"Sparse dataset {group!r} is missing required keys: {missing}")

    @staticmethod
    def _unpack_sparse_array(
        array: csc_matrix | csr_matrix | coo_matrix | np.ndarray,
    ) -> tuple[np.ndarray | dict, dict]:
        """Unpack sparse array."""
        if not issparse(array):
            return array, {}

        # CSR/CSC matrices have common attributes
        if array.format in ["csr", "csc"]:  # type: ignore[union-attr]
            data = {
                "format": array.format,  # type: ignore[union-attr]
                "shape": array.shape,
                "data": array.data,
                "indices": array.indices,  # type: ignore[union-attr]
                "indptr": array.indptr,  # type: ignore[union-attr]
            }
        elif array.format == "coo":  # type: ignore[union-attr]
            data = {
                "format": array.format,  # type: ignore[union-attr]
                "shape": array.shape,
                "data": array.data,
                "col": array.col,  # type: ignore[union-attr]
                "row": array.row,  # type: ignore[union-attr]
            }
        else:
            raise ValueError("Cannot serialise this sparse format")
        return data, {"format": array.format, "shape": array.shape, "is_sparse": True}  # type: ignore[union-attr]

    def _initialize_dataset(
        self,
        h5: h5py.Group,
        groups: list[str] | None = None,
        attributes: dict | None = None,
    ) -> None:
        """Initialize storage."""
        if groups is None:
            groups = []
        if attributes is None:
            attributes = {}
        check_base_attributes(attributes)
        # check whether any attribute/group needs to be added to the dataset
        groups, attributes = self._pre_initialize_dataset(h5, groups, attributes)
        for attr_key, attr_value in attributes.items():
            self._add_attribute_to_group(h5, attr_key, attr_value)
        for group in groups:
            self._add_group(h5, group)

    @staticmethod
    def _pre_initialize_dataset(h5: h5py.File, groups: list[str], attributes: dict) -> tuple[list[str], dict]:
        """Check whether dataset needs initialization."""
        needs_group, needs_attributes = [], {}
        for key in groups:
            if key not in h5:
                needs_group.append(key)
        for key, value in attributes.items():
            if key not in h5.attrs:
                needs_attributes[key] = value
        return needs_group, needs_attributes

    def _add_data_to_group(
        self,
        h5: h5py.Group,
        name: str,
        data: dict,
        attributes: dict | None = None,
        dtype: ty.Any = None,
        **kwargs: ty.Any,
    ) -> None:
        # if "/" not in name and name not in self.HDF5_GROUPS:
        #     logger.warning(f"Group {name} not in {self.HDF5_GROUPS}...")

        if attributes is None:
            attributes = {}
        if not isinstance(attributes, dict):
            raise TypeError("'Attributes' must be a dictionary with key:value pairs!")
        if not isinstance(data, dict):
            raise TypeError("'Data' must be a dictionary with key:value pairs!")

        group_obj = self._add_group(h5, name)
        # add attributes to the group
        for attribute in attributes:
            self._add_attribute_to_group(group_obj, attribute, attributes[attribute])

        # add data to the group
        for value_key, value_data in data.items():
            self._add_array_to_group(group_obj, value_key, value_data, dtype=dtype, **kwargs)
        self._flush(h5)

    def _add_sparse_data_to_group(
        self,
        h5: h5py.Group,
        name: str,
        data: dict,
        attributes: dict | None = None,
        dtype: ty.Any = None,
        **kwargs: ty.Any,
    ) -> None:
        """Add sparse data to the dataset."""
        if attributes is None:
            attributes = {}
        if not isinstance(attributes, dict):
            raise TypeError("'Attributes' must be a dictionary with key:value pairs!")

        # unpack sparse array
        data, data_attributes = self._unpack_sparse_array(data)  # type: ignore[assignment]
        attributes = {**attributes, **data_attributes}
        self._add_data_to_group(h5, name, data, attributes=attributes, dtype=dtype, **kwargs)

    @staticmethod
    def _add_attribute_to_group(h5: h5py.Group, attr: str, value: ty.Any) -> None:
        try:
            h5.attrs[attr] = parse_to_attribute(value)
        except TypeError:
            raise TypeError(
                f"Object dtype {type(value)} does not have native HDF5 equivalent. (key={attr}; value={value})",
            ) from None

    @staticmethod
    def _add_group(h5: h5py.File, group: str | None, flush: bool = True) -> h5py.Group:
        if group is None:
            return h5
        try:
            group_obj = h5[group]
        except KeyError:
            group_obj = h5.create_group(group)
            if flush:
                h5.flush()
        return group_obj

    @staticmethod
    def _get_group(h5: h5py.Group, group: str | None) -> h5py.Group:
        """Get group."""
        if group is None:
            return h5
        return h5[group]

    @staticmethod
    def _add_array_to_group(
        h5: h5py.Group,
        name: str,
        data: dict | np.ndarray,
        dtype: ty.Any = None,
        chunks: tuple | None = None,
        maxshape: tuple | None = None,
        compression: str | int | None = None,
        compression_opts: dict | None = None,
        shape: tuple | None = None,
    ) -> None:
        """Add data to group."""
        replaced_dataset = False

        if dtype is None and hasattr(data, "dtype"):
            dtype = data.dtype
        if shape is None and hasattr(data, "shape"):
            shape = data.shape
        if isinstance(compression, dict):
            compression_opts = compression.get("compression_opts", None)
            compression = compression.get("compression", None)

        if name in list(h5.keys()):
            if dtype is not None and h5[name].dtype == dtype:
                try:
                    if h5[name].ndim == 0:
                        h5[name][()] = data  # scalar dataset
                    else:
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
    def _read_dataset(dataset: h5py.Dataset) -> ty.Any:
        """Read scalar and array datasets."""
        if dataset.shape == ():
            return dataset[()]
        return dataset[:]

    @staticmethod
    def _get_group_data(group: h5py.Group) -> dict:
        data = {}
        for obj_name in group:
            obj = group[obj_name]
            if isinstance(obj, h5py.Dataset):
                data[obj_name] = Store._read_dataset(obj)
        return data

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
    def _get_group_data_and_attrs(group: h5py.Group) -> tuple[dict, dict]:
        data, attrs = {}, {}
        for obj_name in group:
            obj = group[obj_name]
            if isinstance(obj, h5py.Dataset):
                data[obj_name] = Store._read_dataset(obj)
        for obj_name in group.attrs:
            try:
                attrs[obj_name] = parse_from_attribute(group.attrs[obj_name])
            except TypeError:
                logger.error(f"Failed to load attribute '{obj_name}'")
        return data, attrs

    @staticmethod
    def _get_group_data_with_attrs(group: h5py.Group) -> dict:
        data = {}
        for obj_name in group:
            obj = group[obj_name]
            if isinstance(obj, h5py.Group):
                data[obj_name] = Store._get_group_data_with_attrs(obj)
            elif isinstance(obj, h5py.Dataset):
                data[obj_name] = Store._read_dataset(obj)
        for obj_name in group.attrs:
            with suppress(TypeError):
                data[obj_name] = parse_from_attribute(group.attrs[obj_name])
        return data

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
        with suppress(KeyError):
            del h5[name]
        if flush:
            h5.flush()

    if HAS_PANDAS:

        def add_df(self, group: str, df: pd.DataFrame, **_kwargs: ty.Any) -> None:
            """Add dataframe to storage."""
            self.check_can_write()
            with self.open() as h5:
                self._add_df(h5, group, df)

        def _add_df(self, h5: h5py.Group, group: str, df: pd.DataFrame, **kwargs: ty.Any) -> None:
            """Add dataframe to storage."""
            import pickle

            group_obj = self._add_group(h5, group)
            array = pickle.dumps(df.to_dict())
            array_bytes = np.frombuffer(array, dtype=np.uint8)
            self._add_array_to_group(group_obj, "table", array_bytes, dtype=array_bytes.dtype, **kwargs)

        def get_df(self, group: str) -> pd.DataFrame:
            """Get dataframe from storage."""
            import pickle

            array = self.get_array(group, "table")
            return pd.DataFrame.from_dict(pickle.loads(array.tobytes()))  # noqa: S301 - trusted local HDF5 stores.

    else:

        def add_df(self, group: str, df: pd.DataFrame, **_kwargs: ty.Any) -> None:
            """Add dataframe to storage."""
            check_pandas()

        def _add_df(self, h5: h5py.Group, group: str, df: pd.DataFrame, **kwargs: ty.Any) -> None:
            """Add dataframe to storage."""
            check_pandas()

        def get_df(self, group: str) -> pd.DataFrame:
            """Get dataframe from storage."""
            check_pandas()
