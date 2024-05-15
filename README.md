# yoki5

[![License](https://img.shields.io/pypi/l/yoki5.svg?color=green)](https://github.com/lukasz-migas/yoki5/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/yoki5.svg?color=green)](https://pypi.org/project/yoki5)
[![Python Version](https://img.shields.io/pypi/pyversions/yoki5.svg?color=green)](https://python.org)
[![CI](https://github.com/lukasz-migas/yoki5/actions/workflows/ci.yml/badge.svg)](https://github.com/lukasz-migas/yoki5/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/lukasz-migas/yoki5/branch/main/graph/badge.svg)](https://codecov.io/gh/lukasz-migas/yoki5)

## Overview

A simple wrapper around h5py to give easier interface around complex HDF5 files.

This library is designed to simplify the process of reading and writing HDF5 files by providing several methods
that make it easy to add/remove datasets or groups. 

## Getting started

```python
import numpy as np
from yoki5.base import Store
from scipy.sparse import csr_matrix

# Create a new HDF5 file
store = Store('path/to/file.h5')
store.add_data_to_dataset(
    "group-1", 
    {"data": np.random.rand(100, 100)}, 
    {"attribute": "value"},
    compression="gzip",
    chunks=(10, 10),
)
# Retrieve the array
array = store.get_array("group-1", "data")

# Add sparse matrix
store.add_data_to_dataset(
    "group-2", 
     csr_matrix(np.random.randint(0, 255, (100, 100))),
    {"attribute": "value"},
)
# Retrieve the data
matrix = store.get_sparse_array("group-2")

```


## Contributing

Contributions are always welcome. Please feel free to submit PRs with new features, bug fixes, or documentation improvements.

```bash
git clone https://github.com/lukasz-migas/yoki5.git

pip install -e .[dev]
```