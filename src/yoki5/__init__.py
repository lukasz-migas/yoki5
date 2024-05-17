"""Wrapper around h5py to give easier interface around complex files."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("yoki5")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Lukasz G. Migas"
__email__ = "lukas.migas@yahoo.com"
