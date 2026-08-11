"""Export functions."""

from __future__ import annotations

import typing as ty
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from koyo.typing import PathLike
from loguru import logger


def _report_exception(exc: Exception) -> bool:
    """Report an exception before retrying an atomic write."""
    logger.warning(f"Task failed with {type(exc).__name__}: {exc}", exc_info=True)
    return True


def _get_temporary_path(path: Path) -> Path:
    """Return a unique temporary path in the target directory."""
    return path.parent / f".{path.name}.{uuid4().hex}.tmp"


def _atomic_write(path: Path, writer: ty.Callable[[Path], None], *, override: bool = False) -> Path:
    """Write to a temporary path and atomically replace the target path."""
    target_path = Path(path)
    temp_path = _get_temporary_path(target_path)
    if target_path.exists() and not override:
        raise FileExistsError(f"The file {target_path} already exists. Please specify the override=True.")
    try:
        writer(temp_path)
        if target_path.exists() and override:
            target_path.unlink()
        temp_path.rename(target_path)
    except Exception:
        with suppress(FileNotFoundError):
            temp_path.unlink()
        raise
    return target_path


def save_store_atomic(
    store_cls: type,
    target_path: Path,
    data_dir: PathLike | None = None,
    save_fn: ty.Callable | None = None,
) -> None:
    """Save an HDF5 store atomically using a temporary file.

    Writes to a .tmp file first, then atomically renames to the target path.
    On failure, the temporary file is removed to prevent partial/corrupted files.

    Parameters
    ----------
    store_cls : type
        The store class (e.g., UnsupervisedStore, SupervisedStore).
    target_path : Path
        The final destination path for the HDF5 file.
    data_dir : PathLike, optional
        The data directory passed to the store constructor. Omit for stores
        that only accept ``path``.
    save_fn : callable, optional
        A callable that takes the store instance and performs the writes
        (e.g., ``lambda s: s.from_fit_instance(fit)``). If None, the
        store is simply created and closed — useful when the caller
        manages writes externally.
    """
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    if data_dir is not None:
        store = store_cls(tmp_path, data_dir)
    else:
        store = store_cls(tmp_path)
    try:
        if save_fn is not None:
            save_fn(store)
    except Exception:
        store.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    store.close()
    tmp_path.rename(target_path)
