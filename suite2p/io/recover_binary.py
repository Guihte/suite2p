"""Recover a local Suite2p functional-channel binary from registered TIFF batches."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile


def _local_path(path: str | os.PathLike[str]) -> Path:
    """Reject remote volumes and links before accessing the requested path."""
    raw = os.fspath(path)
    if raw.startswith(("\\\\", "//")):
        raise ValueError("Only local paths are supported; UNC paths are refused")
    local = Path(os.path.abspath(raw))
    if os.name == "nt":
        import ctypes

        drive_type = int(ctypes.windll.kernel32.GetDriveTypeW(local.anchor))
        if drive_type in (0, 1, 4):
            raise ValueError("Only known local drives are supported; remote drives are refused")
    # Inspect each existing component without traversing a symbolic link or junction.
    for component in reversed((local, *local.parents)):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if component.is_symlink() or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Symbolic links and reparse points are refused: {component}")
    return local


def _positive_int(ops: dict[str, object], key: str) -> int:
    value = ops.get(key)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"ops.npy must contain a positive integer {key}")
    return int(value)


def recover_binary(
    plane_dir: str | os.PathLike[str],
    *,
    progress: Callable[[int, int], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Path:
    """Recover Suite2p data.bin from full-frame registered TIFFs on a local drive.

    Parameters
    ----------
    plane_dir : str or os.PathLike
        Local Suite2p plane directory containing trusted pickled ops.npy and
        reg_tif registered TIFF batches, plus optional F.npy.
    progress : callable, optional
        Called with (frames_written, total_frames) after validation and during
        writing, on the calling thread.
    cancel_requested : callable, optional
        Polled during validation and writing; True cancels before publication.

    Returns
    -------
    pathlib.Path
        Path to the newly created data.bin. Existing data.bin is never replaced.

    Raises
    ------
    InterruptedError
        Cancellation was requested. Any incomplete temporary binary is removed.

    Notes
    -----
    TIFF pages must be signed int16 full-size registered frames in (T, Y, X)
    order. Their pixel values and order are preserved; metadata is not modified.
    """
    if cancel_requested is not None and cancel_requested():
        raise InterruptedError("Binary recovery cancelled")
    plane = _local_path(plane_dir)
    ops_path = _local_path(plane / "ops.npy")
    reg_dir = _local_path(plane / "reg_tif")
    destination = plane / "data.bin"
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"Refusing to overwrite {destination}")

    loaded = np.load(ops_path, allow_pickle=True)
    try:
        ops = loaded.item()
    except (ValueError, AttributeError) as exc:
        raise ValueError("ops.npy must contain a Suite2p ops dictionary") from exc
    if not isinstance(ops, dict):
        raise ValueError("ops.npy must contain a Suite2p ops dictionary")
    ly, lx, nframes = (_positive_int(ops, key) for key in ("Ly", "Lx", "nframes"))

    f_path = _local_path(plane / "F.npy")
    if f_path.exists():
        traces = np.load(f_path, mmap_mode="r", allow_pickle=False)
        if traces.ndim != 2 or traces.shape[1] != nframes:
            raise ValueError("F.npy must have shape (n_rois, nframes) matching ops.npy")
        del traces

    batches: dict[int, Path] = {}
    naming: str | None = None
    for entry in reg_dir.iterdir():
        if entry.suffix.lower() not in (".tif", ".tiff"):
            continue
        _local_path(entry)
        match = re.fullmatch(r"file(\d+)(_chan0)?\.tif", entry.name)
        if match is None:
            raise ValueError(f"Unexpected or ambiguous registered TIFF name: {entry.name}")
        kind = "legacy" if match.group(2) else "current"
        if naming is not None and naming != kind:
            raise ValueError("Mixed registered TIFF naming conventions are ambiguous")
        naming = kind
        index = int(match.group(1))
        if index in batches:
            raise ValueError(f"Duplicate registered TIFF batch index: {index}")
        batches[index] = entry
    if not batches:
        raise ValueError("No registered TIFF batches found in reg_tif")
    indices = sorted(batches)
    if naming == "current" and indices != list(range(len(indices))):
        raise ValueError("Registered TIFF batch indices must start at 0 and have no gaps")

    count = 0
    for index in indices:
        if naming == "legacy" and index != count:
            raise ValueError("Legacy TIFF start-frame indices must match preceding frame counts")
        with tifffile.TiffFile(batches[index]) as tif:
            pages = len(tif.pages)
            if pages == 0:
                raise ValueError(f"Registered TIFF contains no pages: {batches[index]}")
            for page in tif.pages:
                if cancel_requested is not None and cancel_requested():
                    raise InterruptedError("Binary recovery cancelled")
                if page.shape != (ly, lx) or page.dtype != np.dtype("int16"):
                    raise ValueError(f"TIFF page must have shape ({ly}, {lx}) and dtype int16: {batches[index]}")
                count += 1
    if count != nframes:
        raise ValueError(f"TIFF frame count {count} does not match ops.npy nframes {nframes}")
    if progress is not None:
        progress(0, nframes)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".data.bin.", suffix=".tmp", dir=plane, delete=False) as output:
            temporary = Path(output.name)
            written_frames = 0
            for index in indices:
                with tifffile.TiffFile(batches[index]) as tif:
                    for page in tif.pages:
                        if cancel_requested is not None and cancel_requested():
                            raise InterruptedError("Binary recovery cancelled")
                        frame = page.asarray()
                        if frame.shape != (ly, lx) or frame.dtype != np.dtype("int16"):
                            raise ValueError(f"TIFF page changed during recovery: {batches[index]}")
                        output.write(frame.astype("<i2", copy=False).tobytes(order="C"))
                        written_frames += 1
                        if progress is not None and (
                            written_frames % max(1, nframes // 100) == 0
                            or written_frames == nframes
                        ):
                            progress(written_frames, nframes)
            output.flush()
            if written_frames != nframes or output.tell() != nframes * ly * lx * 2:
                raise ValueError("Binary output size does not match expected frame count")
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Binary recovery cancelled")
        # Linking in the same directory atomically fails if data.bin already exists.
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    """Run local binary recovery from the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Recover missing data.bin from local reg_tif TIFFs containing signed int16 "
            "registered frames. The local plane directory must contain a trusted "
            "ops.npy (loaded with pickle); F.npy is checked if present."
        )
    )
    parser.add_argument("plane_dir", help="Local Suite2p plane directory (no network paths or links)")
    args = parser.parse_args()
    try:
        output = recover_binary(args.plane_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Recovery failed: {exc}\n")
    print(f"Recovered: {output}")
    print("Open this local plane's stat.npy in the Suite2p GUI and retry manual ROI extraction.")


if __name__ == "__main__":
    main()