"""Recover a Suite2p functional-channel binary from registered TIFF batches."""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import tifffile


TiffSet = Literal["spaced", "numbered", "legacy"]


class TiffExportSelectionRequired(ValueError):
    """Multiple TIFF export sets require the caller to select one explicitly."""

    def __init__(self, exports: dict[TiffSet, str]) -> None:
        self.exports = exports
        choices = ", ".join(f"{kind} ({example})" for kind, example in exports.items())
        super().__init__(
            f"Multiple registered TIFF export sets found: {choices}. "
            "Select the set from the same run as the ROI results with --tiff-set."
        )


def _positive_int(ops: dict[str, object], key: str) -> int:
    value = ops.get(key)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"ops.npy must contain a positive integer {key}")
    return int(value)


def recover_binary(
    plane_dir: str | os.PathLike[str],
    *,
    tiff_set: TiffSet | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> Path:
    """Recover Suite2p data.bin from full-frame registered TIFFs.

    Parameters
    ----------
    plane_dir : str or os.PathLike
        Suite2p plane directory on a local or network drive, containing trusted ops.npy and
        reg_tif registered TIFF batches, plus optional F.npy.
    tiff_set : {"spaced", "numbered", "legacy"}, optional
        Explicit export set: file 0000.tif, file00000.tif, or
        file00000_chan0.tif respectively. Required when styles coexist; sets
        are never concatenated or selected from timestamps/version metadata.
    progress : callable, optional
        Called with (frames_written, total_frames) after validation and during
        writing, on the calling thread.
    cancel_requested : callable, optional
        Polled during validation and writing; True cancels before publication.
        An in-flight filesystem operation must finish or time out first.

    Returns
    -------
    pathlib.Path
        Path to the newly created data.bin. Existing data.bin is never replaced.

    Raises
    ------
    TiffExportSelectionRequired
        Multiple export sets are present and none was explicitly selected.
    InterruptedError
        Cancellation was requested. The temporary binary is removed when accessible.
    OSError
        Filesystem access failed. If cleanup also fails (for example after a
        network disconnect), the error identifies the temporary file left behind.

    Notes
    -----
    TIFF pages must be signed int16 full-size registered frames in (T, Y, X)
    order. Their pixel values and order are preserved; metadata is not modified.
    The plane folder must be readable and writable. A temporary binary is written
    beside the destination and published only after completion, without overwrite.
    """
    if cancel_requested is not None and cancel_requested():
        raise InterruptedError("Binary recovery cancelled")
    plane = Path(plane_dir).expanduser().absolute()
    ops_path = plane / "ops.npy"
    reg_dir = plane / "reg_tif"
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

    f_path = plane / "F.npy"
    if f_path.exists():
        traces = np.load(f_path, mmap_mode="r", allow_pickle=False)
        if traces.ndim != 2 or traces.shape[1] != nframes:
            raise ValueError("F.npy must have shape (n_rois, nframes) matching ops.npy")
        del traces

    exports: dict[TiffSet, dict[int, Path]] = {}
    for entry in reg_dir.iterdir():
        if cancel_requested is not None and cancel_requested():
            raise InterruptedError("Binary recovery cancelled")
        if entry.suffix.lower() not in (".tif", ".tiff"):
            continue
        match = re.fullmatch(r"file( ?)(\d+)(_chan0)?\.tif", entry.name)
        if match is None or (match.group(1) and match.group(3)):
            raise ValueError(f"Unexpected or ambiguous registered TIFF name: {entry.name}")
        kind: TiffSet = "legacy" if match.group(3) else "spaced" if match.group(1) else "numbered"
        index = int(match.group(2))
        batches = exports.setdefault(kind, {})
        if index in batches:
            raise ValueError(f"Duplicate registered TIFF batch index: {index}")
        batches[index] = entry
    if not exports:
        raise ValueError("No registered TIFF batches found in reg_tif")
    if cancel_requested is not None and cancel_requested():
        raise InterruptedError("Binary recovery cancelled")
    if tiff_set is None:
        if len(exports) > 1:
            order: tuple[TiffSet, ...] = ("spaced", "numbered", "legacy")
            raise TiffExportSelectionRequired({
                kind: exports[kind][min(exports[kind])].name for kind in order if kind in exports
            })
        tiff_set = next(iter(exports))
    if tiff_set not in exports:
        raise ValueError(f"Registered TIFF export set {tiff_set!r} was not found")
    batches = exports[tiff_set]
    indices = sorted(batches)
    if tiff_set != "legacy" and indices != list(range(len(indices))):
        raise ValueError("Registered TIFF batch indices must start at 0 and have no gaps")

    count = 0
    for index in indices:
        if tiff_set == "legacy" and index != count:
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
        if os.name == "nt":
            # Windows rename refuses an existing destination and also works on
            # network shares/filesystems that do not support hard links.
            os.rename(temporary, destination)
            temporary = None
        else:
            # POSIX rename would overwrite a destination created during recovery.
            os.link(temporary, destination)
    except BaseException as error:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise OSError(
                    f"Recovery failed: {error}\nCould not remove temporary file "
                    f"{temporary}: {cleanup_error}. Remove that temporary file "
                    "after access is restored, then retry."
                ) from error
        raise
    else:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                warnings.warn(
                    f"Recovered {destination}, but could not remove temporary file "
                    f"{temporary}: {cleanup_error}", RuntimeWarning, stacklevel=2,
                )
    return destination


def main() -> None:
    """Run binary recovery from the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Recover missing data.bin from reg_tif TIFFs containing signed int16 "
            "registered frames. The plane directory must be writable and contain a trusted "
            "ops.npy (loaded with pickle); F.npy is checked if present."
        )
    )
    parser.add_argument("plane_dir", help="Suite2p plane directory (local, UNC, or mapped network drive)")
    parser.add_argument(
        "--tiff-set", choices=("spaced", "numbered", "legacy"),
        help="Choose file 0000.tif, file00000.tif, or file00000_chan0.tif exports when they coexist",
    )
    args = parser.parse_args()
    try:
        output = recover_binary(args.plane_dir, tiff_set=args.tiff_set)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Recovery failed: {exc}\n")
    print(f"Recovered: {output}")
    print("Open this plane's stat.npy in the Suite2p GUI and retry manual ROI extraction.")


if __name__ == "__main__":
    main()