"""Recover missing per-plane metadata for combining existing Suite2p results."""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from numbers import Real
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..parameters import convert_settings_orig, default_db, default_settings


def recover_plane_metadata(plane_dir: str | os.PathLike[str]) -> tuple[Path, ...]:
    """Recreate missing db.npy/settings.npy from a trusted per-plane ops.npy.

    Parameters
    ----------
    plane_dir : str or os.PathLike
        Local or network planeN directory containing saved results. Existing
        metadata files are preserved. Missing files require a trusted ops.npy,
        loaded with pickle, with explicit geometry, channel and timing values.

    Returns
    -------
    tuple of pathlib.Path
        Newly created metadata files, or an empty tuple if none were missing.

    Notes
    -----
    No movies are read and no registration, detection or extraction is run.
    Database fields and saved output arrays come from ops.npy, without database
    defaults. Settings use Suite2p's legacy converter: recorded settings are
    retained and newer options use current defaults. This recovers metadata for
    combining existing results, not the exact original processing environment.
    Source paths are preserved; result metadata paths refer to the current plane.
    Failed writes remove only files created by this call when access permits.
    """
    plane = Path(plane_dir).expanduser().absolute()
    if not (plane.name.startswith("plane") and plane.name[5:].isdigit()):
        raise ValueError("Metadata recovery requires a single planeN directory.")
    missing: list[Path] = []
    for name in ("db.npy", "settings.npy"):
        destination = plane / name
        try:
            destination.lstat()
        except FileNotFoundError:
            missing.append(destination)
        else:
            if not destination.is_file():
                raise ValueError(f"Existing metadata is not a regular file: {destination}")
    if not missing:
        return ()

    ops_path = plane / "ops.npy"
    if not ops_path.is_file():
        raise FileNotFoundError(f"Cannot recover missing metadata without saved ops.npy: {plane}")
    try:
        ops = np.load(ops_path, allow_pickle=True).item()
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"ops.npy must contain a Suite2p dictionary: {plane}") from exc
    if not isinstance(ops, dict):
        raise ValueError(f"ops.npy must contain a Suite2p dictionary: {plane}")
    for key in ("Ly", "Lx", "nframes", "nchannels"):
        value = ops.get(key)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"Cannot recover metadata: ops.npy needs a positive integer {key} ({plane})")
    for key in ("fs", "tau"):
        value = ops.get(key)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(float(value)) or value <= 0):
            raise ValueError(f"Cannot recover metadata: ops.npy needs a positive finite {key} ({plane})")
    for key in ("meanImg", "xrange", "yrange", "Vcorr"):
        if key not in ops:
            raise ValueError(f"Cannot recover combined-view metadata: ops.npy is missing {key} ({plane})")
    for key in ("meanImg", "meanImgE", "meanImg_chan2", "meanImg_chan2_corrected"):
        if key in ops and np.shape(ops[key]) != (ops["Ly"], ops["Lx"]):
            raise ValueError(f"ops.npy {key} shape does not match Ly/Lx: {plane}")
    crop_shape: list[int] = []
    for key, dimension in (("yrange", ops["Ly"]), ("xrange", ops["Lx"])):
        values = np.asarray(ops[key])
        if (values.ndim != 1 or values.size < 2 or values.dtype.kind not in "iu"
                or not 0 <= values[0] < values[-1] <= dimension):
            raise ValueError(f"ops.npy {key} must have integer crop endpoints within the image: {plane}")
        crop_shape.append(int(values[-1] - values[0]))
    for key in ("Vcorr", "max_proj"):
        if key in ops and np.shape(ops[key]) != tuple(crop_shape):
            raise ValueError(f"ops.npy {key} shape does not match yrange/xrange: {plane}")
    trace_path = plane / "F.npy"
    if trace_path.is_file():
        traces = np.load(trace_path, mmap_mode="r")
        if traces.ndim != 2 or traces.shape[1] != ops["nframes"]:
            raise ValueError(f"F.npy frame count does not match ops.npy: {plane}")
        del traces

    # Supply fresh containers: the converter mutates its arguments. Restrict
    # the database schema to recorded keys so acquisition values are not guessed.
    db, settings, outputs = convert_settings_orig(
        deepcopy(ops), db={key: None for key in default_db() if key in ops},
        settings=default_settings(),
    )
    db = {**db, **outputs}
    db.update(save_path=str(plane), db_path=str(plane / "db.npy"),
              settings_path=str(plane / "settings.npy"))
    payloads: dict[str, dict[str, Any]] = {"db.npy": db, "settings.npy": settings}
    created: list[Path] = []
    try:
        for destination in missing:
            # Exclusive creation also protects a file created after the check.
            with destination.open("xb") as stream:
                created.append(destination)
                np.save(stream, payloads[destination.name])
    except BaseException as error:
        cleanup_errors: list[str] = []
        for destination in reversed(created):
            try:
                destination.unlink()
            except OSError as cleanup_error:
                cleanup_errors.append(f"{destination}: {cleanup_error}")
        if cleanup_errors:
            raise OSError(
                f"Metadata recovery failed: {error}. Could not remove newly created files: "
                + "; ".join(cleanup_errors)
            ) from error
        raise
    return tuple(created)


def main() -> None:
    """Recover one plane's missing metadata from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plane_dir", help="Plane directory containing trusted ops.npy")
    args = parser.parse_args()
    try:
        created = recover_plane_metadata(args.plane_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    for path in created:
        print(f"Recovered {path}")
    if not created:
        print("db.npy and settings.npy already exist; no files changed.")


if __name__ == "__main__":
    main()
