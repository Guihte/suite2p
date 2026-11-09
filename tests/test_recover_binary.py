"""Recovery of Suite2p binaries from local registered TIFF batches."""

import builtins
import io as stdlib_io
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from suite2p.io import BinaryFile, recover_binary as suite2p_binary


def make_plane(root: Path, count: int = 1) -> tuple[Path, list[np.ndarray]]:
    plane = root / "plane0"
    reg = plane / "reg_tif"
    reg.mkdir(parents=True)
    frames = [np.array([[-32768 + i, -i, 0], [1, i, 32767 - i]], dtype=np.int16)
              for i in range(count)]
    for i, frame in enumerate(frames):
        tifffile.imwrite(reg / f"file{i:05d}.tif", frame)
    np.save(plane / "ops.npy", {"Ly": 2, "Lx": 3, "nframes": count, "data_path": r"\\\\server\\ignored"})
    np.save(plane / "F.npy", np.zeros((2, count), dtype=np.float32))
    return plane, frames


def test_signed_values_numeric_batch_order_and_metadata_unchanged(tmp_path: Path) -> None:
    plane, frames = make_plane(tmp_path, 12)
    for index in range(12):
        (plane / 'reg_tif' / f'file{index:05d}.tif').rename(plane / 'reg_tif' / f'file{index}.tif')
    before = {p: p.read_bytes() for p in (plane / "ops.npy", plane / "F.npy")}
    result = suite2p_binary.recover_binary(plane)
    with BinaryFile(Ly=2, Lx=3, filename=str(result)) as binary:
        actual = binary[:].copy()
    np.testing.assert_array_equal(actual, np.stack(frames))
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("name", ["file00002.tif", "file00000_chan1.tif", "file00000.tiff"])
def test_missing_or_ambiguous_batch_rejected(tmp_path: Path, name: str) -> None:
    plane, _ = make_plane(tmp_path)
    (plane / "reg_tif" / "file00000.tif").rename(plane / "reg_tif" / name)
    with pytest.raises(ValueError, match="indices|ambiguous"):
        suite2p_binary.recover_binary(plane)
    assert not (plane / "data.bin").exists()


@pytest.mark.parametrize("change", ["shape", "dtype", "frames", "traces", "missing_ops"])
def test_invalid_metadata_or_tiff_rejected_before_output(tmp_path: Path, change: str) -> None:
    plane, _ = make_plane(tmp_path)
    image = plane / "reg_tif" / "file00000.tif"
    if change == "shape":
        tifffile.imwrite(image, np.zeros((2, 4), dtype=np.int16))
    elif change == "dtype":
        tifffile.imwrite(image, np.zeros((2, 3), dtype=np.uint16))
    elif change == "frames":
        np.save(plane / "ops.npy", {"Ly": 2, "Lx": 3, "nframes": 2})
        (plane / "F.npy").unlink()
    elif change == "traces":
        np.save(plane / "F.npy", np.zeros((2, 2), dtype=np.float32))
    else:
        np.save(plane / "ops.npy", {"Ly": 2, "Lx": 3})
    with pytest.raises(ValueError):
        suite2p_binary.recover_binary(plane)
    assert not list(plane.glob(".data.bin.*"))
    assert not (plane / "data.bin").exists()


def test_existing_binary_is_preserved(tmp_path: Path) -> None:
    plane, _ = make_plane(tmp_path)
    destination = plane / "data.bin"
    destination.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        suite2p_binary.recover_binary(plane)
    assert destination.read_bytes() == b"original"


@pytest.mark.parametrize("failure", ["decode", "publish"])
def test_failure_removes_temporary_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    plane, _ = make_plane(tmp_path)
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("injected failure")
    if failure == "decode":
        monkeypatch.setattr(tifffile.TiffPage, "asarray", fail)
    else:
        monkeypatch.setattr(suite2p_binary.os, "rename" if os.name == "nt" else "link", fail)
    with pytest.raises(OSError, match="injected"):
        suite2p_binary.recover_binary(plane)
    assert not (plane / "data.bin").exists()
    assert not list(plane.glob(".data.bin.*"))


def forbid_network_io(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make accidental real network filesystem calls fail before performing I/O."""
    def guard(operation):
        def checked(*args, **kwargs):
            for value in (*args, *kwargs.values()):
                if isinstance(value, (str, Path)):
                    spelling = str(value).replace("\\", "/").lower()
                    if spelling.startswith(("//", "z:")):
                        raise AssertionError(f"Unexpected real network access: {value}")
            return operation(*args, **kwargs)
        return checked

    for module, name in (
        (builtins, "open"), (stdlib_io, "open"), (os, "open"), (os, "stat"),
        (os, "lstat"), (os, "scandir"), (os, "rename"), (os, "link"), (os, "unlink"),
    ):
        monkeypatch.setattr(module, name, guard(getattr(module, name)))


@pytest.mark.parametrize("alias", [r"\\nas.invalid\share\plane0", "//nas.invalid/share/plane0", "Z:/suite2p/plane0"])
def test_network_spelling_with_local_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str,
) -> None:
    plane, frames = make_plane(tmp_path, 3)
    absolute = Path.absolute
    spelling = str(Path(alias))

    def mapped_absolute(path: Path) -> Path:
        if str(path) == spelling:
            return plane
        return absolute(path)

    # Redirect the alias before any filesystem call; never contact a real NAS.
    monkeypatch.setattr(Path, "absolute", mapped_absolute)
    forbid_network_io(monkeypatch)
    result = suite2p_binary.recover_binary(alias)
    np.testing.assert_array_equal(np.fromfile(result, dtype="<i2").reshape(-1, 2, 3), np.stack(frames))


def test_write_failure_removes_temporary_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plane, _ = make_plane(tmp_path)
    original = suite2p_binary.tempfile.NamedTemporaryFile

    class FailingOutput:
        def __init__(self, **kwargs: object) -> None:
            self.file = original(**kwargs)
            self.name = self.file.name

        def __enter__(self) -> "FailingOutput":
            return self

        def __exit__(self, *args: object) -> None:
            self.file.close()

        def write(self, data: bytes) -> None:
            raise OSError("injected write failure")

    monkeypatch.setattr(suite2p_binary.tempfile, "NamedTemporaryFile", FailingOutput)
    with pytest.raises(OSError, match="injected write failure"):
        suite2p_binary.recover_binary(plane)
    assert not (plane / "data.bin").exists()
    assert not list(plane.glob(".data.bin.*"))


def test_symlinked_f_is_supported(tmp_path: Path) -> None:
    plane, _ = make_plane(tmp_path)
    traces = plane / "F.npy"
    original = plane / "F_original.npy"
    traces.rename(original)
    try:
        traces.symlink_to(original)
    except OSError:
        pytest.skip("Creating symbolic links is unavailable")
    assert suite2p_binary.recover_binary(plane) == plane / "data.bin"


def test_legacy_offsets_follow_number_of_pages(tmp_path: Path) -> None:
    plane, frames = make_plane(tmp_path, 5)
    reg = plane / "reg_tif"
    for path in reg.glob("*.tif"):
        path.unlink()
    tifffile.imwrite(reg / "file00000_chan0.tif", np.stack(frames[:3]), photometric="minisblack")
    tifffile.imwrite(reg / "file00003_chan0.tif", np.stack(frames[3:]), photometric="minisblack")
    result = suite2p_binary.recover_binary(plane)
    np.testing.assert_array_equal(np.fromfile(result, dtype="<i2").reshape(-1, 2, 3), np.stack(frames))


def test_legacy_missing_frames_are_rejected(tmp_path: Path) -> None:
    plane, frames = make_plane(tmp_path, 5)
    reg = plane / "reg_tif"
    for path in reg.glob("*.tif"):
        path.unlink()
    tifffile.imwrite(reg / "file00000_chan0.tif", np.stack(frames[:3]), photometric="minisblack")
    tifffile.imwrite(reg / "file00004_chan0.tif", np.stack(frames[3:]), photometric="minisblack")
    with pytest.raises(ValueError, match="start-frame"):
        suite2p_binary.recover_binary(plane)
    assert not (plane / "data.bin").exists()


@pytest.mark.parametrize("other", ["file000000.tif", "file00001_chan0.tif"])
def test_duplicate_or_mixed_names_are_rejected(tmp_path: Path, other: str) -> None:
    plane, frames = make_plane(tmp_path)
    tifffile.imwrite(plane / "reg_tif" / other, frames[0])
    with pytest.raises(ValueError, match="Duplicate|Multiple"):
        suite2p_binary.recover_binary(plane)
    assert not (plane / "data.bin").exists()


@pytest.mark.parametrize("cancel_at", [0, 1, 3])
def test_cancellation_never_publishes_partial_binary(tmp_path: Path, cancel_at: int) -> None:
    plane, _ = make_plane(tmp_path, 3)
    written = -1
    updates: list[tuple[int, int]] = []

    def progress(current: int, total: int) -> None:
        nonlocal written
        written = current
        updates.append((current, total))

    with pytest.raises(InterruptedError, match="cancelled"):
        suite2p_binary.recover_binary(
            plane, progress=progress, cancel_requested=lambda: written >= cancel_at,
        )
    assert updates[0] == (0, 3)
    assert not (plane / "data.bin").exists()
    assert not list(plane.glob(".data.bin.*"))


def test_concurrent_destination_is_not_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plane, _ = make_plane(tmp_path)
    operation = "rename" if os.name == "nt" else "link"
    publish = getattr(os, operation)

    def create_before_publish(source, destination):
        Path(destination).write_bytes(b"concurrent output")
        return publish(source, destination)

    monkeypatch.setattr(suite2p_binary.os, operation, create_before_publish)
    with pytest.raises(FileExistsError):
        suite2p_binary.recover_binary(plane)
    assert (plane / "data.bin").read_bytes() == b"concurrent output"
    assert not list(plane.glob(".data.bin.*"))


@pytest.mark.skipif(os.name != "nt", reason="Windows no-overwrite rename publication")
def test_windows_publish_needs_no_hardlink_or_cleanup_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plane, _ = make_plane(tmp_path)

    def unexpected(*args, **kwargs):
        raise AssertionError("Successful Windows publication must not link or unlink")

    monkeypatch.setattr(suite2p_binary.os, "link", unexpected)
    monkeypatch.setattr(Path, "unlink", unexpected)
    assert suite2p_binary.recover_binary(plane) == plane / "data.bin"


@pytest.mark.parametrize("failure", ["read", "cancel"])
def test_cleanup_failure_reports_original_error_and_leftover_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    plane, _ = make_plane(tmp_path)
    cancel = False
    unlink = Path.unlink

    def cleanup(path: Path, *args, **kwargs):
        if path.name.startswith(".data.bin."):
            raise OSError("share disconnected during cleanup")
        return unlink(path, *args, **kwargs)

    def progress(written: int, total: int) -> None:
        nonlocal cancel
        cancel = failure == "cancel"

    def failed_read(*args, **kwargs):
        raise OSError("share disconnected during read")

    monkeypatch.setattr(Path, "unlink", cleanup)
    if failure == "read":
        monkeypatch.setattr(tifffile.TiffPage, "asarray", failed_read)
    with pytest.raises(OSError, match="Could not remove temporary file") as caught:
        suite2p_binary.recover_binary(plane, progress=progress, cancel_requested=lambda: cancel)
    assert caught.value.__cause__ is not None
    assert "cancelled" in str(caught.value) if failure == "cancel" else "disconnected during read" in str(caught.value)
    assert "disconnected during cleanup" in str(caught.value)
    leftovers = list(plane.glob(".data.bin.*"))
    assert len(leftovers) == 1 and str(leftovers[0]) in str(caught.value)
    assert not (plane / "data.bin").exists()


def make_mixed_exports(root: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    plane, frames = make_plane(root, 5)
    current = np.stack(frames)
    legacy = current.copy()
    legacy[:, 0, 0] = -123
    reg = plane / "reg_tif"
    for path in reg.glob("*.tif"):
        path.unlink()
    tifffile.imwrite(reg / "file 0000.tif", current[:3], photometric="minisblack")
    tifffile.imwrite(reg / "file 0001.tif", current[3:], photometric="minisblack")
    tifffile.imwrite(reg / "file00000_chan0.tif", legacy[:3], photometric="minisblack")
    tifffile.imwrite(reg / "file00003_chan0.tif", legacy[3:], photometric="minisblack")
    np.save(plane / "ops.npy", {"Ly": 2, "Lx": 3, "nframes": 5, "nchannels": 2, "functional_chan": 2})
    return plane, current, legacy


def test_actual_suite2p_space_format_is_supported(tmp_path: Path) -> None:
    plane, frames = make_plane(tmp_path, 12)
    for index in range(12):
        (plane / "reg_tif" / f"file{index:05d}.tif").rename(plane / "reg_tif" / f"file{index: 05d}.tif")
    result = suite2p_binary.recover_binary(plane)
    np.testing.assert_array_equal(np.fromfile(result, dtype="<i2").reshape(-1, 2, 3), np.stack(frames))


def test_complete_exports_require_explicit_selection(tmp_path: Path) -> None:
    plane, _, _ = make_mixed_exports(tmp_path)
    with pytest.raises(suite2p_binary.TiffExportSelectionRequired) as caught:
        suite2p_binary.recover_binary(plane)
    assert caught.value.exports == {"spaced": "file 0000.tif", "legacy": "file00000_chan0.tif"}
    assert not (plane / "data.bin").exists()


@pytest.mark.parametrize("tiff_set", ["spaced", "legacy"])
def test_explicit_export_never_mixes_movies(tmp_path: Path, tiff_set: suite2p_binary.TiffSet) -> None:
    plane, current, legacy = make_mixed_exports(tmp_path)
    expected = current if tiff_set == "spaced" else legacy
    result = suite2p_binary.recover_binary(plane, tiff_set=tiff_set)
    actual = np.fromfile(result, dtype="<i2").reshape(-1, 2, 3)
    assert actual.shape[0] == 5
    np.testing.assert_array_equal(actual, expected)


def test_cancel_during_discovery_does_not_request_export_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plane, _, _ = make_mixed_exports(tmp_path)
    cancelled = False
    iterdir = Path.iterdir

    def listing(path: Path):
        nonlocal cancelled
        for entry in iterdir(path):
            cancelled = True
            yield entry

    monkeypatch.setattr(Path, "iterdir", listing)
    with pytest.raises(InterruptedError, match="cancelled"):
        suite2p_binary.recover_binary(plane, cancel_requested=lambda: cancelled)
    assert not (plane / "data.bin").exists()
