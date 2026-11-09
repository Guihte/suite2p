"""Recovery of Suite2p binaries from local registered TIFF batches."""

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
        monkeypatch.setattr(suite2p_binary.os, "link", fail)
    with pytest.raises(OSError, match="injected"):
        suite2p_binary.recover_binary(plane)
    assert not (plane / "data.bin").exists()
    assert not list(plane.glob(".data.bin.*"))


def test_unc_refused_before_filesystem_access(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected filesystem access")
    monkeypatch.setattr(Path, "lstat", fail)
    with pytest.raises(ValueError, match="UNC"):
        suite2p_binary.recover_binary(r"\\\\server\\share\\plane0")

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


def test_symlinked_f_is_refused_before_loading(tmp_path: Path) -> None:
    plane, _ = make_plane(tmp_path)
    traces = plane / "F.npy"
    traces.unlink()
    try:
        traces.symlink_to(tmp_path / "missing")
    except OSError:
        pytest.skip("Creating symbolic links is unavailable")
    with pytest.raises(ValueError, match="links and reparse points"):
        suite2p_binary.recover_binary(plane)
    assert not (plane / "data.bin").exists()

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
    with pytest.raises(ValueError, match="Duplicate|Mixed"):
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
