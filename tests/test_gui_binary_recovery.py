"""Offscreen menu, recovery, cancellation, and manual-ROI error behavior."""

import os
import time
from threading import Event
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from qtpy import QtCore
from qtpy.QtWidgets import QApplication, QMainWindow, QPushButton

from suite2p.gui import drawroi, io, menus
from test_recover_binary import forbid_network_io, make_mixed_exports, make_plane

if TYPE_CHECKING:
    from suite2p.gui.gui2p import MainWindow


@pytest.fixture(scope="module")
def app() -> QApplication:
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
    assert isinstance(application, QApplication)
    return application


@pytest.fixture
def messages(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(io.QMessageBox, "warning", lambda parent, title, text: recorded.append(("warning", text)))
    monkeypatch.setattr(io.QMessageBox, "information", lambda parent, title, text: recorded.append(("information", text)))
    return recorded


class _Window(QMainWindow):
    def __init__(self, plane: str | Path) -> None:
        super().__init__()
        self.loaded = True
        self.basename = str(plane)
        self.ops = {"reg_file": "old-location/data.bin"}
        self.stat = object()


def test_file_menu_action_is_next_to_propagation(app: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    window = QMainWindow()
    calls: list[QMainWindow] = []
    monkeypatch.setattr(io, "recover_registered_binary", lambda parent: calls.append(parent))
    menus.mainmenu(window)
    menu_bar = window.menuBar()
    file_action = menu_bar.actions()[0]
    file_menu = file_action.menu()
    assert file_menu is not None
    actions = file_menu.actions()
    labels = [action.text() for action in actions]
    index = labels.index("Rebuild registered binary from TIFFs")
    assert labels[index - 1] == "Propagate combined labels to plane folders"
    actions[index].trigger()
    assert calls == [window]
    window.close()


def test_gui_recovery_updates_path_without_reloading_rois(
    app: QApplication, tmp_path: Path, messages: list[tuple[str, str]],
) -> None:
    plane, _ = make_plane(tmp_path, 3)
    window = _Window(plane)
    original_stat = window.stat
    original_ops = (plane / "ops.npy").read_bytes()
    io.recover_registered_binary(cast("MainWindow", window))
    assert window.ops["reg_file"] == str(plane / "data.bin")
    assert window.stat is original_stat
    assert (plane / "ops.npy").read_bytes() == original_ops
    assert messages[0][0] == "information"
    assert "retry Extract ROIs" in messages[0][1]
    window.close()


@pytest.mark.parametrize("case", ["unloaded", "combined", "existing"])
def test_gui_refusal_preserves_loaded_state(
    app: QApplication, tmp_path: Path, messages: list[tuple[str, str]], case: str,
) -> None:
    plane, _ = make_plane(tmp_path)
    window = _Window(plane)
    if case == "unloaded":
        window.loaded = False
    elif case == "combined":
        window.basename = str(tmp_path / "combined")
    else:
        (plane / "data.bin").write_bytes(b"keep existing")
    io.recover_registered_binary(cast("MainWindow", window))
    assert window.ops["reg_file"] == "old-location/data.bin"
    assert messages[0][0] == "warning"
    if case == "existing":
        assert (plane / "data.bin").read_bytes() == b"keep existing"
    else:
        assert not (plane / "data.bin").exists()
    window.close()


def test_cancel_keeps_gui_state_and_waits_for_worker_cleanup(
    app: QApplication, tmp_path: Path, messages: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    plane, _ = make_plane(tmp_path)
    window = _Window(plane)
    cancelled: list[bool] = []

    def recovery(plane_dir, *, tiff_set=None, progress, cancel_requested):
        deadline = time.monotonic() + 3
        while not cancel_requested() and time.monotonic() < deadline:
            time.sleep(0.005)
        cancelled.append(cancel_requested())
        raise InterruptedError("Binary recovery cancelled")

    def click_cancel() -> None:
        dialog = app.activeModalWidget()
        assert isinstance(dialog, io._BinaryRecoveryDialog)
        button = dialog.findChild(QPushButton)
        assert button is not None
        button.click()

    monkeypatch.setattr(io, "recover_binary", recovery)
    QtCore.QTimer.singleShot(0, click_cancel)
    io.recover_registered_binary(cast("MainWindow", window))
    assert cancelled == [True]
    assert not messages
    assert window.ops["reg_file"] == "old-location/data.bin"
    assert not (plane / "data.bin").exists()
    window.close()


def test_missing_binary_keeps_manual_rois_and_points_to_menu(
    messages: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    rois = [object()]
    host = SimpleNamespace(parent=SimpleNamespace(ops={}, basename="plane0"), ROIs=rois)

    def missing(*args: object) -> None:
        raise FileNotFoundError("No data.bin")

    monkeypatch.setattr(drawroi, "_resolve_manual_roi_binaries", missing)
    drawroi.ROIDraw.proc_ROI(cast(drawroi.ROIDraw, host))
    assert host.ROIs is rois
    assert "File > Rebuild registered binary from TIFFs" in messages[0][1]


@pytest.mark.parametrize("alias", [r"\\nas.invalid\share\plane0", "//nas.invalid/share/plane0", "Z:/suite2p/plane0"])
def test_gui_network_alias_uses_shared_recovery_without_real_network_io(
    app: QApplication, tmp_path: Path, messages: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch, alias: str,
) -> None:
    plane, _ = make_plane(tmp_path)
    window = _Window(alias)
    recover = io.recover_binary
    calls: list[str] = []

    def mapped_recovery(source, **callbacks):
        assert str(source) == str(Path(alias))
        calls.append(str(source))
        return recover(plane, **callbacks)

    monkeypatch.setattr(io, "recover_binary", mapped_recovery)
    forbid_network_io(monkeypatch)
    io.recover_registered_binary(cast("MainWindow", window))
    assert len(calls) == 1
    assert window.ops["reg_file"] == str(plane / "data.bin")
    assert messages[0][0] == "information"
    window.close()


def test_cancellation_remains_responsive_while_io_is_outstanding(
    app: QApplication, tmp_path: Path, messages: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    plane, _ = make_plane(tmp_path)
    window = _Window(plane)
    release_io = Event()
    observations: list[bool] = []

    def recovery(source, *, tiff_set=None, progress, cancel_requested):
        assert release_io.wait(3), "GUI event loop blocked during cancellation"
        assert cancel_requested()
        raise InterruptedError("Binary recovery cancelled")

    def observe_and_release(dialog: io._BinaryRecoveryDialog) -> None:
        observations.append(dialog.isVisible() and not dialog.buttons.isEnabled())
        release_io.set()

    def click_cancel() -> None:
        dialog = app.activeModalWidget()
        assert isinstance(dialog, io._BinaryRecoveryDialog)
        button = dialog.findChild(QPushButton)
        assert button is not None
        button.click()
        QtCore.QTimer.singleShot(10, lambda: observe_and_release(dialog))

    monkeypatch.setattr(io, "recover_binary", recovery)
    QtCore.QTimer.singleShot(0, click_cancel)
    io.recover_registered_binary(cast("MainWindow", window))
    assert observations == [True]
    assert not messages
    window.close()


def test_recovery_does_not_attach_binary_to_replaced_gui_result(
    app: QApplication, tmp_path: Path, messages: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    plane, _ = make_plane(tmp_path)
    window = _Window(plane)
    release = Event()
    original_recover = io.recover_binary
    replacement_ops = {"reg_file": "different-plane/data.bin"}

    def recovery(source, **callbacks):
        assert release.wait(3)
        return original_recover(source, **callbacks)

    def replace_loaded_result() -> None:
        window.basename = str(tmp_path / "plane1")
        window.ops = replacement_ops
        release.set()

    monkeypatch.setattr(io, "recover_binary", recovery)
    QtCore.QTimer.singleShot(0, replace_loaded_result)
    io.recover_registered_binary(cast("MainWindow", window))
    assert window.ops is replacement_ops
    assert replacement_ops["reg_file"] == "different-plane/data.bin"
    assert (plane / "data.bin").exists()
    assert "loaded result changed" in messages[0][1]
    window.close()


@pytest.mark.parametrize("select", [True, False])
def test_gui_selects_one_export_or_cancels_without_writing(
    app: QApplication, tmp_path: Path, messages: list[tuple[str, str]],
    monkeypatch: pytest.MonkeyPatch, select: bool,
) -> None:
    plane, current, _ = make_mixed_exports(tmp_path)
    window = _Window(plane)
    offered: list[str] = []

    def choose(parent, title, label, choices, *args):
        offered.extend(choices)
        return choices[0], select

    monkeypatch.setattr(io.QInputDialog, "getItem", choose)
    io.recover_registered_binary(cast("MainWindow", window))
    assert offered == ["Files like file 0000.tif", "Files like file00000_chan0.tif"]
    if select:
        actual = np.fromfile(plane / "data.bin", dtype="<i2").reshape(-1, 2, 3)
        np.testing.assert_array_equal(actual, current)
        assert window.ops["reg_file"] == str(plane / "data.bin")
        assert messages[0][0] == "information"
    else:
        assert not (plane / "data.bin").exists()
        assert window.ops["reg_file"] == "old-location/data.bin"
        assert not messages
    window.close()


def test_cancelled_discovery_never_prompts_or_restarts(
    app: QApplication, tmp_path: Path, messages: list[tuple[str, str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    plane, _ = make_plane(tmp_path)
    window = _Window(plane)
    calls: list[Path] = []

    def recovery(source, *, tiff_set=None, progress, cancel_requested):
        calls.append(source)
        deadline = time.monotonic() + 3
        while not cancel_requested() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert cancel_requested()
        raise io.TiffExportSelectionRequired({"spaced": "file 0000.tif", "legacy": "file00000_chan0.tif"})

    def unexpected_selector(*args):
        raise AssertionError("Cancellation must not prompt for an export or restart recovery")

    def click_cancel() -> None:
        dialog = app.activeModalWidget()
        assert isinstance(dialog, io._BinaryRecoveryDialog)
        dialog.reject()

    monkeypatch.setattr(io, "recover_binary", recovery)
    monkeypatch.setattr(io.QInputDialog, "getItem", unexpected_selector)
    QtCore.QTimer.singleShot(0, click_cancel)
    io.recover_registered_binary(cast("MainWindow", window))
    assert calls == [plane]
    assert not messages
    assert not (plane / "data.bin").exists()
    window.close()
