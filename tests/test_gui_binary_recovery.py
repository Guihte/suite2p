"""Offscreen menu, recovery, cancellation, and manual-ROI error behavior."""

import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from qtpy import QtCore
from qtpy.QtWidgets import QApplication, QMainWindow, QProgressDialog, QPushButton

from suite2p.gui import drawroi, io, menus
from test_recover_binary import make_plane

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


@pytest.mark.parametrize("case", ["unloaded", "combined", "remote", "existing"])
def test_gui_refusal_preserves_loaded_state(
    app: QApplication, tmp_path: Path, messages: list[tuple[str, str]], case: str,
) -> None:
    plane, _ = make_plane(tmp_path)
    window = _Window(plane)
    if case == "unloaded":
        window.loaded = False
    elif case == "combined":
        window.basename = str(tmp_path / "combined")
    elif case == "remote":
        window.basename = r"\\nas.invalid\share\plane0"
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

    def recovery(plane_dir, *, progress, cancel_requested):
        deadline = time.monotonic() + 3
        while not cancel_requested() and time.monotonic() < deadline:
            time.sleep(0.005)
        cancelled.append(cancel_requested())
        raise InterruptedError("Binary recovery cancelled")

    def click_cancel() -> None:
        dialog = app.activeModalWidget()
        assert isinstance(dialog, QProgressDialog)
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
