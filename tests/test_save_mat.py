import importlib.util
from pathlib import Path

import numpy as np


SAVE_MODULE_PATH = Path(__file__).parents[1] / "suite2p" / "io" / "save.py"
SAVE_MODULE_SPEC = importlib.util.spec_from_file_location(
    "suite2p_io_save_for_test", SAVE_MODULE_PATH
)
save_module = importlib.util.module_from_spec(SAVE_MODULE_SPEC)
SAVE_MODULE_SPEC.loader.exec_module(save_module)
_sanitize_for_matlab = save_module._sanitize_for_matlab
save_mat = save_module.save_mat


def test_sanitize_for_matlab_handles_nested_dict_with_none():
    value = {"outer": {"inner": None}}

    sanitized = _sanitize_for_matlab(value)

    assert isinstance(sanitized["outer"]["inner"], np.ndarray)
    assert sanitized["outer"]["inner"].size == 0


def test_sanitize_for_matlab_handles_object_array_with_nested_none():
    value = np.empty((1, 2), dtype=object)
    value[0, 0] = {"x": None}
    value[0, 1] = {"y": 1}

    sanitized = _sanitize_for_matlab(value)

    assert sanitized.shape == value.shape
    assert sanitized.dtype == object
    assert isinstance(sanitized[0, 0]["x"], np.ndarray)
    assert sanitized[0, 0]["x"].size == 0
    assert sanitized[0, 1]["y"] == 1


def test_sanitize_for_matlab_handles_pathlib_path():
    path = Path("suite2p") / "plane0"

    sanitized = _sanitize_for_matlab({"path": path})

    assert sanitized["path"] == str(path)


def test_sanitize_for_matlab_handles_mixed_list_and_tuple():
    value = [None, (Path("a"), {"b": None})]

    sanitized = _sanitize_for_matlab(value)

    assert isinstance(sanitized[0], np.ndarray)
    assert sanitized[0].size == 0
    assert sanitized[1][0] == str(Path("a"))
    assert isinstance(sanitized[1][1]["b"], np.ndarray)
    assert sanitized[1][1]["b"].size == 0


def test_sanitize_for_matlab_does_not_mutate_input_objects():
    value = {"outer": {"inner": None}, "items": [None]}

    sanitized = _sanitize_for_matlab(value)

    assert value["outer"]["inner"] is None
    assert value["items"][0] is None
    assert sanitized is not value
    assert sanitized["outer"] is not value["outer"]
    assert sanitized["items"] is not value["items"]


def test_save_mat_writes_fall_mat_with_nested_none(tmp_path):
    ops = {
        "save_path": tmp_path,
        "nested": {"none_value": None},
        "path_value": tmp_path / "input.tif",
    }
    stat = np.empty(1, dtype=object)
    stat[0] = {"ypix": np.array([0]), "xpix": np.array([1]), "nested": {"bad": None}}
    F = np.array([[1.0, 2.0]])
    Fneu = np.array([[0.1, 0.2]])
    spks = np.array([[0.0, 1.0]])
    iscell = np.array([[1.0, 0.9]])
    redcell = None

    save_mat(ops, stat, F, Fneu, spks, iscell, redcell)

    assert (tmp_path / "Fall.mat").is_file()
