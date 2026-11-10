"""Missing plane metadata recovery and combined regeneration on local fixtures."""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from suite2p.io.recover_metadata import recover_plane_metadata
from suite2p.io.save import combined
from suite2p.parameters import default_settings
from test_recover_binary import forbid_network_io


def make_plane(root: Path, index: int = 0, *, modern: bool = False) -> Path:
    plane = root / f"plane{index}"
    plane.mkdir(parents=True)
    ops = {
        "Ly": 3, "Lx": 4, "nframes": 5, "nchannels": 1, "nplanes": 2,
        "functional_chan": 1, "fs": 7.5, "tau": 0.7, "save_mat": False,
        "save_path": "old/location", "reg_file": r"\\nas.invalid\old\data.bin",
        "data_path": [r"\\nas.invalid\raw"], "version": "legacy-saved-version",
        "meanImg": np.full((3, 4), index + 1.0), "Vcorr": np.full((3, 4), 10 + index),
        "xrange": [0, 4], "yrange": [0, 3], "custom_recorded_value": [index, 42],
    }
    if modern:
        settings = default_settings()
        settings.update(fs=7.5, tau=0.7)
        np.save(plane / "settings.npy", settings)
        np.save(plane / "db.npy", {key: value for key, value in ops.items()
                                  if key not in ("meanImg", "Vcorr", "fs", "tau", "save_mat")})
        np.save(plane / "reg_outputs.npy", {"meanImg": ops["meanImg"]})
        np.save(plane / "detect_outputs.npy", {"Vcorr": ops["Vcorr"]})
    else:
        np.save(plane / "ops.npy", ops)
    stat = np.array([{"ypix": np.array([1]), "xpix": np.array([2]), "med": [1, 2]}], dtype=object)
    np.save(plane / "stat.npy", stat)
    for name, value in (("F.npy", index + 2), ("Fneu.npy", index + 3), ("spks.npy", index + 4)):
        np.save(plane / name, np.full((1, 5), value, dtype=np.float32))
    np.save(plane / "iscell.npy", np.array([[index % 2, 0.2 + 0.5 * index]], dtype=np.float32))
    np.save(plane / "redcell.npy", np.array([[1, 0.8]], dtype=np.float32))
    return plane


def snapshot(plane: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in plane.glob("*.npy")}


def test_recovers_metadata_preserving_saved_values_and_existing_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plane = make_plane(tmp_path)
    before = snapshot(plane)
    forbid_network_io(monkeypatch)
    created = recover_plane_metadata(plane)
    assert created == (plane / "db.npy", plane / "settings.npy")
    assert {path: path.read_bytes() for path in before} == before
    db = np.load(created[0], allow_pickle=True).item()
    settings = np.load(created[1], allow_pickle=True).item()
    assert (db['Ly'], db['Lx'], db['nframes'], db['nchannels'], db['nplanes'], db['functional_chan']) == (3, 4, 5, 1, 2, 1)
    assert db['save_path'] == str(plane)
    assert db['db_path'] == str(created[0])
    assert db['settings_path'] == str(created[1])
    assert db['reg_file'] == r"\\nas.invalid\old\data.bin"
    assert db['custom_recorded_value'] == [0, 42]
    assert 'keep_movie_raw' not in db  # No unrecorded database defaults.
    assert settings['fs'] == 7.5 and settings['tau'] == 0.7
    assert settings['version'] == 'legacy-saved-version'
    assert settings['io']['save_mat'] is False
    after = snapshot(plane)
    assert recover_plane_metadata(plane) == ()
    assert snapshot(plane) == after


@pytest.mark.parametrize('existing', ['db.npy', 'settings.npy'])
def test_only_missing_metadata_is_written(tmp_path: Path, existing: str) -> None:
    plane = make_plane(tmp_path)
    np.save(plane / existing, {'preserve': 'existing metadata'})
    before = (plane / existing).read_bytes()
    assert len(recover_plane_metadata(plane)) == 1
    assert (plane / existing).read_bytes() == before


@pytest.mark.parametrize('first_modern', [False, True])
def test_combined_recovers_mixed_planes_without_movies_or_changing_sources(tmp_path: Path, first_modern: bool) -> None:
    planes = [make_plane(tmp_path, index, modern=(first_modern == (index == 0))) for index in range(2)]
    before = {path: content for plane in planes for path, content in snapshot(plane).items()}
    result = combined(str(tmp_path), save=True)
    stat, db, settings, traces, neuropil, spikes, cells, probs, red, red_probs, has_red = result
    assert {path: path.read_bytes() for path in before} == before
    assert all((plane / 'db.npy').is_file() and (plane / 'settings.npy').is_file() for plane in planes)
    assert not list(tmp_path.rglob('*.bin')) and not list(tmp_path.rglob('*.tif'))
    assert traces.shape == neuropil.shape == spikes.shape == (2, 5)
    np.testing.assert_array_equal(traces[:, 0], [2, 3])
    np.testing.assert_array_equal(cells, [0, 1])
    np.testing.assert_allclose(probs, [0.2, 0.7])
    assert has_red and np.all(red == 1)
    np.testing.assert_allclose(red_probs, 0.8)
    assert [entry['source_plane'] for entry in stat] == [0, 1]
    assert [entry['source_plane_roi_index'] for entry in stat] == [0, 0]
    assert [entry['xpix'].tolist() for entry in stat] == [[2], [6]]
    assert (db['Ly'], db['Lx']) == (3, 8)
    np.testing.assert_array_equal(db['meanImg'][:, :4], 1)
    np.testing.assert_array_equal(db['meanImg'][:, 4:], 2)
    np.testing.assert_array_equal(db['Vcorr'][:, :4], 10)
    np.testing.assert_array_equal(db['Vcorr'][:, 4:], 11)
    assert settings['fs'] == 7.5 and settings['tau'] == 0.7
    assert (tmp_path / 'combined' / 'db.npy').is_file()
    repeated = combined(str(tmp_path), save=True)
    np.testing.assert_array_equal(repeated[3], traces)
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize('case', ['missing_ops', 'not_dict', 'missing_nframes', 'missing_fs', 'missing_tau',
                                  'missing_meanImg', 'wrong_meanImg', 'wrong_frames', 'nonpositive', 'boolean',
                                  'wrong_Vcorr', 'wrong_max_proj', 'bad_xrange', 'bad_yrange', 'float_xrange'])
def test_invalid_metadata_is_rejected_before_creation(tmp_path: Path, case: str) -> None:
    plane = make_plane(tmp_path)
    path = plane / 'ops.npy'
    ops = np.load(path, allow_pickle=True).item()
    if case == 'missing_ops':
        path.unlink()
    elif case == 'not_dict':
        np.save(path, np.arange(2))
    else:
        if case.startswith('missing_'):
            del ops[case.removeprefix('missing_')]
        elif case == 'wrong_meanImg':
            ops['meanImg'] = np.zeros((1, 1))
        elif case in ('wrong_Vcorr', 'wrong_max_proj'):
            ops[case.removeprefix('wrong_')] = np.zeros((1, 1))
        elif case == 'bad_xrange':
            ops['xrange'] = [0, 5]
        elif case == 'bad_yrange':
            ops['yrange'] = [2, 1]
        elif case == 'float_xrange':
            ops['xrange'] = [0.0, 4.0]
        elif case == 'wrong_frames':
            ops['nframes'] = 6
        elif case == 'nonpositive':
            ops['nchannels'] = 0
        else:
            ops['Ly'] = True
        np.save(path, ops)
    with pytest.raises((ValueError, FileNotFoundError)):
        recover_plane_metadata(plane)
    assert not (plane / 'db.npy').exists() and not (plane / 'settings.npy').exists()


def test_save_failure_removes_only_new_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plane = make_plane(tmp_path)
    before = snapshot(plane)
    save = np.save

    def fail_settings(stream, payload):
        if str(stream.name).endswith('settings.npy'):
            stream.write(b'partial')
            raise OSError('simulated write failure')
        save(stream, payload)

    monkeypatch.setattr(np, 'save', fail_settings)
    with pytest.raises(OSError, match='simulated write failure'):
        recover_plane_metadata(plane)
    assert snapshot(plane) == before


@pytest.mark.parametrize('alias', [r"\\nas.invalid\share\plane0", '//nas.invalid/share/plane0', 'Z:/suite2p/plane0'])
def test_network_spelling_redirected_to_local_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str) -> None:
    plane = make_plane(tmp_path)
    absolute = Path.absolute
    spelling = str(Path(alias))
    monkeypatch.setattr(Path, 'absolute', lambda path: plane if str(path) == spelling else absolute(path))
    forbid_network_io(monkeypatch)
    assert len(recover_plane_metadata(alias)) == 2


def test_gui_regenerate_recovers_legacy_plane_and_reloads_combined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from suite2p.gui import io

    plane = make_plane(tmp_path)
    parent = SimpleNamespace(basename=str(plane), loaded=True)
    messages = []
    loaded = []
    monkeypatch.setattr(io.QMessageBox, 'information', lambda _parent, _title, text: messages.append(text))
    monkeypatch.setattr(io.QMessageBox, 'critical', lambda _parent, _title, text: pytest.fail(text))
    monkeypatch.setattr(io, '_load_combined_output_to_gui', lambda _parent, root, output: loaded.append((root, output)))
    io.regenerate_combined_folder(parent)
    assert len(loaded) == len(messages) == 1
    assert loaded[0][0] == tmp_path
    assert (plane / 'db.npy').is_file() and (plane / 'settings.npy').is_file()
    assert (tmp_path / 'combined' / 'stat.npy').is_file()


def test_gui_still_rejects_missing_roi_traces(tmp_path: Path) -> None:
    from suite2p.gui import io

    plane = make_plane(tmp_path)
    (plane / 'F.npy').unlink()
    with pytest.raises(ValueError, match='F.npy'):
        io._validate_plane_outputs_for_combined(tmp_path)
    assert not (plane / 'db.npy').exists()
