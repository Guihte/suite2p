"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os, time
from pathlib import Path
import numpy as np
import scipy.io
from natsort import natsorted
from scipy.ndimage import gaussian_filter1d
from qtpy import QtGui
from qtpy.QtWidgets import QFileDialog, QMessageBox

from . import utils, masks, views, graphics, traces, classgui
from .. import io


MANUAL_ROI_COMBINED_WARNING = (
    "Manual ROI addition must be performed on a single plane folder. "
    "Add ROIs in planeN, then regenerate combined."
)
MISSING_PROVENANCE_WARNING = (
    "Combined provenance is missing. Regenerate combined before propagating labels."
)
STALE_COMBINED_WARNING = (
    "Plane ROI counts no longer match this combined folder. "
    "Regenerate combined before propagating labels."
)
MISSING_REDCELL_WARNING = (
    "Redcell propagation was requested, but redcell.npy is missing and this run "
    "could not be confirmed as a two-channel/red-cell workflow."
)


def is_combined_folder(path):
    return Path(path).name == "combined"


def is_plane_folder(path):
    name = Path(path).name
    return name.startswith("plane") and name[5:].isdigit()


def get_suite2p_root_from_basename(basename):
    if not basename:
        raise ValueError("No Suite2P folder is currently loaded.")
    path = Path(basename)
    if is_combined_folder(path) or is_plane_folder(path):
        return path.parent
    raise ValueError("Choose a folder containing planeX folders.")


def find_plane_folders(suite2p_root):
    suite2p_root = Path(suite2p_root)
    if not suite2p_root.is_dir():
        raise ValueError(f"Suite2P folder does not exist: {suite2p_root}")
    return list(natsorted(
        [p for p in suite2p_root.iterdir() if p.is_dir() and p.name.startswith("plane")],
        key=lambda p: p.name,
    ))


def _validate_plane_outputs_for_combined(suite2p_root):
    plane_folders = find_plane_folders(suite2p_root)
    if not plane_folders:
        raise ValueError("No processed planeX folders in folder.")
    for plane_folder in plane_folders:
        plane_name = plane_folder.name
        for filename in ("stat.npy", "db.npy", "settings.npy"):
            if not (plane_folder / filename).is_file():
                raise ValueError(
                    f"Cannot regenerate combined because {plane_name} is missing "
                    f"required file: {filename}"
                )
        stat = np.load(plane_folder / "stat.npy", allow_pickle=True)
        if len(stat) == 0:
            continue
        for filename in ("iscell.npy", "F.npy", "Fneu.npy", "spks.npy"):
            if not (plane_folder / filename).is_file():
                raise ValueError(
                    f"Cannot regenerate combined because {plane_name} is missing "
                    f"required file: {filename}"
                )
    return plane_folders


def _validate_label_array(array, *, name):
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_rois, 2).")
    return array


def _validate_label_row_count(array, n_rois, *, name):
    array = _validate_label_array(array, name=name)
    if array.shape[0] != n_rois:
        raise ValueError(f"{name} row count does not match stat/F rows.")
    return array


def _validate_trace_row_count(array, n_rois, *, name):
    if array.shape[0] != n_rois:
        raise ValueError(f"{name} row count does not match stat.npy.")


def _load_label_array(path, *, name):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing required label file: {path}")
    return _validate_label_array(np.load(path), name=name)


def _zero_label_vectors(n_rois):
    return np.zeros((n_rois,), "bool"), np.zeros((n_rois,), np.float32)


def _leading_manual_roi_count(stat):
    count = 0
    for stat_entry in stat:
        if not bool(stat_entry.get("manual", 0)):
            break
        count += 1
    return count


def _coerce_optional_redcell_for_gui(redcell, n_rois, stat=None):
    redcell = _validate_label_array(redcell, name="redcell.npy")
    if redcell.shape[0] != n_rois:
        n_manual = _leading_manual_roi_count(stat) if stat is not None else 0
        if n_manual > 0 and redcell.shape[0] + n_manual == n_rois:
            redcell = np.concatenate(
                (np.zeros((n_manual, 2), dtype=redcell.dtype), redcell),
                axis=0,
            )
        else:
            raise ValueError("redcell.npy row count does not match stat/F rows.")
    return redcell[:, 0].astype("bool"), redcell[:, 1].copy(), True


def _combined_label_array_from_parent(parent, label_attr, prob_attr, fallback_path, *, name):
    labels = getattr(parent, label_attr, None)
    probs = getattr(parent, prob_attr, None)
    if labels is not None and probs is not None:
        labels = np.asarray(labels)
        probs = np.asarray(probs)
        if labels.shape[0] == probs.shape[0]:
            return _validate_label_array(
                np.concatenate((labels[:, np.newaxis], probs[:, np.newaxis]), axis=1),
                name=name,
            )
    return _load_label_array(fallback_path, name=name)


def _collect_provenance(stat):
    provenance = {}
    for combined_index, roi_stat in enumerate(stat):
        if ("source_plane" not in roi_stat or
                "source_plane_roi_index" not in roi_stat):
            raise ValueError(MISSING_PROVENANCE_WARNING)
        source_plane = int(roi_stat["source_plane"])
        source_index = int(roi_stat["source_plane_roi_index"])
        provenance.setdefault(source_plane, []).append((combined_index, source_index))
    return provenance


def _stat_has_combined_provenance(stat):
    return all(
        "source_plane" in stat_entry and "source_plane_roi_index" in stat_entry
        for stat_entry in stat
    )


def _validate_provenance_against_plane(plane_iscell, entries):
    source_indices = sorted(source_index for _, source_index in entries)
    if source_indices != list(range(plane_iscell.shape[0])):
        raise ValueError(STALE_COMBINED_WARNING)


def _plane_has_two_channel_metadata(plane_folder):
    for filename in ("ops.npy", "db.npy", "settings.npy"):
        path = Path(plane_folder) / filename
        if not path.is_file():
            continue
        try:
            metadata = np.load(path, allow_pickle=True).item()
        except Exception:
            continue
        if int(metadata.get("nchannels", 1)) > 1:
            return True
        if "meanImg_chan2" in metadata or "reg_file_chan2" in metadata:
            return True
    return False


def _load_or_create_plane_redcell(plane_folder, shape, redcell_missing_policy):
    redcell_path = Path(plane_folder) / "redcell.npy"
    if redcell_path.is_file():
        redcell = _load_label_array(redcell_path, name=f"{Path(plane_folder).name}/redcell.npy")
        if redcell.shape != shape:
            raise ValueError(
                f"{Path(plane_folder).name}/redcell.npy must match iscell.npy shape."
            )
        return redcell
    if (redcell_missing_policy == "zeros_if_two_channel" and
            _plane_has_two_channel_metadata(plane_folder)):
        return np.zeros(shape, dtype=np.float32)
    raise ValueError(MISSING_REDCELL_WARNING)


def _ensure_combined_stat_optional_fields(stat_entry):
    stat_entry.setdefault("snr", np.nan)
    return stat_entry


def propagate_combined_label_arrays_to_planes(
        suite2p_root,
        combined_stat,
        combined_iscell,
        *,
        propagate_redcell=False,
        combined_redcell=None,
        redcell_missing_policy="refuse"):
    suite2p_root = Path(suite2p_root)
    combined_stat = np.asarray(combined_stat, dtype=object)
    combined_iscell = _validate_label_array(combined_iscell, name="combined/iscell.npy")
    if combined_stat.shape[0] != combined_iscell.shape[0]:
        raise ValueError("combined/stat.npy and combined/iscell.npy row counts differ.")

    provenance = _collect_provenance(combined_stat)
    plane_iscells = {}
    plane_redcells = {}
    for source_plane, entries in provenance.items():
        plane_folder = suite2p_root / f"plane{source_plane}"
        plane_iscell = _load_label_array(
            plane_folder / "iscell.npy",
            name=f"plane{source_plane}/iscell.npy",
        )
        _validate_provenance_against_plane(plane_iscell, entries)
        plane_iscells[source_plane] = plane_iscell.copy()

        if propagate_redcell:
            if combined_redcell is None:
                raise ValueError("combined/redcell.npy is required for redcell propagation.")
            plane_redcells[source_plane] = _load_or_create_plane_redcell(
                plane_folder,
                plane_iscell.shape,
                redcell_missing_policy,
            ).copy()

    if propagate_redcell:
        combined_redcell = _validate_label_array(combined_redcell, name="combined/redcell.npy")
        if combined_redcell.shape != combined_iscell.shape:
            raise ValueError("combined/redcell.npy must match combined/iscell.npy shape.")

    rows_updated = 0
    redcell_rows_updated = 0
    for source_plane, entries in provenance.items():
        for combined_index, source_index in entries:
            plane_iscells[source_plane][source_index, :] = combined_iscell[combined_index, :]
            rows_updated += 1
            if propagate_redcell:
                plane_redcells[source_plane][source_index, :] = combined_redcell[combined_index, :]
                redcell_rows_updated += 1

    for source_plane, plane_iscell in plane_iscells.items():
        np.save(suite2p_root / f"plane{source_plane}" / "iscell.npy", plane_iscell)
    for source_plane, plane_redcell in plane_redcells.items():
        np.save(suite2p_root / f"plane{source_plane}" / "redcell.npy", plane_redcell)

    return {
        "planes_updated": len(plane_iscells),
        "rows_updated": rows_updated,
        "redcell_rows_updated": redcell_rows_updated,
    }


def _combined_output_for_gui(output):
    output = list(output)
    output[1] = {**output[1], **output[2]}
    del output[2]
    return output


def _load_combined_output_to_gui(parent, suite2p_root, output):
    parent.basename = os.path.join(str(suite2p_root), "combined")
    load_to_GUI(parent, parent.basename, _combined_output_for_gui(output))
    parent.loaded = True


def _choose_suite2p_root(parent):
    try:
        return get_suite2p_root_from_basename(getattr(parent, "basename", None))
    except ValueError:
        name = QFileDialog.getExistingDirectory(
            parent=parent,
            caption="Open folder with planeX folders",
        )
        if not name:
            raise ValueError("No Suite2P folder selected.")
        return Path(name)


def regenerate_combined_folder(parent):
    try:
        suite2p_root = _choose_suite2p_root(parent)
        _validate_plane_outputs_for_combined(suite2p_root)
        output = io.combined(str(suite2p_root), save=True)
        _load_combined_output_to_gui(parent, suite2p_root, output)
        QMessageBox.information(
            parent,
            "Regenerate combined folder",
            f"Regenerated combined folder from {len(find_plane_folders(suite2p_root))} plane folders.",
        )
    except Exception as e:
        QMessageBox.critical(parent, "Regenerate combined folder", str(e))


def propagate_combined_labels_to_planes(
        parent,
        *,
        propagate_redcell=False,
        redcell_missing_policy="refuse"):
    try:
        if not is_combined_folder(getattr(parent, "basename", "")):
            QMessageBox.warning(
                parent,
                "Propagate combined labels",
                "Load the combined folder before propagating labels.",
            )
            return
        suite2p_root = get_suite2p_root_from_basename(parent.basename)
        combined_folder = suite2p_root / "combined"
        combined_iscell = _combined_label_array_from_parent(
            parent,
            "iscell",
            "probcell",
            combined_folder / "iscell.npy",
            name="combined/iscell.npy",
        )
        combined_redcell = None
        if propagate_redcell:
            combined_redcell = _combined_label_array_from_parent(
                parent,
                "redcell",
                "probredcell",
                combined_folder / "redcell.npy",
                name="combined/redcell.npy",
            )
        result = propagate_combined_label_arrays_to_planes(
            suite2p_root,
            parent.stat,
            combined_iscell,
            propagate_redcell=propagate_redcell,
            combined_redcell=combined_redcell,
            redcell_missing_policy=redcell_missing_policy,
        )
        output = io.combined(str(suite2p_root), save=True)
        _load_combined_output_to_gui(parent, suite2p_root, output)
        message = (
            f"Propagated {result['rows_updated']} iscell rows across "
            f"{result['planes_updated']} plane folders."
        )
        if propagate_redcell:
            message += f"\nPropagated {result['redcell_rows_updated']} redcell rows."
        QMessageBox.information(parent, "Propagate combined labels", message)
    except ValueError as e:
        QMessageBox.warning(parent, "Propagate combined labels", str(e))
    except Exception as e:
        QMessageBox.critical(parent, "Propagate combined labels", str(e))


def export_fig(parent):
    parent.win.scene().contextMenuItem = parent.p1
    parent.win.scene().showExportDialog()


def make_masks_and_enable_buttons(parent):
    parent.checkBox.setChecked(True)
    parent.ops_plot["color"] = 0
    parent.ops_plot["view"] = 0
    parent.colors["cols"] = 0
    parent.colors["istat"] = 0
    if parent.checkBoxN.isChecked():
        parent.roi_text(False)
    parent.roi_text_labels = []
    parent.roitext = False
    parent.checkBoxN.setChecked(False)
    parent.checkBoxN.setEnabled(True)
    parent.loadBeh.setEnabled(True)
    parent.saveMat.setEnabled(True)
    parent.saveNWB.setEnabled(True)
    parent.saveMerge.setEnabled(True)
    parent.sugMerge.setEnabled(True)
    parent.manual.setEnabled(not is_combined_folder(parent.basename))
    parent.bloaded = False
    parent.ROI_remove()
    parent.isROI = False
    parent.setWindowTitle(parent.fname)
    # set bin size to be 0.5s by default
    parent.bin = int(parent.ops["tau"] * parent.ops["fs"] / 2)
    parent.binedit.setText(str(parent.bin))
    if "chan2_thres" not in parent.ops:
        parent.ops["chan2_thres"] = 0.6
    parent.chan2prob = parent.ops["chan2_thres"]
    parent.chan2edit.setText(str(parent.chan2prob))
    # add boundaries to stat for ROI overlays
    ncells = len(parent.stat)
    for n in range(0, ncells):
        ypix = parent.stat[n]["ypix"].flatten()
        xpix = parent.stat[n]["xpix"].flatten()
        yext, xext = utils.boundary(ypix, xpix)
        parent.stat[n]["yext"] = yext
        parent.stat[n]["xext"] = xext
        ycirc, xcirc = utils.circle(parent.stat[n]["med"], parent.stat[n]["radius"])
        goodi = ((ycirc >= 0) & (xcirc >= 0) & (ycirc < parent.ops["Ly"]) &
                 (xcirc < parent.ops["Lx"]))
        parent.stat[n]["ycirc"] = ycirc[goodi]
        parent.stat[n]["xcirc"] = xcirc[goodi]
        parent.stat[n]["inmerge"] = 0
    # enable buttons
    enable_views_and_classifier(parent)
    # make views
    views.init_views(parent)
    # make color arrays for various views
    masks.make_colors(parent)
    manual_count = sum(bool(stat.get("manual", 0)) for stat in parent.stat)
    parent.manualRoiCheck.setEnabled(manual_count > 0)
    parent.manualRoiCheck.setChecked(manual_count > 0)

    if parent.iscell.sum() > 0:
        icells = np.nonzero(parent.iscell)[0]
        nonmanual_cells = [
            i for i in icells if not bool(parent.stat[i].get("manual", 0))
        ]
        ich = nonmanual_cells[0] if len(nonmanual_cells) > 0 else icells[0]
    else:
        ich = 0
    parent.ichosen = int(ich)
    parent.imerge = [int(ich)]
    parent.iflip = int(ich)
    parent.ichosen_stats()
    parent.comboBox.setCurrentIndex(2)
    # colorbar
    parent.colormat = masks.draw_colorbar()
    masks.plot_colorbar(parent)
    tic = time.time()
    masks.init_masks(parent)
    M = masks.draw_masks(parent)
    masks.plot_masks(parent, M)
    print(f"time to draw and plot masks: {time.time() - tic : .4f} sec")
    parent.lcell1.setText("%d" % (ncells - parent.iscell.sum()))
    parent.lcell0.setText("%d" % (parent.iscell.sum()))
    graphics.init_range(parent)
    traces.plot_trace(parent)
    parent.xyrat = 1.0
    if (isinstance(parent.ops["diameter"], (list, np.ndarray)) and
            len(parent.ops["diameter"]) > 1 and parent.ops.get("aspect", 1.0)):
        parent.xyrat = parent.ops["diameter"][0] / parent.ops["diameter"][1]
    else:
        parent.xyrat = parent.ops.get("aspect", 1.0)

    parent.p1.setAspectLocked(lock=True, ratio=parent.xyrat)
    parent.p2.setAspectLocked(lock=True, ratio=parent.xyrat)
    #parent.p2.setXLink(parent.p1)
    #parent.p2.setYLink(parent.p1)
    parent.loaded = True
    parent.mode_change(2)
    parent.show()
    # no classifier loaded
    classgui.activate(parent, False)


def enable_views_and_classifier(parent):
    for b in range(9):
        parent.quadbtns.button(b).setEnabled(True)
        parent.quadbtns.button(b).setStyleSheet(parent.styleUnpressed)
    for b in range(len(parent.view_names)):
        parent.viewbtns.button(b).setEnabled(True)
        parent.viewbtns.button(b).setStyleSheet(parent.styleUnpressed)
        # parent.viewbtns.button(b).setShortcut(QtGui.QKeySequence("R"))
        if b == 0:
            parent.viewbtns.button(b).setChecked(True)
            parent.viewbtns.button(b).setStyleSheet(parent.stylePressed)
    # check for second channel
    if "meanImg_chan2_corrected" not in parent.ops:
        parent.viewbtns.button(5).setEnabled(False)
        parent.viewbtns.button(5).setStyleSheet(parent.styleInactive)
        if "meanImg_chan2" not in parent.ops:
            parent.viewbtns.button(6).setEnabled(False)
            parent.viewbtns.button(6).setStyleSheet(parent.styleInactive)

    for b in range(len(parent.color_names)):
        if b == 5:
            if parent.hasred:
                parent.colorbtns.button(b).setEnabled(True)
                parent.colorbtns.button(b).setStyleSheet(parent.styleUnpressed)
        elif b == 0:
            parent.colorbtns.button(b).setEnabled(True)
            parent.colorbtns.button(b).setChecked(True)
            parent.colorbtns.button(b).setStyleSheet(parent.stylePressed)
        elif b < 8:
            parent.colorbtns.button(b).setEnabled(True)
            parent.colorbtns.button(b).setStyleSheet(parent.styleUnpressed)

    #parent.applyclass.setStyleSheet(parent.styleUnpressed)
    #parent.applyclass.setEnabled(True)
    b = 0
    for btn in parent.sizebtns.buttons():
        btn.setStyleSheet(parent.styleUnpressed)
        btn.setEnabled(True)
        if b == 0:
            btn.setChecked(True)
            btn.setStyleSheet(parent.stylePressed)
            btn.press(parent)
        b += 1
    for b in range(3):
        if b == 0:
            parent.topbtns.button(b).setEnabled(True)
            parent.topbtns.button(b).setStyleSheet(parent.styleUnpressed)
        else:
            parent.topbtns.button(b).setEnabled(False)
            parent.topbtns.button(b).setStyleSheet(parent.styleInactive)
    # enable classifier menu
    parent.loadClass.setEnabled(True)
    parent.loadTrain.setEnabled(True)
    parent.loadUClass.setEnabled(True)
    parent.loadSClass.setEnabled(True)
    parent.resetDefault.setEnabled(True)
    parent.visualizations.setEnabled(True)
    parent.custommask.setEnabled(True)
    # parent.p1.scene().showExportDialog()


def load_dialog(parent):
    dlg_kwargs = {
        "parent": parent,
        "caption": "Open stat.npy",
        "filter": "stat.npy",
    }
    name = QFileDialog.getOpenFileName(**dlg_kwargs)
    parent.fname = name[0]
    load_proc(parent)

def load_dialog_NWB(parent):
    dlg_kwargs = {
        "parent": parent,
        "caption": "Open ophys.nwb",
        "filter": "*.nwb",
    }
    name = QFileDialog.getOpenFileName(**dlg_kwargs)
    parent.fname = name[0]
    load_NWB(parent)

def load_dialog_folder(parent):
    dlg_kwargs = {
        "parent": parent,
        "caption": "Open folder with planeX folders",
    }    
    name = QFileDialog.getExistingDirectory(**dlg_kwargs)
    parent.fname = name
    load_folder(parent)

def load_NWB(parent):
    name = parent.fname
    print(name)
    try:
        procs = list(io.read_nwb(name))
        if procs[1]["nchannels"] == 2:
            hasred = True
        else:
            hasred = False
        procs.append(hasred)
        load_to_GUI(parent, os.path.split(name)[0], procs)

        parent.loaded = True
    except Exception as e:
        print("ERROR with NWB: %s" % e)


def load_folder(parent):
    print(parent.fname)
    save_folder = parent.fname
    plane_folders = [
        f.path for f in os.scandir(save_folder) if f.is_dir() and f.name[:5] == "plane"
    ]
    stat_found = False
    if len(plane_folders) > 0:
        stat_found = all(
            [os.path.isfile(os.path.join(f, "stat.npy")) for f in plane_folders])
    if not stat_found:
        print("No processed planeX folders in folder")
        return

    # create a combined folder to hold iscell and redcell
    output = io.combined(save_folder, save=False)
    output = list(output)
    output[1] = {**output[1], **output[2]}  # combine db and settings
    del output[2]
    parent.basename = os.path.join(parent.fname, "combined")
    load_to_GUI(parent, parent.basename, output)
    parent.loaded = True
    print(parent.fname)


def load_files(name):
    """ give stat.npy path and load all needed files for suite2p """
    try:
        stat = np.load(name, allow_pickle=True)
        ypix = stat[0]["ypix"]
    except (ValueError, KeyError, OSError, RuntimeError, TypeError, NameError):
        print("ERROR: this is not a stat.npy file :( "
              "(needs stat[n]['ypix']!)")
        stat = None
    goodfolder = False
    if stat is not None:
        basename, fname = os.path.split(name)
        goodfolder = True
        try:
            Fcell = np.load(basename + "/F.npy")
            Fneu = np.load(basename + "/Fneu.npy")
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("ERROR: there are no fluorescence traces in this folder "
                  "(F.npy/Fneu.npy)")
            goodfolder = False
        try:
            Spks = np.load(basename + "/spks.npy")
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("there are no spike deconvolved traces in this folder "
                  "(spks.npy)")
            goodfolder = False
        noops = True
        try:
            ops = np.load(os.path.join(basename, "ops.npy"), allow_pickle=True).item()
            noops = False
        except:
            noops = True
        if noops:
            try:
                settings = np.load(basename + "/settings.npy", allow_pickle=True).item()
                db = np.load(basename + "/db.npy", allow_pickle=True).item()
                try:
                    reg_outputs = np.load(basename + "/reg_outputs.npy", allow_pickle=True).item()
                    detect_outputs = np.load(basename + "/detect_outputs.npy", allow_pickle=True).item()
                    ops = {**db, **settings, **reg_outputs, **detect_outputs}
                except:
                    ops = {**db, **settings}
                    print("no reg_outputs.npy or detect_outputs.npy found")
            except (ValueError, OSError, RuntimeError, TypeError, NameError):
                if noops:
                    print("ERROR: there is no settings or db file in this folder (settings.npy / db.npy)")
                    goodfolder = False
        try:
            iscell = np.load(basename + "/iscell.npy")
            probcell = iscell[:, 1]
            iscell = iscell[:, 0].astype("bool")
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("no manual labels found (iscell.npy)")
            if goodfolder:
                NN = Fcell.shape[0]
                iscell = np.ones((NN,), "bool")
                probcell = np.ones((NN,), np.float32)
        try:
            redcell = np.load(basename + "/redcell.npy")
            NN = Fcell.shape[0]
            redcell, probredcell, hasred = _coerce_optional_redcell_for_gui(
                redcell, NN, stat=stat)
        except (ValueError, OSError, RuntimeError, TypeError, NameError):
            print("no channel 2 labels found (redcell.npy)")
            hasred = False
            if goodfolder:
                NN = Fcell.shape[0]
                redcell, probredcell = _zero_label_vectors(NN)
    else:
        print("incorrect file, not a stat.npy")
        return None

    if goodfolder:
        return stat, ops, Fcell, Fneu, Spks, iscell, probcell, redcell, probredcell, hasred
    else:
        print("stat.npy found, but other files not in folder")
        return None


def load_proc(parent):
    name = parent.fname
    print(name)
    basename, fname = os.path.split(name)
    if fname == "stat.npy" and is_combined_folder(basename):
        try:
            stat = np.load(name, allow_pickle=True)
            if not _stat_has_combined_provenance(stat):
                suite2p_root = get_suite2p_root_from_basename(basename)
                _validate_plane_outputs_for_combined(suite2p_root)
                output = io.combined(str(suite2p_root), save=True)
                _load_combined_output_to_gui(parent, suite2p_root, output)
                parent.loaded = True
                QMessageBox.information(
                    parent,
                    "Regenerate combined folder",
                    "Regenerated combined folder before loading because provenance was missing.",
                )
                return
        except Exception as e:
            QMessageBox.critical(parent, "Load combined folder", str(e))
            return
    output = load_files(name)
    if output is not None:
        load_to_GUI(parent, basename, output)
        parent.loaded = True
    else:
        Text = "Incorrect files, choose another?"
        load_again(parent, Text)


def load_to_GUI(parent, basename, procs):
    stat, ops, Fcell, Fneu, Spks, iscell, probcell, redcell, probredcell, hasred = procs
    parent.basename = basename
    parent.stat = stat
    parent.ops = ops
    parent.Fcell = Fcell
    parent.Fneu = Fneu
    parent.Spks = Spks
    # Handle both 1D and 2D iscell formats
    if iscell.ndim == 2:
        parent.iscell = iscell[:, 0].astype("bool")
        parent.probcell = iscell[:, 1]
    else:
        parent.iscell = iscell.astype("bool")
        parent.probcell = probcell
    # Handle both 1D and 2D redcell formats
    if redcell.ndim == 2:
        try:
            parent.redcell, parent.probredcell, hasred = _coerce_optional_redcell_for_gui(
                redcell, parent.iscell.shape[0], stat=parent.stat)
        except ValueError:
            print("redcell labels do not match ROI count; ignoring redcell labels for this load")
            parent.redcell, parent.probredcell = _zero_label_vectors(parent.iscell.shape[0])
            hasred = False
    else:
        parent.redcell = redcell.astype("bool")
        parent.probredcell = probredcell
    parent.hasred = hasred
    if (parent.redcell.shape[0] != parent.iscell.shape[0] or
            parent.probredcell.shape[0] != parent.iscell.shape[0]):
        print("redcell labels do not match ROI count; ignoring redcell labels for this load")
        parent.redcell, parent.probredcell = _zero_label_vectors(parent.iscell.shape[0])
        parent.hasred = False
    parent.notmerged = np.ones_like(parent.iscell).astype("bool")
    for n in range(len(parent.stat)):
        if parent.hasred:
            parent.stat[n]["chan2_prob"] = parent.probredcell[n]
        _ensure_combined_stat_optional_fields(parent.stat[n])
        parent.stat[n]["inmerge"] = 0
    parent.stat = np.array(parent.stat)
    make_masks_and_enable_buttons(parent)
    parent.ichosen = 0
    parent.imerge = [0]
    for n in range(len(parent.stat)):
        if "imerge" not in parent.stat[n]:
            parent.stat[n]["imerge"] = []


def load_behavior(parent):
    name = QFileDialog.getOpenFileName(parent, "Open *.npy", filter="*.npy")
    name = name[0]
    bloaded = False
    try:
        beh = np.load(name)
        bresample = False
        if beh.ndim > 1:
            if beh.shape[1] < 2:
                beh = beh.flatten()
                if beh.shape[0] == parent.Fcell.shape[1]:
                    parent.bloaded = True
                    beh_time = np.arange(0, parent.Fcell.shape[1])
            else:
                parent.bloaded = True
                beh_time = beh[:, 1]
                beh = beh[:, 0]
                bresample = True
        else:
            if beh.shape[0] == parent.Fcell.shape[1]:
                parent.bloaded = True
                beh_time = np.arange(0, parent.Fcell.shape[1])
    except (ValueError, KeyError, OSError, RuntimeError, TypeError, NameError):
        print("ERROR: this is not a 1D array with length of data")
    if parent.bloaded:
        beh -= beh.min()
        beh /= beh.max()
        parent.beh = beh
        parent.beh_time = beh_time
        if bresample:
            parent.beh_resampled = resample_frames(parent.beh, parent.beh_time,
                                                   np.arange(0, parent.Fcell.shape[1]))
        else:
            parent.beh_resampled = parent.beh
        b = 8
        parent.colorbtns.button(b).setEnabled(True)
        parent.colorbtns.button(b).setStyleSheet(parent.styleUnpressed)
        masks.beh_masks(parent)
        traces.plot_trace(parent)
        if hasattr(parent, "VW"):
            parent.VW.bloaded = parent.bloaded
            parent.VW.beh = parent.beh
            parent.VW.beh_time = parent.beh_time
            parent.VW.plot_traces()
        parent.show()
    else:
        print("ERROR: this is not a 1D array with length of data")


def resample_frames(y, x, xt):
    """ resample y (defined at x) at times xt """
    ts = x.size / xt.size
    y = gaussian_filter1d(y, np.ceil(ts / 2), axis=0)
    f = interp1d(x, y, fill_value="extrapolate")
    yt = f(xt)
    return yt


def save_redcell(parent):
    if not getattr(parent, "hasred", False):
        print("redcell labels unavailable; not saving redcell.npy")
        return
    if (parent.redcell.shape[0] != parent.notmerged.shape[0] or
            parent.probredcell.shape[0] != parent.notmerged.shape[0]):
        print("redcell labels do not match ROI count; not saving redcell.npy")
        return
    np.save(
        os.path.join(parent.basename, "redcell.npy"),
        np.concatenate((np.expand_dims(parent.redcell[parent.notmerged], axis=1),
                        np.expand_dims(parent.probredcell[parent.notmerged], axis=1)),
                       axis=1))


def save_iscell(parent):
    np.save(
        parent.basename + "/iscell.npy",
        np.concatenate(
            (
                np.expand_dims(parent.iscell[parent.notmerged], axis=1),
                np.expand_dims(parent.probcell[parent.notmerged], axis=1),
            ),
            axis=1,
        ),
    )
    parent.lcell0.setText("%d" % (parent.iscell.sum()))
    parent.lcell1.setText("%d" % (parent.iscell.size - parent.iscell.sum()))


def save_mat(parent):
    print("saving to mat")
    matpath = os.path.join(parent.basename, "Fall.mat")
    if "date_proc" in parent.ops:
        parent.ops["date_proc"] = []
    scipy.io.savemat(
        matpath, {
            "stat":
                parent.stat,
            "settings":
                parent.ops,
            "F":
                parent.Fcell,
            "Fneu":
                parent.Fneu,
            "spks":
                parent.Spks,
            "iscell":
                np.concatenate(
                    (parent.iscell[:, np.newaxis], parent.probcell[:, np.newaxis]),
                    axis=1),
            "redcell":
                np.concatenate((np.expand_dims(parent.redcell, axis=1),
                                np.expand_dims(parent.probredcell, axis=1)), axis=1)
        })


def save_merge(parent):
    print("saving to NPY")
    np.save(os.path.join(parent.basename, "settings.npy"), parent.ops)
    np.save(os.path.join(parent.basename, "stat.npy"), parent.stat)
    np.save(os.path.join(parent.basename, "F.npy"), parent.Fcell)
    np.save(os.path.join(parent.basename, "Fneu.npy"), parent.Fneu)
    if parent.hasred:
        np.save(os.path.join(parent.basename, "F_chan2.npy"), parent.F_chan2)
        np.save(os.path.join(parent.basename, "Fneu_chan2.npy"), parent.Fneu_chan2)
        np.save(
            os.path.join(parent.basename, "redcell.npy"),
            np.concatenate((np.expand_dims(
                parent.redcell, axis=1), np.expand_dims(parent.probredcell, axis=1)),
                           axis=1))
    np.save(os.path.join(parent.basename, "spks.npy"), parent.Spks)
    iscell = np.concatenate(
        (parent.iscell[:, np.newaxis], parent.probcell[:, np.newaxis]), axis=1)
    np.save(os.path.join(parent.basename, "iscell.npy"), iscell)

    parent.notmerged = np.ones(parent.iscell.size, "bool")


def load_custom_mask(parent):
    name = QFileDialog.getOpenFileName(parent, "Open *.npy", filter="*.npy")
    name = name[0]
    cloaded = False
    try:
        mask = np.load(name)
        mask = mask.flatten()
        if mask.size == parent.Fcell.shape[0]:
            b = len(parent.color_names) - 1
            parent.colorbtns.button(b).setEnabled(True)
            parent.colorbtns.button(b).setStyleSheet(parent.styleUnpressed)
            cloaded = True
    except (ValueError, KeyError, OSError, RuntimeError, TypeError, NameError):
        print("ERROR: this is not a 1D array with length of data")
    if cloaded:
        parent.custom_mask = mask
        masks.custom_masks(parent)
        M = masks.draw_masks(parent)
        b = len(parent.colors) + 1
        parent.colorbtns.button(b).setEnabled(True)
        parent.colorbtns.button(b).setStyleSheet(parent.styleUnpressed)
        parent.colorbtns.button(b).setChecked(True)
        parent.colorbtns.button(b).press(parent, b)
        parent.show()
    else:
        print("ERROR: this is not a 1D array with length of # of ROIs")


def load_again(parent, Text):
    tryagain = QMessageBox.question(parent, "ERROR", Text,
                                    QMessageBox.Yes | QMessageBox.No)

    if tryagain == QMessageBox.Yes:
        load_dialog(parent)
