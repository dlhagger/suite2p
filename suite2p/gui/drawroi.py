"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import os
import time
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyqtgraph as pg
from qtpy import QtGui, QtCore
from qtpy.QtWidgets import (
    QPushButton,
    QLabel,
    QLineEdit,
    QMainWindow,
    QGridLayout,
    QButtonGroup,
    QMessageBox,
    QWidget,
    QVBoxLayout,
    QGroupBox,
)
from matplotlib.colors import hsv_to_rgb
from matplotlib.path import Path
from scipy import stats
from scipy.ndimage import rotate

from . import io
from ..extraction import masks
from ..detection.stats import roi_stats
from ..extraction import preprocess
from ..extraction.dcnv import oasis
from ..extraction.extract import extract_traces
from ..io.binary import BinaryFile
from ..manual_rois import (
    build_prompt_image,
    clean_mask,
    default_sam3_checkpoint_path,
    mask_to_stat,
    normalize_image,
    stat_to_mask,
    UltralyticsSam3MaskProvider,
)
from ..parameters import default_settings
from ..run_s2p import _assign_torch_device


def masks_and_traces(settings, stat_manual, stat_orig):
    """ main extraction function
        inputs: settings and stat
        creates cell and neuropil masks and extracts traces
        returns: F (ROIs x time), Fneu (ROIs x time), F_chan2, Fneu_chan2, settings, stat
        F_chan2 and Fneu_chan2 will be empty if no second channel
    """
    # Merge with defaults to ensure all required keys are present
    settings = {**default_settings(), **settings}

    t0 = time.time()
    
    # Concatenate stat so a good neuropil function can be formed
    stat_all = stat_manual.copy()
    for n in range(len(stat_orig)):
        stat_all.append(stat_orig[n])

    stat_all = np.array(stat_all)
    stat_all = roi_stats(stat_all, settings["Ly"], settings["Lx"],
                         diameter=settings["diameter"])
    cell_masks = [
        masks.create_cell_mask(stat, Ly=settings["Ly"], Lx=settings["Lx"],
                               allow_overlap=settings["extraction"]["allow_overlap"]) for stat in stat_all
    ]
    manual_roi_stats = stat_all[:len(stat_manual)]
    manual_cell_masks = cell_masks[:len(stat_manual)]
    if settings["extraction"]["neuropil_extract"]:
        cell_pix = masks.create_cell_pix(
            stat_all,
            Ly=settings["Ly"],
            Lx=settings["Lx"],
            lam_percentile=settings["extraction"]["lam_percentile"],
        )
        manual_neuropil_masks = masks.create_neuropil_masks(
            ypixs=[stat["ypix"] for stat in manual_roi_stats],
            xpixs=[stat["xpix"] for stat in manual_roi_stats],
            cell_pix=cell_pix,
            inner_neuropil_radius=settings["extraction"]["inner_neuropil_radius"],
            min_neuropil_pixels=settings["extraction"]["min_neuropil_pixels"],
            circular=settings["extraction"]["circular_neuropil"],
        )
    else:
        manual_neuropil_masks = None
    print("Masks made in %0.2f sec." % (time.time() - t0))

    # Extract traces from binary file
    Ly, Lx = settings["Ly"], settings["Lx"]
    batch_size = settings["extraction"]["batch_size"]
    device = _assign_torch_device(settings["torch_device"])
    f_reg = BinaryFile(Ly, Lx, settings["reg_file"])
    F, Fneu = extract_traces(f_reg, manual_cell_masks, manual_neuropil_masks, batch_size=batch_size, device=device)
    f_reg.close()

    # Handle chan2 if present
    if "reg_file_chan2" in settings and settings["reg_file_chan2"]:
        f_reg_chan2 = BinaryFile(Ly, Lx, settings["reg_file_chan2"])
        F_chan2, Fneu_chan2 = extract_traces(f_reg_chan2, manual_cell_masks, manual_neuropil_masks, batch_size=batch_size, device=device)
        f_reg_chan2.close()
    else:
        F_chan2, Fneu_chan2 = None, None

    # compute activity statistics for classifier
    npix = np.array(
        [stat_orig[n]["npix"] for n in range(len(stat_orig))], dtype="float32"
    )
    if npix.size:
        npix_reference = float(np.mean(npix[:100]))
    else:
        npix_reference = float(
            np.mean([stat["npix"] for stat in manual_roi_stats[:100]])
        )
    if not np.isfinite(npix_reference) or npix_reference <= 0:
        npix_reference = 1.0
    for n in range(len(manual_roi_stats)):
        manual_roi_stats[n]["npix_norm"] = (
            manual_roi_stats[n]["npix"] / npix_reference
        )
        manual_roi_stats[n]["compact"] = 1
        manual_roi_stats[n]["footprint"] = 2
        manual_roi_stats[n]["manual"] = 1  # Add manual key
        if len(stat_orig) and "iplane" in stat_orig[0]:
            manual_roi_stats[n]["iplane"] = stat_orig[0]["iplane"]

    # subtract neuropil and compute activity statistics
    dF = F - settings["extraction"]["neuropil_coefficient"] * Fneu
    sk = stats.skew(dF, axis=1)
    sd = np.std(dF, axis=1)
    if dF.shape[1] > 1:
        diff_var = np.diff(dF, axis=1).var(axis=1)
    else:
        diff_var = np.zeros(dF.shape[0], dtype=dF.dtype)
    trace_var = dF.var(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 1 - 0.5 * np.divide(
            diff_var,
            trace_var,
            out=np.zeros_like(diff_var),
            where=trace_var > 0,
        )
    snr = np.nan_to_num(snr, nan=0.0, posinf=0.0, neginf=0.0)

    for n in range(F.shape[0]):
        manual_roi_stats[n]["snr"] = snr[n]
        manual_roi_stats[n]["skew"] = sk[n]
        manual_roi_stats[n]["std"] = sd[n]
        manual_roi_stats[n]["med"] = [
            np.mean(manual_roi_stats[n]["ypix"]),
            np.mean(manual_roi_stats[n]["xpix"])
        ]

    dF = preprocess(F=dF, baseline=settings["dcnv_preprocess"]["baseline"], win_baseline=settings["dcnv_preprocess"]["win_baseline"],
                    sig_baseline=settings["dcnv_preprocess"]["sig_baseline"], fs=settings["fs"],
                    prctile_baseline=settings["dcnv_preprocess"]["prctile_baseline"], device=device)
    spks = oasis(F=dF, batch_size=settings["extraction"]["batch_size"], tau=settings["tau"], fs=settings["fs"])

    return F, Fneu, F_chan2, Fneu_chan2, spks, settings, manual_roi_stats


class ViewButton(QPushButton):

    def __init__(self, bid, Text, parent=None):
        super(ViewButton, self).__init__(parent)
        self.setText(Text)
        self.setCheckable(True)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()

    def press(self, parent, bid):
        for b in range(len(parent.views)):
            parent.viewbtns.button(b).setStyleSheet(parent.styleUnpressed)
        parent.viewbtns.button(bid).setStyleSheet(parent.stylePressed)
        parent.img0.setImage(parent.masked_images[:, :, :, bid])
        if hasattr(parent, "sam_provider"):
            parent.reset_sam_provider()
            if getattr(parent, "sam_points", []):
                parent.clear_sam_proposal()
                parent.samStatus.setText(
                    "SAM3 image source changed; prompts cleared"
                )
        parent.win.show()
        parent.show()


class Sam3AsyncSignals(QtCore.QObject):
    """Thread-safe bridge from background SAM3 work back to the Qt GUI."""

    loaded = QtCore.Signal(object, str, int)
    proposalReady = QtCore.Signal(object, int)
    failed = QtCore.Signal(str, str, int)


class ROIDraw(QMainWindow):

    def __init__(self, parent):
        super(ROIDraw, self).__init__(parent)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.parent = parent
        self.setGeometry(70, 70, 1400, 800)
        self.setWindowTitle("extract ROI activity")
        self.cwidget = QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QGridLayout()
        # layout = QtGui.QFormLayout()
        self.cwidget.setLayout(self.l0)
        self.stylePressed = ("QPushButton {Text-align: left; "
                             "background-color: rgb(100,100,100); "
                             "color:white;}")
        self.styleUnpressed = ("QPushButton {Text-align: left; "
                               "background-color: rgb(50,50,50); "
                               "color:white;}")
        self.groupStyle = (
            "QGroupBox {color: white; font-weight: bold; border: 1px solid "
            "rgb(85,85,85); border-radius: 4px; margin-top: 8px; padding: 6px;}"
            "QGroupBox::title {subcontrol-origin: margin; left: 8px; padding: 0 4px;}"
            "QLabel {color: white;}"
        )

        # self.p0 = pg.ViewBox(lockAspect=False,name="plot1",border=[100,100,100],invertY=True)
        self.win = pg.GraphicsLayoutWidget()
        # --- cells image
        self.win = pg.GraphicsLayoutWidget()
        self.l0.addWidget(self.win, 4, 0, 13, 14)
        layout = self.win.ci.layout
        # A plot area (ViewBox + axes) for displaying the image
        self.p1 = self.win.addPlot(row=0, col=1)
        self.p1.setMouseEnabled(x=True, y=False)
        self.p1.setMenuEnabled(False)
        self.p1.scene().sigMouseMoved.connect(self.mouse_moved)

        self.p0 = self.win.addViewBox(name="plot1", lockAspect=True, row=0, col=0,
                                      invertY=True)
        self.img0 = pg.ImageItem()
        self.p0.addItem(self.img0)

        self.win.scene().sigMouseClicked.connect(self.plot_clicked)

        self.ROIs = []
        self.cell_pos = []
        self.nROIs = 0
        self.extracted = False
        self.saveGUI = False
        self.saved_manual_rois = False

        manual_box = QGroupBox("Manual ROI tools")
        manual_box.setStyleSheet(self.groupStyle)
        manual_layout = QGridLayout()
        manual_layout.setContentsMargins(8, 12, 8, 8)
        manual_layout.setHorizontalSpacing(6)
        manual_layout.setVerticalSpacing(4)
        manual_box.setLayout(manual_layout)

        self.instructions = QLabel(
            "Draw masks now; use SAM3 later for irregular point-prompt masks."
        )
        self.instructions.setWordWrap(True)
        manual_layout.addWidget(self.instructions, 0, 0, 1, 5)

        shortcut = QLabel("Shortcuts: Alt+click ellipse • D remove selected • W/E/R/T switch image")
        shortcut.setWordWrap(True)
        shortcut.setStyleSheet("color: rgb(210,210,210);")
        manual_layout.addWidget(shortcut, 1, 0, 1, 5)

        self.addROI = QPushButton("add ROI")
        self.addROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.addROI.clicked.connect(lambda: self.add_ROI(pos=None))
        self.addROI.setEnabled(True)
        self.addROI.setStyleSheet(self.styleUnpressed)
        manual_layout.addWidget(self.addROI, 2, 0, 1, 1)

        self.addPolygon = QPushButton("add polygon")
        self.addPolygon.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.addPolygon.clicked.connect(lambda: self.add_polygon(pos=None))
        self.addPolygon.setStyleSheet(self.styleUnpressed)
        manual_layout.addWidget(self.addPolygon, 2, 1, 1, 1)

        self.removeROI = QPushButton("remove selected")
        self.removeROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.removeROI.clicked.connect(self.remove_current_roi)
        self.removeROI.setEnabled(False)
        self.removeROI.setStyleSheet(self.styleUnpressed)
        manual_layout.addWidget(self.removeROI, 2, 2, 1, 1)

        lbl = QLabel("diameter:")
        lbl.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        manual_layout.addWidget(lbl, 2, 3, 1, 1)
        self.diam = QLineEdit(self)
        self.diam.setValidator(QtGui.QIntValidator(0, 10000))
        self.diam.setText("12")
        self.diam.setFixedWidth(35)
        self.diam.setAlignment(QtCore.Qt.AlignRight)
        manual_layout.addWidget(self.diam, 2, 4, 1, 1)

        self.roiStatus = QLabel("")
        self.roiStatus.setStyleSheet("color: rgb(180, 220, 255);")
        manual_layout.addWidget(self.roiStatus, 3, 0, 1, 5)
        self.l0.addWidget(manual_box, 0, 0, 4, 4)

        self.sam_provider = None
        self.sam_points = []
        self.sam_labels = []
        self.sam_proposal = None
        self.sam_loading = False
        self.sam_busy = False
        self.sam_pending_proposal = False
        self.sam_generation = 0
        self.sam_prompt_revision = 0
        self.sam_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="suite2p-sam3"
        )
        self.sam_signals = Sam3AsyncSignals(self)
        self.sam_signals.loaded.connect(self._sam_loaded)
        self.sam_signals.proposalReady.connect(self._sam_proposal_ready)
        self.sam_signals.failed.connect(self._sam_failed)
        self.sam_overlay = pg.ImageItem()
        self.sam_overlay.setZValue(20)
        self.sam_prompt_scatter = pg.ScatterPlotItem()
        self.sam_prompt_scatter.setZValue(21)
        self.p0.addItem(self.sam_overlay)
        self.p0.addItem(self.sam_prompt_scatter)

        self.samMode = QPushButton("SAM3 points")
        self.samMode.setCheckable(True)
        self.samMode.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.samMode.toggled.connect(self.sam_mode_changed)
        self.samMode.setStyleSheet(self.styleUnpressed)
        self.acceptSAM = QPushButton("accept SAM")
        self.acceptSAM.clicked.connect(self.accept_sam_proposal)
        self.acceptSAM.setEnabled(False)
        self.acceptSAM.setStyleSheet(self.styleUnpressed)
        self.undoSAM = QPushButton("undo point")
        self.undoSAM.clicked.connect(self.undo_sam_point)
        self.undoSAM.setEnabled(False)
        self.undoSAM.setStyleSheet(self.styleUnpressed)
        self.clearSAM = QPushButton("clear SAM")
        self.clearSAM.clicked.connect(self.clear_sam_proposal)
        self.clearSAM.setStyleSheet(self.styleUnpressed)

        sam_box = QGroupBox("SAM3 mask assist")
        sam_box.setStyleSheet(self.groupStyle)
        sam_layout = QGridLayout()
        sam_layout.setContentsMargins(8, 12, 8, 8)
        sam_layout.setHorizontalSpacing(6)
        sam_layout.setVerticalSpacing(4)
        sam_box.setLayout(sam_layout)
        self.samStatus = QLabel(
            "SAM3 not enabled yet. Manual ellipse/polygon tools are ready."
        )
        self.samStatus.setWordWrap(True)
        sam_layout.addWidget(self.samStatus, 0, 0, 1, 4)
        sam_layout.addWidget(self.samMode, 1, 0, 1, 1)
        sam_layout.addWidget(self.undoSAM, 1, 1, 1, 1)
        sam_layout.addWidget(self.acceptSAM, 1, 2, 1, 1)
        sam_layout.addWidget(self.clearSAM, 1, 3, 1, 1)
        self.l0.addWidget(sam_box, 0, 7, 4, 4)

        action_box = QGroupBox("Trace extraction")
        action_box.setStyleSheet(self.groupStyle)
        action_layout = QGridLayout()
        action_layout.setContentsMargins(8, 12, 8, 8)
        action_layout.setHorizontalSpacing(6)
        action_layout.setVerticalSpacing(4)
        action_box.setLayout(action_layout)
        self.procROI = QPushButton("extract ROIs")
        self.procROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.procROI.setStyleSheet(self.styleUnpressed)
        self.procROI.setCheckable(False)
        self.procROI.clicked.connect(self.proc_ROI)
        action_layout.addWidget(self.procROI, 0, 0, 1, 1)

        # CloseGUI button - once done save the ROIs to stat.npy
        self.closeGUI = QPushButton("Save and Quit")
        self.closeGUI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.closeGUI.clicked.connect(lambda: self.close_GUI())
        self.closeGUI.setEnabled(False)
        self.closeGUI.setStyleSheet(self.styleUnpressed)
        action_layout.addWidget(self.closeGUI, 1, 0, 1, 1)
        action_hint = QLabel("Extract traces before saving manual masks.")
        action_hint.setWordWrap(True)
        action_hint.setStyleSheet("color: rgb(210,210,210);")
        action_layout.addWidget(action_hint, 2, 0, 1, 1)
        self.l0.addWidget(action_box, 0, 11, 4, 3)

        # view buttons
        self.views = [
            "W: mean img", "E: mean img (enhanced)", "R: correlation map",
            "T: max projection"
        ]
        self.has_chan2 = "meanImg_chan2" in parent.ops
        if self.has_chan2:
            self.views.append("Y: mean img chan2")
        b = 0
        self.viewbtns = QButtonGroup(self)
        view_container = QGroupBox("Image source")
        view_container.setStyleSheet(self.groupStyle)
        view_vbox = QVBoxLayout()
        view_vbox.setContentsMargins(8, 12, 8, 8)
        view_vbox.setSpacing(4)
        view_container.setLayout(view_vbox)
        for names in self.views:
            btn = ViewButton(b, "&" + names, self)
            self.viewbtns.addButton(btn, b)
            view_vbox.addWidget(btn)
            btn.setEnabled(True)
            b += 1
        view_vbox.addStretch()
        self.l0.addWidget(view_container, 0, 4, 4, 3)
        for b in range(len(self.views)):
            self.viewbtns.button(b).setStyleSheet(self.styleUnpressed)
        self.viewbtns.button(0).setChecked(True)
        self.viewbtns.button(0).setStyleSheet(self.stylePressed)
        self.configure_sam_controls()
        self.update_roi_status()

        self.l0.addWidget(QLabel("neuropil"), 13, 13, 1, 1)

        self.Ly = self.parent.ops["Ly"]
        self.Lx = self.parent.ops["Lx"]
        self.iscell = self.parent.iscell

        # Get maskf for pixels that are cells
        self.masked_images = self.normalize_img_add_masks()
        # Mean image as default
        self.img0.setImage(self.masked_images[:, :, :, 0])

    def sam_setting(self, key, default=None):
        run_settings = self.parent.ops.get("run", {})
        if isinstance(run_settings, dict) and key in run_settings:
            return run_settings[key]
        return self.parent.ops.get(key, default)

    def sam_checkpoint_path(self):
        return default_sam3_checkpoint_path()

    def sam_configured(self):
        """Return True when this Suite2p session has SAM3 enabled."""
        return bool(self.sam_setting("sam3_enabled", False))

    def sam_backend_available(self):
        """Return True when SAM3 is enabled and importable."""
        if not self.sam_configured():
            return False
        if not UltralyticsSam3MaskProvider.installed():
            return False
        checkpoint = self.sam_checkpoint_path()
        if not checkpoint or not os.path.isfile(os.path.expanduser(checkpoint)):
            return False
        return True

    def sam_unavailable_reason(self):
        if not self.sam_configured():
            return (
                "SAM3 disabled for this session. Enable it in Run Suite2p > "
                "ROI source: manual > Enable SAM3 assist."
            )
        if not UltralyticsSam3MaskProvider.installed():
            return (
                "SAM3 enabled, but the optional 'ultralytics' package is not "
                "installed in this environment."
            )
        checkpoint = self.sam_checkpoint_path()
        if not checkpoint:
            return (
                "SAM3 enabled, but no checkpoint was found. Put sam3.pt in "
                "models/sam3.pt."
            )
        if not os.path.isfile(os.path.expanduser(checkpoint)):
            return f"SAM3 checkpoint was not found at {checkpoint}."
        return ""

    def configure_sam_controls(self):
        available = self.sam_backend_available()
        self.samMode.setEnabled(available)
        self.clearSAM.setEnabled(available)
        self.undoSAM.setEnabled(False)
        self.acceptSAM.setEnabled(False)
        if available:
            checkpoint = self.sam_checkpoint_path()
            checkpoint_note = f" ({checkpoint})" if checkpoint else ""
            self.samStatus.setText(
                f"SAM3 via Ultralytics detected{checkpoint_note}. Enable "
                "points, then left-click includes and Shift+left-click excludes."
            )
        else:
            self.samMode.setChecked(False)
            self.samStatus.setText(
                f"{self.sam_unavailable_reason()} Manual ellipse/polygon tools are ready."
            )

    def current_view_index(self):
        """Return the currently selected manual-labelling image view."""
        try:
            bid = int(self.viewbtns.checkedId())
        except Exception:
            bid = -1
        if bid >= 0:
            return bid
        for bid in range(len(self.views)):
            button = self.viewbtns.button(bid)
            if button is not None and button.isChecked():
                return bid
        return 0

    def current_view_name(self):
        return self.views[self.current_view_index()].split(":", 1)[-1].strip()

    def _raw_view_image(self, bid):
        if bid == 0:
            return self.parent.ops.get("meanImg")
        if bid == 1:
            return self.parent.ops.get("meanImgE")
        if bid == 2:
            return self.parent.ops.get("Vcorr")
        if bid == 3:
            return self.parent.ops.get("max_proj")
        if bid == 4 and self.has_chan2:
            return self.parent.ops.get("meanImg_chan2")
        return None

    def sam_prompt_image(self):
        """Build a SAM prompt image from the currently visible image source."""
        bid = self.current_view_index()
        src = self._raw_view_image(bid)
        if src is None:
            return build_prompt_image(self.parent.ops), "composite image"

        shape = (int(self.parent.ops["Ly"]), int(self.parent.ops["Lx"]))
        yrange = self.parent.ops.get("yrange", [0, shape[0]])
        xrange = self.parent.ops.get("xrange", [0, shape[1]])
        src = np.asarray(src)
        image = np.zeros(shape, dtype=np.float32)
        if src.shape == shape:
            image = src
        else:
            expected = (
                int(yrange[1] - yrange[0]),
                int(xrange[1] - xrange[0]),
            )
            if src.shape == expected:
                image[slice(*yrange), slice(*xrange)] = src
            else:
                return build_prompt_image(self.parent.ops), "composite image"

        image = normalize_image(image)
        if not np.any(image):
            return build_prompt_image(self.parent.ops), "composite image"
        rgb = np.round(255 * np.repeat(image[:, :, np.newaxis], 3, axis=2))
        return rgb.astype(np.uint8), self.current_view_name()

    def reset_sam_provider(self):
        """Invalidate cached model/image state after changing the prompt image."""
        self.sam_generation += 1
        self.sam_prompt_revision += 1
        self.sam_provider = None
        self.sam_loading = False
        self.sam_busy = False
        self.sam_pending_proposal = False

    def update_roi_status(self):
        n_rois = len(self.ROIs)
        trace_state = "traces extracted" if self.extracted else "traces not extracted"
        if self.saved_manual_rois:
            save_state = "saved"
        elif self.extracted and n_rois:
            save_state = "ready to save"
        else:
            save_state = "extract before saving"
        self.roiStatus.setText(
            f"ROIs: {n_rois}  |  {trace_state}  |  {save_state}"
        )
        self.removeROI.setEnabled(n_rois > 0)
        self.procROI.setEnabled(n_rois > 0)
        self.closeGUI.setEnabled(self.extracted and n_rois > 0)

    def set_sam_controls_busy(self, busy):
        self.sam_busy = bool(busy)
        available = self.sam_backend_available()
        self.samMode.setEnabled(available and not self.sam_busy)
        self.undoSAM.setEnabled(
            available and not self.sam_busy and bool(self.sam_points)
        )
        self.acceptSAM.setEnabled(
            available and not self.sam_busy and self.sam_proposal is not None
        )
        self.clearSAM.setEnabled(available)

    def mark_rois_dirty(self):
        self.extracted = False
        self.saved_manual_rois = False
        self.closeGUI.setEnabled(False)
        self.update_roi_status()

    def remove_current_roi(self):
        if not self.ROIs:
            return
        index = getattr(self, "iROI", len(self.ROIs) - 1)
        index = min(len(self.ROIs) - 1, max(0, index))
        self.ROIs[index].remove(self)
        self.update_roi_status()

    def closeEvent(self, event):
        print("closing GUI")
        if self.should_prompt_before_close():
            self.check_proc(event)
        if event.isAccepted():
            self.shutdown_sam_executor()

    def should_prompt_before_close(self):
        """Ask only when closing would discard manual ROI work."""
        if self.saveGUI or self.saved_manual_rois:
            return False
        if len(self.ROIs) == 0:
            return False
        return True

    def shutdown_sam_executor(self):
        if getattr(self, "sam_executor", None) is None:
            return
        self.sam_generation += 1
        self.sam_prompt_revision += 1
        self.sam_executor.shutdown(wait=False, cancel_futures=True)
        self.sam_executor = None

    def check_proc(self, event):
        if self.extracted:
            message = (
                "You have extracted manual ROI traces but have not saved them. "
                "Save before closing?"
            )
        else:
            message = (
                "You have manual ROIs that have not been extracted/saved. "
                "Click Yes to extract and save, No to discard, or Cancel to keep editing."
            )
        cproc = QMessageBox.question(
            self, "PROC",
            message,
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if cproc == QMessageBox.Yes:
            self.close_GUI()
        elif cproc == QMessageBox.Cancel:
            event.ignore()

    def close_GUI(self):
        self.nROIs = len(self.ROIs)
        if self.nROIs == 0:
            self.saveGUI = True
            self.close()
            return
        if not self.extracted:
            self.proc_ROI()
            if not self.extracted:
                return
        # Replace old stat file
        print("Saving old stat")
        np.save(os.path.join(self.parent.basename, "stat_orig.npy"), self.parent.stat)

        # Save iscell
        print("Num cells", self.nROIs)

        # Append new stat file with old and save
        print("Saving new stat")
        stat_all = np.concatenate((self.new_stat, self.parent.stat))
        np.save(os.path.join(self.parent.basename, "stat.npy"), stat_all)
        iscell_prob = np.concatenate(
            (self.parent.iscell[:, np.newaxis], self.parent.probcell[:, np.newaxis]),
            axis=1)

        new_iscell = np.ones((self.nROIs, 2))
        new_iscell = np.concatenate((new_iscell, iscell_prob), axis=0)
        np.save(os.path.join(self.parent.basename, "iscell.npy"), new_iscell)

        # Save fluorescence traces
        Fcell = np.concatenate((self.Fcell, self.parent.Fcell), axis=0)
        Fneu = np.concatenate((self.Fneu, self.parent.Fneu), axis=0)
        Spks = np.concatenate(
            (self.Spks, self.parent.Spks), axis=0
        )  # For now convert spikes to 0 for the new ROIS and then fix it later
        np.save(os.path.join(self.parent.basename, "F.npy"), Fcell)
        np.save(os.path.join(self.parent.basename, "Fneu.npy"), Fneu)
        np.save(os.path.join(self.parent.basename, "spks.npy"), Spks)

        if "reg_file_chan2" in self.parent.ops:
            F_chan2 = np.load(os.path.join(self.parent.basename, "F_chan2.npy"))
            Fneu_chan2 = np.load(os.path.join(self.parent.basename, "Fneu_chan2.npy"))
            redorig = np.load(os.path.join(self.parent.basename, "redcell.npy"))
            F_chan2 = np.concatenate((self.F_chan2, F_chan2), axis=0)
            Fneu_chan2 = np.concatenate((self.Fneu_chan2, Fneu_chan2), axis=0)
            Fneu = np.concatenate((self.Fneu, self.parent.Fneu), axis=0)
            new_redcell = np.zeros((self.nROIs, 2))
            new_redcell = np.concatenate((new_redcell, redorig), axis=0)
            np.save(os.path.join(self.parent.basename, "F_chan2.npy"), F_chan2)
            np.save(os.path.join(self.parent.basename, "Fneu_chan2.npy"), Fneu_chan2)
            np.save(os.path.join(self.parent.basename, "redcell.npy"), new_redcell)

        print(np.shape(Fcell), np.shape(Fneu), np.shape(Spks), np.shape(new_iscell),
              np.shape(stat_all))

        # close GUI
        io.load_proc(self.parent)
        self.saved_manual_rois = True
        self.saveGUI = True
        self.close()

    def normalize_img_add_masks(self):
        nviews = len(self.views)
        masked_image = np.zeros((self.Ly, self.Lx, 3, nviews))
        yr = slice(self.parent.ops["yrange"][0], self.parent.ops["yrange"][1])
        xr = slice(self.parent.ops["xrange"][0], self.parent.ops["xrange"][1])
        for i in np.arange(nviews):
            if i == 0:
                src = self.parent.ops.get("meanImg")
            elif i == 1:
                src = self.parent.ops.get("meanImgE")
            elif i == 2:
                src = self.parent.ops.get("Vcorr")
            elif i == 3:
                src = self.parent.ops.get("max_proj", None)
            elif i == 4 and self.has_chan2:
                src = self.parent.ops.get("meanImg_chan2")
            else:
                src = None

            mimg = np.zeros((self.Ly, self.Lx), np.float32)
            if src is not None:
                mimg1 = np.percentile(src, 1)
                mimg99 = np.percentile(src, 99)
                if mimg99 > mimg1:
                    src = (src - mimg1) / (mimg99 - mimg1)
                else:
                    src = np.zeros_like(src)
                src = np.clip(src, 0, 1).astype(np.float32)
                if src.shape[0] == self.Ly and src.shape[1] == self.Lx:
                    mimg = src
                else:
                    mimg[yr, xr] = src

            masked_image[:, :, :, i] = self.create_masks_of_cells(mimg)

        return masked_image

    def create_masks_of_cells(self, mean_img):
        H = np.zeros_like(mean_img)
        S = np.zeros_like(mean_img)
        columncol = self.parent.colors["istat"][0]

        for n in np.arange(np.shape(self.parent.iscell)[0]):
            if self.parent.iscell[n] == 1:
                ypix = self.parent.stat[n]["ypix"].flatten()
                xpix = self.parent.stat[n]["xpix"].flatten()
                H[ypix, xpix] = np.random.rand()
                S[ypix, xpix] = 1

        pix = np.concatenate(
            ((H[:, :, np.newaxis]), S[:, :, np.newaxis], mean_img[:, :, np.newaxis]),
            axis=-1)
        pix = hsv_to_rgb(pix)
        return pix

    def mouse_moved(self, pos):
        if self.extracted:
            if self.p1.sceneBoundingRect().contains(pos):
                x = self.p1.vb.mapSceneToView(pos).x()
                y = self.p1.vb.mapSceneToView(pos).y()
                self.ineuron = self.nROIs - y + 1
                # print(self.ineuron)

    def keyPressEvent(self, event):
        if event.modifiers() != QtCore.Qt.AltModifier and event.modifiers(
        ) != QtCore.Qt.ShiftModifier:
            if event.key() == QtCore.Qt.Key_D and self.ROIs:
                self.ROIs[self.iROI].remove(self)
            elif event.key() == QtCore.Qt.Key_W:
                self.viewbtns.button(0).setChecked(True)
                self.viewbtns.button(0).press(self, 0)
            elif event.key() == QtCore.Qt.Key_E:
                self.viewbtns.button(1).setChecked(True)
                self.viewbtns.button(1).press(self, 1)
            elif event.key() == QtCore.Qt.Key_R:
                self.viewbtns.button(2).setChecked(True)
                self.viewbtns.button(2).press(self, 2)
            elif event.key() == QtCore.Qt.Key_T:
                self.viewbtns.button(3).setChecked(True)
                self.viewbtns.button(3).press(self, 3)
            elif event.key() == QtCore.Qt.Key_Y and self.has_chan2:
                self.viewbtns.button(4).setChecked(True)
                self.viewbtns.button(4).press(self, 4)

    def add_ROI(self, pos=None):
        self.iROI = len(self.ROIs)
        self.nROIs = len(self.ROIs)
        self.ROIs.append(
            sROI(iROI=self.nROIs, parent=self, pos=pos, diameter=int(self.diam.text())))
        self.ROIs[-1].position(self)
        self.nROIs += 1
        print("%d cells added to manual GUI" % self.nROIs)
        self.mark_rois_dirty()

    def add_polygon(self, pos=None):
        self.iROI = len(self.ROIs)
        if pos is None:
            view = self.p0.viewRange()
            cx = (view[0][0] + view[0][1]) / 2
            cy = (view[1][0] + view[1][1]) / 2
        else:
            cx, cy = float(pos[0]), float(pos[1])
        diameter = max(3, int(self.diam.text()))
        self.ROIs.append(
            PolygonROI(
                iROI=self.iROI,
                parent=self,
                center=(cx, cy),
                diameter=diameter,
            )
        )
        self.nROIs = len(self.ROIs)
        self.mark_rois_dirty()
        print(f"{self.nROIs} cells added to manual GUI")

    def sam_mode_changed(self, enabled):
        if enabled and not self.sam_backend_available():
            self.samMode.setChecked(False)
            self.samStatus.setText(
                f"{self.sam_unavailable_reason()} Manual ellipse/polygon tools are ready."
            )
            return
        self.samMode.setStyleSheet(
            self.stylePressed if enabled else self.styleUnpressed
        )
        if enabled:
            self.samStatus.setText(
                "SAM3: left-click includes, Shift+left-click excludes. First "
                "prompt loads the model in the background."
            )

    def _ensure_sam_provider(self):
        if not self.sam_backend_available():
            self.samStatus.setText(
                f"{self.sam_unavailable_reason()} Manual ellipse/polygon tools are ready."
            )
            return False
        if self.sam_provider is not None:
            return True
        if self.sam_loading:
            self.sam_pending_proposal = True
            return False
        self.sam_loading = True
        self.sam_pending_proposal = True
        self.set_sam_controls_busy(True)
        generation = self.sam_generation
        try:
            prompt_image, prompt_name = self.sam_prompt_image()
            checkpoint_path = self.sam_checkpoint_path()
            device = self.parent.ops.get("torch_device")
            self.samStatus.setText(
                f"SAM3: loading model for {prompt_name} in the background. "
                "First load can take a minute."
            )
            self.sam_executor.submit(
                self._load_sam_provider_background,
                checkpoint_path,
                device,
                prompt_image,
                prompt_name,
                generation,
            )
        except Exception as exc:
            self._sam_failed("load", str(exc), generation)
            return False
        return False

    def _load_sam_provider_background(
        self, checkpoint_path, device, prompt_image, prompt_name, generation
    ):
        try:
            provider = UltralyticsSam3MaskProvider(
                checkpoint_path=checkpoint_path,
                device=device,
            )
            provider.set_image(prompt_image)
            self.sam_signals.loaded.emit(provider, prompt_name, generation)
        except Exception as exc:
            self.sam_signals.failed.emit("load", str(exc), generation)

    @QtCore.Slot(object, str, int)
    def _sam_loaded(self, provider, prompt_name, generation):
        if generation != self.sam_generation:
            return
        self.sam_provider = provider
        self.sam_loading = False
        self.set_sam_controls_busy(False)
        self.samStatus.setText(f"SAM3 ready using {prompt_name}")
        if self.sam_pending_proposal and self.sam_points:
            self._submit_sam_proposal()

    def _draw_sam_prompts(self):
        spots = []
        for (x, y), label in zip(self.sam_points, self.sam_labels):
            spots.append(
                {
                    "pos": (x, y),
                    "symbol": "+",
                    "size": 13,
                    "pen": pg.mkPen("g" if label else "r", width=3),
                }
            )
        self.sam_prompt_scatter.setData(spots)
        self.undoSAM.setEnabled(
            bool(self.sam_points) and self.sam_backend_available() and not self.sam_busy
        )

    def _update_sam_proposal(self):
        if self.sam_busy or self.sam_loading:
            self.sam_pending_proposal = True
            self.samStatus.setText("SAM3 busy; proposal will update when ready")
            return
        if not self._ensure_sam_provider():
            return
        self._submit_sam_proposal()

    def _submit_sam_proposal(self):
        if self.sam_provider is None:
            self._ensure_sam_provider()
            return
        if not self.sam_points:
            return
        revision = self.sam_prompt_revision
        points = np.asarray(self.sam_points, dtype=np.float32).copy()
        labels = np.asarray(self.sam_labels, dtype=np.int32).copy()
        self.sam_pending_proposal = False
        self.sam_proposal = None
        self.acceptSAM.setEnabled(False)
        self.set_sam_controls_busy(True)
        self.samStatus.setText("SAM3: updating proposal in the background...")
        self.sam_executor.submit(
            self._propose_sam_background, points, labels, revision
        )

    def _propose_sam_background(self, points, labels, revision):
        try:
            proposal = self.sam_provider.propose(points, labels)
            self.sam_signals.proposalReady.emit(proposal, revision)
        except Exception as exc:
            self.sam_signals.failed.emit("proposal", str(exc), revision)

    @QtCore.Slot(object, int)
    def _sam_proposal_ready(self, proposal, revision):
        self.set_sam_controls_busy(False)
        if revision != self.sam_prompt_revision:
            if self.sam_pending_proposal and self.sam_points:
                self._submit_sam_proposal()
            return
        positive_points = [
            point
            for point, label in zip(self.sam_points, self.sam_labels)
            if label == 1
        ]
        anchor = positive_points[-1] if positive_points else None
        try:
            mask = clean_mask(
                proposal.mask,
                positive_point=anchor,
                yrange=self.parent.ops.get("yrange"),
                xrange=self.parent.ops.get("xrange"),
            )
            if not mask.any():
                raise ValueError("SAM3 proposal was empty after mask cleanup")
        except Exception as exc:
            self._sam_failed("proposal", str(exc), revision)
            return

        proposal.mask = mask
        self.sam_proposal = proposal
        rgba = np.zeros((self.Ly, self.Lx, 4), dtype=np.uint8)
        rgba[mask] = (30, 144, 255, 115)
        self.sam_overlay.setImage(rgba, levels=(0, 255))
        self.acceptSAM.setEnabled(True)
        self.samStatus.setText(
            f"SAM3 proposal: {mask.sum()} pixels, score {proposal.score:.3f}; "
            f"{sum(self.sam_labels)} include / "
            f"{len(self.sam_labels) - sum(self.sam_labels)} exclude points"
        )

    @QtCore.Slot(str, str, int)
    def _sam_failed(self, stage, message, generation_or_revision):
        if stage == "load" and generation_or_revision != self.sam_generation:
            return
        if stage == "proposal" and generation_or_revision != self.sam_prompt_revision:
            if self.sam_pending_proposal and self.sam_points:
                self.set_sam_controls_busy(False)
                self._submit_sam_proposal()
            return

        self.sam_loading = False
        self.sam_pending_proposal = False
        self.set_sam_controls_busy(False)
        if stage == "load":
            self.sam_provider = None
            self.samMode.setChecked(False)
            QMessageBox.critical(self, "SAM3 unavailable", message)
            self.samStatus.setText(
                "SAM3 unavailable; polygon and ellipse tools still work"
            )
        else:
            QMessageBox.warning(self, "SAM3 proposal failed", message)
            self.samStatus.setText("SAM3 proposal failed; adjust or clear prompts")

    def sam_label_for_click(self, event):
        """Return SAM point label for supported clicks, or None to ignore."""
        if event.button() != QtCore.Qt.LeftButton:
            return None
        modifiers = event.modifiers()
        try:
            exclude = bool(modifiers & QtCore.Qt.ShiftModifier)
        except TypeError:
            exclude = modifiers == QtCore.Qt.ShiftModifier
        return 0 if exclude else 1

    def undo_sam_point(self):
        if not self.sam_points:
            return
        self.sam_points.pop()
        self.sam_labels.pop()
        self.sam_prompt_revision += 1
        self.sam_pending_proposal = bool(self.sam_points)
        self.sam_proposal = None
        self.sam_overlay.clear()
        self.acceptSAM.setEnabled(False)
        if self.sam_provider is not None:
            self.sam_provider.reset_prompts()
        self._draw_sam_prompts()
        if not self.sam_points:
            self.samStatus.setText("SAM3 point removed; add an include point")
            return
        if not any(label == 1 for label in self.sam_labels):
            self.samStatus.setText("SAM3 needs at least one include point")
            return
        self._update_sam_proposal()

    def clear_sam_proposal(self):
        self.sam_prompt_revision += 1
        self.sam_pending_proposal = False
        self.sam_points = []
        self.sam_labels = []
        self.sam_proposal = None
        self.sam_overlay.clear()
        self.sam_prompt_scatter.clear()
        self.undoSAM.setEnabled(False)
        self.acceptSAM.setEnabled(False)
        if self.sam_provider is not None:
            self.sam_provider.reset_prompts()
        if self.sam_backend_available():
            self.samStatus.setText("SAM3 prompts cleared")
        else:
            self.samStatus.setText(
                f"{self.sam_unavailable_reason()} Manual ellipse/polygon tools are ready."
            )

    def accept_sam_proposal(self):
        if self.sam_proposal is None:
            return
        provenance = self.sam_provider.provenance()
        provenance.update(
            {
                "score": self.sam_proposal.score,
                "points": np.asarray(self.sam_points).tolist(),
                "labels": list(self.sam_labels),
                "coordinate_system": "registered",
            }
        )
        self.ROIs.append(
            MaskROI(
                iROI=len(self.ROIs),
                parent=self,
                mask=self.sam_proposal.mask,
                source="sam3",
                provenance=provenance,
            )
        )
        self.nROIs = len(self.ROIs)
        self.clear_sam_proposal()
        self.mark_rois_dirty()
        print(f"{self.nROIs} cells added to manual GUI")

    def plot_clicked(self, event):
        items = self.win.scene().items(event.scenePos())
        for x in items:
            if x == self.img0:
                pos = self.p0.mapSceneToView(event.scenePos())
                posy = pos.x()
                posx = pos.y()
                if self.samMode.isChecked():
                    label = self.sam_label_for_click(event)
                    if label is None:
                        return
                    self.sam_points.append([float(posy), float(posx)])
                    self.sam_labels.append(label)
                    self.sam_prompt_revision += 1
                    self.sam_pending_proposal = True
                    self._draw_sam_prompts()
                    self._update_sam_proposal()
                    return
                if event.modifiers() == QtCore.Qt.AltModifier:
                    self.add_ROI(pos=np.array([
                        posx - 5, posy - 5,
                        int(self.diam.text()),
                        int(self.diam.text())
                    ]))
                if event.double():
                    self.p0.setXRange(0, self.Lx)
                    self.p0.setYRange(0, self.Ly)
            elif x == self.p1:
                if event.double():
                    self.p1.setXRange(0, self.trange.size)
                    self.p1.setYRange(self.fmin, self.fmax)

    def proc_ROI(self):
        self.nROIs = len(self.ROIs)
        if self.nROIs == 0:
            QMessageBox.information(self, "Manual ROIs", "Draw at least one ROI first.")
            return
        stat0 = []
        if self.extracted:
            for t, s in zip(self.scatter, self.tlabel):
                self.p0.removeItem(s)
                self.p0.removeItem(t)
        self.scatter = []
        self.tlabel = []
        occupied = np.zeros((self.Ly, self.Lx), dtype=bool)
        overlap_pixels = 0
        for n in range(self.nROIs):
            stat = self.ROIs[n].to_stat()
            roi_mask = clean_mask(
                stat_to_mask(stat, (self.Ly, self.Lx)),
                yrange=self.parent.ops.get("yrange"),
                xrange=self.parent.ops.get("xrange"),
            )
            if not roi_mask.any():
                QMessageBox.warning(
                    self,
                    "Invalid manual ROI",
                    f"ROI {n} has no pixels inside the valid registered image.",
                )
                return
            stat = mask_to_stat(
                roi_mask,
                source=stat.get("roi_source", "manual"),
                provenance=stat.get("manual_provenance"),
            )
            stat0.append(stat)
            ypix = stat["ypix"]
            xpix = stat["xpix"]
            overlap_pixels += int(occupied[ypix, xpix].sum())
            occupied[ypix, xpix] = True
            self.tlabel.append(pg.TextItem(str(n), self.ROIs[n].color, anchor=(0, 0)))
            self.tlabel[-1].setPos(xpix.mean(), ypix.mean())
            self.p0.addItem(self.tlabel[-1])
            self.scatter.append(
                pg.ScatterPlotItem([xpix.mean()], [ypix.mean()], pen=self.ROIs[n].color,
                                   symbol="+"))
            self.p0.addItem(self.scatter[-1])
        if overlap_pixels:
            QMessageBox.warning(
                self,
                "Overlapping manual ROIs",
                f"{overlap_pixels} ROI pixels overlap. Suite2p will exclude shared "
                "pixels unless extraction.allow_overlap is enabled.",
            )
        if not os.path.isfile(self.parent.ops["reg_file"]):
            self.parent.ops["reg_file"] = os.path.join(self.parent.basename, "data.bin")

        F, Fneu, F_chan2, Fneu_chan2, spks, settings, stat = masks_and_traces(
            self.parent.ops, stat0, self.parent.stat)
        self.Fcell = F
        self.Fneu = Fneu
        self.F_chan2 = F_chan2
        self.Fneu_chan2 = Fneu_chan2
        self.Spks = spks
        self.plot_trace()
        self.extracted = True
        self.new_stat = stat
        self.update_roi_status()

    def plot_trace(self):
        self.trange = np.arange(0, self.Fcell.shape[1])
        self.p1.clear()
        kspace = 1.0
        ax = self.p1.getAxis("left")
        favg = 0
        k = self.nROIs - 1
        ttick = list()
        for n in range(self.nROIs):
            f = self.Fcell[n, :]
            fneu = self.Fneu[n, :]
            favg += f.flatten()
            fmax = f.max()
            fmin = f.min()
            scale = fmax - fmin
            if not np.isfinite(scale) or scale <= 1e-12:
                f = np.zeros_like(f)
                fneu = np.zeros_like(fneu)
            else:
                f = (f - fmin) / scale
                fneu = (fneu - fmin) / scale
            rgb = self.ROIs[n].color
            self.p1.plot(self.trange, f + k * kspace, pen=rgb)
            if self.nROIs == 1:
                self.p1.plot(self.trange, fneu + k * kspace, pen="r")
            ttick.append((k * kspace + f.mean(), str(n)))
            k -= 1
        self.fmax = (self.nROIs - 1) * kspace + 1
        self.fmin = 0
        ax.setTicks([ttick])
        self.p1.setXRange(0, self.Fcell.shape[1])
        # self.p1.setYRange(self.fmin, self.fmax)


class sROI():

    def __init__(self, iROI, parent=None, pos=None, diameter=None, color=None,
                 yrange=None, xrange=None):
        # what type of ROI it is

        self.iROI = iROI
        self.shape = (parent.Ly, parent.Lx)
        self.xrange = xrange
        self.yrange = yrange
        if color is None:
            self.color = hsv_to_rgb(np.array([np.random.rand() / 1.4 + 0.1, 1, 1]))
            self.color = tuple(255 * self.color)
        else:
            self.color = color
        if pos is None:
            view = parent.p0.viewRange()
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            dx = (view[0][1] - view[0][0]) / 4
            dy = (view[1][1] - view[1][0]) / 4
            if diameter is None:
                dx = np.maximum(3, np.minimum(dx, parent.Ly * 0.2))
                dy = np.maximum(3, np.minimum(dy, parent.Lx * 0.2))
            else:
                dx = diameter
                dy = diameter
            imx = imx - dx / 2
            imy = imy - dy / 2
        else:
            imy = pos[0]
            imx = pos[1]
            dy = pos[2]
            dx = pos[3]

        self.draw(parent, imy, imx, dy, dx)
        self.position(parent)
        self.ROI.sigRegionChangeFinished.connect(lambda: self.position(parent))
        self.ROI.sigClicked.connect(lambda: self.position(parent))
        self.ROI.sigRemoveRequested.connect(lambda: self.remove(parent))

    def draw(self, parent, imy, imx, dy, dx):
        roipen = pg.mkPen(self.color, width=3, style=QtCore.Qt.SolidLine)
        self.ROI = pg.EllipseROI([imx, imy], [dx, dy], pen=roipen, removable=True)
        self.ROI.handleSize = 8
        self.ROI.handlePen = roipen
        self.ROI.addScaleHandle([1, 0.5], [0., 0.5])
        self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.ROI.addRotateHandle([0.5, 1], [0.5, 0.5])
        self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        self.med = [imy, imx]
        parent.p0.addItem(self.ROI)

    def remove(self, parent):
        parent.p0.removeItem(self.ROI)
        for i in range(len(parent.ROIs)):
            if i > self.iROI:
                parent.ROIs[i].iROI -= 1
        del parent.ROIs[self.iROI]
        parent.iROI = min(len(parent.ROIs) - 1, max(0, parent.iROI))
        parent.nROIs = len(parent.ROIs)
        parent.extracted = False
        parent.closeGUI.setEnabled(False)
        if hasattr(parent, "update_roi_status"):
            parent.update_roi_status()
        parent.win.show()
        parent.show()

    def rotate_ROI(self, parent, ellipse, xrange, yrange, center_x, center_y):
        #Rotates ROI depending on Rotatehandle degree
        ellipse = rotate(ellipse, angle=math.floor(self.ROI.angle()), order=0)
        ellipse = np.flip(ellipse, axis=0)
        w, h = ellipse.shape[1], ellipse.shape[0]
        xrange = (np.arange(-(w // 2), w - w // 2) + int(center_x)).astype(np.int32)
        yrange = (np.arange(-(h // 2), h - h // 2) + int(center_y)).astype(np.int32)
        return ellipse, xrange, yrange

    def position(self, parent):
        parent.iROI = self.iROI
        pos0 = self.ROI.getSceneHandlePositions()
        sizex, sizey = self.ROI.size()
        pos = parent.p0.mapSceneToView(pos0[0][1])
        br = self.ROI.boundingRect()
        posy = pos.y()
        posx = pos.x()

        xrange = (np.arange(-1 * int(sizex), 1) + int(posx)).astype(np.int32)
        yrange = (np.arange(-1 * int(sizey), 1) + int(posy)).astype(np.int32)
        yrange += int(np.floor(sizey / 2)) + 1
        # what is ellipse circling?
        ellipse = np.zeros((yrange.size, xrange.size), "bool")
        x, y = np.meshgrid(np.arange(0, xrange.size, 1), np.arange(0, yrange.size, 1))
        ellipse = ((y - br.center().y())**2 / (br.height() / 2)**2 +
                   (x - br.center().x())**2 / (br.width() / 2)**2) <= 1
        center_scene = self.ROI.mapToScene(br.center())
        center_view = parent.p0.mapSceneToView(center_scene)
        center_x = center_view.x()
        center_y = center_view.y()
        if self.ROI.angle() not in (0, 180, -180):
            ellipse, xrange, yrange = self.rotate_ROI(parent, ellipse, xrange, yrange, center_x, center_y)
        #ensures that ROI is not placed outside of movie coordinates
        ellipse = ellipse[:, np.logical_and(xrange >= 0, xrange < parent.Lx)]
        xrange = xrange[np.logical_and(xrange >= 0, xrange < parent.Lx)]
        ellipse = ellipse[np.logical_and(yrange >= 0, yrange < parent.Ly), :]
        yrange = yrange[np.logical_and(yrange >= 0, yrange < parent.Ly)]

        self.ellipse = ellipse
        self.xrange = xrange
        self.yrange = yrange
        if hasattr(parent, "extracted"):
            parent.extracted = False
            parent.closeGUI.setEnabled(False)
            if hasattr(parent, "update_roi_status"):
                parent.update_roi_status()

    def to_stat(self):
        mask = np.zeros(self.shape, dtype=bool)
        mask[np.ix_(self.yrange, self.xrange)] = self.ellipse
        return mask_to_stat(mask, source="manual_ellipse")


class PolygonROI:
    """Editable polygon whose rasterized support becomes a Suite2p mask."""

    def __init__(self, iROI, parent, center, diameter, color=None):
        self.iROI = iROI
        self.parent = parent
        self.shape = (parent.Ly, parent.Lx)
        if color is None:
            rgb = hsv_to_rgb(np.array([np.random.rand() / 1.4 + 0.1, 1, 1]))
            self.color = tuple(255 * rgb)
        else:
            self.color = color

        cx, cy = center
        radius = diameter / 2
        points = [
            (cx - radius, cy - radius),
            (cx + radius, cy - radius),
            (cx + radius, cy + radius),
            (cx - radius, cy + radius),
        ]
        pen = pg.mkPen(self.color, width=3, style=QtCore.Qt.SolidLine)
        self.ROI = pg.PolyLineROI(points, closed=True, pen=pen, removable=True)
        self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        parent.p0.addItem(self.ROI)
        self.position()
        self.ROI.sigRegionChangeFinished.connect(self.position)
        self.ROI.sigClicked.connect(self.position)
        self.ROI.sigRemoveRequested.connect(self.remove)

    def _vertices(self):
        state = self.ROI.saveState()
        vertices = []
        for point in state["points"]:
            mapped = self.ROI.mapToParent(QtCore.QPointF(float(point[0]), float(point[1])))
            vertices.append([mapped.x(), mapped.y()])
        return np.asarray(vertices, dtype=np.float32)

    def position(self):
        self.parent.iROI = self.iROI
        vertices = self._vertices()
        xmin = max(0, int(np.floor(vertices[:, 0].min())))
        xmax = min(self.shape[1], int(np.ceil(vertices[:, 0].max())) + 1)
        ymin = max(0, int(np.floor(vertices[:, 1].min())))
        ymax = min(self.shape[0], int(np.ceil(vertices[:, 1].max())) + 1)
        self.mask = np.zeros(self.shape, dtype=bool)
        if xmax > xmin and ymax > ymin:
            xpix, ypix = np.meshgrid(
                np.arange(xmin, xmax) + 0.5,
                np.arange(ymin, ymax) + 0.5,
            )
            inside = Path(vertices).contains_points(
                np.column_stack((xpix.ravel(), ypix.ravel()))
            )
            self.mask[ymin:ymax, xmin:xmax] = inside.reshape(ypix.shape)
        self.parent.extracted = False
        self.parent.closeGUI.setEnabled(False)
        if hasattr(self.parent, "update_roi_status"):
            self.parent.update_roi_status()

    def remove(self, parent=None):
        self.parent.p0.removeItem(self.ROI)
        _remove_manual_roi(self.parent, self.iROI)

    def to_stat(self):
        self.position()
        return mask_to_stat(self.mask, source="manual_polygon")


class MaskROI:
    """A fixed irregular mask, normally accepted from a prompt model."""

    def __init__(self, iROI, parent, mask, source, provenance=None, color=None):
        self.iROI = iROI
        self.parent = parent
        self.mask = np.asarray(mask, dtype=bool).copy()
        self.source = source
        self.provenance = provenance or {}
        if color is None:
            rgb = hsv_to_rgb(np.array([np.random.rand() / 1.4 + 0.1, 1, 1]))
            self.color = tuple(255 * rgb)
        else:
            self.color = color

        rgba = np.zeros((*self.mask.shape, 4), dtype=np.uint8)
        rgba[self.mask, :3] = np.asarray(self.color, dtype=np.uint8)
        rgba[self.mask, 3] = 105
        self.item = pg.ImageItem(rgba)
        self.item.setZValue(10)
        parent.p0.addItem(self.item)

    def remove(self, parent=None):
        self.parent.p0.removeItem(self.item)
        _remove_manual_roi(self.parent, self.iROI)

    def to_stat(self):
        return mask_to_stat(
            self.mask,
            source=self.source,
            provenance=self.provenance,
        )


def _remove_manual_roi(parent, index):
    for roi in parent.ROIs[index + 1:]:
        roi.iROI -= 1
    del parent.ROIs[index]
    parent.nROIs = len(parent.ROIs)
    parent.iROI = min(parent.nROIs - 1, max(0, parent.iROI))
    parent.extracted = False
    parent.closeGUI.setEnabled(False)
    if hasattr(parent, "update_roi_status"):
        parent.update_roi_status()
    parent.win.show()
    parent.show()
