"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import gc
import glob
import json
import math
import os
import time
import logging 
logger = logging.getLogger(__name__)

import numpy as np
from tifffile import imread, read_scanimage_metadata, TiffFile, TiffWriter

from . import utils

try:
    from ScanImageTiffReader import ScanImageTiffReader
    HAS_SCANIMAGE = True
except ImportError:
    ScanImageTiffReader = None
    HAS_SCANIMAGE = False


def _metadata_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)


def _metadata_float(metadata, key):
    value = metadata.get(key)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) and value > 0 else None


def _metadata_int(metadata, key, default=0):
    try:
        return int(metadata.get(key, default))
    except (TypeError, ValueError):
        return default


def _metadata_int_list(metadata, key):
    value = metadata.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        value = value.strip("[]")
        values = value.replace(",", " ").split()
    elif isinstance(value, np.ndarray):
        values = value.ravel().tolist()
    elif isinstance(value, (list, tuple)):
        values = np.asarray(value, dtype=object).ravel().tolist()
    else:
        values = [value]

    ints = []
    for item in values:
        try:
            ints.append(int(float(item)))
        except (TypeError, ValueError):
            continue
    return ints


def _metadata_fov_um(metadata):
    value = metadata.get("SI.hRoiManager.imagingFovUm")
    if value is None:
        return None
    try:
        corners = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if corners.ndim != 2 or corners.shape[1] < 2:
        return None
    width = float(np.ptp(corners[:, 0]))
    height = float(np.ptp(corners[:, 1]))
    if not (np.isfinite(width) and np.isfinite(height) and width > 0 and height > 0):
        return None
    return [height, width]


def _scanimage_roi_count(roi_metadata):
    if not isinstance(roi_metadata, dict):
        return None
    roi_groups = roi_metadata.get("RoiGroups", {})
    imaging = roi_groups.get("imagingRoiGroup", {})
    rois = imaging.get("rois")
    if rois is None:
        return None
    if isinstance(rois, dict) and "_ArrayData_" in rois:
        return 0
    if isinstance(rois, (list, tuple)):
        return len(rois)
    if isinstance(rois, dict):
        return 1
    return None


def scanimage_acquisition_metadata(
        metadata, roi_metadata=None, configured_nplanes=None):
    """Infer safe Suite2p acquisition settings from ScanImage TIFF metadata."""
    frame_key = "SI.hRoiManager.scanFrameRate"
    volume_key = "SI.hRoiManager.scanVolumeRate"
    stack_key = "SI.hStackManager.enable"
    slices_key = "SI.hStackManager.numSlices"
    actual_slices_key = "SI.hStackManager.actualNumSlices"
    frames_per_volume_key = (
        "SI.hStackManager.numFramesPerVolumeWithFlyback"
    )
    discard_flyback_key = "SI.hFastZ.discardFlybackFrames"
    discard_flyback_count_key = "SI.hFastZ.numDiscardFlybackFrames"

    frame_rate = _metadata_float(metadata, frame_key)
    volume_rate = _metadata_float(metadata, volume_key)
    stack_enabled = _metadata_bool(metadata.get(stack_key, False))
    num_slices = max(
        1, _metadata_int(
            metadata, actual_slices_key, _metadata_int(metadata, slices_key, 1)
        )
    )
    frames_per_volume = max(
        1, _metadata_int(metadata, frames_per_volume_key, num_slices)
    )
    discard_flyback = _metadata_bool(metadata.get(discard_flyback_key, False))
    num_discard_flyback = max(
        0, _metadata_int(
            metadata,
            discard_flyback_count_key,
            max(0, frames_per_volume - num_slices),
        )
    )
    metadata_is_multiplane = stack_enabled and (
        num_slices > 1 or frames_per_volume > 1
    )
    if metadata_is_multiplane and frames_per_volume > num_slices and not discard_flyback:
        nplanes = frames_per_volume
        ignore_flyback = list(range(num_slices, frames_per_volume))
    elif metadata_is_multiplane:
        nplanes = num_slices
        ignore_flyback = []
    else:
        nplanes = 1
        ignore_flyback = []

    if configured_nplanes is not None:
        configured_nplanes = max(1, int(configured_nplanes))
        plane_mismatch = (
            slices_key in metadata
            and configured_nplanes not in {num_slices, frames_per_volume, nplanes}
        )
    else:
        plane_mismatch = False

    if metadata_is_multiplane:
        if volume_rate is not None:
            sampling_frequency = volume_rate
            source_key = volume_key
        elif frame_rate is not None:
            sampling_frequency = frame_rate / frames_per_volume
            source_key = f"{frame_key}/{frames_per_volume_key}"
        else:
            return None
    elif frame_rate is not None:
        sampling_frequency = frame_rate
        source_key = frame_key
    else:
        sampling_frequency = None
        source_key = None

    saved_channels = _metadata_int_list(metadata, "SI.hChannels.channelSave")
    if not saved_channels:
        saved_channels = _metadata_int_list(metadata, "SI.hScan2D.channels")
    nchannels = len(saved_channels) if saved_channels else None
    functional_chan = 1 if nchannels == 1 else None
    lines_per_frame = _metadata_int(metadata, "SI.hRoiManager.linesPerFrame", None)
    pixels_per_line = _metadata_int(metadata, "SI.hRoiManager.pixelsPerLine", None)
    fov_um = _metadata_fov_um(metadata)
    microns_per_pixel = None
    if fov_um and lines_per_frame and pixels_per_line:
        microns_per_pixel = [
            fov_um[0] / lines_per_frame,
            fov_um[1] / pixels_per_line,
        ]

    return {
        "sampling_frequency": (
            None if sampling_frequency is None else float(sampling_frequency)
        ),
        "source": "scanimage_tiff_metadata",
        "source_key": source_key,
        "scan_frame_rate": frame_rate,
        "scan_volume_rate": volume_rate,
        "stack_enabled": stack_enabled,
        "num_slices": num_slices,
        "frames_per_volume_with_flyback": frames_per_volume,
        "discard_flyback": discard_flyback,
        "num_discard_flyback": num_discard_flyback,
        "nplanes": nplanes,
        "ignore_flyback": ignore_flyback,
        "configured_nplanes": configured_nplanes,
        "plane_mismatch": plane_mismatch,
        "saved_channels": saved_channels,
        "nchannels": nchannels,
        "functional_chan": functional_chan,
        "lines_per_frame": lines_per_frame,
        "pixels_per_line": pixels_per_line,
        "bidirectional": _metadata_bool(
            metadata.get("SI.hScan2D.bidirectional", False)
        ),
        "scan_mode": metadata.get("SI.hScan2D.scanMode"),
        "scan_type": metadata.get("SI.hRoiManager.scanType"),
        "scan_zoom_factor": _metadata_float(metadata, "SI.hRoiManager.scanZoomFactor"),
        "fov_um": fov_um,
        "microns_per_pixel": microns_per_pixel,
        "mroi_enabled": _metadata_bool(metadata.get("SI.hRoiManager.mroiEnable", False)),
        "scanimage_roi_count": _scanimage_roi_count(roi_metadata),
    }


def sampling_frequency_from_scanimage_metadata(metadata, configured_nplanes=None):
    """Determine Suite2p's per-plane sampling rate from ScanImage metadata."""
    result = scanimage_acquisition_metadata(
        metadata, configured_nplanes=configured_nplanes
    )
    if result["sampling_frequency"] is None:
        return None
    return result


def read_scanimage_acquisition_metadata(filename, configured_nplanes=None):
    """Read ScanImage static metadata without loading TIFF image data."""
    try:
        with open(filename, "rb") as file_handle:
            metadata_result = read_scanimage_metadata(file_handle)
        metadata = metadata_result[0]
        roi_metadata = metadata_result[1] if len(metadata_result) > 1 else None
    except (OSError, TypeError, ValueError, KeyError):
        return None
    if not isinstance(metadata, dict):
        return None
    result = scanimage_acquisition_metadata(
        metadata,
        roi_metadata=roi_metadata,
        configured_nplanes=configured_nplanes,
    )
    result["filename"] = os.fspath(filename)
    return result


def read_scanimage_sampling_frequency(filename, configured_nplanes=None):
    result = read_scanimage_acquisition_metadata(
        filename, configured_nplanes=configured_nplanes
    )
    if result is None or result["sampling_frequency"] is None:
        return None
    return result


def _sample_filenames(filenames, max_files):
    filenames = [os.fspath(filename) for filename in filenames]
    if len(filenames) > max_files:
        indices = np.linspace(0, len(filenames) - 1, max_files, dtype=int)
        filenames = [filenames[index] for index in np.unique(indices)]
    return filenames


def infer_tiff_acquisition_metadata(
        filenames, configured_nplanes=None, max_files=20, rtol=1e-4):
    """Infer and cross-check safe ScanImage acquisition metadata across TIFF files."""
    filenames = _sample_filenames(filenames, max_files=max_files)
    if not filenames:
        return None

    results = [
        result
        for filename in filenames
        if (result := read_scanimage_acquisition_metadata(
            filename, configured_nplanes=configured_nplanes
        )) is not None
    ]
    if not results:
        return None
    if len(results) != len(filenames):
        raise ValueError(
            "ScanImage acquisition metadata was missing from "
            f"{len(filenames) - len(results)} of {len(filenames)} sampled TIFFs"
        )

    rates = np.array([
        result["sampling_frequency"]
        for result in results
        if result["sampling_frequency"] is not None
    ], dtype=float)
    if rates.size and not np.allclose(rates, rates[0], rtol=rtol, atol=1e-6):
        raise ValueError(
            "ScanImage TIFF sampling frequencies are inconsistent: "
            + ", ".join(f"{rate:.8g}" for rate in np.unique(rates))
        )
    for key in ("nplanes", "nchannels", "ignore_flyback", "saved_channels"):
        values = [result[key] for result in results]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(
                f"ScanImage TIFF {key} metadata are inconsistent: {values}"
            )
    if any(result["plane_mismatch"] for result in results):
        metadata_planes = sorted(
            {result["num_slices"] for result in results}
        )
        metadata_frames = sorted(
            {result["frames_per_volume_with_flyback"] for result in results}
        )
        raise ValueError(
            f"Configured nplanes={configured_nplanes} does not match "
            f"ScanImage numSlices={metadata_planes} or "
            f"framesPerVolumeWithFlyback={metadata_frames}"
        )

    result = dict(results[0])
    result["files_checked"] = len(results)
    result["files_requested"] = len(filenames)
    return result


def infer_tiff_sampling_frequency(
        filenames, configured_nplanes=None, max_files=20, rtol=1e-4):
    """Infer and cross-check a ScanImage sampling rate across TIFF files."""
    filenames = _sample_filenames(filenames, max_files=max_files)
    if not filenames:
        return None

    results = [
        result
        for filename in filenames
        if (result := read_scanimage_sampling_frequency(
            filename, configured_nplanes=configured_nplanes
        )) is not None
    ]
    if not results:
        return None
    if len(results) != len(filenames):
        raise ValueError(
            "ScanImage sampling-rate metadata was missing from "
            f"{len(filenames) - len(results)} of {len(filenames)} sampled TIFFs"
        )

    rates = np.array(
        [result["sampling_frequency"] for result in results], dtype=float
    )
    if not np.allclose(rates, rates[0], rtol=rtol, atol=1e-6):
        raise ValueError(
            "ScanImage TIFF sampling frequencies are inconsistent: "
            + ", ".join(f"{rate:.8g}" for rate in np.unique(rates))
        )
    if any(result["plane_mismatch"] for result in results):
        metadata_planes = sorted(
            {result["num_slices"] for result in results}
        )
        metadata_frames = sorted(
            {result["frames_per_volume_with_flyback"] for result in results}
        )
        raise ValueError(
            f"Configured nplanes={configured_nplanes} does not match "
            f"ScanImage numSlices={metadata_planes} or "
            f"framesPerVolumeWithFlyback={metadata_frames}"
        )

    result = dict(results[0])
    result["files_checked"] = len(results)
    result["files_requested"] = len(filenames)
    return result


def open_tiff(file, sktiff):
    """
    Open a TIFF file and return the reader object and the number of pages.

    Uses either tifffile or ScanImageTiffReader depending on `sktiff`.

    Parameters
    ----------
    file : str
        Path to the TIFF file.
    sktiff : bool
        If True, use tifffile (TiffFile). If False, use ScanImageTiffReader.

    Returns
    -------
    tif : TiffFile or ScanImageTiffReader
        Opened TIFF reader object.
    Ltif : int
        Number of pages (frames) in the TIFF file.
    """
    if sktiff:
        tif = TiffFile(file)
        Ltif = len(tif.pages)
    else:
        tif = ScanImageTiffReader(file)
        Ltif = 1 if len(tif.shape()) < 3 else tif.shape()[0]  # single page tiffs
    return tif, Ltif


def use_sktiff_reader(tiff_filename, batch_size):
    """
    Test whether ScanImageTiffReader can read a TIFF file.

    Attempts to read a small batch from the file using ScanImageTiffReader. If it
    succeeds, returns False (use ScanImageTiffReader). If it fails or the package is
    not installed, returns True (fall back to tifffile).

    Parameters
    ----------
    tiff_filename : str
        Path to the TIFF file to test.
    batch_size : int
        Number of frames to attempt reading as a test.

    Returns
    -------
    use_sktiff : bool
        True if tifffile should be used, False if ScanImageTiffReader works.
    """
    if HAS_SCANIMAGE:
        try:
            with ScanImageTiffReader(tiff_filename) as tif:
                tif.data() if len(tif.shape()) < 3 else tif.data(
                    beg=0, end=np.minimum(batch_size,
                                          tif.shape()[0] - 1))
            return False
        except:
            logger.info(
                "NOTE: ScanImageTiffReader not working for this tiff type, using tifffile"
            )
            return True
    else:
        logger.info("NOTE: ScanImageTiffReader not installed, using tifffile")
        return True
    

def read_tiff(file, tif, Ltif, ix, batch_size, use_sktiff):
    """
    Read a batch of frames from an open TIFF file starting at index `ix`.

    Reads up to `batch_size` frames and converts the data to int16. Returns None
    if `ix` is past the end of the file.

    Parameters
    ----------
    file : str
        Path to the TIFF file (used by tifffile's `imread` when `use_sktiff` is True).
    tif : TiffFile or ScanImageTiffReader
        Already-opened TIFF reader object.
    Ltif : int
        Total number of pages (frames) in the TIFF file.
    ix : int
        Starting frame index to read from.
    batch_size : int
        Maximum number of frames to read in this batch.
    use_sktiff : bool
        If True, read with tifffile (`imread`). If False, read with ScanImageTiffReader.

    Returns
    -------
    im : numpy.ndarray or None
        Frames as an int16 array of shape (nfr, Ly, Lx), or None if `ix >= Ltif`.
    """
    if ix >= Ltif:
        return None
    nfr = min(Ltif - ix, batch_size)
    if use_sktiff:
        im = imread(file, key=range(ix, ix + nfr))
    elif Ltif == 1:
        im = tif.data()
    else:
        im = tif.data(beg=ix, end=ix + nfr)
    # for single-page tiffs, add 1st dim
    if len(im.shape) < 3:
        im = np.expand_dims(im, axis=0)

    # check if uint16
    if im.dtype.type == np.uint16:
        im = (im // 2).astype(np.int16)
    elif im.dtype.type == np.int32:
        im = (im // 2).astype(np.int16)
    elif im.dtype.type != np.int16:
        im = im.astype(np.int16)

    if im.shape[0] > nfr:
        im = im[:nfr, :, :]

    return im

def tiff_to_binary(dbs, settings, reg_file, reg_file_chan2):
    """
    Read TIFF files and write interleaved plane/channel data to binary files.

    Iterates over all TIFF files listed in `dbs[0]["file_list"]`, de-interleaves
    planes and channels, and writes each plane's frames to the corresponding binary
    file. Also computes per-plane mean images and frame counts.

    Parameters
    ----------
    dbs : list of dict
        Database dictionaries for each plane/ROI. Must contain keys "file_list",
        "first_files", "batch_size", "force_sktiff", "nplanes", "nchannels", and
        optionally "nrois", "swap_order", "lines". Updated in-place with "Ly", "Lx",
        "nframes", "frames_per_file", "frames_per_folder", "meanImg", and
        "meanImg_chan2".
    settings : dict
        Suite2p settings dictionary, saved alongside each plane's database.
    reg_file : list of file objects
        Opened binary files for writing each plane's functional channel data.
    reg_file_chan2 : list of file objects
        Opened binary files for writing each plane's second channel data
        (used only when nchannels > 1).

    Returns
    -------
    dbs : list of dict
        Updated database dictionaries with image dimensions, frame counts, and
        mean images populated.
    """

    t0 = time.time()
        
    fs = dbs[0]["file_list"]
    first_files = dbs[0]["first_files"]

    # try tiff readers
    batch_size = dbs[0]["batch_size"]
    use_sktiff = True if dbs[0]["force_sktiff"] else use_sktiff_reader(fs[0], batch_size=batch_size)

    nplanes, nchannels = dbs[0]["nplanes"], dbs[0]["nchannels"]
    # processing for multiple ROIs (nrois > 1 if mesoscope recording)
    nrois = dbs[0].get("nrois", 1)
    batch_size = nplanes * nchannels * math.ceil(batch_size / (nplanes * nchannels))

    # loop over all tiffs
    which_folder = -1
    ntotal = 0
    swap = dbs[0].get("swap_order", False)
    for ifile, file in enumerate(fs):
        # open tiff
        tif, Ltif = open_tiff(file, use_sktiff)
        # keep track of the plane identity of the first frame (channel identity is assumed always 0)
        if first_files[ifile]:
            which_folder += 1
            iplane = 0
        ix = 0
        while 1:
            im = read_tiff(file, tif, Ltif, ix, batch_size, use_sktiff)
            if im is None:
                break          
            nframes = im.shape[0]
            for j in range(0, nplanes):
                if ifile == 0 and ix == 0:
                    Ly, Lx = im.shape[1], im.shape[2]
                    for k in range(nrois):
                        jk = j*nrois + k
                        Ly = (dbs[jk]["lines"][-1] + 1 - dbs[jk]["lines"][0] 
                              if nrois > 1 else Ly)
                        dbs[jk]["Ly"], dbs[jk]["Lx"] = Ly, Lx
                        dbs[jk]["nframes"] = 0
                        dbs[jk]["frames_per_file"] = np.zeros(len(fs), "int")
                        dbs[jk]["frames_per_folder"] = np.zeros(first_files.sum(), "int")
                        dbs[jk]["meanImg"] = np.zeros((Ly, Lx), "float64")
                        if nchannels > 1:
                            dbs[jk]["meanImg_chan2"] = np.zeros((Ly, Lx), "float64")

                if not swap:
                    i0 = nchannels * ((iplane + j) % nplanes)
                else:
                    i0 = (iplane + j) % (nplanes*nchannels)

                if nchannels > 1:
                    nfunc = dbs[jk]["functional_chan"] - 1
                else:
                    nfunc = 0

                #print(i0, int(i0) + (swap+1)*nfunc)

                im2write = im[int(i0) + (swap+1)*nfunc:nframes:nplanes * nchannels]
                
                for k in range(nrois):
                    jk = j*nrois + k
                    if nrois > 1:
                        imk = im2write[:, dbs[jk]["lines"][0] : dbs[jk]["lines"][-1] + 1]
                    else:
                        imk = im2write
                    reg_file[jk].write(bytearray(imk))
                    dbs[jk]["meanImg"] += imk.sum(axis=0).astype("float64")
                    dbs[jk]["nframes"] += imk.shape[0]
                    dbs[jk]["frames_per_file"][ifile] += imk.shape[0]
                    dbs[jk]["frames_per_folder"][which_folder] += imk.shape[0]
                    
                if nchannels > 1:
                    #print(int(i0) + (swap+1)*(1 - nfunc))
                    im2write = im[int(i0) + (swap+1)*(1 - nfunc):nframes:nplanes * nchannels]
                    for k in range(nrois):
                        jk = j*nrois + k
                        if nrois > 1:
                            imk = im2write[:, dbs[jk]["lines"][0] : dbs[jk]["lines"][-1] + 1]
                        else:
                            imk = im2write
                        reg_file_chan2[jk].write(bytearray(imk))
                        dbs[jk]["meanImg_chan2"] += imk.sum(axis=0).astype("float64")
            if not swap:
                iplane = (iplane - nframes / nchannels) % nplanes
            else:
                iplane = (iplane - nframes) % (nchannels * nplanes)
            ix += nframes
            ntotal += nframes
            if ntotal % (batch_size * 4) == 0:
                logger.info("%d frames of binary, time %0.2f sec." %
                      (ntotal, time.time() - t0))
        gc.collect()
    # write dbs and settings files
    for db in dbs:
        db["meanImg"] /= db["nframes"]
        if nchannels > 1:
            db["meanImg_chan2"] /= db["nframes"]
        np.save(db["db_path"], db)
        np.save(db["settings_path"], settings)
    
    # close all binary files and write settings files
    for jk in range(0, nplanes * nrois):
        reg_file[jk].close()
        if nchannels > 1:
            reg_file_chan2[jk].close()
    return dbs

def ome_to_binary(dbs, settings, reg_file, reg_file_chan2):
    """
    Convert non-interleaved OME-TIFF files to binary, splitting channels by filename.

    Designed for recordings where channels are stored in separate TIFF files
    (distinguished by "Ch1"/"Ch2" in the filename) rather than interleaved pages.
    Supports both single-page and multi-page TIFFs, and handles bidirectional
    Bruker plane ordering.

    Parameters
    ----------
    dbs : list of dict
        Database dictionaries for each plane. Must contain keys "file_list",
        "first_files", "batch_size", "nplanes", "nchannels", "functional_chan",
        and optionally "bruker_bidirectional". Updated in-place with "Ly", "Lx",
        "nframes", "frames_per_file", "frames_per_folder", "meanImg", and
        "meanImg_chan2".
    settings : dict
        Suite2p settings dictionary, saved alongside each plane's database.
    reg_file : list of file objects
        Opened binary files for writing each plane's functional channel data.
    reg_file_chan2 : list of file objects
        Opened binary files for writing each plane's second channel data
        (used only when nchannels > 1).

    Returns
    -------
    dbs : list of dict
        Updated database dictionaries with image dimensions, frame counts, and
        mean images populated.
    """
    t0 = time.time()

    nplanes = dbs[0]["nplanes"]
    nchannels = dbs[0]["nchannels"]

    # get file list from dbs (already populated by run_s2p.py)
    fs = dbs[0]["file_list"]
    first_files = dbs[0]["first_files"]
    batch_size = dbs[0]["batch_size"]
    use_sktiff = not HAS_SCANIMAGE

    fs_Ch1, fs_Ch2 = [], []
    for f in fs:
        if f.find("Ch1") > -1:
            if dbs[0]["functional_chan"] == 1:
                fs_Ch1.append(f)
            else:
                fs_Ch2.append(f)
        else:
            if dbs[0]["functional_chan"] == 1:
                fs_Ch2.append(f)
            else:
                fs_Ch1.append(f)

    if len(fs_Ch2) == 0:
        dbs[0]["nchannels"] = 1
        nchannels = 1
    logger.info(f"nchannels = {nchannels}")
    
    # loop over all tiffs
    TiffReader = ScanImageTiffReader if HAS_SCANIMAGE else TiffFile
    with TiffReader(fs_Ch1[0]) as tif:
        if HAS_SCANIMAGE:
            n_pages = tif.shape()[0] if len(tif.shape()) > 2 else 1
            shape = tif.shape()[-2:]
        else:
            n_pages = len(tif.pages)
            im0 = tif.pages[0].asarray()
            shape = im0.shape

    for db in dbs:
        db["nframes"] = 0
        db["frames_per_folder"] = np.zeros(first_files.sum(), "int")
        db["frames_per_file"] = np.ones(len(fs_Ch1), "int") if n_pages==1 else np.zeros(len(fs_Ch1), "int")
        db["meanImg"] = np.zeros(shape, np.float32)
        if nchannels > 1:
            db["meanImg_chan2"] = np.zeros(shape, np.float32)

    bruker_bidirectional = dbs[0].get("bruker_bidirectional", False)
    iplanes = np.arange(0, nplanes)
    if not bruker_bidirectional:
        iplanes = np.tile(iplanes[np.newaxis, :],
                          int(np.ceil(len(fs_Ch1) / nplanes))).flatten()
        iplanes = iplanes[:len(fs_Ch1)]
    else:
        iplanes = np.hstack((iplanes, iplanes[::-1]))
        iplanes = np.tile(iplanes[np.newaxis, :],
                          int(np.ceil(len(fs_Ch1) / (2 * nplanes)))).flatten()
        iplanes = iplanes[:len(fs_Ch1)]

    itot = 0
    for ik, file in enumerate(fs_Ch1):
        ip = iplanes[ik]
        # read tiff
        if n_pages==1:    
            with TiffReader(file) as tif:
                im = tif.data()  if HAS_SCANIMAGE else tif.pages[0].asarray()
            if im.dtype.type == np.uint16:
                im = (im // 2)
            im = im.astype(np.int16)

            # write to binary
            dbs[ip]["nframes"] += 1
            dbs[ip]["frames_per_folder"][0] += 1
            dbs[ip]["meanImg"] += im.astype(np.float32)
            reg_file[ip].write(bytearray(im))
            #gc.collect()
        else:
            tif, Ltif = open_tiff(file, not HAS_SCANIMAGE)
            # keep track of the plane identity of the first frame (channel identity is assumed always 0)
            ix = 0
            while 1:
                im = read_tiff(file, tif, Ltif, ix, batch_size, use_sktiff)
                if im is None:
                    break
                nframes = im.shape[0]
                ix += nframes
                itot += nframes
                reg_file[ip].write(bytearray(im))
                dbs[ip]["meanImg"] += im.astype(np.float32).sum(axis=0)
                dbs[ip]["nframes"] += im.shape[0]
                dbs[ip]["frames_per_file"][ik] += nframes
                dbs[ip]["frames_per_folder"][0] += nframes
                if itot % 1000 == 0:
                    logger.info("%d frames of binary, time %0.2f sec." % (itot, time.time() - t0))
                gc.collect()            

    if nchannels > 1:
        itot = 0
        for ik, file in enumerate(fs_Ch2):
            ip = iplanes[ik]
            if n_pages==1:
                with TiffReader(file) as tif:
                    im = tif.data() if HAS_SCANIMAGE else tif.pages[0].asarray()
                if im.dtype.type == np.uint16:
                    im = (im // 2)
                im = im.astype(np.int16)
                dbs[ip]["meanImg_chan2"] += im.astype(np.float32)
                reg_file_chan2[ip].write(bytearray(im))
            else:
                tif, Ltif = open_tiff(file, not HAS_SCANIMAGE)
                ix = 0
                while 1:
                    im = read_tiff(file, tif, Ltif, ix, batch_size, use_sktiff)
                    if im is None:
                        break
                    nframes = im.shape[0]
                    ix += nframes
                    itot += nframes
                    dbs[ip]["meanImg_chan2"] += im.astype(np.float32).sum(axis=0)
                    reg_file_chan2[ip].write(bytearray(im))
                    if itot % 1000 == 0:
                        logger.info("%d frames of binary, time %0.2f sec." % (itot, time.time() - t0))
                    gc.collect()

    # update dbs with image dimensions and mean images
    for db in dbs:
        db["Ly"], db["Lx"] = shape
        db["meanImg"] /= db["nframes"]
        if nchannels > 1:
            db["meanImg_chan2"] /= db["nframes"]
        # Save db and settings to each plane folder
        np.save(db["db_path"], db)
        np.save(db["settings_path"], settings)

    return dbs
