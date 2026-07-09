import numpy as np
import pytest
import torch
from importlib.metadata import PackageNotFoundError

from suite2p import default_settings
from suite2p.manual_rois import (
    UltralyticsSam3MaskProvider,
    build_prompt_image,
    clean_mask,
    mask_to_stat,
    normalize_image,
    stat_to_mask,
)
from suite2p.pipeline_s2p import pipeline


def test_normalize_image_handles_nonfinite_and_constant_images():
    image = np.array([[1.0, np.nan], [3.0, 2.0]], dtype=np.float32)
    normalized = normalize_image(image, lower=0, upper=100)
    assert normalized.dtype == np.float32
    assert normalized.min() == 0
    assert normalized.max() == 1
    assert np.isfinite(normalized).all()
    assert not normalize_image(np.ones((3, 3))).any()


def test_build_prompt_image_embeds_cropped_products():
    ops = {
        "Ly": 6,
        "Lx": 7,
        "yrange": [1, 5],
        "xrange": [2, 6],
        "meanImg": np.arange(42, dtype=np.float32).reshape(6, 7),
        "meanImgE": np.arange(42, dtype=np.float32).reshape(6, 7),
        "Vcorr": np.arange(16, dtype=np.float32).reshape(4, 4),
        "max_proj": np.arange(16, dtype=np.float32).reshape(4, 4),
    }
    image = build_prompt_image(ops)
    assert image.shape == (6, 7, 3)
    assert image.dtype == np.uint8
    assert not image[0, :, 1:].any()
    assert not image[:, :2, 1:].any()
    assert image[1:5, 2:6, 1].max() == 255


def test_clean_mask_keeps_clicked_component_and_valid_crop():
    mask = np.zeros((10, 10), bool)
    mask[1:3, 1:3] = True
    mask[5:9, 5:9] = True
    mask[6, 6] = False

    cleaned = clean_mask(
        mask,
        positive_point=(6, 6),
        yrange=[2, 9],
        xrange=[2, 9],
        min_pixels=3,
    )
    assert cleaned[6, 6]
    assert not cleaned[1:3, 1:3].any()
    assert cleaned.sum() == 16


def test_mask_stat_roundtrip_uses_uniform_weights():
    mask = np.zeros((8, 9), bool)
    mask[2:5, 3:7] = True
    stat = mask_to_stat(mask, source="sam3", provenance={"score": 0.91})

    assert stat["npix"] == 12
    assert stat["manual"] == 1
    assert stat["roi_source"] == "sam3"
    assert np.all(stat["lam"] == 1)
    assert stat["manual_provenance"]["score"] == pytest.approx(0.91)
    np.testing.assert_array_equal(stat_to_mask(stat, mask.shape), mask)


def test_mask_to_stat_rejects_empty_mask():
    with pytest.raises(ValueError, match="empty mask"):
        mask_to_stat(np.zeros((4, 4), bool))


def test_ultralytics_sam3_provider_selects_highest_scoring_mask():
    class FakeMasks:
        def __init__(self):
            self.data = np.zeros((2, 5, 6), bool)
            self.data[0, 1, 1] = True
            self.data[1, 2:4, 2:4] = True

    class FakeBoxes:
        conf = np.array([0.2, 0.8], dtype=np.float32)

    class FakeResult:
        masks = FakeMasks()
        boxes = FakeBoxes()

    class FakeModel:
        def __init__(self):
            self.kwargs = None

        def predict(self, **kwargs):
            self.kwargs = kwargs
            return [FakeResult()]

    provider = UltralyticsSam3MaskProvider(
        checkpoint_path="/tmp/checkpoint.pt",
        device="cpu",
    )
    provider.model = FakeModel()
    provider.image = np.zeros((5, 6, 3), dtype=np.uint8)
    provider.image_shape = provider.image.shape[:2]

    proposal = provider.propose([[3, 3], [4, 4]], [1, 0])

    assert proposal.score == pytest.approx(0.8)
    assert proposal.mask.sum() == 4
    assert proposal.labels.tolist() == [1, 0]
    assert provider.model.kwargs["source"].mode == "RGB"
    assert provider.model.kwargs["source"].size == (6, 5)
    assert provider.model.kwargs["points"] == [[3.0, 3.0], [4.0, 4.0]]
    assert provider.model.kwargs["labels"] == [1, 0]
    assert provider.model.kwargs["conf"] == 0.0
    assert provider.model.kwargs["device"] == "cpu"


def test_ultralytics_provider_installed_falls_back_to_importable_module(monkeypatch):
    monkeypatch.setattr(
        "suite2p.manual_rois.version",
        lambda package: (_ for _ in ()).throw(PackageNotFoundError(package)),
    )
    monkeypatch.setattr(
        "suite2p.manual_rois.importlib_util.find_spec",
        lambda package: object() if package == "ultralytics" else None,
    )

    assert UltralyticsSam3MaskProvider.installed()


def test_ultralytics_sam3_provider_provenance_records_config(monkeypatch):
    monkeypatch.setattr("suite2p.manual_rois.version", lambda package: "test-version")
    provider = UltralyticsSam3MaskProvider(
        checkpoint_path="/tmp/checkpoint.pt",
        device="cpu",
    )

    provenance = provider.provenance()

    assert provenance["backend"] == "ultralytics_sam3"
    assert provenance["ultralytics_version"] == "test-version"
    assert provenance["checkpoint_path"] == "/tmp/checkpoint.pt"
    assert provenance["device"] == "cpu"


def test_sam_click_labels_use_shift_left_for_negative():
    from qtpy import QtCore
    from suite2p.gui.drawroi import ROIDraw

    class FakeEvent:
        def __init__(self, button, modifiers=QtCore.Qt.NoModifier):
            self._button = button
            self._modifiers = modifiers

        def button(self):
            return self._button

        def modifiers(self):
            return self._modifiers

    assert ROIDraw.sam_label_for_click(
        None, FakeEvent(QtCore.Qt.LeftButton)
    ) == 1
    assert ROIDraw.sam_label_for_click(
        None, FakeEvent(QtCore.Qt.LeftButton, QtCore.Qt.ShiftModifier)
    ) == 0
    assert ROIDraw.sam_label_for_click(
        None, FakeEvent(QtCore.Qt.RightButton)
    ) is None


def test_manual_roi_close_prompt_only_for_unsaved_work():
    from suite2p.gui.drawroi import ROIDraw

    class FakeDraw:
        saveGUI = False
        saved_manual_rois = False
        ROIs = []

    window = FakeDraw()
    assert not ROIDraw.should_prompt_before_close(window)

    window.ROIs = [object()]
    assert ROIDraw.should_prompt_before_close(window)

    window.saved_manual_rois = True
    assert not ROIDraw.should_prompt_before_close(window)

    window.saved_manual_rois = False
    window.saveGUI = True
    assert not ROIDraw.should_prompt_before_close(window)


def test_manual_pipeline_initializes_extractable_empty_session(tmp_path):
    settings = default_settings()
    settings["run"]["roi_source"] = "manual"
    settings["run"]["do_registration"] = 0
    settings["torch_device"] = "cpu"
    movie = np.random.default_rng(4).integers(
        0, 100, size=(30, 12, 13), dtype=np.int16
    )

    outputs = pipeline(
        str(tmp_path),
        movie,
        run_registration=False,
        settings=settings,
        device=torch.device("cpu"),
    )

    detect_outputs, stat, F = outputs[1], outputs[2], outputs[3]
    assert "max_proj" in detect_outputs
    assert "meanImg" in detect_outputs
    assert stat.shape == (0,)
    assert F.shape == (0, movie.shape[0])
    for filename in ("stat.npy", "F.npy", "Fneu.npy", "spks.npy", "iscell.npy"):
        assert (tmp_path / filename).is_file()


@pytest.mark.parametrize("neuropil_extract", [True, False])
def test_manual_mask_extracts_trace_from_registered_binary(
    tmp_path, neuropil_extract
):
    from suite2p.gui.drawroi import masks_and_traces

    rng = np.random.default_rng(8)
    movie = rng.integers(0, 100, size=(40, 20, 20), dtype=np.int16)
    movie[:, 7:11, 8:12] += np.arange(40, dtype=np.int16)[:, None, None]
    reg_file = tmp_path / "data.bin"
    movie.tofile(reg_file)

    settings = default_settings()
    settings.update(
        {
            "Ly": 20,
            "Lx": 20,
            "reg_file": str(reg_file),
            "torch_device": "cpu",
        }
    )
    settings["extraction"]["neuropil_extract"] = neuropil_extract
    settings["extraction"]["circular_neuropil"] = neuropil_extract
    mask = np.zeros((20, 20), bool)
    mask[7:11, 8:12] = True
    stat = mask_to_stat(mask, source="manual_polygon")

    F, Fneu, _, _, spks, _, extracted_stat = masks_and_traces(
        settings, [stat], np.array([], dtype=object)
    )

    assert F.shape == (1, movie.shape[0])
    assert Fneu.shape == F.shape
    if not neuropil_extract:
        assert not Fneu.any()
    assert spks.shape == F.shape
    assert extracted_stat[0]["manual"] == 1
    assert extracted_stat[0]["roi_source"] == "manual_polygon"
    dF = F - settings["extraction"]["neuropil_coefficient"] * Fneu
    expected_snr = 1 - 0.5 * np.diff(dF, axis=1).var(axis=1) / dF.var(axis=1)
    assert extracted_stat[0]["snr"] == pytest.approx(expected_snr[0])
