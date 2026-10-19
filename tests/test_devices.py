"""Device selection must work even when tests run without a GPU."""

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from suite2p.devices import assign_torch_device
from suite2p.manual_rois import UltralyticsSam3MaskProvider


class DeviceSelectionTests(unittest.TestCase):
    def test_legacy_user_default_cuda_becomes_auto_without_changing_other_settings(self):
        from suite2p import parameters

        with tempfile.TemporaryDirectory() as folder:
            saved = {"torch_device": "cuda", "fs": 23.5}
            np.save(Path(folder) / "settings_user.npy", saved)
            with patch.object(parameters, "SETTINGS_FOLDER", Path(folder)):
                settings = parameters.user_settings()
            self.assertEqual(settings["torch_device"], "auto")
            self.assertEqual(settings["fs"], 23.5)
            self.assertEqual(parameters.default_settings()["torch_device"], "auto")
            # Loading defaults does not rewrite the user's saved file.
            self.assertEqual(np.load(Path(folder) / "settings_user.npy",
                                     allow_pickle=True).item(), saved)

    def test_explicit_user_default_devices_are_preserved(self):
        from suite2p import parameters

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(parameters, "SETTINGS_FOLDER", Path(folder)):
                for device in ("auto", "cpu", "mps", "cuda:1"):
                    with self.subTest(device=device):
                        np.save(Path(folder) / "settings_user.npy", {"torch_device": device})
                        self.assertEqual(parameters.user_settings()["torch_device"], device)

    def test_auto_selects_mps_on_mac(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=True), \
             patch("torch.zeros") as allocate:
            self.assertEqual(assign_torch_device().type, "mps")
            allocate.assert_called_once_with(1, device=torch.device("mps"))

    def test_auto_prefers_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.backends.mps.is_available", return_value=True), \
             patch("torch.zeros"):
            self.assertEqual(assign_torch_device("auto").type, "cuda")

    def test_auto_uses_cpu_without_accelerators(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=False):
            self.assertEqual(assign_torch_device(None).type, "cpu")

    def test_allocation_failure_falls_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False), \
             patch("torch.backends.mps.is_available", return_value=True), \
             patch("torch.zeros", side_effect=RuntimeError("device unavailable")):
            self.assertEqual(assign_torch_device().type, "cpu")
            self.assertEqual(assign_torch_device("mps").type, "cpu")

    def test_explicit_cpu_does_not_use_gpu(self):
        with patch("torch.zeros") as allocate:
            self.assertEqual(assign_torch_device("cpu").type, "cpu")
            allocate.assert_not_called()

    def test_explicit_cuda_index_is_preserved(self):
        with patch("torch.zeros"):
            self.assertEqual(str(assign_torch_device("cuda:1")), "cuda:1")

    def test_sam_resolves_auto_before_passing_device_to_ultralytics(self):
        with patch("suite2p.manual_rois.assign_torch_device",
                   return_value=torch.device("mps")) as resolve:
            provider = UltralyticsSam3MaskProvider(device="auto")
            self.assertEqual(provider.device, "mps")
            resolve.assert_called_once_with("auto")

    def test_sam_uses_cpu_if_requested_device_is_unavailable(self):
        with patch("torch.zeros", side_effect=RuntimeError("no CUDA")):
            provider = UltralyticsSam3MaskProvider(device="cuda")
            self.assertEqual(provider.device, "cpu")

    def test_sam_predict_receives_resolved_mps_device(self):
        with patch("suite2p.manual_rois.assign_torch_device",
                   return_value=torch.device("mps")):
            provider = UltralyticsSam3MaskProvider(device="auto")
        provider.image = np.zeros((8, 8, 3), dtype=np.uint8)
        provider.image_shape = (8, 8)
        masks = np.ones((1, 8, 8), dtype=bool)
        result = SimpleNamespace(masks=SimpleNamespace(data=masks),
                                 boxes=SimpleNamespace(conf=np.array([0.9])))
        with patch.object(provider, "model", create=True) as model:
            model.predict.return_value = [result]
            proposal = provider.propose([[4, 4]], [1])
            self.assertEqual(model.predict.call_args.kwargs["device"], "mps")
            self.assertTrue(proposal.mask.all())

    def test_cellpose_receives_selected_device(self):
        from suite2p.detection import anatomical

        mask = np.zeros((16, 16), dtype=np.int32)
        mask[4:8, 4:8] = 1
        with patch.object(anatomical, "CellposeModel") as model:
            model.return_value.eval.return_value = [mask]
            anatomical.roi_detect(np.ones((16, 16), dtype=np.float32),
                                  diameter=[4, 4], settings={"params": {}},
                                  device=torch.device("mps"))
            self.assertEqual(model.call_args.kwargs["device"], torch.device("mps"))
            self.assertTrue(model.call_args.kwargs["gpu"])
