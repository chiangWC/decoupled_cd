import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


class _FakeTorchDevice:
    def __init__(self, spec: str) -> None:
        self.spec = spec

    def __str__(self) -> str:
        return self.spec


def _load_device_module():
    import sys

    fake_torch = ModuleType("torch")
    fake_torch.device = _FakeTorchDevice
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
    sys.modules["torch"] = fake_torch

    device_path = Path(__file__).resolve().parents[1] / "utils" / "device.py"
    spec = spec_from_file_location("device_module_for_test", device_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


DEVICE_MODULE = _load_device_module()


class ResolveDeviceTest(unittest.TestCase):
    @patch.object(DEVICE_MODULE.torch.cuda, "is_available", return_value=True)
    @patch.object(DEVICE_MODULE, "select_best_gpu", return_value=3)
    def test_auto_maps_physical_gpu_to_visible_logical_index(self, mock_select_best_gpu, _mock_is_available) -> None:
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1,3"}, clear=False):
            device = DEVICE_MODULE.resolve_device("auto")

        self.assertEqual(str(device), "cuda:1")
        mock_select_best_gpu.assert_called_once_with([1, 3])

    @patch.object(DEVICE_MODULE.torch.cuda, "is_available", return_value=True)
    @patch.object(DEVICE_MODULE, "select_best_gpu", return_value=1)
    def test_auto_supports_logical_gpu_candidates_inside_visible_set(self, mock_select_best_gpu, _mock_is_available) -> None:
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "1,3"}, clear=False):
            device = DEVICE_MODULE.resolve_device("auto", gpu_candidates="0")

        self.assertEqual(str(device), "cuda:0")
        mock_select_best_gpu.assert_called_once_with([1])

    @patch.object(DEVICE_MODULE.torch.cuda, "is_available", return_value=True)
    @patch.object(DEVICE_MODULE, "select_best_gpu", return_value=3)
    def test_auto_preserves_physical_ids_without_visible_device_mask(self, mock_select_best_gpu, _mock_is_available) -> None:
        with patch.dict("os.environ", {}, clear=True):
            device = DEVICE_MODULE.resolve_device("auto")

        self.assertEqual(str(device), "cuda:3")
        mock_select_best_gpu.assert_called_once_with([])


if __name__ == "__main__":
    unittest.main()
