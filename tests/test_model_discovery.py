"""Offline checks for the ComfyUI LoRA path used by the paired PDD model."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_nodes(lora_dir):
    paths = types.ModuleType("folder_paths")

    def filenames(_category):
        assert _category == "loras"
        return sorted(str(path.relative_to(lora_dir)) for path in lora_dir.rglob("*")
                      if path.is_file())

    def full_path(_category, filename):
        assert _category == "loras"
        if filename not in filenames(_category):
            raise FileNotFoundError(filename)
        return str(lora_dir / filename)

    paths.get_filename_list = filenames
    paths.get_full_path_or_raise = full_path

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    stubs = {"folder_paths": paths, "torch": types.ModuleType("torch"), "comfy": comfy}
    for name in ("lora", "model_base", "model_management", "sample", "samplers", "utils"):
        child = types.ModuleType("comfy." + name)
        stubs["comfy." + name] = child
        setattr(comfy, name, child)
    comfy.samplers.CFGGuider = type("CFGGuider", (), {})

    spec = importlib.util.spec_from_file_location("fun_acc_nodes_test", ROOT / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class ModelDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.loras = root / "loras"
        self.loras.mkdir()
        self.nodes = load_nodes(self.loras)
        self.nodes.MODEL_DIR = root / "legacy_models"
        self.nodes.MODEL_DIR.mkdir()

    def test_lora_model_appears_and_resolves(self):
        model = self.loras / self.nodes.MODEL_FILENAME
        model.touch()
        self.assertIn(model.name, self.nodes.FunAccPDD4StepSampler.INPUT_TYPES()["required"]["model_file"][0])
        self.assertEqual(self.nodes._model_path(model.name), model)

    def test_nested_lora_path_resolves(self):
        model = self.loras / "Qwen" / self.nodes.MODEL_FILENAME
        model.parent.mkdir()
        model.touch()
        choice = str(model.relative_to(self.loras))
        self.assertIn(choice, self.nodes._model_choices())
        self.assertEqual(self.nodes._model_path(choice), model)

    def test_legacy_model_remains_available(self):
        model = self.nodes.MODEL_DIR / self.nodes.MODEL_FILENAME
        model.touch()
        self.assertIn(model.name, self.nodes._model_choices())
        self.assertEqual(self.nodes._model_path(model.name), model)

    def test_lora_path_takes_priority_and_unknown_model_fails(self):
        name = self.nodes.MODEL_FILENAME
        (self.loras / name).touch()
        (self.nodes.MODEL_DIR / name).touch()
        self.assertEqual(self.nodes._model_path(name), self.loras / name)
        with self.assertRaisesRegex(ValueError, "ComfyUI/models/loras"):
            self.nodes._model_path("missing.safetensors")
        with self.assertRaises(ValueError):
            self.nodes._model_path("../missing.safetensors")


if __name__ == "__main__":
    unittest.main()
