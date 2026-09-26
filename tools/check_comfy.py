"""Check the paired model against a local native Qwen-Image-2.1 diffusion model."""

import argparse
import gc
import importlib.util
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfyui", type=Path, help="ComfyUI source directory")
    parser.add_argument("base_model", type=Path, help="Qwen-Image-2.1 diffusion model")
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfyui.resolve()))

    import comfy.sd
    import comfy.utils

    node_path = Path(__file__).resolve().parents[1] / "nodes.py"
    spec = importlib.util.spec_from_file_location("t8_funacc_pdd_nodes", node_path)
    nodes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nodes)

    model = comfy.sd.load_diffusion_model(str(args.base_model.resolve()))
    filename = next((name for name in nodes._model_choices()
                     if Path(name).name == nodes.MODEL_FILENAME), None)
    if filename is None:
        raise FileNotFoundError(f"Place {nodes.MODEL_FILENAME} in ComfyUI/models/loras")
    path = nodes._model_path(filename)
    state, metadata = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
    patches, head_key, heads = nodes._prepare_patches(model, state, metadata)
    original_patch_count = len(model.patches)
    for step in range(4):
        stage = model.clone()
        stage_patches = patches.copy()
        stage_patches[head_key] = ("set", (heads[step],))
        applied = stage.add_patches(stage_patches, strength_patch=1.0)
        if len(applied) != 297:
            raise RuntimeError(f"Step {step + 1}: only {len(applied)} of 297 patches applied")
        if len(model.patches) != original_patch_count:
            raise RuntimeError("The input base model was modified while checking stage patches")
        print(f"Step {step + 1}: {len(applied)} patches mapped")
        del stage
    print(f"Compatible: {type(model.model).__name__}; {path.name}")
    del model
    gc.collect()


if __name__ == "__main__":
    main()
