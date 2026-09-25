"""Generate a small local image with the paired node and native ComfyUI models."""

import argparse
import gc
import importlib.util
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfyui", type=Path)
    parser.add_argument("diffusion_model", type=Path)
    parser.add_argument("text_encoder", type=Path)
    parser.add_argument("vae", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt", default="A red apple on a wooden table, studio photograph")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfyui.resolve()))

    import numpy as np
    import torch
    from PIL import Image
    import comfy.sd
    import comfy.utils

    node_path = Path(__file__).resolve().parents[1] / "nodes.py"
    spec = importlib.util.spec_from_file_location("t8_funacc_pdd_nodes", node_path)
    nodes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nodes)

    model = comfy.sd.load_diffusion_model(str(args.diffusion_model.resolve()))
    clip = comfy.sd.load_clip(
        ckpt_paths=[str(args.text_encoder.resolve())],
        embedding_directory=[],
        clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
    )
    positive = clip.encode_from_tokens_scheduled(
        clip.tokenize(args.prompt, images=[], keep_vision=True, prevent_empty_text=True)
    )
    del clip
    gc.collect()

    # Match the standard ComfyUI EmptyLatentImage node (4 channels, 8x ratio).
    latent = {
        "samples": torch.zeros((1, 4, args.size // 8, args.size // 8)),
        "downscale_ratio_spacial": 8,
    }
    filename = next(nodes.MODEL_DIR.glob("*.safetensors")).name
    sampled = nodes.FunAccPDD4StepSampler().sample(model, positive, latent, filename, args.seed)[0]
    print("Sampled", tuple(sampled["samples"].shape), flush=True)

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(args.vae.resolve())))
    image = vae.decode(sampled["samples"])[0]
    arr = np.clip(image.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(args.output)
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
