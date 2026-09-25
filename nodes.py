"""Four-step PDD sampling for the Alibaba PAI Qwen-Image-2.1 Fun-Acc release."""

from __future__ import annotations

import json
from pathlib import Path

import torch

import comfy.lora
import comfy.model_base
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils


MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_FORMAT = "t8_qwenimage21_funacc_pdd_4step_v1"
SIGMAS = (1.0, 0.9169867038726807, 0.7861579060554504, 0.5494909882545471, 0.0)
EXPECTED_TENSOR_COUNT = 528
EXPECTED_LORA_PAIRS = 231
EXPECTED_FULL_WEIGHTS = 65


class _PositiveGuider(comfy.samplers.CFGGuider):
    def set_positive(self, conditioning):
        self.inner_set_conds({"positive": conditioning})


def _model_path(filename: str) -> Path:
    choices = {p.name: p for p in MODEL_DIR.glob("*.safetensors") if p.is_file()}
    if filename not in choices:
        raise ValueError("Fun-Acc PDD model file is missing from the custom node's models folder.")
    return choices[filename]


def _prepare_patches(model, state: dict, metadata: dict):
    if metadata.get("t8_format") != MODEL_FORMAT:
        raise ValueError("This node needs its paired T8 Fun-Acc PDD model file.")
    if tuple(json.loads(metadata.get("pdd_sigmas", "[]"))) != SIGMAS:
        raise ValueError("Unexpected Fun-Acc PDD sigma schedule in model metadata.")
    if len(state) != EXPECTED_TENSOR_COUNT:
        raise ValueError(f"Expected {EXPECTED_TENSOR_COUNT} PDD tensors, found {len(state)}.")

    head = state.pop("proj_out.weight", None)
    if head is None or tuple(head.shape) != (4, 64, 4096):
        raise ValueError("The four PDD output heads are missing or have the wrong shape.")

    mapped = {}
    up = down = full = 0
    for name, tensor in state.items():
        if name.endswith(".lora_up"):
            mapped[name + ".weight"] = tensor
            up += 1
        elif name.endswith(".lora_down"):
            mapped[name + ".weight"] = tensor
            down += 1
        elif name.endswith(".weight"):
            mapped["diffusion_model." + name[:-len(".weight")] + ".set_weight"] = tensor
            full += 1
        else:
            raise ValueError(f"Unexpected PDD tensor: {name}")
    if (up, down, full) != (EXPECTED_LORA_PAIRS, EXPECTED_LORA_PAIRS, EXPECTED_FULL_WEIGHTS):
        raise ValueError(f"Unexpected PDD tensor groups: up={up}, down={down}, full={full}.")

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    modules = {name[:-len(".lora_up.weight")] for name in mapped if name.endswith(".lora_up.weight")}
    full_modules = {name[:-len(".set_weight")] for name in mapped if name.endswith(".set_weight")}
    missing = (modules | full_modules | {"diffusion_model.proj_out"}) - key_map.keys()
    if missing:
        raise ValueError(f"ComfyUI's Qwen-Image-2.1 LoRA mapping is missing: {sorted(missing)[:3]}")

    patches = comfy.lora.load_lora(mapped, key_map, log_missing=False)
    if len(patches) != EXPECTED_LORA_PAIRS + EXPECTED_FULL_WEIGHTS:
        raise ValueError(f"ComfyUI loaded {len(patches)} of 296 expected PDD patches.")

    base_shapes = model.model.state_dict()
    head_key = key_map["diffusion_model.proj_out"]
    if tuple(base_shapes[head_key].shape) != tuple(head.shape[1:]):
        raise ValueError("The base model's output head shape is incompatible with Fun-Acc PDD.")
    for name in full_modules:
        target = key_map[name]
        tensor = mapped[name + ".set_weight"]
        if tuple(base_shapes[target].shape) != tuple(tensor.shape):
            raise ValueError(f"Base model weight shape differs from Fun-Acc PDD: {target}")
    return patches, head_key, head


class FunAccPDD4StepSampler:
    @classmethod
    def INPUT_TYPES(cls):
        choices = sorted(p.name for p in MODEL_DIR.glob("*.safetensors") if p.is_file())
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "latent_image": ("LATENT",),
            "model_file": (choices or ["Place the paired model in models/ and restart ComfyUI"],),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/qwen_image21"

    def sample(self, model, positive, latent_image, model_file, seed):
        if not isinstance(model.model, comfy.model_base.QwenImage21):
            raise ValueError("Load a native Qwen-Image-2.1 base model before this node.")

        path = _model_path(model_file)
        state, metadata = comfy.utils.load_torch_file(str(path), safe_load=True, return_metadata=True)
        patches, head_key, heads = _prepare_patches(model, state, metadata or {})

        latent = latent_image.copy()
        samples = comfy.sample.fix_empty_latent_channels(
            model, latent["samples"], latent.get("downscale_ratio_spacial"),
            latent.get("downscale_ratio_temporal"),
        )
        noise = comfy.sample.prepare_noise(samples, seed, latent.get("batch_index"))
        sampler = comfy.samplers.sampler_object("euler")
        sigmas = torch.tensor(SIGMAS, dtype=torch.float32)

        for step in range(4):
            stage = model.clone()
            stage_patches = patches.copy()
            stage_patches[head_key] = ("set", (heads[step],))
            applied = stage.add_patches(stage_patches, strength_patch=1.0)
            if len(applied) != EXPECTED_LORA_PAIRS + EXPECTED_FULL_WEIGHTS + 1:
                raise ValueError("Some Fun-Acc PDD weights could not be applied to the base model.")

            guider = _PositiveGuider(stage)
            guider.set_positive(positive)
            samples = guider.sample(
                noise, samples, sampler, sigmas[step:step + 2],
                denoise_mask=latent.get("noise_mask"), disable_pbar=False, seed=seed,
            ).to(comfy.model_management.intermediate_device())
            noise = comfy.sample.prepare_empty_noise(samples)

        latent["samples"] = samples
        latent.pop("downscale_ratio_spacial", None)
        latent.pop("downscale_ratio_temporal", None)
        return (latent,)
