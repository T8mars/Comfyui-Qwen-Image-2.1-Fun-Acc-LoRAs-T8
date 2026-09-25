[简体中文](README.md) | [English](README_EN.md)

# Qwen-Image-2.1 Fun-Acc PDD 4-Step · ComfyUI

A ComfyUI sampler for [Alibaba PAI's Qwen-Image-2.1 Fun-Acc 4-Step](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs). It uses the original four-step schedule and selects the matching PDD output head at each step. It supports text-to-image generation and image editing.

## Install and use

1. Update ComfyUI to `0.36.0` or newer. Search for **Qwen-Image-2.1 Fun-Acc PDD 4-Step (T8)** in the node manager, or clone this repository into `ComfyUI/custom_nodes/`.
2. Download `Qwen-Image-2.1-Fun-Acc-4Step-PDD-T8.safetensors` from the [model repository](https://huggingface.co/t8star/Qwen-Image-2.1-Fun-Acc-LoRAs-Comfy) into this node's `models/` folder.
3. Get the Qwen-Image-2.1 base assets from [Comfy-Org](https://huggingface.co/Comfy-Org/Qwen-Image-2.1): place the diffusion model in `ComfyUI/models/diffusion_models/`, the Qwen3-VL text encoder in `ComfyUI/models/text_encoders/`, and the VAE in `ComfyUI/models/vae/`.
4. Restart ComfyUI and open the [text-to-image workflow](example_workflows/Qwen-Image-2.1-Fun-Acc-PDD-4Step-T2I.json). Check the three base model filenames before running it.

For image editing, connect the native `Text Encode Qwen Image 2.1` node's `positive` and `latent` outputs to this sampler, and provide that encoder with a reference image and VAE.

The paired model contains four separate output heads. Load it with this node; a standard LoRA Loader and KSampler do not reproduce its four-step sampling. Its tensor data matches the [source release](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs); only identification metadata was added. See the Hugging Face repository for the model license and attribution. Text-to-image and image editing were tested at 512×512 on ComfyUI `0.36.0`.

| Text-to-image | Image editing |
|---|---|
| ![Text-to-image test](assets/preview_t2i.png) | ![Image editing test](assets/preview_edit.png) |

## Links

| Platform | Link |
|---|---|
| Bilibili | [T8star](https://space.bilibili.com/385085361) |
| YouTube | [T8star-Aix](https://www.youtube.com/@T8star-Aix/) |
| API | [Seedance API](https://api.seedance.nz/sign-up?aff=5f4w) |
| Free gallery | [OpenZhenzhen](https://www.openzhenzhen.com) |
| Online AI apps | [RunningHub](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) |
| ComfyUI package | [Quark Drive](https://pan.quark.cn/s/264edb7e36bd) |
| Hugging Face | [t8star](https://huggingface.co/t8star) |
