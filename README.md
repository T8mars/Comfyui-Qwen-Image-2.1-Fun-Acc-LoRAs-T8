[简体中文](README.md) | [English](README_EN.md)

# Qwen-Image-2.1 Fun-Acc PDD 4-Step · ComfyUI

为 [Alibaba PAI 的 Qwen-Image-2.1 Fun-Acc 4-Step](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs) 提供 ComfyUI 采样节点。节点自动使用原版四步时间表，并在每一步切换对应的 PDD 输出头。支持文生图和图片编辑。

## 安装与使用

1. 更新 ComfyUI 至 `0.36.0` 或更新版本，在节点管理器中搜索 **Qwen-Image-2.1 Fun-Acc PDD 4-Step (T8)** 安装；也可将本仓库克隆到 `ComfyUI/custom_nodes/`。
2. 从[模型仓库](https://huggingface.co/t8star/Qwen-Image-2.1-Fun-Acc-LoRAs-Comfy)下载 `Qwen-Image-2.1-Fun-Acc-4Step-PDD-T8.safetensors`，放入本节点的 `models/` 文件夹。
3. 从 [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) 准备基础模型：扩散模型放 `ComfyUI/models/diffusion_models/`，Qwen3-VL 文本编码器放 `ComfyUI/models/text_encoders/`，VAE 放 `ComfyUI/models/vae/`。
4. 重启 ComfyUI，将下方工作流 JSON 下载并拖入画布，核对基础模型文件名后运行。

| 工作流 | 使用前准备 |
|---|---|
| [文生图 · 下载 JSON](example_workflows/Qwen-Image-2.1-Fun-Acc-PDD-4Step-T2I.json) | 选择已下载的基础模型与配套 PDD 文件 |
| [图片编辑 · 下载 JSON](example_workflows/Qwen-Image-2.1-Fun-Acc-PDD-4Step-Edit.json) | 在 `Load Image` 中上传参考图，并选择模型文件 |

两份均为 ComfyUI 画布工作流格式，可直接拖入画布。首次运行前需安装本节点并下载上述模型文件。

配套模型含四个独立输出头，需由本节点加载；普通 LoRA Loader 和 KSampler 无法复现其四步采样过程。模型张量与[原版](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs)一致，仅增加识别元数据。模型授权与署名见 Hugging Face 仓库。已在 ComfyUI `0.36.0` 上完成 512×512 文生图和图片编辑测试。

| 文生图 | 图片编辑 |
|---|---|
| ![文生图测试](assets/preview_t2i.png) | ![图片编辑测试](assets/preview_edit.png) |

## 更多链接

| 平台 | 链接 |
|---|---|
| B站 | [T8star](https://space.bilibili.com/385085361) |
| YouTube | [T8star-Aix](https://www.youtube.com/@T8star-Aix/) |
| API | [Seedance API](https://api.seedance.nz/sign-up?aff=5f4w) |
| 免费画廊 | [OpenZhenzhen](https://www.openzhenzhen.com) |
| 在线 AI 应用 | [RunningHub](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) |
| ComfyUI 整合包 | [夸克网盘](https://pan.quark.cn/s/264edb7e36bd) |
| Hugging Face | [t8star](https://huggingface.co/t8star) |
