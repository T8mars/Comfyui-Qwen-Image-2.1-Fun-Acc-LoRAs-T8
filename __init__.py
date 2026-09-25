from .nodes import FunAccPDD4StepSampler

NODE_CLASS_MAPPINGS = {
    "T8QwenImage21FunAccPDD4Step": FunAccPDD4StepSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "T8QwenImage21FunAccPDD4Step": "Qwen-Image-2.1 Fun-Acc PDD 4 Step (T8)",
}
