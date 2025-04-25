#  test_qwen_cpu.py

from pathlib import Path
from typing  import Any, Dict, List
from PIL     import Image
import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

def _gather_media(msgs: List[Dict[str, Any]], media_type: str):
    blob, sizes = [], []
    for m in msgs:
        for chunk in m["content"]:
            if chunk.get("type") == media_type:
                blob.append(chunk[media_type])
                sizes.append(chunk.get("size", None))
    return blob if blob else None, sizes if sizes else None


def process_vision_info(msgs):
    """
    Returns   image_inputs, video_inputs
    Each is either a list (matching what AutoProcessor expects) or None.
    """
    images, _  = _gather_media(msgs, "image")
    videos, _  = _gather_media(msgs, "video")
    return images, videos
# ──────────────────────────────────────────────────────────────

MODEL_ID   = "Qwen/Qwen2.5-VL-3B-Instruct"
DTYPE      = torch.float32
DEVICE_MAP = {"": "cpu"}

print("🔄  Loading model …")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype      = DTYPE,
    device_map       = DEVICE_MAP,
    low_cpu_mem_usage= True,
    local_files_only = True,
)

processor = AutoProcessor.from_pretrained(
    MODEL_ID, local_files_only=True
)

# ───────  your demo  ───────
IMG_PATH = Path("assets/sample.png")
if not IMG_PATH.exists():
    raise FileNotFoundError(IMG_PATH)

img = Image.open(IMG_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": "Describe this image in one sentence."},
        ],
    }
]

# build the prompt and the vision tensors
prompt                  = processor.apply_chat_template(
                              messages, tokenize=False, add_generation_prompt=True
                          )
image_inputs, _         = process_vision_info(messages)
inputs                  = processor(
                              text   = [prompt],
                              images = image_inputs,
                              return_tensors="pt"
                          ).to(model.device)

print("🚀  Generating …")
gen_out = model.generate(
    **inputs,
    max_new_tokens = 32,
    do_sample      = False
)

out_ids = gen_out[0, inputs.input_ids.shape[1]:]   # trim prompt tokens
print("\n📝  Model answer:", processor.decode(out_ids, skip_special_tokens=True).strip())
