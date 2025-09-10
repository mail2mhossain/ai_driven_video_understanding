import os, math, json
from typing import List, Union
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Tips to stay within 4–6 GB VRAM

# Quantization: keep load_in_4bit=True. If you still OOM, reduce MAX_SIDE (e.g., 768 → 640).
# Batching: send several images in one prompt (as above) but keep each image’s longest side modest.
# Throughput: set do_sample=False, temperature=0 for deterministic, shorter outputs.
# Fallback: if your GPU is 4 GB and very tight, try load_in_8bit=True or offload some layers to CPU (device_map="auto" keeps you safe).

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"  # primary pick
DEVICE_MAP = "auto"                        # lets HF place layers to GPU/CPU
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# --- VRAM-friendly load (INT4) ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map=DEVICE_MAP,
    torch_dtype=DTYPE,
    load_in_4bit=True,                 # quantize to 4-bit
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# --- Practical image resizing to cap visual tokens ---
# For ~6 GB VRAM, keeping longest side ~896 px is a good starting point.
# For 4 GB, try 768 px. Increase only if you don't hit OOM.
MAX_SIDE = 896  # change to 768 for tighter VRAM; 1024+ if you have headroom

def load_and_downscale(img: Union[str, Image.Image]) -> Image.Image:
    im = Image.open(img).convert("RGB") if isinstance(img, str) else img.convert("RGB")
    w, h = im.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / float(max(w, h))
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return im

# --- Prompt: ask for strict OCR per image, JSON only, no hallucination ---
SYSTEM_INSTRUCTIONS = (
    "You are a careful OCR assistant. Read text from each image precisely.\n"
    "Output JSON only with this schema:\n"
    "{ 'images': [ { 'index': <int>, 'language_guess': <string>, 'text_lines': [<string>, ...] } ] }\n"
    "Rules: 1) Preserve casing/punctuation; 2) Use reading order if obvious; "
    "3) If a line is unclear, include best guess and mark with '(?)'; 4) Do not add text not visible."
)

def build_messages(pil_images: List[Image.Image], user_note: str = "OCR all images.") -> list:
    # Qwen-VL expects interleaved messages with image items and a text query
    content = []
    for idx, im in enumerate(pil_images):
        content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": f"[Image {idx}]"})
    content.append({"type": "text", "text": user_note})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTIONS}]},
        {"role": "user",   "content": content},
    ]

@torch.inference_mode()
def multi_image_ocr(image_paths: List[str], max_new_tokens: int = 800) -> dict:
    images = [load_and_downscale(p) for p in image_paths]
    messages = build_messages(images)

    # Apply chat template and collect vision inputs
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Qwen2.5-VL (trust_remote_code=True) accepts PIL images directly via kwargs
    # (the custom forward grabs them from 'images' kwarg). Pass the list as is.
    gen_ids = model.generate(
        **inputs,
        images=images,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
    )
    out = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    # Try to parse JSON; if the model wrapped text, attempt to extract JSON block.
    try:
        return json.loads(out)
    except Exception:
        # simple fallback: find the first {...} block
        import re
        m = re.search(r"\{.*\}", out, flags=re.DOTALL)
        return json.loads(m.group(0)) if m else {"raw": out}

if __name__ == "__main__":
    # Example usage: replace with your paths
    paths = [
        "samples/doc_page_1.png",
        "samples/doc_page_2.jpg",
        "samples/signboard_cn_en.png",
    ]
    result = multi_image_ocr(paths)
    print(json.dumps(result, ensure_ascii=False, indent=2))
