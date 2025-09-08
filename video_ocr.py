#!/usr/bin/env python3
"""
Video-frame OCR with MiniCPM-V-4_5 (multi-images code path)

Dependencies:
  pip install decord pillow transformers accelerate sentencepiece torch

Tip for 6 GB GPUs:
  - Resize frames to <= 448 px long side (already done below).
  - Use fp16 on CUDA.
  - Keep 6–10 frames per call; clear VRAM between calls if you loop.
"""

import re
import argparse
import json
from typing import List, Tuple
import pytesseract
import cv2

import numpy as np
from PIL import Image
from decord import VideoReader, cpu
import torch
from transformers import AutoModel, AutoTokenizer


# ----------------------------
# Utilities
# ----------------------------
def seconds_to_mmss(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t//60:02d}:{t%60:02d}"

def sample_uniform_inclusive(n_total: int, n_want: int) -> List[int]:
    """Evenly spaced indices from [0, n_total-1], including endpoints."""
    if n_want <= 0:
        return []
    if n_want >= n_total:
        return list(range(n_total))
    return [int(round(i * (n_total - 1) / (n_want - 1))) for i in range(n_want)]

def resize_keep_ar(img: Image.Image, long_side: int = 448) -> Image.Image:
    """Resize keeping aspect ratio so max(H, W) == long_side (or smaller)."""
    w, h = img.size
    m = max(w, h)
    if m <= long_side:
        return img
    scale = long_side / float(m)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.BILINEAR)


def _pick_important_ocr_lines(lines: List[str], max_lines: int = 6) -> List[str]:
    """Keep lines likely to matter: digits, ALL-CAPS tokens, URLs/emails. Dedup + score."""
    if not lines:
        return []
    def score(line: str) -> int:
        s = 0
        if re.search(r"\d", line): s += 3
        if re.search(r"https?://|www\.", line): s += 2
        s += min(2, sum(1 for w in re.findall(r"\b[A-Z0-9]{3,}\b", line)))
        if "@" in line or "#" in line: s += 1
        return s
    # normalize & dedup
    uniq, seen = [], set()
    for ln in lines:
        ln = " ".join((ln or "").split())
        if ln and ln not in seen:
            seen.add(ln); uniq.append(ln)
    uniq.sort(key=score, reverse=True)
    return uniq[:max_lines]

def ocr_excerpt_from_item(ocr_item: dict, max_lines_per_chunk: int = 6) -> str:
    """Build a short, timestamped OCR excerpt string from one chunk's OCR result."""
    lines: List[str] = []
    if "json" in ocr_item and isinstance(ocr_item["json"], dict):
        for f in ocr_item["json"].get("frames", []):
            ts = f.get("timestamp") or ""
            for ln in (f.get("lines") or []):
                ln = " ".join((ln or "").split())
                if ln:
                    lines.append(f"[{ts}] {ln}")
    else:
        raw = ocr_item.get("raw", "")
        for ln in raw.splitlines():
            ln = " ".join(ln.split())
            if ln:
                lines.append(ln)
    picks = _pick_important_ocr_lines(lines, max_lines=max_lines_per_chunk)
    return "\n".join(picks)

# ----------------------------
# Frame extraction
# ----------------------------
def extract_frames_for_ocr(
    video_path: str,
    total_frames: int = 8,
    long_side: int = 448
) -> Tuple[List[Image.Image], List[int], List[float]]:
    """
    Returns:
      frames  : list[PIL.Image] resized for VLM
      indices : list[int]      selected frame indices
      stamps  : list[float]    timestamps in seconds
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    n = len(vr)
    if n == 0:
        return [], [], []

    # Sample uniformly across the entire video (includes first & last)
    num = min(max(1, total_frames), n)
    indices = sample_uniform_inclusive(n, num)

    # Decode and resize
    batch = vr.get_batch(indices).asnumpy()  # (num, H, W, 3), uint8
    frames = [resize_keep_ar(Image.fromarray(x.astype("uint8")), long_side=long_side) for x in batch]

    fps = float(vr.get_avg_fps()) or 25.0
    stamps = [round(i / fps, 2) for i in indices]
    return frames, indices, stamps


# ----------------------------
# OCR via MiniCPM-V-4_5 (multi-images)
# ----------------------------
def load_minicpm_v(model_id: str = "openbmb/MiniCPM-V-4_5"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="sdpa",   # or "flash_attention_2" if installed
        dtype=dtype,
    ).eval().to(device)

    return tokenizer, model

def build_ocr_prompt(frame_times: List[float]) -> str:
    """
    Ask the model to return OCR grouped by frame.
    We provide the (index -> timestamp) mapping in the prompt so the model can label output.
    """
    mapping_lines = [f"{i}: {seconds_to_mmss(t)}" for i, t in enumerate(frame_times)]
    mapping = "\n".join(mapping_lines)

    prompt = (
        "You are an OCR engine. The images are frames from a video, in chronological order.\n"
        "For EACH frame, extract ALL readable on-screen text VERBATIM (keep case, punctuation). "
        "If text is not clearly readable, write '[unclear]'.\n\n"
        "Return a strict JSON object with this schema:\n"
        "{\n"
        '  "frames": [\n'
        '    {"index": <int>, "timestamp": "MM:SS", "lines": ["text line 1", "text line 2", ...]},\n'
        "    ...\n"
        "  ]\n"
        "}\n\n"
        "Frame index to timestamp mapping:\n"
        f"{mapping}\n\n"
        "Only output JSON. Do not add explanations."
    )
    return prompt

def ocr_multi_images(
    model,
    tokenizer,
    frames: List[Image.Image],
    prompt: str,
    max_slice_nums: int = 8,
    max_new_tokens: int = 512
) -> str:
    """
    Multi-images code path: pass a list of PIL Images + one prompt.
    """
    msgs = [{"role": "user", "content": frames + [prompt]}]
    with torch.inference_mode():
        out = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
        )

    out = out if isinstance(out, str) else str(out)
    ocr_excerpt = ocr_excerpt_from_item(out)
    if ocr_excerpt:
        print(f"[OCR] Text extracted: {ocr_excerpt})")
    return ocr_excerpt

def _preprocess_for_ocr(img_pil: Image.Image) -> np.ndarray:
    """Lightweight preprocessing that usually helps Tesseract."""
    arr = np.array(img_pil)  # RGB uint8
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr

    # Upscale small frames a bit (helps small text)
    h, w = gray.shape[:2]
    if max(h, w) < 480:
        scale = 480.0 / max(h, w)
        gray = cv2.resize(gray, (int(round(w*scale)), int(round(h*scale))), interpolation=cv2.INTER_CUBIC)

    # Gentle denoise that preserves edges, then Otsu binarization
    gray = cv2.bilateralFilter(gray, 7, 75, 75)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def ocr_multi_images(
    frames: List[Image.Image],
) -> str:
    """
    Fast OCR across multiple PIL images using pytesseract.
    Returns a short 'excerpt' of the most useful lines across frames.
    """
    all_lines: List[str] = []
    for img in frames:
        proc = _preprocess_for_ocr(img)
        text = pytesseract.image_to_string(proc)
        # Split into non-empty lines
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        all_lines.extend(lines)

    # Keep a compact, high-signal excerpt
    excerpt_lines = _pick_important_ocr_lines(all_lines, max_lines=12)
    ocr_excerpt = "\n".join(excerpt_lines)

    # if ocr_excerpt:
    #     print(f"[OCR] Text extracted:\n{ocr_excerpt}")
    # else:
    #     print("[OCR] No readable text found.")

    return ocr_excerpt


# ----------------------------
# Main (CLI)
# ----------------------------
def main():
    ap = argparse.ArgumentParser("Video-frame OCR with MiniCPM-V-4_5 (multi-images)")
    ap.add_argument("video", help="Path to video file")
    ap.add_argument("--frames", type=int, default=8, help="Total frames to sample (6–10 recommended on 6GB)")
    ap.add_argument("--long-side", type=int, default=448, help="Resize long side for low VRAM")
    ap.add_argument("--max-slice-nums", type=int, default=8, help="MiniCPM-V slice count (raise if OOM)")
    ap.add_argument("--max-new", type=int, default=512, help="Max new tokens for OCR output")
    args = ap.parse_args()

    # 1) Sample frames
    frames, idxs, times = extract_frames_for_ocr(args.video, total_frames=args.frames, long_side=args.long_side)
    if not frames:
        print("No frames extracted.")
        return
    print(f"[Info] Sampled {len(frames)} frames at indices {idxs}")
    print("[Info] Timestamps (s):", times)
    print("[Info] Timestamps (mm:ss):", [seconds_to_mmss(t) for t in times])

    # 2) Load model
    tokenizer, model = load_minicpm_v()

    # 3) Build OCR prompt (JSON output grouped by frame)
    prompt = build_ocr_prompt(times)

    # 4) Run OCR (multi-images)
    try:
        result = ocr_multi_images(
            model,
            tokenizer,
            frames,
            prompt,
            max_slice_nums=args.max_slice_nums,
            max_new_tokens=args.max_new,
        )
    except RuntimeError as e:
        # Simple OOM fallback: try more slices; if still bad, suggest fewer frames or smaller long_side
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            print("[Warn] OOM occurred. Retrying with higher max_slice_nums...")
            result = ocr_multi_images(
                model, tokenizer, frames, prompt,
                max_slice_nums=min(16, args.max_slice_nums * 2),
                max_new_tokens=args.max_new,
            )
        else:
            raise

    # 5) Print raw output and try to pretty-print JSON
    print("\n==== RAW MODEL OUTPUT ====\n")
    print(result)

    print("\n==== PARSED JSON (if valid) ====\n")
    try:
        data = json.loads(result)
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception:
        print("[Note] Output was not strict JSON; keep the raw text above.")

if __name__ == "__main__":
    main()
