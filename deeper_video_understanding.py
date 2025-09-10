# deeper_video_understanding.py
# FrameSense AI
import math
import os
import gc
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from decord import VideoReader, cpu

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transcribe_audio import transcribe
from video_ocr import build_ocr_prompt, ocr_multi_images


# ---------------------------
# Configuration
# ---------------------------
MODEL_PATH = "AIDC-AI/Ovis2.5-2B"

# Chunking & sampling
CHUNK_SECONDS = 20    # 20–30
OVERLAP_SECONDS = 5
FRAMES_PER_CHUNK = 4  # 12–16
BURST = 2
STRIDE = 2
GLOBAL_KEYFRAMES = 12              # extra global frames for the final pass (optional)

# MiniCPM runtime params
USE_IMAGE_ID = False
MAX_SLICE_NUMS = 8                  # try 6–8 on 6 GB GPUs                
GEN_KWARGS = {
    "max_new_tokens": 320,
    "temperature": 0.2,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
}

# Thinking mode & budget
enable_thinking = True
enable_thinking_budget = True  # Only effective if enable_thinking is True.

# Total tokens for thinking + answer. Ensure: max_new_tokens > thinking_budget + 25
max_new_tokens = 3072
thinking_budget = 2048

def _load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda()
    return model


# ---------------------------
# Utilities
# ---------------------------
def seconds_to_mmss(t: float) -> str:
    t = max(0, int(round(t)))
    m, s = divmod(t, 60)
    return f"{m:02d}:{s:02d}"

def sample_uniform_inclusive(n_total: int, n_want: int) -> List[int]:
    """Evenly spaced indices from [0, n_total-1], including endpoints."""
    if n_want <= 0:
        return []
    if n_want >= n_total:
        return list(range(n_total))
    return [int(round(i * (n_total - 1) / (n_want - 1))) for i in range(n_want)]

def sample_uniform_with_bursts(n_total: int, n_want: int) -> List[int]:
    """
    Uniform anchors + local bursts to preserve micro-motion.
    Example: n_want=16, burst=3 → ~5 anchors, each with a small +/- neighborhood.
    """
    if n_want <= 0:
        return []
    if n_want >= n_total:
        return list(range(n_total))

    anchors = sample_uniform_inclusive(n_total, max(1, n_want // BURST))
    picks = []
    half = BURST // 2
    for a in anchors:
        for k in range(-half, half + 1):
            idx = a + k * STRIDE
            if 0 <= idx < n_total:
                picks.append(idx)
    picks = sorted(set(picks))
    # Thin to exactly n_want if overshooting
    if len(picks) > n_want:
        sel = sample_uniform_inclusive(len(picks), n_want)
        picks = [picks[i] for i in sel]
    return picks

def resize_keep_ar(img: Image.Image, long_side: int = 448) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= long_side:
        return img
    scale = long_side / float(m)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.BILINEAR)

def indices_for_range(start_idx: int, end_idx: int, n_want: int) -> List[int]:
    """Indices within [start_idx, end_idx] inclusive, with bursts."""
    n_total = max(1, end_idx - start_idx + 1)
    rel = sample_uniform_with_bursts(n_total, min(n_want, n_total))
    return [start_idx + r for r in rel]

# def chunk_frame_ranges(n_total_frames: int, fps: float, chunk_seconds: int) -> List[Tuple[int, int, float, float]]:
#     """
#     Returns a list of (start_idx, end_idx, start_sec, end_sec) per chunk.
#     end_idx is inclusive.
#     """
#     total_seconds = n_total_frames / max(1e-6, fps)
#     n_chunks = int(math.ceil(total_seconds / chunk_seconds))
#     ranges = []
#     for c in range(n_chunks):
#         cs = c * chunk_seconds
#         ce = min((c + 1) * chunk_seconds, total_seconds)
#         s_idx = int(round(cs * fps))
#         e_idx = min(n_total_frames - 1, int(round(ce * fps)) - 1)
#         if e_idx < s_idx:
#             e_idx = s_idx
#         ranges.append((s_idx, e_idx, cs, ce))
#     print(f"[Chunking] Total chunks: {len(ranges)}")
#     return ranges


import math
from typing import List, Tuple

def chunk_frame_ranges(
    n_total_frames: int,
    fps: float,
    chunk_seconds: float,
    overlap_seconds: float = 0.0,
) -> List[Tuple[int, int, float, float]]:
    """
    Return a list of (start_idx, end_idx, start_sec, end_sec) per chunk.
    - Chunks are time windows of length `chunk_seconds`.
    - Consecutive chunks overlap by `overlap_seconds`.
    - `end_idx` is inclusive.

    Example (fps=30, chunk_seconds=20, overlap_seconds=5):
      [0–20], [15–35], [30–50], ... (seconds)
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be > 0")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be >= 0")
    if overlap_seconds >= chunk_seconds:
        # Prevent zero or negative stride which would loop forever
        raise ValueError("overlap_seconds must be < chunk_seconds")

    total_seconds = n_total_frames / max(1e-6, fps)
    stride = chunk_seconds - overlap_seconds  # forward step between chunk starts

    ranges: List[Tuple[int, int, float, float]] = []
    cs = 0.0  # chunk start (seconds)

    while cs < total_seconds:
        ce = min(cs + chunk_seconds, total_seconds)  # chunk end (seconds)
        s_idx = int(round(cs * fps))
        # end_idx is inclusive -> subtract 1 frame at the boundary
        e_idx = min(n_total_frames - 1, int(round(ce * fps)) - 1)
        if e_idx < s_idx:
            e_idx = s_idx

        ranges.append((s_idx, e_idx, cs, ce))
        cs += stride  # advance by stride (allows overlap)

        # Safety: float drift guard (optional)
        if len(ranges) > 1_000_000:
            raise RuntimeError("Too many chunks; check parameters for stride/overlap.")

        print(f"[Chunking] Total chunks: {len(ranges)}  "
          f"(chunk={chunk_seconds}s, overlap={overlap_seconds}s, stride={stride}s)")
    return ranges


def get_frames(vr: VideoReader, frame_idx: List[int]) -> Tuple[List[Image.Image], List[float]]:
    frames_np = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")) for v in frames_np]
    # timestamps in seconds (approx)
    fps = float(vr.get_avg_fps()) or 1.0
    timestamps = [i / max(1e-6, fps) for i in frame_idx]
    return frames, timestamps


# ---------------------------
# Optional ASR (Whisper)
# ---------------------------


def transcript_for_range(transcript: dict, start_s: float, end_s: float) -> str:
    if not transcript or "segments" not in transcript:
        return ""
    parts = []
    for seg in transcript["segments"]:
        s, e, txt = float(seg["start"]), float(seg["end"]), str(seg["text"]).strip()
        # keep seg if it overlaps with the chunk window
        if e >= start_s and s <= end_s and txt:
            parts.append(txt)
    return " ".join(parts).strip()


# ---------------------------
# MiniCPM-V prompts
# ---------------------------
def build_chunk_prompt(start_s: float, end_s: float, asr_excerpt: str = "", ocr_excerpt: str = "") -> str:
    st, et = seconds_to_mmss(start_s), seconds_to_mmss(end_s)

    # (Optional) Trim very long excerpts to keep the prompt lean.
    def _trim(txt, lim=800):
        return (txt[:lim] + " …") if len(txt) > lim else txt

    base = (
        f"You are a careful video analyst. The frames come from {st}–{et}.\n"
        "Return ONLY the sections below, nothing else.\n\n"
        "## 1) Summary (2–4 bullets)\n"
        "- Use short, factual bullets describing what happens.\n"
        "- Mention actions and transitions.\n\n"
        "## 2) Key objects/people\n"
        "- List important objects/people and their roles.\n\n"
        "## 3) On-screen text (verbatim)\n"
        "- Quote exactly as seen; if none, write: None.\n\n"
        "## 4) Uncertainties\n"
        "- Explicitly state what is unclear/ambiguous.\n\n"
        "Be concise and specific.\n"
        "Do not invent details. If you are unsure, say 'uncertain'.\n"
        "End your response with: Final answer:"
    )

    if asr_excerpt:
        base += f'\n\nASR excerpt (verbatim context):\n"{_trim(asr_excerpt)}"'

    if ocr_excerpt:
        base += (
            "\n\nOCR evidence (verbatim; prefer this for exact text/numbers):\n"
            f"{_trim(ocr_excerpt)}"
        )
    return base


def build_global_prompt(chunk_summaries: List[str]) -> str:
    joined = "\n\n---\n\n".join(chunk_summaries)
    return (
        "You are aggregating analyses from multiple time ranges of one video.\n"
        "Based on the chunk summaries below, produce:\n"
        "A) A 4–7 sentence overall summary.\n"
        "B) A consolidated timeline with approximate timestamps (mm:ss) for key events.\n"
        "C) Main entities and their roles.\n"
        "D) Notable actions, numbers, and on-screen text (verbatim where possible).\n"
        "E) Open questions or ambiguities.\n"
        "Be faithful to the evidence and avoid speculation.\n\n"
        "Chunk summaries:\n"
        f"{joined}"
    )


# ---------------------------
# MiniCPM-V chat wrapper
# ---------------------------
def minicpm_chat(model, tokenizer, frames: List[Image.Image], prompt: str, max_slice_nums: int) -> str:
    # MiniCPM-V expects msgs = [{'role': 'user', 'content': frames + [prompt]}]
    frames = [resize_keep_ar(f, long_side=448) for f in frames]
    msgs = [{"role": "user", "content": frames + [prompt]}]
    params = {"use_image_id": USE_IMAGE_ID, "max_slice_nums": max_slice_nums}
    
    try:
        out = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, **GEN_KWARGS, **params)
        return out if isinstance(out, str) else str(out)
    except RuntimeError as e:
        # Simple OOM fallback: reduce slice count and retry once
        if "out of memory" in str(e).lower() and max_slice_nums > 1:
            torch.cuda.empty_cache()
            params["max_slice_nums"] = max(1, max_slice_nums - 1)
            out = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, **GEN_KWARGS, **params)
            return out if isinstance(out, str) else str(out)
        raise



def get_video_info(vr):
    total_frames = len(vr)
    fps = float(vr.get_avg_fps()) or 1.0
    duration = total_frames / max(1e-6, fps)
    print(f"[Video] Frames: {total_frames}, FPS: {fps:.2f}, Duration: {seconds_to_mmss(duration)}")
    return {"frames": total_frames, "fps": fps, "duration_s": duration}


def analyze_video(video_path: str):
    # Device & dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    dtype = torch.float16  # instead of bfloat16 for widest GPU support
    print(f"[Init] Device: {device}, dtype: {dtype}")

    # Load model / tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        attn_implementation="sdpa",
        dtype=dtype,
    )
    
    model = model.eval().to(device)

    video_info = get_video_info(video_path)
    total_frames = video_info["frames"]
    fps = video_info["fps"]
    duration = video_info["duration_s"]

    # Optional ASR
    transcript = transcribe(video_path)

    # Build chunks
    ranges = chunk_frame_ranges(total_frames, fps, CHUNK_SECONDS)
    print(f"[Chunking] {len(ranges)} chunk(s) of ~{CHUNK_SECONDS}s each")

    chunk_summaries = []
    for idx, (s_idx, e_idx, s_sec, e_sec) in enumerate(ranges, 1):
        # Sample frames for this chunk
        fidx = indices_for_range(s_idx, e_idx, FRAMES_PER_CHUNK)
        frames, ts = get_frames(vr, fidx)

        print(f"[Chunk {idx}] Frames: {len(frames)}")

        prompt = build_ocr_prompt(ts)
        ocr_excerpt = ocr_multi_images(model, tokenizer, frames, prompt)

        asr_excerpt = transcript_for_range(transcript, s_sec, e_sec) if transcript else ""

        # Build and run chunk prompt
        prompt = build_chunk_prompt(s_sec, e_sec, asr_excerpt=asr_excerpt, ocr_excerpt=ocr_excerpt)
        print(f"[Chunk {idx}] Frames: {len(frames)} | {seconds_to_mmss(s_sec)}–{seconds_to_mmss(e_sec)}")
        local_res = minicpm_chat(model, tokenizer, frames, prompt, MAX_SLICE_NUMS)

        # Save a compact header + response
        header = f"### Chunk {idx} ({seconds_to_mmss(s_sec)}–{seconds_to_mmss(e_sec)})"
        chunk_summaries.append(f"{header}\n{local_res}")
        
        del frames  # and any big tensors
        gc.collect()
        torch.cuda.empty_cache()

    # Optional: add a small set of global keyframes to help the final pass
    global_frames = []
    if GLOBAL_KEYFRAMES > 0 and total_frames > 0:
        g_idx = sample_uniform_inclusive(total_frames, min(GLOBAL_KEYFRAMES, total_frames))
        global_frames, _ = get_frames(vr, g_idx)
        print(f"[Global] Added {len(global_frames)} keyframes for final aggregation")

    # Global (reduce) pass
    global_prompt = build_global_prompt(chunk_summaries)
    final_res = minicpm_chat(model, tokenizer, global_frames, global_prompt, MAX_SLICE_NUMS)

    # Print results
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(final_res)

    # (Optional) Save to file
    out_md = os.path.splitext(os.path.basename(video_path))[0] + "_analysis.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Video Analysis\n\n")
        f.write(f"- File: {video_path}\n")
        f.write(f"- Duration: {seconds_to_mmss(duration)} (fps {fps:.2f}, frames {total_frames})\n\n")
        f.write("## Chunk Analyses\n\n")
        for cs in chunk_summaries:
            f.write(cs + "\n\n")
        f.write("## Final Summary\n\n")
        f.write(final_res + "\n")
    print(f"[Saved] {out_md}")

def describe_video(model, frames, prompt) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": frames},   # chronological order
            {"type": "text",  "text": prompt},
        ]
    }]

    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        max_pixels=896*896,  # keep memory & latency predictable
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    pixel_values = pixel_values.to(device, dtype=model.dtype) if pixel_values is not None else None
    grid_thws = grid_thws.to(device) if grid_thws is not None else None

    # For reproducible outputs, you can fix a seed (optional):
    # torch.manual_seed(42)

    # If you prefer some creativity in wording, set do_sample=True with temperature=0.7, top_p=0.9; keep repetition_penalty=1.05. 
    # For production summaries, I recommend do_sample=False.

    gen_kwargs = dict(
        max_new_tokens=768,             # ample for 4 sections + bullets
        do_sample=False,                # determinism for pipelines
        temperature=None,               # ignored when do_sample=False
        top_p=None,                     # ignored when do_sample=False
        repetition_penalty=1.05,        # curb loops on small models
        eos_token_id=model.text_tokenizer.eos_token_id,
        pad_token_id=model.text_tokenizer.pad_token_id,
    )

    with torch.inference_mode():
        outputs = model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            **gen_kwargs
        )

    text = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract up to the end marker if present
    end_marker = "Final answer:"
    if end_marker in text:
        text = text.split(end_marker, 1)[0].rstrip() + "\n" + end_marker

    # Optional: print for quick debugging
    # print(f"\n{text}\n")

    return text


def ocr_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    video_info = get_video_info(vr)
    total_frames = video_info["frames"]
    fps = video_info["fps"]
    duration = video_info["duration_s"]

    ranges = chunk_frame_ranges(total_frames, fps, CHUNK_SECONDS, OVERLAP_SECONDS)
    chunk_summaries = []
    for idx, (s_idx, e_idx, s_sec, e_sec) in enumerate(ranges, 1):
        fidx = indices_for_range(s_idx, e_idx, FRAMES_PER_CHUNK)
        frames, ts = get_frames(vr, fidx)
        
        ocr_excerpt = ocr_multi_images(frames)
        if ocr_excerpt:
            mapping_lines = [f"{i}: {seconds_to_mmss(t)}" for i, t in enumerate(ts)]
            mapping = "\n".join(mapping_lines)
            ocr_excerpt = mapping + "\n" + ocr_excerpt

        chunk_summaries.append({
            "chunk_idx": idx,
            "start_sec": s_sec,
            "end_sec": e_sec,
            "frame_indices": fidx,
            "timestamps": ts,
            "description": ocr_excerpt,
        })
        del frames
        gc.collect()
        torch.cuda.empty_cache()

    return chunk_summaries


def vlm_ocr_asr(video_path, segments):
    vr = VideoReader(video_path, ctx=cpu(0))
    video_info = get_video_info(vr)
    total_frames = video_info["frames"]
    fps = video_info["fps"]
    duration = video_info["duration_s"]

    ranges = chunk_frame_ranges(total_frames, fps, CHUNK_SECONDS, OVERLAP_SECONDS)
    model = _load_model()
    chunk_summaries = []

    for idx, (s_idx, e_idx, s_sec, e_sec) in enumerate(ranges, 1):
        # Sample frames for this chunk
        fidx = indices_for_range(s_idx, e_idx, FRAMES_PER_CHUNK)
        frames, ts = get_frames(vr, fidx)

        print(f"[Chunk {idx}] Frames: {len(frames)}")
        
        ocr_excerpt = ocr_multi_images(frames)
        if ocr_excerpt:
            mapping_lines = [f"{i}: {seconds_to_mmss(t)}" for i, t in enumerate(ts)]
            ocr_excerpt = "\n".join(mapping_lines) + "\n" + ocr_excerpt

        asr_excerpt = transcript_for_range(segments, s_sec, e_sec) if segments else ""

        # Build and run chunk prompt
        prompt = build_chunk_prompt(s_sec, e_sec, asr_excerpt=asr_excerpt, ocr_excerpt=ocr_excerpt)
        print(f"[Chunk {idx}] Frames: {len(frames)} | {seconds_to_mmss(s_sec)}–{seconds_to_mmss(e_sec)}")
        desc = describe_video(model, frames, prompt)

        chunk_summaries.append({
            "chunk_idx": idx,
            "start_sec": s_sec,
            "end_sec": e_sec,
            "frame_indices": fidx,
            "timestamps": ts,
            "description": desc,
        })
        
        del frames  # and any big tensors
        gc.collect()
        torch.cuda.empty_cache()

    return chunk_summaries

def qwen_ocr_multi_images(frames: List[Image.Image]) -> str:
    from qwen_2_5_VL import multi_image_understanding
    return multi_image_understanding(frames)

# ---------------------------
# Entry
# ---------------------------
if __name__ == "__main__":
    # Example: change to your file path
    video_path = "badminton.mp4"   # e.g., "badminton.mp4" Intellegent.mp4
    vr = VideoReader(video_path, ctx=cpu(0))
    video_info = get_video_info(vr)
    total_frames = video_info["frames"]
    fps = video_info["fps"]
    duration = video_info["duration_s"]

    ranges = chunk_frame_ranges(total_frames, fps, CHUNK_SECONDS, OVERLAP_SECONDS)
    model = _load_model()

    for idx, (s_idx, e_idx, s_sec, e_sec) in enumerate(ranges, 1):
        # Sample frames for this chunk
        fidx = indices_for_range(s_idx, e_idx, FRAMES_PER_CHUNK)
        frames, ts = get_frames(vr, fidx)
        desc = qwen_ocr_multi_images(frames)
        print(f"[Chunk {idx}] Description: {desc}")


