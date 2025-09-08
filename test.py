import gc
import torch
from moviepy.editor import VideoFileClip  # pip install moviepy==1.0.3
from decord import VideoReader, cpu
from PIL import Image
from typing import List, Tuple, Optional
from transformers import AutoModelForCausalLM
from text_summarizer import summarize_text

MODEL_PATH = "AIDC-AI/Ovis2.5-2B"
CHUNK_SECONDS = 20    # 20–30
FRAMES_PER_CHUNK = 8  # 12–16
BURST = 2
STRIDE = 2

# Thinking mode & budget
enable_thinking = True
enable_thinking_budget = True  # Only effective if enable_thinking is True.

# Total tokens for thinking + answer. Ensure: max_new_tokens > thinking_budget + 25
max_new_tokens = 3072
thinking_budget = 2048

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    trust_remote_code=True
).cuda()

video_file = "badminton.mp4"


# def describe_video(frames, prompt) -> str:
#     messages = [{"role": "user", "content": [
#         {"type": "video", "video": frames},
#         {"type": "text", "text": prompt},
#     ]}]

#     input_ids, pixel_values, grid_thws = model.preprocess_inputs(messages=messages, add_generation_prompt=True, max_pixels=896*896)
#     input_ids = input_ids.cuda()
#     pixel_values = pixel_values.cuda().to(model.dtype) if pixel_values is not None else None
#     grid_thws = grid_thws.cuda() if grid_thws is not None else None

#     with torch.no_grad():
#         outputs = model.generate(inputs=input_ids, pixel_values=pixel_values, grid_thws=grid_thws,
#                                 max_new_tokens=1024, do_sample=True,
#                                 eos_token_id=model.text_tokenizer.eos_token_id,
#                                 pad_token_id=model.text_tokenizer.pad_token_id)
#     description = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(f"\n{description}\n")
#     return description

# import torch

def describe_video(frames, prompt) -> str:
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


from deeper_video_understanding import (
    chunk_frame_ranges, 
    indices_for_range, 
    get_frames, 
    get_video_info,
    build_chunk_prompt,
    seconds_to_mmss
)

vr = VideoReader(video_file, ctx=cpu(0))
video_info = get_video_info(vr)
total_frames = video_info["frames"]
fps = video_info["fps"]
duration = video_info["duration_s"]

ranges = chunk_frame_ranges(total_frames, fps, CHUNK_SECONDS)
print(f"[Chunking] {len(ranges)} chunk(s) of ~{CHUNK_SECONDS}s each")

def analyze_video():
    chunk_summaries = []
    for idx, (s_idx, e_idx, s_sec, e_sec) in enumerate(ranges, 1):
        # Sample frames for this chunk
        fidx = indices_for_range(s_idx, e_idx, FRAMES_PER_CHUNK)
        frames, ts = get_frames(vr, fidx)
        print(f"[Chunk {idx}] Frames: {len(frames)}")
        prompt = build_chunk_prompt(s_sec, e_sec, asr_excerpt=None, ocr_excerpt=None)
        desc = describe_video(frames, prompt)
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

chunk_summaries = analyze_video()
print("\n".join(chunk_summaries))

summary = summarize_text(chunk_summaries)
print(summary)
