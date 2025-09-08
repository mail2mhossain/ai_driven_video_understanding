import os, re, math, torch
from dataclasses import dataclass
import pytesseract
from typing import List, Tuple, Optional, Any
from faster_whisper.transcribe import Segment
import numpy as np
import cv2
from transcribe_audio import transcribe, compute_asr_metrics

# ---------- Metrics container ----------
@dataclass
class ProbeMetrics:
    duration_s: float
    speech_ratio: float       # fraction of time with detected speech
    wpm: float                # words per minute (approx)
    ocr_density: float        # avg non-space chars per sampled frame
    ocr_digits: int           # total digits found in OCR sample
    motion_index: float       # mean frame-to-frame diff (0..~100+)
    visual_keywords_hits: int # count of keywords in ASR text
    segments: List[Segment]  # Whisper result dict

VISUAL_KWS = [
    "as you can see", "shown here", "look at", "see here", "this slide",
    "chart", "graph", "plot", "diagram", "figure", "table", "screenshot",
    "demo", "demonstrate", "gesture", "points", "on screen", "accuracy", "fps", "percentage"
]
KW_REGEX = re.compile("|".join([re.escape(k) for k in VISUAL_KWS]), re.IGNORECASE)

def _mmss(t: float) -> str:
    t = max(0, int(round(t))); return f"{t//60:02d}:{t%60:02d}"

# ---------- Quick ASR probe (first window) ----------
def asr_probe(segments, window_s: int = 90, lang: Optional[str] = None):
    """
    Returns (speech_ratio, wpm, concat_text, duration_used)
    Uses faster-whisper if installed; else returns zeros.
    """
    
    segs = []
    t_end = 0.0
    for s in segments:
        t_end = max(t_end, float(s.end))
        if s.start <= window_s:
            segs.append((float(s.start), float(s.end), s.text))
    if not segs:
        return 0.0, 0.0, "", min(window_s, t_end or window_s)
    used_dur = min(window_s, t_end or window_s)
    speech = sum(max(0.0, min(e, used_dur) - max(0.0, s)) for s, e, _ in segs)
    text = " ".join(t for _,_,t in segs).strip()
    words = len(text.split())
    speech_ratio = float(speech) / max(1.0, used_dur)
    wpm = (words / max(1.0, used_dur)) * 60.0

    print(f"[ASR] Speech ratio: {speech_ratio}, WPM: {wpm}, Text: {text}, Used duration: {used_dur}")
    return speech_ratio, wpm, text, used_dur


# ---------- OCR probe on k frames ----------
def ocr_probe(video_path: str, k: int = 6) -> Tuple[float, int]: 
    print(f"[OCR] Probing {video_path} with {k} frames...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps*60)
    if frames <= 0: return 0.0, 0
    idxs = [int(round(i*(frames-1)/(k-1))) for i in range(k)] if k>1 else [0]
    total_chars, total_digits = 0, 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        txt = pytesseract.image_to_string(gray)
        txt = "".join(txt.split())
        total_chars += len(txt)
        total_digits += sum(ch.isdigit() for ch in txt)
    cap.release()
    density = total_chars / max(1, k)
    print(f"[OCR] Density: {density}, Digits: {total_digits}")
    return float(density), int(total_digits)

# ---------- Motion probe (fast) ----------
def motion_probe(video_path: str, stride_s: float = 0.75) -> Tuple[float, float]:
    print(f"[Motion] Probing {video_path} with stride {stride_s}...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frames / fps if frames>0 else 0.0
    step = max(1, int(round(fps * stride_s)))
    idxs = list(range(0, frames, step))[:400]  # cap work
    prev = None
    diffs = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok: break
        small = cv2.resize(frame, (160, 90))
        small = small.astype("float32") / 255.0
        if prev is not None:
            diffs.append(float(np.mean(np.abs(small - prev))) * 100.0)
        prev = small
    cap.release()
    return (float(np.mean(diffs)) if diffs else 0.0), duration

# ---------- Master probe ----------
def probe_video(video_path: str, lang: Optional[str] = None) -> ProbeMetrics:
    segments = transcribe(video_path, language=lang)
    speech_ratio, wpm, text, used = compute_asr_metrics(segments)
    ocr_density, ocr_digits = ocr_probe(video_path, k=6)
    motion_idx, duration = motion_probe(video_path, stride_s=0.75)
    vk_hits = len(re.findall(KW_REGEX, text or ""))

    return ProbeMetrics(
        duration_s=duration or used,
        speech_ratio=speech_ratio,
        wpm=wpm,
        ocr_density=ocr_density,
        ocr_digits=ocr_digits,
        motion_index=motion_idx,
        visual_keywords_hits=vk_hits,
        segments=segments,
    )

# ---------- Decision ----------
def decide_pipeline(video_path: str) -> str:
    """
    Returns one of: 'ASR_ONLY', 'ASR_OCR', 'ASR_OCR_VLM', 'VLM_OCR'
    """
    print(f"[Decision] Deciding pipeline ...")
    metrics = probe_video(video_path, lang="en") 
    speech, wpm = metrics.speech_ratio, metrics.wpm
    ocr_dense = metrics.ocr_density >= 12 or metrics.ocr_digits >= 4
    visual_cues = (metrics.visual_keywords_hits >= 1)
    high_motion = metrics.motion_index >= 6.0  # scale is ~0..20+ depending on content

    # Visual-first
    if speech < 0.15 and (high_motion or ocr_dense):
        return ("VLM_OCR", None)  # MiniCPM-V + OCR, no ASR

    # Speech-first
    if speech >= 0.40 and wpm >= 80:
        return ("ASR_OCR", metrics.segments) if ocr_dense else ("ASR_ONLY", metrics.segments)

    # Mixed
    if (0.15 <= speech < 0.40) or visual_cues or ocr_dense:
        return ("ASR_OCR_VLM", metrics.segments)

    # Fallback
    return ("ASR_OCR", metrics.segments)

# ---------- Example usage ----------
if __name__ == "__main__":
    video_path = "badminton.mp4"
    decision, segments = decide_pipeline(video_path)
    # print(m)
    print("Decision:", decision)
