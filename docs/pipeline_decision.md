Here’s a simple, reliable way to choose your pipeline **per video**:

## TL;DR decision tree

1. **Probe the video first (60–90 s or first 15%)**

   * Run **light ASR** (faster-whisper) → get *speech ratio* and *words/minute (WPM)*.
   * Run **OCR** on \~6 evenly spaced frames → get *OCR density* and *digit count*.
   * Measure **motion** via quick frame differences (every \~0.5–1 s).

2. **Pick a path**

* **ASR-first (ASR + OCR + text LLM)**
  Use when: `speech_ratio ≥ 0.40` **and** `WPM ≥ 80`.
  Add MiniCPM-V only for specific visual questions (numbers on slides, charts, actions).
* **Mixed (ASR + OCR → + targeted MiniCPM-V)**
  Use when: `0.15 ≤ speech_ratio < 0.40` **or** OCR shows many digits/slide text **or** transcript has visual keywords (“as you can see”, “chart”, “slide”).
  Call MiniCPM-V on **2–3 short windows** with **6–10 frames** each.
* **Visual-first (MiniCPM-V + OCR, no ASR)**
  Use when: `speech_ratio < 0.15` **and** (high motion **or** OCR dense).
  Do shot detection → sample 24–32 total frames → OCR → MiniCPM-V captions/QA → summarize.

> Default on a 6-GB GPU: start with **ASR + OCR + text LLM**; escalate to **targeted MiniCPM-V** only when the probe says visuals matter.

---

## Drop-in probe + decision code

Paste these helpers into your project; they return a decision and (if needed) timestamps to query with MiniCPM-V.

```python
import os, re, math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2

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

VISUAL_KWS = [
    "as you can see", "shown here", "look at", "see here", "this slide",
    "chart", "graph", "plot", "diagram", "figure", "table", "screenshot",
    "demo", "demonstrate", "gesture", "points", "on screen", "accuracy", "fps", "percentage"
]
KW_REGEX = re.compile("|".join([re.escape(k) for k in VISUAL_KWS]), re.IGNORECASE)

def _mmss(t: float) -> str:
    t = max(0, int(round(t))); return f"{t//60:02d}:{t%60:02d}"

# ---------- Quick ASR probe (first window) ----------
def asr_probe(video_path: str, window_s: int = 90, lang: Optional[str] = None):
    """
    Returns (speech_ratio, wpm, concat_text, duration_used)
    Uses faster-whisper if installed; else returns zeros.
    """
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("small", device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES","") != "-1" else "cpu",
                             compute_type="float16" if os.environ.get("CUDA_VISIBLE_DEVICES","") != "-1" else "int8")
        segments, info = model.transcribe(video_path, vad_filter=True, language=lang)
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
        return speech_ratio, wpm, text, used_dur
    except Exception:
        # No ASR lib; treat as no speech
        return 0.0, 0.0, "", float(window_s)

# ---------- OCR probe on k frames ----------
def ocr_probe(video_path: str, k: int = 6) -> Tuple[float, int]:
    try:
        import pytesseract
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
        return float(density), int(total_digits)
    except Exception:
        return 0.0, 0

# ---------- Motion probe (fast) ----------
def motion_probe(video_path: str, stride_s: float = 0.75) -> Tuple[float, float]:
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
    speech_ratio, wpm, text, used = asr_probe(video_path, window_s=90, lang=lang)
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
    )

# ---------- Decision ----------
def decide_pipeline(metrics: ProbeMetrics) -> str:
    """
    Returns one of: 'ASR_ONLY', 'ASR_OCR', 'ASR_OCR_VLM', 'VLM_OCR'
    """
    speech, wpm = metrics.speech_ratio, metrics.wpm
    ocr_dense = metrics.ocr_density >= 12 or metrics.ocr_digits >= 4
    visual_cues = (metrics.visual_keywords_hits >= 1)
    high_motion = metrics.motion_index >= 6.0  # scale is ~0..20+ depending on content

    # Visual-first
    if speech < 0.15 and (high_motion or ocr_dense):
        return "VLM_OCR"  # MiniCPM-V + OCR, no ASR

    # Speech-first
    if speech >= 0.40 and wpm >= 80:
        return "ASR_OCR" if ocr_dense else "ASR_ONLY"

    # Mixed
    if (0.15 <= speech < 0.40) or visual_cues or ocr_dense:
        return "ASR_OCR_VLM"

    # Fallback
    return "ASR_OCR"

# ---------- Example usage ----------
if __name__ == "__main__":
    path = "your_video.mp4"
    m = probe_video(path, lang=None)  # set lang="bn" to force Bangla
    decision = decide_pipeline(m)
    print(m)
    print("Decision:", decision)
```

### How to act on the decision

* **ASR\_ONLY** → Transcribe → summarize with a small text LLM.
* **ASR\_OCR** → Transcribe → OCR 6–8 frames → summarize (include OCR).
* **ASR\_OCR\_VLM** → Do the above **plus** 2–3 targeted MiniCPM-V calls at flagged times (6–10 frames each, 448 px long side, `fp16`, `max_slice_nums=6–12`).
* **VLM\_OCR** → Shot detection → sample 24–32 frames total → OCR → MiniCPM-V captions/QA → summarize.

---

## Practical thresholds (tune if needed)

* **speech\_ratio**: 0.40 (speech-heavy), 0.15 (speech-light)
* **WPM**: 80 (lecture/news), <40 often music/silence
* **OCR density**: ≥12 non-space chars/frame or ≥4 digits across 6 frames
* **Motion index**: ≥6 means noticeable action/transitions

---

## 6-GB GPU guardrails for MiniCPM-V (when used)

* **Frames per call**: 6–10
* **Total calls**: ≤3 per video
* **Resize**: long side ≤ **448 px**
* **Precision**: `torch.float16`
* **Slicing**: `max_slice_nums = 6–12`
* **Clear VRAM** between calls: `torch.cuda.empty_cache()`

This gives you an **automatic, explainable** way to decide when to rely on **ASR+OCR** and when to bring in **MiniCPM-V**.
