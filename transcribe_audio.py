import os
import torch
from typing import List, Tuple, Optional, Dict, Any
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

ASR_MODEL_SIZE = "medium"           # "tiny"/"base"/"small"/"medium"/"large/large-v3"
ASR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if ASR_DEVICE == "cuda" else "int8"

def transcribe(video_path: str, language: str = "en") -> List[Segment]:
    """ Internal function to transcribe audio using Whisper. """
    print(f"[ASR] Transcribing {video_path}...")
    model = WhisperModel(ASR_MODEL_SIZE, device=ASR_DEVICE, compute_type=COMPUTE_TYPE)
    segments, _ = model.transcribe(video_path, language=language, vad_filter=False, task="transcribe")
    segments = list(segments)
    # print(f"\nData Type: {type(segments[0])}")
    # print(f"\nData: {segments}")
    return segments

def get_extracted_asr(segments: List[Segment]) -> Optional[dict]:
    """
    Returns Whisper result dict with 'text' and 'segments' if available.
    If Whisper isn't installed, returns None gracefully.
    """
    print(f"[ASR] Extracting ASR from {len(segments)} segments...")
    out = {"text": "", "segments": []}
    text_accum = []
    for seg in segments:
        out["segments"].append({"start": seg.start, "end": seg.end, "text": seg.text})
        text_accum.append(seg.text)
    out["text"] = " ".join(text_accum)

    # if out:
    #     print(f"[ASR] Transcript extracted: {out['text']}")

    return out



def compute_asr_metrics(
    segments: List[Segment],
    window_s: int = 90,
) -> Tuple[float, float, str, float]:
    """
    Returns (speech_ratio, wpm, concat_text, duration_used)
    Uses faster-whisper if installed; else returns zeros.
    """
    print(f"[ASR] Computing ASR metrics for {len(segments)} segments...")
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

    print(f"[ASR] Speech ratio: {speech_ratio}, WPM: {wpm}, Used duration: {used_dur}")
    return speech_ratio, wpm, text, used_dur

if __name__ == "__main__":
    video_path = "Intellegent.mp4"
    # transcribed_text = transcribe_video(video_path, model='medium', language='en')  # en
    # print(f"Transcribed text: {transcribed_text}")
    segments = transcribe(video_path)
    transcription = get_extracted_asr(segments)
    speech_ratio, wpm, concat_text, duration_used = compute_asr_metrics(segments)
