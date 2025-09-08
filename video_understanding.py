from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer
import torch

model_path = 'openbmb/MiniCPM-V-4_5'
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    attn_implementation='sdpa',  # sdpa or flash_attention_2
    dtype=torch.float16,
)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True)


# If you see OOM on consumer GPUs, keep MAX_NUM_FRAMES ≤ 24, set params["max_slice_nums"] = 3, or switch to dtype=torch.float16.
MAX_NUM_FRAMES = 24  # if cuda OOM set a smaller number

def sample_uniform_inclusive(n_total, n_want):
    if n_want <= 0: return []
    if n_want >= n_total: return list(range(n_total))
    # even spacing from 0 to n_total-1, inclusive
    return [int(round(i * (n_total - 1) / (n_want - 1))) for i in range(n_want)]


# Burst sampling (captures motion without more frames)
# Keep uniform anchors, then add tiny local bursts around each anchor:
def sample_uniform_with_bursts(n_total, n_want, burst=3, stride=3):
    # Choose base anchors first
    base = sample_uniform_inclusive(n_total, max(1, n_want // burst))
    picks = []
    for a in base:
        for k in range(-(burst//2), burst//2 + 1):
            i = a + k * stride
            if 0 <= i < n_total:
                picks.append(i)
    # de-dup and trim to n_want
    picks = sorted(set(picks))
    if len(picks) > n_want:
        # uniformly thin to exactly n_want
        picks = [picks[i] for i in sample_uniform_inclusive(len(picks), n_want)]
    return picks



def uniform_sample(l, n):
    if n <= 0:
        return []
    if n >= len(l):
        return list(l)  # or l[:] if you want a copy

    gap = len(l) / n
    idxs = [min(len(l) - 1, int(i * gap + gap / 2)) for i in range(n)]
    return [l[i] for i in idxs]


def encode_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    n = min(MAX_NUM_FRAMES, total_frames)
    frame_idx = sample_uniform_inclusive(total_frames, n)
    print(f"Total Frames: {total_frames}")
    print(f"Frame Index: {frame_idx}")
    sample_fps = max(1, int(round(vr.get_avg_fps())))
    print('FPS:', sample_fps)
    # frame_idx = [i for i in range(0, len(vr), sample_fps)]
    # if len(frame_idx) > MAX_NUM_FRAMES:
    #     frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    print('num frames:', len(frames))
    return frames

video_path = "badminton.mp4" # badminton.mp4  Intellegent.mp4
frames = encode_video(video_path)

question = "Describe the video"
msg = [
    {'role': 'user', 'content': frames + [question]},
]

# Set decode params for video
params = {}
params["use_image_id"] = False
# use 1 if cuda OOM and video resolution > 448*448
params["max_slice_nums"] = 2

res = model.chat(
    image=None,
    msgs=msg,
    tokenizer=tokenizer,
    **params
)

print("Response:")
print(res)