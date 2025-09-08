

## How to choose the number

Think in terms of sampling rate (frames you send per second of video):

| Goal                               | Suggested sampling | Frames for 20 s |
| ---------------------------------- | -----------------: | --------------: |
| Quick caption / summary            |    **0.5–0.8 fps** |       **10–16** |
| Rich description / multiple events |    **1.0–1.5 fps** |       **20–30** |
| Fast action / compliance review    |    **2.0–3.0 fps** |       **40–60** |

Short answer: for a 20-second clip (600 frames @ 30 fps), a good default is **12–16 frames**.
That usually gives the LLM enough coverage of the story without wasting tokens or sending near-duplicates.

## How to choose the number

Think in terms of **sampling rate** (frames you send per second of video):

| Goal                               | Suggested sampling | Frames for 20 s |
| ---------------------------------- | -----------------: | --------------: |
| Quick caption / summary            |    **0.5–0.8 fps** |       **10–16** |
| Rich description / multiple events |    **1.0–1.5 fps** |       **20–30** |
| Fast action / compliance review    |    **2.0–3.0 fps** |       **40–60** |

> Rule of thumb: start at **\~0.6–1.0 fps** and scale up only if events are missed.

## Why 12–16 works well

* Covers **start, midpoints, and end** plus transitions.
* Keeps **token/image budget** low (many vision LLMs work best with ≤20 images).
* With your “uniform + burst” sampler, those 12–16 frames also capture **micro-motion** around key anchors.

Recommended sampling (works well with Ovis2.5-2B)

CHUNK_SECONDS = 30
FRAMES_PER_CHUNK = 12
burst = 2, stride = 2


If you want **exactly** N frames (not fewer due to edge clipping), either:

* reduce `burst` to **2** or `stride` to **1**, or
* after `sample_uniform_with_bursts`, **pad** with `sample_uniform_inclusive` until length == N.

## Extra tips

* **Downscale** frames (e.g., to 448–512 px on the long side) to cut bandwidth without losing semantics.
* If the clip has **scene cuts**, first run a simple **shot-change detector** and sample per shot (2–4 per shot).
* If there is on-screen **text/graphics**, add a couple of **focused crops** when text appears/changes.
* Send a **short caption per frame** (“Frame shows X entering room”) and then ask for a **global summary**—this often improves overall descriptions.


