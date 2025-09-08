# FrameSense AI — branding & product starter

**Tagline options (pick one)**

* “From footage to facts.”
* “Understand every video, faster.”
* “Summaries, timelines, and insight—automatically.”

**One-liner**
FrameSense AI turns long videos into concise summaries, timelines, and actionable notes—locally and fast.

**30-second pitch**
FrameSense AI ingests any video, transcribes speech, reads on-screen text, samples key frames, and produces a trustworthy summary with a timestamped timeline. It’s optimized for 6-GB GPUs and low-spec machines using efficient sampling, OCR, and lightweight LLMs—so teams can understand content without watching it end-to-end.

**Core features (MVP)**

* ASR-driven summaries (multi-language incl. Bangla/English)
* Timestamped timeline of key events
* On-screen text via OCR; merges into transcript
* Smart frame sampling (uniform + bursts)
* Export to Markdown/JSON; copy-ready notes
* Local processing/privacy-first toggle
* CLI + simple desktop UI (optional)

**MVP scope (what to build first)**

1. Ingest (mp4/mov) → Faster-Whisper transcript
2. OCR 6–8 sampled frames → merge into transcript
3. Summarize with a small local LLM (e.g., Phi-3 Mini)
4. Output: overview + timeline + entities + quotes (Markdown)
5. Basic UI: drop file → progress → result pane with export

**Architecture sketch**

* Decode: `decord` (frame indices)
* ASR: `faster-whisper` (CUDA/CPU)
* OCR: Tesseract (`pytesseract`) or EasyOCR
* Summarizer: small instruct LLM (CPU/GPU)
* Optional: VLM step for visual Q\&A (behind a toggle)
* Storage: local workspace folder; Markdown + JSON

**Naming & visual identity**

* Colors: deep blue (#1F3A5F), teal accent (#2BBBAD), soft gray (#F4F6F8)
* Logo idea: stylized play button + “eye” glyph forming an “F”
* Iconography: timeline, captions, checkmarks for “facts”

**Website/App store blurb (short)**
FrameSense AI converts long videos into clear summaries and timestamped timelines. ASR + OCR + smart sampling deliver fast, reliable insights—right on your machine.

