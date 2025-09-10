Here’s a tight, side-by-side for your shortlist (edge PC, 4–6 GB VRAM, multilingual OCR, open-source).

| Model                                                   |  Params | License             | Multi-image in one prompt                                                                      | OCR / document strengths                                                                                                                         | Multilingual note                                                                                                              | Edge-fit tips (4–6 GB VRAM)                                                                                                         |
| ------------------------------------------------------- | ------: | ------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| **AIDC-AI/Ovis2.5-2B**                                  | \~2.57B | Apache-2.0          | **Yes** (built-in multi-image & video examples; supports `max_pixels` cap) ([Hugging Face][1]) | Native-resolution ViT (NaViT) preserves tiny text; “Chart & Document OCR” called out; optional “thinking mode” for accuracy. ([Hugging Face][1]) | EN/zh focus (tags); general OCR across docs/charts shown in card. ([Hugging Face][1])                                          | Use the multi-image example with `max_pixels≈896×896`; enable Flash-Attn if available. ([Hugging Face][1])                          |
| **Qwen2.5-VL-3B-Instruct**                              | \~3.75B | Qwen (open weights) | **Yes** (interleaved images/video; ready multi-image snippet) ([Hugging Face][2])              | Strong DocVQA/TextVQA; **structured extraction** for invoices/forms/tables; localization (boxes/points). ([Hugging Face][2])                     | Qwen2.5-VL inherits broad text-reading across languages from Qwen2-VL; 3B targets edge. ([Hugging Face][2], [Qwen][3])         | Set `min_pixels`/`max_pixels` to limit visual tokens; Flash-Attn 2 recommended for memory/speed. ([Hugging Face][2])                |
| **Qwen2-VL-2B-Instruct** *(closest to “Qwen2.5-VL-2B”)* | \~2.21B | Apache-2.0          | **Yes** (multi-image/video; sample code) ([Hugging Face][4])                                   | Good on TextVQA/DocVQA; dynamic-resolution + M-ROPE. ([Hugging Face][4])                                                                         | Explicit **multilingual OCR** support: most European languages, Japanese, Korean, Arabic, Vietnamese, etc. ([Hugging Face][4]) | Throttle with `min_pixels`/`max_pixels`; Flash-Attn 2 helps with multi-image batches. ([Hugging Face][4])                           |
| **Phi-3-Vision-128k-Instruct**                          |  \~4.2B | MIT                 | *Best-suited for a single image per prompt* (multi-turn for more) ([Hugging Face][5])          | Designed for OCR + charts/tables; **128k** context for long page QA. ([Hugging Face][5])                                                         | Primarily English; card flags non-EN as weaker. ([Hugging Face][5])                                                            | Works on consumer GPUs; if needed set `_attn_implementation='eager'` (disable Flash-Attn). ONNX variants exist. ([Hugging Face][5]) |

### Notes that matter

* **About “Qwen2.5-VL-2B”:** as of today, Qwen2.5-VL is released in **3B/7B/72B**. There is **no official 2B** for 2.5-VL; the closest small model is **Qwen2-VL-2B** (previous gen). ([Hugging Face][6])
* **Multi-image OCR workflow:** Ovis and both Qwen variants natively accept multiple images in one turn; for tight VRAM, cap image pixels (`max_pixels`), which limits visual tokens and avoids OOM. ([Hugging Face][1])
* **Licensing:** Ovis (Apache-2.0), Qwen2-VL-2B (Apache-2.0), Phi-3-Vision (MIT). ([Hugging Face][1])

### Quick recommendation for your 4–6 GB VRAM PC

* **Best all-rounder (multilingual OCR + reasoning + multi-image):** **Qwen2.5-VL-3B-Instruct** with INT8/INT4 and `min/max_pixels` set. ([Hugging Face][2])
* **Smallest with strong tiny-text reading:** **Ovis2.5-2B** using `max_pixels≈896×896` for batches. ([Hugging Face][1])
* **Long-document OCR QA (single-image per turn, long outputs):** **Phi-3-Vision-128k-Instruct**. ([Hugging Face][5])

If you want, I can share minimal, VRAM-safe loader snippets (one per model) with sensible `min/max_pixels`, quantization hints, and a multi-image OCR prompt template.

[1]: https://huggingface.co/AIDC-AI/Ovis2.5-2B "AIDC-AI/Ovis2.5-2B · Hugging Face"
[2]: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct "Qwen/Qwen2.5-VL-3B-Instruct · Hugging Face"
[3]: https://qwenlm.github.io/blog/qwen2.5-vl/?utm_source=chatgpt.com "Qwen2.5 VL! Qwen2.5 VL! Qwen2.5 VL! | Qwen"
[4]: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct "Qwen/Qwen2-VL-2B-Instruct · Hugging Face"
[5]: https://huggingface.co/microsoft/Phi-3-vision-128k-instruct "microsoft/Phi-3-vision-128k-instruct · Hugging Face"
[6]: https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5?utm_source=chatgpt.com "Qwen2.5-VL - a Qwen Collection"
