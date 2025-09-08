from pipeline_decider import decide_pipeline
from text_summarizer import summarize_text, build_second_pass_input
from transcribe_audio import get_extracted_asr
from deeper_video_understanding import ocr_video, vlm_ocr_asr

def _trim(txt, lim=800):
        return (txt[:lim] + " …") if len(txt) > lim else txt

def execute_workflow(video_path, report_cb):
    report_cb(step=1, total=5, message="Step 1: Deciding pipeline…")
    decision, segments = decide_pipeline(video_path)
    report_cb(step=2, total=5, message="Step 2: Decided: {}".format(decision))

    if decision == "VLM_OCR":
        report_cb(step=3, total=5, message="Step 3: Extracting text (OCR) and Describing video using VLM…")
        chunk_summaries = vlm_ocr_asr(video_path, None)
        final_extract = build_second_pass_input(chunk_summaries)
        report_cb(step=4, total=5, message="Step 4: Summarizing…")
        video_summary = summarize_text(final_extract)
        report_cb(step=5, total=5, message="Step 5: Completed", done=True)
        return video_summary
        
    if decision == "ASR_ONLY":
        report_cb(step=3, total=5, message="Step 3: Transcribing speech…")
        asr_result = get_extracted_asr(segments)
        report_cb(step=4, total=5, message="Step 4: Summarizing…")
        video_summary = summarize_text(asr_result["text"])
        report_cb(step=5, total=5, message="Step 5: Completed", done=True)
        return video_summary

    if decision == "ASR_OCR":
        report_cb(step=3, total=5, message="Step 3: Extracting text (OCR) from video…")
        chunk_summaries = ocr_video(video_path)
        report_cb(step=4, total=5, message="Step 4: Transcribing speech…")
        ocr_excerpt = build_second_pass_input(chunk_summaries)
        asr_extract = get_extracted_asr(segments)

        final_extract = f'\n\nASR excerpt (verbatim context):\n"{_trim(asr_extract["text"])}"'
        if ocr_excerpt:
            final_extract += (
                "\n\nOCR evidence (verbatim; prefer this for exact text/numbers):\n"
                f"{_trim(ocr_excerpt)}"
            )
        report_cb(step=5, total=5, message="Step 5: Summarizing…")
        video_summary = summarize_text(final_extract)
        report_cb(step=5, total=5, message="Step 5: Completed", done=True)
        return video_summary

    if decision == "ASR_OCR_VLM":
        report_cb(step=3, total=5, message="Step 3: Extracting text (OCR), Transcribing and Describing video using VLM…")
        chunk_summaries = vlm_ocr_asr(video_path, segments)
        final_extract = build_second_pass_input(chunk_summaries)
        report_cb(step=4, total=5, message="Step 4: Summarizing…")
        video_summary = summarize_text(final_extract)
        report_cb(step=5, total=5, message="Step 5: Completed", done=True)
        return video_summary


if __name__ == "__main__":
    video_path = "badminton.mp4"
    video_description = execute_workflow(video_path)
    print(video_description)