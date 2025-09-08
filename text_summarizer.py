import gc
import torch
from typing import List, Dict, Tuple, Any
import json
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from deeper_video_understanding import seconds_to_mmss

final_prompt = """ 
    You are given chunked summaries of a video transcript (and optional on-screen text).
    Produce: (A) 4–7 sentence overall summary; (B) timeline with timestamps; 
    (C) key entities and roles; (D) notable numbers and quotes; (E) ambiguities.
    Do not duplicate any information.
    \n\n
    {text}
    Output should be in markdown format.
    """

SECOND_PASS_PROMPT = """
You will receive chunk-level notes in time order.

Tasks (produce Markdown with EXACTLY these sections, nothing else):
### Global Summary (5–10 bullets; ≤150 words)
- Concise, factual bullets covering the video end-to-end.

### Timeline
- One line per event, format: [mm:ss–mm:ss]: event
- Use mm:ss; if an instantaneous moment, use [mm:ss].
- Keep chronological order and merge duplicates.

### Entities
- Bulleted list: Name — role/description (one line each).

### Notable Numbers & Quotes
- Bulleted; numbers with units; quotes verbatim in quotation marks with attribution if present.
- If none, write: None.

### Uncertainties
- Bulleted; list gaps, ambiguities, or contradictions.
- If none, write: None.

Constraints:
- Use ONLY the provided notes; do not invent details.
- Do NOT duplicate the same fact across sections.

Notes:
{text}

"""

def _load_summarizer():
    repo_id = "Qwen/Qwen2.5-7B-Instruct"
    model = HuggingFaceEndpoint(repo_id=repo_id, temperature=0)
    llm = ChatHuggingFace(llm=model)
    return llm

def build_second_pass_input(chunk_results: List[Dict[str, Any]]) -> str:
    # Stitch chunk outputs into a single plain-text document.
    parts = []
    for r in chunk_results:
        st, et = seconds_to_mmss(r["start_sec"]), seconds_to_mmss(r["end_sec"])
        parts.append(f"[{st}–{et}]\n{r['description']}\n")
    return "\n".join(parts).strip()


def summarize_text(text: str) -> str:
    llm = _load_summarizer()
    
    prompt = PromptTemplate(
        template=SECOND_PASS_PROMPT,
        input_variables=["text"],
    )
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"text": text})

    gc.collect()
    torch.cuda.empty_cache()

    return summary