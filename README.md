**From raw footage to concise insights:** Build an adaptive video-understanding pipeline that probes each video and automatically chooses the right mix of ASR, OCR, and a vision-language model. Process in chunks (Decord), transcribe speech (Whisper), read on-screen text (Tesseract), describe frames (Ovis2.5), then condense everything into a clear summary (Qwen2.5).

conda create --prefix D:\\conda_env\\video_understanding Python=3.11 -y && conda activate D:\conda_env\video_understanding 



### Install requirements:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install git+https://github.com/huggingface/transformers

pip install git+https://github.com/huggingface/accelerate


pip install -r requirements.txt
```

### To remove the environment when done:
```bash
conda remove --prefix D:\\conda_env\\video_understanding --all
```

### Run the App:
```bash
streamlit run app.py
```