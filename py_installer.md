conda activate D:\conda_env\video_understanding

pyinstaller --onedir --noconsole ^
  --name VideoUnderstanding ^
  --paths D:\conda_env\video_understanding\Lib\site-packages ^
  ^
  --collect-all PyQt5 ^
  --collect-all torch ^
  --collect-all torchvision ^
  --collect-all torchaudio ^
  --collect-all nvidia ^
  ^
  --collect-all transformers ^
  --collect-all tokenizers ^
  --collect-all huggingface_hub ^
  --collect-all safetensors ^
  --collect-all accelerate ^
  ^
  --collect-all decord ^
  --collect-all sentencepiece ^
  --collect-all PIL ^
  --collect-all opencv_python_headless ^
  ^
  --collect-all faster_whisper ^
  --collect-all ctranslate2 ^
  ^
  --collect-all moviepy ^
  --collect-all imageio ^
  --collect-binaries imageio_ffmpeg ^
  ^
  --collect-all qwen_vl_utils ^
  --collect-all hf_xet ^
  ^
  --collect-data markdown ^
  ^
  --collect-submodules langchain ^
  --collect-submodules langchain_core ^
  --collect-submodules langchain_community ^
  --collect-submodules langchain_huggingface ^
  ^
  --hidden-import decide_pipeline ^
  --hidden-import ocr_video ^
  --hidden-import vlm_ocr_asr ^
  ^
  --exclude-module streamlit ^
  desktop_app.py
