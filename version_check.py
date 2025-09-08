import sys, torch, ctranslate2
print("Python:", sys.version)
try:
    print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
except Exception as e:
    print("Torch not installed or CPU-only:", e)
print("CT2:", ctranslate2.__version__, "GPUs:", ctranslate2.get_cuda_device_count())

