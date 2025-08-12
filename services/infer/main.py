from fastapi import FastAPI
import os, json

app = FastAPI(title="Levizion Inference (GPU)")

def gpu_status():
    info = {"cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "nvidia_env": os.environ.get("NVIDIA_VISIBLE_DEVICES", "")}
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        devices = []
        if cuda_ok:
            count = torch.cuda.device_count()
            for i in range(count):
                devices.append(torch.cuda.get_device_name(i))
        info.update({"torch": True, "cuda_available": cuda_ok, "devices": devices})
    except Exception as e:
        info.update({"torch": False, "error": str(e)})
    return info

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/gpu")
def gpu():
    return gpu_status()
