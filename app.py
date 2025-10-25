# app.py
import modal
import subprocess
import os
import requests
import tempfile
from pathlib import Path

# --------------------------------------------------------------------------- #
# Modal App
# --------------------------------------------------------------------------- #
app = modal.App("sd-webui-controlnet")

# Volume — auto-created
vol = modal.Volume.from_name("sd-models", create_if_missing=True)

# --------------------------------------------------------------------------- #
# GPU: L40S (48GB VRAM)
# --------------------------------------------------------------------------- #
GPU = "L40S"

# --------------------------------------------------------------------------- #
# Image — proper pip_install, no run_commands for pip
# --------------------------------------------------------------------------- #
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "wget", "curl", "libglib2.0-0", "libsm6", "libxext6", "libxrender1",
        "libgl1-mesa-glx", "libglib2.0-0"
    )
    .run_commands(
        "pip install --upgrade pip",
        "pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121",
        "pip uninstall -y pydantic pydantic-core",
        "pip install pydantic==1.10.13"
    )
    .pip_install(
        "torchvision",
        "gradio==3.41.2", "transformers==4.30.2", "accelerate", "bitsandbytes",
        "opencv-python", "pillow", "pillow-avif-plugin==1.4.3", "numpy",
        "scipy", "tqdm", "omegaconf", "einops", "controlnet-aux",
        "pytorch-lightning==1.9.0", "safetensors", "timm", "kornia",
        "GitPython", "blendmodes", "clean-fid", "diskcache",
        "facexlib", "fastapi>=0.90.1", "inflection", "jsonmerge",
        "lark", "open-clip-torch", "piexif", "protobuf==3.20.0",
        "psutil", "requests", "resize-right", "scikit-image>=0.19",
        "tomesd", "torch", "torchdiffeq", "torchsde",
        "ftfy", "regex", "taming-transformers", "clip",
        "pydantic==1.10.13"
    )

    .run_commands(
        "git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /sd-webui",
        "cd /sd-webui && git checkout v1.10.1",  # SDXL
        "cd /sd-webui/extensions && git clone https://github.com/Mikubill/sd-webui-controlnet.git",
        "mkdir -p /sd-webui/embeddings /sd-webui/outputs",
        "pip uninstall -y pydantic pydantic-core",
        "pip install pydantic==1.10.13"
    )
)

# --------------------------------------------------------------------------- #
# WebUI — NO medvram, max 3 users
# --------------------------------------------------------------------------- #
@app.function(
    image=image,
    gpu=GPU,
    volumes={"/models": vol},
    timeout=7200,
    startup_timeout=1200,
)
@modal.web_server(7860)
def run_webui():
    os.chdir("/sd-webui")
    subprocess.run(["ln", "-sfn", "/models", "/sd-webui/models"], check=False)

    import torch
    args = [
        "python", "launch.py",
        "--listen", 
        "--port", "7860",
        "--api",
        "--disable-safe-unpickle",
        "--enable-insecure-extension-access",
        "--skip-torch-cuda-test",
        "--skip-install",  # Skip installing SD WebUI requirements
        "--skip-python-version-check",
    ]

    print(f"Launching WebUI on {torch.cuda.get_device_name(0)}")
    proc = subprocess.Popen(args)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()

# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={"/models": vol})
def upload_model(local_path: str, model_type: str = "stable-diffusion"):
    mapping = {
        "stable-diffusion": "/Stable-diffusion",
        "controlnet": "/ControlNet",
        "lora": "/Lora",
        "vae": "/VAE",
    }
    if model_type not in mapping:
        raise ValueError(f"Invalid type: {list(mapping)}")
    remote = f"{mapping[model_type]}/{Path(local_path).name}"
    vol.put_file(remote, local_path)
    vol.commit()
    print(f"Uploaded {local_path} → {remote}")

@app.function(image=image, volumes={"/models": vol})
def list_models():
    print("=== Volume ===")
    for p in vol.listdir("/"):
        print(f"Folder: {p}")
        try:
            for f in vol.listdir(f"/{p}"):
                print(f"   File: {f}")
        except:
            pass

@app.function(image=image, gpu=GPU)
def test_gpu():
    import torch
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Memory:", f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# --------------------------------------------------------------------------- #
# Local Entrypoint
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "test-gpu":
        test_gpu.remote()
    elif cmd == "list-models":
        list_models.remote()
    elif cmd == "upload-model":
        if len(sys.argv) < 4:
            print("Usage: upload-model <path> <type>")
            return
        upload_model.remote(sys.argv[2], sys.argv[3])
    elif cmd == "deploy":
        print("Deploying (no medvram, max 3 users)…")
        run_webui.deploy(name="sd-webui-controlnet")
    else:
        print("Commands: test-gpu | list-models | upload-model | deploy")