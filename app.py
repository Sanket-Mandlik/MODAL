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
# GPU: L40S → A100-40 → A100-80
# --------------------------------------------------------------------------- #
def select_gpu():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        )
    except Exception as e:
        raise RuntimeError(f"GPU detection failed: {e}")

    for line in out.strip().split("\n"):
        if not line:
            continue
        name, mem = line.split(",", 1)
        name = name.lower().strip()
        mem_gb = int(mem.strip().split()[0])

        if "l40s" in name or "ls40" in name:
            return modal.gpu.L40S(count=1)
        if "a100" in name:
            if mem_gb >= 80:
                return modal.gpu.A100(count=1, memory=80)
            if mem_gb >= 40:
                return modal.gpu.A100(count=1, memory=40)

    raise RuntimeError("Only L40S, A100-40GB, A100-80GB allowed.")

# --------------------------------------------------------------------------- #
# Image — proper pip_install, no run_commands for pip
# --------------------------------------------------------------------------- #
image = (
    modal.Image.debian_slim()
    .apt_install("git", "wget", "curl", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install(
        "torch==2.5.1+cu121",
        "torchvision==0.20.1+cu121",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "gradio", "transformers", "accelerate", "bitsandbytes",
        "xformers==0.0.26", "opencv-python", "pillow", "numpy",
        "scipy", "tqdm", "omegaconf", "einops", "controlnet-aux"
    )
    .run_commands(
        "git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /sd-webui",
        "cd /sd-webui && git checkout v1.10.1",  # SDXL
        "cd /sd-webui/extensions && git clone https://github.com/Mikubill/sd-webui-controlnet.git",
        "mkdir -p /sd-webui/embeddings /sd-webui/outputs",
    )
)

# --------------------------------------------------------------------------- #
# WebUI — NO medvram, max 3 users
# --------------------------------------------------------------------------- #
@app.function(
    image=image,
    gpu=select_gpu(),
    volumes={"/models": vol},
    timeout=7200,
    startup_timeout=1200,
    allow_concurrent_inputs=3,
    _experimental_enable_memory_snapshots=True,
)
@modal.web_server(7860)
def run_webui():
    os.chdir("/sd-webui")
    subprocess.run(["ln", "-sfn", "/models", "/sd-webui/models"], check=False)

    args = [
        "python", "launch.py",
        "--listen", "--port", "7860",
        "--api",
        "--xformers",
        "--disable-safe-unpickle",
        "--enable-insecure-extension-access",
        "--skip-torch-cuda-test",
        # NO --medvram
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

@app.function(image=image, gpu=select_gpu())
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