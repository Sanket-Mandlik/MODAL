# app.py
import modal
import subprocess
import os
import time
import urllib.request
from pathlib import Path
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Modal App
#2# --------------------------------------------------------------------------- #
app = modal.App("sd-webui-controlnet")

# Persistent volume — models go directly into A1111's model path
vol = modal.Volume.from_name("sd-models", create_if_missing=True)

# GPU
GPU = "L40S"

# --------------------------------------------------------------------------- #
# Image — ALL REQUIRED PACKAGES
# --------------------------------------------------------------------------- #
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "wget", "curl", "libglib2.0-0", "libsm6", "libxext6", "libxrender1",
        "libgl1-mesa-glx"
    )
    .run_commands(
        "pip install --upgrade pip",
        "pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision",
        "pip uninstall -y pydantic pydantic-core -q",
        "pip install pydantic==1.10.13"
    )
    .pip_install(
        # Core
        "gradio==3.41.2", "transformers==4.30.2", "accelerate", "bitsandbytes",
        "opencv-python", "pillow", "pillow-avif-plugin==1.4.3", "numpy",
        "scipy", "tqdm", "omegaconf", "einops",

        # ControlNet & Preprocessors
        "controlnet-aux", "facexlib", "gfpgan",

        # Others
        "pytorch-lightning==1.9.0", "safetensors", "timm", "kornia",
        "GitPython", "blendmodes", "clean-fid", "diskcache",
        "fastapi>=0.90.1", "inflection", "jsonmerge",
        "lark", "open-clip-torch", "piexif", "protobuf==3.20.0",
        "psutil", "requests", "resize-right", "scikit-image>=0.19",
        "tomesd", "torchdiffeq", "torchsde", "ftfy", "regex",
        "taming-transformers", "clip", "httpx", "anyio"
    )
    .run_commands(
        "git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /sd-webui",
        "cd /sd-webui && git checkout v1.10.1",
        "cd /sd-webui/extensions && git clone https://github.com/Mikubill/sd-webui-controlnet.git",
        "mkdir -p /sd-webui/embeddings /sd-webui/outputs"
    )
)

# --------------------------------------------------------------------------- #
# WebUI — Direct Volume Mount + No Symlinks
# --------------------------------------------------------------------------- #
@app.function(
    image=image,
    gpu=GPU,
    volumes={"/sd-webui/models": vol},  # Direct mount into A1111's model dir
    timeout=7200,
    startup_timeout=1800,
)
@modal.asgi_app()
def run_webui():
    os.chdir("/sd-webui")

    # === Ensure model directory exists ===
    model_dir = Path("/sd-webui/models/Stable-diffusion")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "v1-5-pruned-emaonly.safetensors"

    # === Download model if not exists (only once) ===
    if not model_path.exists():
        print("Model not found. Downloading v1-5-pruned-emaonly.safetensors...")
        url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
        with urllib.request.urlopen(url) as response, open(model_path, "wb") as f:
            total = int(response.headers.get("content-length", 0))
            with tqdm(total=total, unit="iB", unit_scale=True, desc="Downloading") as pbar:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        vol.commit()
        print(f"Model saved to volume: {model_path} ({model_path.stat().st_size / 1e9:.2f} GB)")
    else:
        print(f"Model ready: {model_path} ({model_path.stat().st_size / 1e9:.2f} GB)")

    # === Start WebUI ===
    print("Launching WebUI (L40S 48GB — full speed)...")
    proc = subprocess.Popen([
        "python", "launch.py",
        "--listen", "--port", "7860", "--api",
        "--disable-safe-unpickle", "--enable-insecure-extension-access",
        "--skip-torch-cuda-test", "--skip-install", "--skip-python-version-check",
        "--opt-sdp-attention", "--no-half-vae", "--medvram"
    ])

    # === Wait for WebUI to be ready ===
    import requests
    def wait_for_api():
        print("Waiting for WebUI API to start...", end="")
        for _ in range(60):
            try:
                r = requests.get("http://localhost:7860/sdapi/v1/sd-models", timeout=5)
                if r.status_code == 200:
                    print(" READY!")
                    return
            except:
                print(".", end="", flush=True)
                time.sleep(3)
        raise RuntimeError("WebUI failed to start after 3 minutes")

    wait_for_api()

    # === FastAPI Proxy ===
    from fastapi import FastAPI, Request
    import httpx

    web_app = FastAPI()

    @web_app.get("/")
    async def root():
        return {
            "message": "A1111 Stable Diffusion WebUI + ControlNet",
            "api": "/sdapi/v1/txt2img",
            "ui": "Gradio UI ready",
            "model": str(model_path.name)
        }

    @web_app.get("/health")
    async def health():
        return {"status": "healthy"}

    async def proxy(request: Request):
        url = f"http://localhost:7860{request.url.path}"
        if request.url.query:
            url += f"?{request.url.query}"
        client = httpx.AsyncClient(timeout=120.0)
        try:
            req = client.build_request(
                request.method, url,
                headers=request.headers,
                content=await request.body()
            )
            resp = await client.send(req, stream=True)
            return resp
        finally:
            await client.aclose()

    web_app.add_route("/{path:path}", proxy, methods=["*"])

    return web_app


# --------------------------------------------------------------------------- #
# Helper: Upload model (optional, but useful)
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={"/sd-webui/models": vol})
def upload_model(local_path: str, model_type: str = "Stable-diffusion"):
    mapping = {
        "stable-diffusion": "/sd-webui/models/Stable-diffusion",
        "controlnet": "/sd-webui/models/ControlNet",
        "lora": "/sd-webui/models/Lora",
        "vae": "/sd-webui/models/VAE",
    }
    if model_type not in mapping:
        raise ValueError(f"Invalid type: {list(mapping)}")
    remote_dir = Path(mapping[model_type])
    remote_dir.mkdir(parents=True, exist_oke=True)
    vol.put_file(str(remote_dir / Path(local_path).name), local_path)
    vol.commit()
    print(f"Uploaded {local_path} → {remote_dir / Path(local_path).name}")


# --------------------------------------------------------------------------- #
# Helper: List models
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={"/sd-webui/models": vol})
def list_models():
    print("=== Persistent Volume Contents ===")
    for p in vol.listdir("/sd-webui/models"):
        print(f"Folder: {p.path}")
        try:
            for f in vol.listdir(p.path):
                size = f.size / 1e9 if f.size else 0
                print(f"  File: {f.path} ({size:.2f} GB)")
        except:
            pass


# --------------------------------------------------------------------------- #
# Helper: Test GPU
# --------------------------------------------------------------------------- #
@app.function(image=image, gpu=GPU)
def test_gpu():
    import torch
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM:", f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# --------------------------------------------------------------------------- #
# Local Entrypoint
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main():
    import sys
    if len(sys.argv) < 2:
        print("Commands: test-gpu | list-models | upload-model <path> <type> | deploy | serve")
        return
    cmd = sys.argv[1]
    if cmd == "test-gpu":
        test_gpu.remote()
    elif cmd == "list-models":
        list_models.remote()
    elif cmd == "upload-model":
        if len(sys.argv) != 4:
            print("Usage: upload-model <path> <type>")
            return
        upload_model.remote(sys.argv[2], sys.argv[3])
    elif cmd == "deploy":
        run_webui.deploy(name="sd-webui-controlnet")
        print("Deployed! URL: https://your-workspace--sd-webui-controlnet.modal.run")
    elif cmd == "serve":
        print("Starting... URL in ~30s")
        run_webui.serve()