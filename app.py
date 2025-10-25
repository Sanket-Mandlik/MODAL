# app.py
import modal
import subprocess
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Modal App
# --------------------------------------------------------------------------- #
app = modal.App("sd-webui-controlnet")  # REMOVED allow_background_volume_commits

# Persistent volume
vol = modal.Volume.from_name("sd-models", create_if_missing=True)

# GPU
GPU = "L40S"

# --------------------------------------------------------------------------- #
# Image
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
        "gradio==3.41.2", "transformers==4.30.2", "accelerate", "bitsandbytes",
        "opencv-python", "pillow", "pillow-avif-plugin==1.4.3", "numpy",
        "scipy", "tqdm", "omegaconf", "einops", "controlnet-aux",
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
# WebUI — PROXY + NO BLOCKING
# --------------------------------------------------------------------------- #
from modal import asgi_app

@app.function(
    image=image,
    gpu=GPU,
    volumes={"/models": vol},
    timeout=7200,
    startup_timeout=1800,
)
@asgi_app()
def run_webui():
    os.chdir("/sd-webui")

    # === Fix model symlink ===
    vol_dir = Path("/models/Stable-diffusion")
    webui_dir = Path("/sd-webui/models/Stable-diffusion")
    vol_dir.mkdir(parents=True, exist_ok=True)

    if webui_dir.exists() and not webui_dir.is_symlink():
        for f in webui_dir.glob("*.safetensors"):
            target = vol_dir / f.name
            if not target.exists():
                print(f"Moving {f.name} to volume...")
                f.rename(target)
        try:
            webui_dir.rmdir()
        except:
            pass

    if not webui_dir.exists():
        os.symlink(vol_dir, webui_dir)

    # === Download model once ===
    model_path = vol_dir / "v1-5-pruned-emaonly.safetensors"
    if not model_path.exists():
        print("Downloading model to volume...")
        import urllib.request
        from tqdm import tqdm

        url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
        with urllib.request.urlopen(url) as r, open(model_path, "wb") as f:
            total = int(r.headers.get("content-length", 0))
            with tqdm(total=total, unit="iB", unit_scale=True) as pbar:
                while True:
                    chunk = r.read(1024*1024)
                    if not chunk: break
                    f.write(chunk)
                    pbar.update(len(chunk))
        vol.commit()
        print("Model saved.")

    # === Start WebUI ===
    print("Starting WebUI on port 7860...")
    subprocess.Popen([
        "python", "launch.py",
        "--listen", "--port", "7860", "--api",
        "--disable-safe-unpickle", "--enable-insecure-extension-access",
        "--skip-torch-cuda-test", "--skip-install", "--skip-python-version-check",
        "--opt-sdp-attention", "--no-half-vae", "--medvram"
    ])

    # === Return proxy ===
    from fastapi import FastAPI, Request
    import httpx

    web_app = FastAPI()

    @web_app.get("/")
    async def root():
        return {"message": "WebUI starting... wait 30s"}

    async def proxy(request: Request):
        url = f"http://localhost:7860{request.url.path}"
        if request.url.query:
            url += f"?{request.url.query}"
        client = httpx.AsyncClient(timeout=60.0)
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
# Helpers
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
        print(f"Folder: {p.path}")
        try:
            for f in vol.listdir(p.path):
                print(f"  File: {f.path}")
        except:
            pass

@app.function(image=image, gpu=GPU)
def test_gpu():
    import torch
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM:", f"{torch.cuda.get_device_props(0).total_memory/1e9:.1f} GB")

# --------------------------------------------------------------------------- #
# Local entrypoint
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
    elif cmd == "serve":
        print("Starting... URL in ~30s")
        run_webui.serve()