# app.py
import modal
import subprocess
import os
from pathlib import Path
import time

# --------------------------------------------------------------------------- #
# Modal App
# --------------------------------------------------------------------------- #
app = modal.App(
    "sd-webui-controlnet",
    allow_background_volume_commits=True  # REQUIRED FOR vol.commit()
)

# Persistent volume for models
vol = modal.Volume.from_name("sd-models", create_if_missing=True)

# GPU
GPU = "L40S"

# --------------------------------------------------------------------------- #
# Image — All dependencies
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
        "facexlib", "fastapi>=0.90.1", "inflection", "jsonmerge",
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
    startup_timeout=1800,  # 30 min for first model download
    _allow_background_volume_commits=True
)
@asgi_app()
def run_webui():
    os.chdir("/sd-webui")

    # === 1. Fix model directory symlink ===
    vol_model_dir = Path("/models/Stable-diffusion")
    webui_model_dir = Path("/sd-webui/models/Stable-diffusion")

    vol_model_dir.mkdir(parents=True, exist_ok=True)

    # Move any existing local models to volume
    if webui_model_dir.exists() and not webui_model_dir.is_symlink():
        for file in webui_model_dir.glob("*.safetensors"):
            target = vol_model_dir / file.name
            if not target.exists():
                print(f"Moving {file.name} → volume")
                file.rename(target)
        try:
            webui_model_dir.rmdir()
        except:
            pass

    # Create symlink: WebUI sees volume
    if not webui_model_dir.exists():
        os.symlink(vol_model_dir, webui_model_dir)
        print(f"Symlinked {webui_model_dir} → {vol_model_dir}")

    # === 2. Download model ONCE to volume ===
    model_path = vol_model_dir / "v1-5-pruned-emaonly.safetensors"
    if not model_path.exists():
        print("Model not found. Downloading to persistent volume...")
        import urllib.request
        from tqdm import tqdm

        url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
        print(f"Downloading: {url}")

        with urllib.request.urlopen(url) as response, open(model_path, "wb") as f:
            total = int(response.headers.get("content-length", 0))
            with tqdm(total=total, unit="iB", unit_scale=True, desc="SD 1.5") as pbar:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

        vol.commit()
        print("Model saved to volume. Future runs will be instant.")

    # === 3. Start WebUI in background (NO WAIT) ===
    print("Starting Stable Diffusion WebUI on port 7860...")
    subprocess.Popen([
        "python", "launch.py",
        "--listen", "--port", "7860",
        "--api",
        "--disable-safe-unpickle",
        "--enable-insecure-extension-access",
        "--skip-torch-cuda-test",
        "--skip-install",
        "--skip-python-version-check",
        "--opt-sdp-attention",
        "--no-half-vae",
        "--medvram"
    ])

    # === 4. Return FastAPI proxy to Gradio ===
    from fastapi import FastAPI, Request
    import httpx

    web_app = FastAPI()

    @web_app.get("/")
    async def root():
        return {
            "message": "Stable Diffusion WebUI is starting...",
            "status": "Wait 20-40 seconds, then refresh",
            "tip": "Use /sd to access UI directly"
        }

    @web_app.get("/health")
    async def health():
        return {"status": "running"}

    async def reverse_proxy(request: Request):
        url = f"http://localhost:7860{request.url.path}"
        if request.url.query:
            url += f"?{request.url.query}"

        client = httpx.AsyncClient(timeout=60.0)
        try:
            req = client.build_request(
                method=request.method,
                url=url,
                headers=request.headers,
                content=await request.body()
            )
            resp = await client.send(req, stream=True)
            return resp
        finally:
            await client.aclose()

    web_app.add_route("/{full_path:path}", reverse_proxy, methods=["GET", "POST", "PUT", "DELETE", "PATCH"])

    return web_app

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
        raise ValueError(f"Invalid type. Use: {list(mapping.keys())}")

    remote_path = f"{mapping[model_type]}/{Path(local_path).name}"
    vol.put_file(remote_path, local_path)
    vol.commit()
    print(f"Uploaded: {local_path} → {remote_path}")

@app.function(image=image, volumes={"/models": vol})
def list_models():
    print("=== Volume Contents ===")
    try:
        for folder in vol.listdir("/"):
            print(f"Folder: {folder.path}")
            try:
                for file in vol.listdir(folder.path):
                    print(f"   └─ {file.path}")
            except:
                pass
    except Exception as e:
        print("Volume empty or error:", e)

@app.function(image=image, gpu=GPU)
def test_gpu():
    import torch
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name} | VRAM: {mem:.1f} GB")

# --------------------------------------------------------------------------- #
# Local Entrypoint
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: modal run app.py <command>")
        print("Commands: test-gpu | list-models | upload-model <path> <type> | deploy | serve")
        return

    cmd = sys.argv[1]

    if cmd == "test-gpu":
        test_gpu.remote()
    elif cmd == "list-models":
        list_models.remote()
    elif cmd == "upload-model":
        if len(sys.argv) != 4:
            print("Usage: upload-model <local_path> <type>")
            return
        upload_model.remote(sys.argv[2], sys.argv[3])
    elif cmd == "deploy":
        print("Deploying WebUI...")
        run_webui.deploy(name="sd-webui-controlnet")
    elif cmd == "serve":
        print("Serving WebUI... (Ctrl+C to stop)")
        print("URL will appear in ~30 seconds")
        run_webui.serve()
    else:
        print("Unknown command.")