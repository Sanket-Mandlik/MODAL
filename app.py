# app.py
import modal
import subprocess
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Modal App
# --------------------------------------------------------------------------- #
app = modal.App("sd-webui-controlnet")

# Persistent volume
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

    # === Volume & Symlink Setup ===
    vol_dir = Path("/models/Stable-diffusion")
    webui_dir = Path("/sd-webui/models/Stable-diffusion")
    vol_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] Volume mount: /models")
    print(f"[DEBUG] Volume dir exists: {vol_dir.exists()}")
    print(f"[DEBUG] WebUI dir exists: {webui_dir.exists()}")

    # List files in volume
    try:
        files = [p.name for p in vol_dir.iterdir()]
        print(f"[DEBUG] Files in /models/Stable-diffusion: {files}")
    except Exception as e:
        print(f"[DEBUG] Could not list volume dir: {e}")

    # Clean up old real directory if exists
    if webui_dir.exists() and not webui_dir.is_symlink():
        print(f"[CLEANUP] Removing old real directory: {webui_dir}")
        import shutil
        shutil.rmtree(webui_dir, ignore_errors=True)

    # Always (re)create symlink
    if not webui_dir.is_symlink():
        print(f"[SYMLINK] Creating: {vol_dir} → {webui_dir}")
        try:
            os.symlink(vol_dir, webui_dir)
            print("[SYMLINK] Success")
        except Exception as e:
            print(f"[SYMLINK] Failed: {e}")
            raise
    else:
        target = os.readlink(webui_dir)
        print(f"[SYMLINK] Already exists → {target}")

    # === Download model once ===
    model_name = "v1-5-pruned-emaonly.safetensors"
    model_path = vol_dir / model_name

    print(f"[MODEL] Checking: {model_path}")

    if model_path.exists():
        size_gb = model_path.stat().st_size / (1024**3)
        print(f"[MODEL] FOUND: {model_name} ({size_gb:.2f} GB) → Skipping download")
    else:
        print(f"[MODEL] NOT FOUND → Downloading {model_name}...")
        import urllib.request
        from tqdm import tqdm

        url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
        try:
            with urllib.request.urlopen(url) as r, open(model_path, "wb") as f:
                total = int(r.headers.get("content-length", 0))
                with tqdm(total=total, unit="iB", unit_scale=True, desc="Download") as pbar:
                    while True:
                        chunk = r.read(1024*1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            print("[MODEL] Download complete")
        except Exception as e:
            print(f"[MODEL] Download failed: {e}")
            if model_path.exists():
                model_path.unlink()
            raise
        finally:
            vol.commit()
            print("[VOLUME] Commit complete")

    # === Start WebUI ===
    print("Starting WebUI on port 7860...")
    
    # CRITICAL: Disable ControlNet hashing
    env = os.environ.copy()
    env["CONTROLNET_NO_MODEL_HASH"] = "1"
    
    # Start WebUI process
    webui_process = subprocess.Popen([
        "python", "launch.py",
        "--listen", "--port", "7860", "--api",
        "--disable-safe-unpickle", "--enable-insecure-extension-access",
        "--skip-torch-cuda-test", "--skip-install", "--skip-python-version-check",
        "--opt-sdp-attention", "--no-half-vae", "--medvram",
        "--disable-model-loading-ram-optimization",
        "--no-hashing"  # ← ADD THIS: Disables ALL hashing (A1111 + ControlNet)
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    # === FastAPI Proxy ===
    from fastapi import FastAPI, Request, HTTPException
    import httpx
    import asyncio
    import time

    web_app = FastAPI()
    webui_ready = False

    async def wait_for_webui():
        """Wait for WebUI to be ready"""
        global webui_ready
        max_wait = 300  # 5 minutes max
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get("http://localhost:7860")
                    if response.status_code == 200:
                        print("[WEBUI] Ready!")
                        webui_ready = True
                        return True
            except Exception as e:
                print(f"[WEBUI] Not ready yet: {e}")
                await asyncio.sleep(5)
        
        print("[WEBUI] Timeout waiting for startup")
        return False

    @web_app.get("/")
    async def root():
        if not webui_ready:
            return {"message": "WebUI starting... please wait", "status": "starting"}
        return {"message": "WebUI ready", "status": "ready"}

    @web_app.get("/health")
    async def health():
        if not webui_ready:
            raise HTTPException(status_code=503, detail="WebUI not ready")
        return {"status": "healthy"}

    async def proxy(request: Request):
        if not webui_ready:
            raise HTTPException(status_code=503, detail="WebUI not ready")
        
        url = f"http://localhost:7860{request.url.path}"
        if request.url.query:
            url += f"?{request.url.query}"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                req = client.build_request(
                    request.method, url,
                    headers=request.headers,
                    content=await request.body()
                )
                resp = await client.send(req, stream=True)
                return resp
            except httpx.ConnectError:
                raise HTTPException(status_code=503, detail="WebUI not available")

    web_app.add_route("/{path:path}", proxy, methods=["*"])
    
    # Start background task to wait for WebUI
    asyncio.create_task(wait_for_webui())
    
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
    print("=== Volume Contents ===")
    try:
        for p in vol.listdir("/"):
            print(f"Folder: {p.path}")
            try:
                for f in vol.listdir(p.path):
                    print(f"  File: {f.path} ({f.size / (1024**3):.2f} GB)")
            except:
                pass
    except Exception as e:
        print(f"Error listing volume: {e}")

@app.function(image=image, gpu=GPU)
def test_gpu():
    import torch
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM:", f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

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