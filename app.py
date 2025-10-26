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
    timeout=180,
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

    # === Check for models in volume ===
    print(f"[MODEL] Checking volume for models...")
    
    try:
        model_files = list(vol_dir.glob("*.safetensors")) + list(vol_dir.glob("*.ckpt"))
        if model_files:
            print(f"[MODEL] Found {len(model_files)} model(s) in volume:")
            for model_file in model_files:
                size_gb = model_file.stat().st_size / (1024**3)
                print(f"  - {model_file.name} ({size_gb:.2f} GB)")
        else:
            print("[MODEL] No models found in volume. Please upload models via upload_model() function.")
            print("[MODEL] You can use: python -m modal run app.py upload-model <path> stable-diffusion")
    except Exception as e:
        print(f"[MODEL] Error checking models: {e}")

    # === Start WebUI ===
    print("Starting WebUI on port 7860...")
    
    # CRITICAL: Disable auto downloads and hashing
    env = os.environ.copy()
    env["CONTROLNET_NO_MODEL_HASH"] = "1"
    # Removed HF_HUB_DISABLE_DOWNLOAD - was blocking SD WebUI startup
    
    # Start WebUI process
    print("[WEBUI] Starting subprocess...")
    webui_process = subprocess.Popen([
        "python", "-u", "launch.py",  # -u flag for unbuffered output
        "--listen", "--port", "7860", "--api",
        "--share",
        "--disable-safe-unpickle", "--enable-insecure-extension-access",
        "--skip-torch-cuda-test", "--skip-install", "--skip-python-version-check",
        "--opt-sdp-attention", "--no-half-vae", "--medvram",
        "--disable-model-loading-ram-optimization",
        "--no-hashing"
    ], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=0)
    
    print(f"[WEBUI] Process started with PID: {webui_process.pid}")

    # === FastAPI Proxy ===
    from fastapi import FastAPI, Request, HTTPException
    from contextlib import asynccontextmanager
    import httpx
    import asyncio
    import time

    webui_ready = False

    async def wait_for_webui():
        """Wait for WebUI to be ready"""
        global webui_ready
        max_wait = 300  # 5 minutes max
        start_time = time.time()
        check_count = 0
        
        print("[WEBUI] Starting health check loop...")
        
        while time.time() - start_time < max_wait:
            check_count += 1
            
            # Check if process is still running
            if webui_process.poll() is not None:
                print(f"[WEBUI] Process died with return code: {webui_process.returncode}")
                return False
            
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get("http://localhost:7860")
                    if response.status_code == 200:
                        print(f"[WEBUI] Ready! (after {check_count} checks)")
                        webui_ready = True
                        return True
                    else:
                        print(f"[WEBUI] Check {check_count}: Got status {response.status_code}")
            except httpx.ConnectError as e:
                print(f"[WEBUI] Check {check_count}: Connection failed - {e}")
            except httpx.TimeoutException:
                print(f"[WEBUI] Check {check_count}: Timeout")
            except Exception as e:
                print(f"[WEBUI] Check {check_count}: Error - {e}")
            
            await asyncio.sleep(10)  # Check every 10 seconds
        
        print(f"[WEBUI] Timeout after {check_count} checks")
        return False

    async def monitor_webui_output():
        """Monitor WebUI process output"""
        global webui_ready
        try:
            while True:
                line = webui_process.stdout.readline()
                if not line:
                    break
                line_str = line.strip()
                print(f"[WEBUI-OUT] {line_str}")
                # Check if WebUI is ready
                if "Running on local URL" in line_str:
                    print("[WEBUI-OUT] WebUI server is ready!")
                    # Give it a moment to fully start
                    await asyncio.sleep(2)
                    webui_ready = True
        except Exception as e:
            print(f"[WEBUI-OUT] Error reading output: {e}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        print("[STARTUP] Starting WebUI health check...")
        asyncio.create_task(monitor_webui_output())
        # Wait for WebUI to be ready
        await wait_for_webui()
        print("[STARTUP] FastAPI app ready!")
        yield
        # Shutdown
        print("[SHUTDOWN] Cleaning up...")
        if webui_process.poll() is None:
            print("[SHUTDOWN] Terminating WebUI process...")
            webui_process.terminate()
            try:
                webui_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("[SHUTDOWN] Force killing WebUI process...")
                webui_process.kill()

    web_app = FastAPI(lifespan=lifespan)

    @web_app.get("/health")
    async def health():
        global webui_ready
        if not webui_ready:
            raise HTTPException(status_code=503, detail="WebUI not ready")
        return {"status": "healthy"}

    async def proxy(request: Request):
        global webui_ready
        if not webui_ready:
            raise HTTPException(status_code=503, detail="WebUI not ready")
        
        # Build target URL
        url = f"http://localhost:7860{request.url.path}"
        if request.url.query:
            url += f"?{request.url.query}"
        
        print(f"[PROXY] {request.method} {request.url.path} -> {url}")
        
        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                req = client.build_request(
                    request.method, url,
                    headers={k: v for k, v in request.headers.items() if k.lower() not in ['host', 'content-length']},
                    content=await request.body()
                )
                resp = await client.send(req, stream=True)
                
                # Wrap stream with proper cleanup
                async def stream_generator():
                    try:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
                    finally:
                        # Ensure cleanup
                        if hasattr(resp, 'aclose'):
                            try:
                                await resp.aclose()
                            except:
                                pass
                
                from fastapi.responses import StreamingResponse
                return StreamingResponse(
                    stream_generator(),
                    status_code=resp.status_code,
                    headers={k: v for k, v in resp.headers.items() if k.lower() not in ['transfer-encoding', 'content-encoding']},
                    media_type=resp.headers.get("content-type")
                )
        except httpx.ConnectError as e:
            print(f"[PROXY] ConnectError: {e}")
            raise HTTPException(status_code=503, detail="WebUI not available")
        except Exception as e:
            print(f"[PROXY] Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Add root route and catch-all proxy
    web_app.add_route("/", proxy, methods=["*"])
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