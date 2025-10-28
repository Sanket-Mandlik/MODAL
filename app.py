# app.py
import modal
import subprocess
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Modal App
# --------------------------------------------------------------------------- #
app = modal.App("sd-webui-controlnet-sdxl")  # Changed app name to force fresh build

# Persistent volume
vol = modal.Volume.from_name("sd-models", create_if_missing=True)

# GPU
GPU = "L40S"

# --------------------------------------------------------------------------- #
# Image — PRE-BUILD SNAPSHOT (WebUI + extensions + symlinks) - SDXL Version
# --------------------------------------------------------------------------- #
sd_webui_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "wget", "curl", "libglib2.0-0", "libsm6", "libxext6", "libxrender1",
        "libgl1-mesa-glx", "cmake", "build-essential", "pkg-config", "libhdf5-dev"
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

        # SDXL-specific dependencies (core only)
        "diffusers>=0.21.0", "compel>=2.0.0",

        # Others
        "pytorch-lightning==1.9.0", "safetensors", "timm", "kornia",
        "GitPython", "blendmodes", "clean-fid", "diskcache",
        "fastapi>=0.90.1", "inflection", "jsonmerge",
        "lark", "open-clip-torch==2.24.0", "piexif", "protobuf==3.20.0",
        "psutil", "requests", "resize-right", "scikit-image>=0.19",
        "tomesd", "torchdiffeq", "torchsde", "ftfy", "regex",
        "taming-transformers", "clip", "httpx", "anyio"
    )
    .run_commands(
        "git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /sd-webui",
        "cd /sd-webui && git checkout v1.10.1",
        "cd /sd-webui/extensions && git clone https://github.com/Mikubill/sd-webui-controlnet.git",
        "mkdir -p /sd-webui/embeddings /sd-webui/outputs /sd-webui/models/ControlNet /sd-webui/models/Stable-diffusion",
        # Clear any existing cache and ensure clean SDXL setup
        "rm -rf /sd-webui/cache /sd-webui/models/*/.*cache* /sd-webui/models/*/.*tmp*",
        # Download base SDXL model as fallback (full size model)

        # Ensure proper permissions for model directories
        "chmod -R 755 /sd-webui/models"
    )
)

# --------------------------------------------------------------------------- #
# WebUI — PROXY + NO BLOCKING
# --------------------------------------------------------------------------- #
from modal import asgi_app
import modal

@app.function(
    image=sd_webui_image,          # Snapshot – build once, reuse forever
    gpu=GPU,                       # L40S GPU
    volumes={"/models": vol},      # Persistent models
    timeout=600,                 # 10 minutes - allow for image generation + response
    startup_timeout=1800,         # BUILD/STARTUP CAN TAKE 30 MIN
    scaledown_window=5,        # Keep alive for 30 seconds after request
)

@modal.concurrent(max_inputs=1)
@asgi_app()
def run_webui():
    os.chdir("/sd-webui")

    # === Clear SD 1.5 Cache and Setup SDXL ===
    print("[CACHE] Clearing SD 1.5 cache and preparing for SDXL...")
    
    # Clear WebUI cache directories
    cache_dirs = [
        "/sd-webui/cache",
        "/sd-webui/models/Stable-diffusion",
        "/sd-webui/models/ControlNet",
        "/sd-webui/models/VAE",
        "/sd-webui/models/Lora"
    ]
    
    for cache_dir in cache_dirs:
        cache_path = Path(cache_dir)
        if cache_path.exists() and not cache_path.is_symlink():
            print(f"[CACHE] Removing old cache: {cache_dir}")
            import shutil
            shutil.rmtree(cache_path, ignore_errors=True)

    # === Volume & Symlink Setup ===
    vol_sd_dir = Path("/models/Stable-diffusion")
    vol_cn_dir = Path("/models/ControlNet")
    webui_sd_dir = Path("/sd-webui/models/Stable-diffusion")
    webui_cn_dir = Path("/sd-webui/models/ControlNet")
    
    # Ensure volume directories exist
    vol_sd_dir.mkdir(parents=True, exist_ok=True)
    vol_cn_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DEBUG] Volume mount: /models")
    print(f"[DEBUG] SD volume dir exists: {vol_sd_dir.exists()}")
    print(f"[DEBUG] ControlNet volume dir exists: {vol_cn_dir.exists()}")

    # List files in volume directories
    try:
        sd_files = [p.name for p in vol_sd_dir.iterdir()]
        cn_files = [p.name for p in vol_cn_dir.iterdir()]
        print(f"[DEBUG] Files in /models/Stable-diffusion: {sd_files}")
        print(f"[DEBUG] Files in /models/ControlNet: {cn_files}")
    except Exception as e:
        print(f"[DEBUG] Could not list volume dirs: {e}")

    # Create symlinks for both directories
    for vol_dir, webui_dir in [(vol_sd_dir, webui_sd_dir), (vol_cn_dir, webui_cn_dir)]:
        if not webui_dir.is_symlink():
            print(f"[SYMLINK] Creating: {vol_dir} → {webui_dir}")
            try:
                os.symlink(vol_dir, webui_dir)
                print(f"[SYMLINK] Success for {webui_dir.name}")
            except Exception as e:
                print(f"[SYMLINK] Failed for {webui_dir.name}: {e}")
                raise
        else:
            target = os.readlink(webui_dir)
            print(f"[SYMLINK] Already exists {webui_dir.name} → {target}")

    # === Check for SDXL models in volume ===
    print(f"[MODEL] Checking volume for SDXL models...")
    
    # Clean up partial downloads from both directories
    for vol_dir in [vol_sd_dir, vol_cn_dir]:
        try:
            partial_files = list(vol_dir.glob("*.partial"))
            if partial_files:
                print(f"[CLEANUP] Removing {len(partial_files)} partial download(s) from {vol_dir.name}...")
                for partial_file in partial_files:
                    partial_file.unlink()
                    print(f"  Removed: {partial_file.name}")
        except Exception as e:
            print(f"[CLEANUP] Error removing partial files from {vol_dir.name}: {e}")
    
    # Check SDXL checkpoints
    try:
        sd_model_files = list(vol_sd_dir.glob("*.safetensors")) + list(vol_sd_dir.glob("*.ckpt"))
        if sd_model_files:
            print(f"[MODEL] Found {len(sd_model_files)} SDXL checkpoint(s):")
            for model_file in sd_model_files:
                size_gb = model_file.stat().st_size / (1024**3)
                print(f"  - {model_file.name} ({size_gb:.2f} GB)")
        else:
            print("[MODEL] No SDXL checkpoints found in volume.")
    except Exception as e:
        print(f"[MODEL] Error checking SDXL checkpoints: {e}")
    
    # Check ControlNet models
    try:
        cn_model_files = list(vol_cn_dir.glob("*.safetensors")) + list(vol_cn_dir.glob("*.pth"))
        if cn_model_files:
            print(f"[MODEL] Found {len(cn_model_files)} ControlNet model(s):")
            for model_file in cn_model_files:
                size_gb = model_file.stat().st_size / (1024**3)
                print(f"  - {model_file.name} ({size_gb:.2f} GB)")
        else:
            print("[MODEL] No ControlNet models found in volume.")
    except Exception as e:
        print(f"[MODEL] Error checking ControlNet models: {e}")

    # === Start WebUI with SDXL optimizations ===
    print("Starting WebUI with SDXL optimizations on port 7860...")
    
    env = os.environ.copy()
    env["CONTROLNET_NO_MODEL_HASH"] = "1"
    # Clear any SD 1.5 cache environment variables
    env.pop("SD_WEBUI_CACHE_DIR", None)
    env.pop("HF_HOME", None)
    env.pop("TRANSFORMERS_CACHE", None)

    webui_process = subprocess.Popen([
        "python", "-u", "launch.py",
        "--listen", "--port", "7860", "--api",
        "--disable-safe-unpickle", "--enable-insecure-extension-access",
        "--skip-torch-cuda-test", "--skip-install", "--skip-python-version-check",
        "--opt-sdp-attention", "--no-half-vae", "--medvram",
        "--disable-model-loading-ram-optimization",
        "--no-hashing", "--no-download-sd-model",  # Don't auto-download SD 1.5 models
        "--xformers",  # Better memory efficiency for SDXL
        "--disable-nan-check",  # SDXL compatibility
        "--api-log"  # Enable API logging for debugging
    ], env=env, stdout=None, stderr=None)  # Don't capture output - let it go directly to Modal logs
    
    print(f"[WEBUI] Process started with PID: {webui_process.pid}")

    # === FastAPI Proxy ===
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import Response, StreamingResponse
    from contextlib import asynccontextmanager
    import httpx
    import asyncio
    import time

    webui_ready = False

    async def wait_for_webui():
        global webui_ready
        max_wait = 300
        start_time = time.time()
        check_count = 0
        
        print("[WEBUI] Starting health check loop...")
        
        while time.time() - start_time < max_wait:
            check_count += 1
            
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
            
            await asyncio.sleep(10)
        
        print(f"[WEBUI] Timeout after {check_count} checks")
        return False

    async def monitor_webui_output():
        global webui_ready
        try:
            while True:
                line = webui_process.stdout.readline()
                if not line:
                    break
                line_str = line.strip()
                print(f"[WEBUI-OUT] {line_str}")
                if "Running on local URL" in line_str:
                    print("[WEBUI-OUT] WebUI server is ready!")
                    await asyncio.sleep(2)
                    webui_ready = True
        except Exception as e:
            print(f"[WEBUI-OUT] Error reading output: {e}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        print("[STARTUP] Waiting for WebUI to be ready...")
        await wait_for_webui()
        print("[STARTUP] FastAPI app ready!")
        
        try:
            yield
        finally:
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
        
        # Actually check if WebUI API is responding
        try:
            async with httpx.AsyncClient(timeout=5.0) as test_client:
                test_resp = await test_client.get("http://localhost:7860/sdapi/v1/progress")
                return {"status": "healthy", "webui_api_responding": test_resp.status_code == 200}
        except:
            return {"status": "healthy", "webui_api_responding": False}

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
            # Increased timeout for image generation POST requests
            # Set read timeout to 1800s (30 min) to allow for large image responses
            is_img_gen = request.url.path in ["/sdapi/v1/txt2img", "/sdapi/v1/img2img"]
            read_timeout = 1800.0 if is_img_gen else 600.0
            timeout = httpx.Timeout(60.0, connect=10.0, read=read_timeout, write=60.0, pool=10.0)
            print(f"[PROXY] Timeout config: read={read_timeout}s for {'image gen' if is_img_gen else 'regular'} request")
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                # For ALL requests (GET, POST, etc), use streaming to avoid hanging
                if request.method in ["GET", "POST", "PUT", "PATCH"]:
                    print(f"[PROXY] Fetching full response for {request.method} {request.url.path}")
                    
                    # Get request body
                    body_data = await request.body()
                    
                    print(f"[PROXY] Request body size: {len(body_data)} bytes")
                    
                    # Build request exactly as the original working code did
                    req = client.build_request(
                        request.method, url,
                        headers={k: v for k, v in request.headers.items() if k.lower() not in ['host', 'content-length']},
                        content=body_data
                    )
                    
                    print(f"[PROXY] Sending to WebUI...")
                    try:
                        resp = await client.send(req, stream=False)
                        print(f"[PROXY] Got response: {resp.status_code}, size: {len(resp.content) if resp.content else 0} bytes")
                    except Exception as e:
                        print(f"[PROXY] Error: {e}")
                        import traceback
                        traceback.print_exc()
                        raise
                    
                    content = resp.content
                    
                    # For debugging, verify the response contains base64 image data
                    if content:
                        try:
                            import json
                            response_data = json.loads(content.decode('utf-8'))
                            print(f"[PROXY] Response is valid JSON with keys: {list(response_data.keys())}")
                            
                            # Check if it has images array (base64 format)
                            if 'images' in response_data:
                                num_images = len(response_data['images'])
                                # Check if first image is base64
                                if response_data['images']:
                                    first_image = response_data['images'][0]
                                    if first_image.startswith('data:image'):
                                        print(f"[PROXY] ✅ Response contains {num_images} image(s) in base64 format")
                                        print(f"[PROXY] First image base64 length: {len(first_image)} chars")
                                    else:
                                        print(f"[PROXY] ⚠️ First image doesn't appear to be base64: {first_image[:100]}")
                        except Exception as e:
                            print(f"[PROXY] Response is not valid JSON: {e}")
                            print(f"[PROXY] First 100 bytes: {content[:100]}")
                    
                    # Filter problematic headers and add content-length
                    filtered_headers = {}
                    for k, v in resp.headers.items():
                        if k.lower() not in ['transfer-encoding', 'content-encoding', 'connection']:
                            filtered_headers[k] = v
                    # Explicitly set content-length for proper response
                    filtered_headers['content-length'] = str(len(content))
                    
                    print(f"[PROXY] Returning response with {len(content)} bytes")
                    return Response(
                        content=content,
                        status_code=resp.status_code,
                        headers=filtered_headers,
                        media_type=resp.headers.get("content-type")
                    )
                else:
                    # For other requests, use streaming
                    req = client.build_request(
                        request.method, url,
                        headers={k: v for k, v in request.headers.items() if k.lower() not in ['host', 'content-length']},
                        content=await request.body()
                    )
                    resp = await client.send(req, stream=True)
                    
                    # Wrap stream with proper cleanup
                    async def stream_generator():
                        try:
                            async for chunk in resp.aiter_bytes(chunk_size=8192):
                                yield chunk
                        except httpx.ReadError as e:
                            print(f"[PROXY] ReadError during streaming: {e}")
                            # Don't raise, just end the stream gracefully
                        finally:
                            # Ensure cleanup
                            if hasattr(resp, 'aclose'):
                                try:
                                    await resp.aclose()
                                except:
                                    pass
                    
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


        
    # Catch-all proxy route
    web_app.add_route("/{full_path:path}", proxy, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
    web_app.api_route("/", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])(proxy)
    
    return web_app

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@app.function(image=sd_webui_image, volumes={"/models": vol})
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

@app.function(image=sd_webui_image, volumes={"/models": vol})
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

@app.function(image=sd_webui_image, gpu=GPU)
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