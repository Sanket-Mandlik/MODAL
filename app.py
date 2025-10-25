import modal
import subprocess
import os
import requests
import tempfile
from pathlib import Path

# Define the Modal app
app = modal.App("sd-webui-controlnet")

# Persistent volume for models
vol = modal.Volume.from_name("sd-models", create_if_missing=True)

# Build the container image with dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git", "wget", "curl", "python3.11", "python3.11-venv", "python3.11-dev", "python3-pip", "build-essential")
    .run_commands(
        "rm -f /usr/local/bin/python /usr/local/bin/python3 /usr/local/bin/pip /usr/local/bin/pip3",
        "ln -s /usr/bin/python3.11 /usr/local/bin/python",
        "ln -s /usr/bin/python3.11 /usr/local/bin/python3",
        "ln -s /usr/bin/pip3 /usr/local/bin/pip",
        "ln -s /usr/bin/pip3 /usr/local/bin/pip3",
        "python --version",
        "pip --version",
        "pip install --upgrade pip --break-system-packages",
        "pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1+cu121 torchvision --break-system-packages"
    )
    .pip_install(
        # Core dependencies
        "gradio",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "xformers",
        "opencv-python",
        "pillow",
        "numpy",
        "scipy",
        "tqdm",
        "omegaconf",
        "einops",
        "controlnet-aux",
    )
    .run_commands(
        "git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /sd-webui",
        "cd /sd-webui && git checkout master",
        "cd /sd-webui/extensions && git clone https://github.com/Mikubill/sd-webui-controlnet.git",
        "mkdir -p /sd-webui/embeddings",
        "mkdir -p /sd-webui/outputs",
    )
)

# Use L4 GPU - good balance of performance and cost for SD WebUI
GPU = "L4"

@app.function(
    image=image,
    gpu=GPU,
    volumes={"/models": vol},
    timeout=600,
)
@modal.web_server(port=7860, startup_timeout=600)
def run_webui():
    """Run Stable Diffusion WebUI with ControlNet."""
    os.chdir("/sd-webui")
    # Create symlink from /sd-webui/models to /models
    subprocess.run(["ln", "-sfn", "/models", "/sd-webui/models"], check=False)
    subprocess.run([
        "python", "launch.py",
        "--listen", "0.0.0.0",
        "--port", "7860",
        "--api",
        "--skip-python-version-check",
        "--disable-safe-unpickle",
        "--no-half-vae",
        "--xformers",
        "--enable-insecure-extension-access",
        "--gradio-img2img-tool", "color-sketch",
        "--gradio-inpaint-tool", "color-sketch",
    ], check=True)

@app.function(image=image, volumes={"/models": vol})
def upload_model(local_path: str, model_type: str = "stable-diffusion"):
    """Upload a local model file to the Modal Volume."""
    model_dirs = {
        "stable-diffusion": "/Stable-diffusion",
        "controlnet": "/ControlNet",
        "lora": "/Lora",
        "vae": "/VAE"
    }
    if model_type not in model_dirs:
        raise ValueError(f"Invalid model_type. Must be one of: {list(model_dirs.keys())}")

    remote_path = model_dirs[model_type]
    vol.put_file(remote_path + "/" + Path(local_path).name, local_path)
    vol.flush()
    print(f"Successfully uploaded {local_path} to {remote_path}")

@app.function(image=image, volumes={"/models": vol})
def download_controlnet_models():
    """Download common ControlNet models to the volume."""
    models = [
        {
            "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth",
            "name": "control_v11p_sd15_canny.pth"
        },
        {
            "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_depth.pth",
            "name": "control_v11p_sd15_depth.pth"
        }
    ]
    for model in models:
        print(f"Downloading {model['name']}...")
        response = requests.get(model["url"], stream=True)
        response.raise_for_status()
        # Save to temp file then upload
        with tempfile.NamedTemporaryFile(delete=False) as tempf:
            for chunk in response.iter_content(chunk_size=8192):
                tempf.write(chunk)
            tempf.flush()
            vol.put_file(f"/ControlNet/{model['name']}", tempf.name)
        os.unlink(tempf.name)
    vol.flush()
    print("All ControlNet models downloaded successfully!")

@app.function(image=image, volumes={"/models": vol})
def list_models():
    """List all models in the volume."""
    print("=== Models in Volume ===")
    for path in vol.listdir("/"):
        print(f"📁 {path}")
        try:
            files = vol.listdir(f"/{path}")
            for file in files:
                print(f"  📄 {file}")
        except Exception:
            pass

@app.function(image=image, gpu=GPU)
def test_gpu():
    """Test GPU availability and PyTorch CUDA support."""
    import torch
    print("=== GPU Test Results ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("❌ CUDA not available!")

@app.function(
    image=image,
    gpu=GPU,
)
def test_snapshot_performance():
    """Test snapshot performance by loading a model."""
    import torch
    import time
    print("=== Snapshot Performance Test ===")
    start_time = time.time()
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model = model.cuda()
    load_time = time.time() - start_time
    print(f"Model load time: {load_time:.2f} seconds")
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    with torch.no_grad():
        output = model(dummy_input)
    print(f"✅ Model loaded and inference successful!")
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return "Snapshot created - next run will be much faster!"

@app.local_entrypoint()
def main():
    """Main entrypoint for local commands."""
    import sys
    if len(sys.argv) < 2:
        print("Usage: modal run app.py [command]")
        print("Commands:")
        print("  test-gpu - Test GPU availability")
        print("  test-snapshots - Test snapshot performance")
        print("  list-models - List all models in volume")
        print("  download-controlnet - Download common ControlNet models")
        print("  upload-model <path> <type> - Upload a local model")
        return

    command = sys.argv[1]
    if command == "test-gpu":
        test_gpu.remote()
    elif command == "test-snapshots":
        test_snapshot_performance.remote()
    elif command == "list-models":
        list_models.remote()
    elif command == "download-controlnet":
        download_controlnet_models.remote()
    elif command == "upload-model":
        if len(sys.argv) < 4:
            print("Usage: modal run app.py upload-model <local_path> <model_type>")
            return
        upload_model.remote(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown command: {command}")
