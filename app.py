import modal
import subprocess
import os
import requests
from pathlib import Path

# Define the Modal app
app = modal.App("sd-webui-controlnet")

# Create and reference a persistent volume for models
vol = modal.Volume.from_name("sd-models", create_if_missing=True)

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git", "wget", "curl")
    .pip_install(
        # PyTorch with CUDA support for Modal
        "torch==2.3.0",
        "torchvision==0.18.0",
        "--extra-index-url https://download.pytorch.org/whl/cu118",
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
        # Clone SD WebUI
        "git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git /sd-webui",
        "cd /sd-webui && git checkout master",
        # Install ControlNet extension
        "cd /sd-webui/extensions && git clone https://github.com/Mikubill/sd-webui-controlnet.git",
        # Create necessary directories
        "mkdir -p /sd-webui/models/Stable-diffusion",
        "mkdir -p /sd-webui/models/ControlNet", 
        "mkdir -p /sd-webui/models/Lora",
        "mkdir -p /sd-webui/models/VAE",
        "mkdir -p /sd-webui/embeddings",
        "mkdir -p /sd-webui/outputs",
    )
)

@app.function(
    image=image,
    gpu="A10G",  # Use A10G for good performance/cost balance
    volumes={"/sd-webui/models": vol},  # Mount volume to models directory
    timeout=600,
    allow_concurrent_inputs=1,  # Limit to 1 concurrent container
    # Enable memory snapshots for faster cold starts
    _experimental_enable_memory_snapshots=True,
)
@modal.web_server(port=7860, startup_timeout=600)
def run_webui():
    """Run Stable Diffusion WebUI with ControlNet"""
    os.chdir("/sd-webui")
    
    # Start the web UI with optimized settings
    subprocess.run([
        "python", "launch.py",
        "--listen", "0.0.0.0",
        "--port", "7860",
        "--api",
        "--disable-safe-unpickle",
        "--no-half-vae",
        "--xformers",
        "--enable-insecure-extension-access",
        "--gradio-auth", "admin:password123",  # Change this!
        "--gradio-img2img-tool", "color-sketch",
        "--gradio-inpaint-tool", "color-sketch",
    ])

@app.function(image=image, volumes={"/sd-webui/models": vol})
def upload_model(local_path: str, model_type: str = "stable-diffusion"):
    """Upload a local model file to the Modal Volume
    
    Args:
        local_path: Path to your local model file
        model_type: Type of model - "stable-diffusion", "controlnet", "lora", "vae"
    """
    vol = modal.Volume.from_name("sd-models")
    
    # Map model types to directories
    model_dirs = {
        "stable-diffusion": "/Stable-diffusion",
        "controlnet": "/ControlNet", 
        "lora": "/Lora",
        "vae": "/VAE"
    }
    
    if model_type not in model_dirs:
        raise ValueError(f"Invalid model_type. Must be one of: {list(model_dirs.keys())}")
    
    remote_path = model_dirs[model_type]
    
    # Upload the file
    vol.put_files(
        remote_path=remote_path,
        local_paths=[local_path],
    )
    vol.flush()
    
    print(f"Successfully uploaded {local_path} to {remote_path}")

@app.function(image=image, volumes={"/sd-webui/models": vol})
def download_controlnet_models():
    """Download common ControlNet models to the volume"""
    vol = modal.Volume.from_name("sd-models")
    
    # Common ControlNet models
    models = [
        {
            "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth",
            "name": "control_v11p_sd15_canny.pth"
        },
        {
            "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth", 
            "name": "control_v11p_sd15_openpose.pth"
        },
        {
            "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_depth.pth",
            "name": "control_v11p_sd15_depth.pth"
        },
        {
            "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_normalbae.pth",
            "name": "control_v11p_sd15_normalbae.pth"
        }
    ]
    
    for model in models:
        print(f"Downloading {model['name']}...")
        response = requests.get(model["url"], stream=True)
        response.raise_for_status()
        
        # Write to volume
        with vol.batch_upload() as batch:
            batch.put_file(
                remote_path=f"/ControlNet/{model['name']}",
                local_file=response.content,
            )
    
    vol.flush()
    print("All ControlNet models downloaded successfully!")

@app.function(image=image, volumes={"/sd-webui/models": vol})
def list_models():
    """List all models in the volume"""
    vol = modal.Volume.from_name("sd-models")
    
    print("=== Models in Volume ===")
    for path in vol.listdir("/"):
        print(f"📁 {path}")
        try:
            files = vol.listdir(f"/{path}")
            for file in files:
                print(f"  📄 {file}")
        except:
            pass

@app.function(image=image, gpu="A10G")
def test_gpu():
    """Test GPU availability and PyTorch CUDA support"""
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
    gpu="A10G",
    _experimental_enable_memory_snapshots=True,
)
def test_snapshot_performance():
    """Test snapshot performance by loading a model"""
    import torch
    import time
    
    print("=== Snapshot Performance Test ===")
    
    # Time the model loading
    start_time = time.time()
    
    # Load a small model to test GPU memory snapshot
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    model = model.cuda()
    
    load_time = time.time() - start_time
    print(f"Model load time: {load_time:.2f} seconds")
    
    # Test inference
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ Model loaded and inference successful!")
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # This function will create a memory snapshot for future runs
    return "Snapshot created - next run will be much faster!"

# CLI commands for easy management
@app.local_entrypoint()
def main():
    """Main entrypoint for local commands"""
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
