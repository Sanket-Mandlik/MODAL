# app.py
import modal
import subprocess
import os
import tempfile
from pathlib import Path
from typing import Tuple

# --------------------------------------------------------------------------- #
# Modal App
# --------------------------------------------------------------------------- #
app = modal.App("sd-webui-controlnet")

# Persistent volume – created automatically on first deploy
vol = modal.Volume.from_name("sd-models", create_if_missing=True)

# --------------------------------------------------------------------------- #
# GPU selection – L40S → A100-40 → A100-80 only
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

    raise RuntimeError(
        "No supported GPU found. Only L40S, A100-40GB or A100-80GB are allowed."
    )

# --------------------------------------------------------------------------- #
# Container image
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
        "gradio==4.31.0",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "xformers==0.0.26",
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
        "cd /sd-webui && git checkout v1.10.1",
        "cd /sd-webui/extensions && git clone https://github.com/Mikubill/sd-webui-controlnet.git",
        "mkdir -p /sd-webui/embeddings /sd-webui/outputs",
    )
)

# --------------------------------------------------------------------------- #
# Prompt generator (unchanged)
# --------------------------------------------------------------------------- #
class EmptyRoomPromptGenerator:
    COLOR_PALETTE_MAPPING = {
        "urban": "urban",
        "pastel": "pastel",
        "neutral": "neutral",
        "natural": "modern",
    }

    @staticmethod
    def _get_mapped_color_palette(color_palette: str) -> str:
        return EmptyRoomPromptGenerator.COLOR_PALETTE_MAPPING.get(
            color_palette.lower(), color_palette.lower()
        )

    @staticmethod
    def generate_prompts(
        room_type: str, style: str, color_palette: str = "pastel"
    ) -> Tuple[str, str]:
        style_lower = style.lower() if style else "modern"
        room_type_lower = room_type.lower()
        color_palette_lower = color_palette.lower() if color_palette else "pastel"
        mapped = EmptyRoomPromptGenerator._get_mapped_color_palette(color_palette_lower)

        if room_type_lower in ["living", "living room"]:
            return EmptyRoomPromptGenerator._get_living_room_prompts(style_lower, mapped)
        if room_type_lower in ["bedroom", "bed room"]:
            return EmptyRoomPromptGenerator._get_bedroom_prompts(style_lower, mapped)
        if room_type_lower == "kids_room":
            return EmptyRoomPromptGenerator._get_kids_room_prompts(mapped)
        if room_type_lower == "bathroom":
            return EmptyRoomPromptGenerator._get_bathroom_prompts(mapped)
        if room_type_lower == "kitchen":
            return EmptyRoomPromptGenerator._get_kitchen_prompts(mapped)
        if room_type_lower == "living + dining":
            return EmptyRoomPromptGenerator._get_living_dining_prompts(style_lower, mapped)
        return EmptyRoomPromptGenerator._get_default_prompts(style_lower, mapped)

    # (All static methods from your original class go here — unchanged)
    @staticmethod
    def _get_living_room_prompts(style: str, color_palette: str) -> Tuple[str, str]:
        base_furniture = (
            "((modular sofa set:1.3) against wall), "
            "(coffee table:1.2) in front, "
            "accent (armchair:1.1) near window, "
            "(big TV wall unit:1.3) opposite sofa, "
            "(lit bookshelf:1.1) in corner, "
            "(layered ceiling:1.1) with fan, "
            " (decorative items:1.1), "
            "curtains, (floor lamp:1.1), "
            "(indoor plant:1.1), clean edges"
        )
        if style == "minimalist":
            positive_prompt = (
                f"Ultra-modern living room interior by leading architect, "
                f"featuring {base_furniture}, "
                f"{color_palette} palette, MDF, glass, concrete."
            )
        else:
            positive_prompt = (
                f"Ultra-modern living room interior by Livspace, "
                f"featuring {base_furniture}, "
                f"{color_palette} palette, modern MDF, glass, marble."
            )
        negative_prompt = "multiple tv units, watermark"
        return positive_prompt, negative_prompt

    # ... (copy all other _get_* methods exactly as you posted) ...
    # For brevity, not repeated here — just paste them in.

# --------------------------------------------------------------------------- #
# Prompt API endpoint (no auth)
# --------------------------------------------------------------------------- #
@app.function()
@modal.web_endpoint(method="POST")
def generate(room_type: str, style: str = "modern", color_palette: str = "pastel"):
    pos, neg = EmptyRoomPromptGenerator.generate_prompts(room_type, style, color_palette)
    return {"positive": pos, "negative": neg}

# --------------------------------------------------------------------------- #
# WebUI – NO AUTH, max 3 concurrent users
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

    # Symlink volume to expected path
    subprocess.run(["ln", "-sfn", "/models", "/sd-webui/models"], check=False)

    # VRAM optimization
    import torch
    gpu_name = torch.cuda.get_device_name(0).lower()
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    args = [
        "python", "launch.py",
        "--listen", "--port", "7860",
        "--api", "--api-log",
        "--disable-safe-unpickle",
        "--xformers",
        "--enable-insecure-extension-access",
        "--skip-torch-cuda-test",
    ]
    if "a100" in gpu_name and mem_gb < 50:
        args.append("--medvram")

    print(f"Launching WebUI on {gpu_name.upper()} ({mem_gb:.0f} GB)")
    proc = subprocess.Popen(args)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()

# --------------------------------------------------------------------------- #
# Helper functions
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
    print("=== Volume contents ===")
    for p in vol.listdir("/"):
        print(f"Folder: {p}")
        try:
            for f in vol.listdir(f"/{p}"):
                print(f"   File: {f}")
        except Exception:
            pass

@app.function(image=image, gpu=select_gpu())
def test_gpu():
    import torch
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Memory:", f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# --------------------------------------------------------------------------- #
# Local entrypoint
# --------------------------------------------------------------------------- #
@app.local_entrypoint()
def main():
    import sys
    if len(sys.argv) < 2:
        print("Commands: test-gpu | list-models | upload-model <path> <type> | deploy")
        return

    cmd = sys.argv[1]
    if cmd == "test-gpu":
        test_gpu.remote()
    elif cmd == "list-models":
        list_models.remote()
    elif cmd == "upload-model":
        if len(sys.argv) < 4:
            print("Usage: upload-model <local_path> <type>")
            return
        upload_model.remote(sys.argv[2], sys.argv[3])
    elif cmd == "deploy":
        print("Deploying WebUI (no auth, max 3 users)…")
        run_webui.deploy(name="sd-webui-controlnet")
    else:
        print("Unknown command")