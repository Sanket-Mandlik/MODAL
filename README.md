# Stable Diffusion WebUI with ControlNet on Modal.com

This project deploys Stable Diffusion WebUI with ControlNet support on Modal.com using Modal Volumes for persistent model storage.

## Features

- ✅ Full Stable Diffusion WebUI with ControlNet extension
- ✅ PyTorch with CUDA support (A10G GPU)
- ✅ Persistent model storage using Modal Volumes
- ✅ Easy model upload/download management
- ✅ Pre-configured ControlNet models
- ✅ Gradio web interface with authentication

## Quick Start

### 1. Install Modal CLI
```bash
pip install modal
```

### 2. Authenticate with Modal
```bash
modal setup
```

### 3. Deploy the Application
```bash
modal deploy app.py
```

This will:
- Build the container with SD WebUI + ControlNet
- Create a persistent volume for models
- Deploy the web interface
- Provide a public URL for access

## Model Management

### Upload Your Local Models
```bash
# Upload a Stable Diffusion checkpoint
modal run app.py upload-model /path/to/your-model.safetensors stable-diffusion

# Upload a ControlNet model
modal run app.py upload-model /path/to/controlnet.pth controlnet

# Upload a LoRA model
modal run app.py upload-model /path/to/lora.safetensors lora
```

### Download Pre-configured ControlNet Models
```bash
modal run app.py download-controlnet
```

This downloads common ControlNet models:
- Canny edge detection
- OpenPose human pose
- Depth estimation
- Normal map

### List All Models
```bash
modal run app.py list-models
```

## Testing

### Test GPU Availability
```bash
modal run app.py test-gpu
```

### Access the Web Interface
After deployment, Modal provides a public URL. The interface includes:
- Text-to-image generation
- Image-to-image with ControlNet
- Inpainting with ControlNet
- All ControlNet conditioning types

Default login: `admin:password123` (change this in the code!)

## Directory Structure

Models are organized in the Modal Volume as:
```
/sd-webui/models/
├── Stable-diffusion/     # Main SD checkpoints (.ckpt, .safetensors)
├── ControlNet/          # ControlNet models (.pth)
├── Lora/                # LoRA models (.safetensors)
├── VAE/                 # VAE models
└── embeddings/          # Textual inversion embeddings
```

## Key Differences from RunPod/FAL

### Modal Advantages:
- **No Docker needed**: Everything defined in Python
- **Automatic scaling**: Serverless with GPU autoscaling
- **Memory snapshots**: Faster cold starts (1-2s vs 10-60s)
- **Persistent volumes**: Models survive redeploys
- **Pay-per-use**: Only pay when running

### Cost Optimization:
- Uses `allow_concurrent_inputs=1` to limit containers
- Models stored in Volume (not re-downloaded)
- Memory snapshots reduce initialization time

## Customization

### Change GPU Type
Edit the `gpu` parameter in `@app.function()`:
```python
@app.function(gpu="A100")  # For higher VRAM needs
```

### Add More Models
Extend the `download_controlnet_models()` function or upload manually.

### Modify WebUI Settings
Edit the `subprocess.run()` call in `run_webui()` to change launch parameters.

## Troubleshooting

### Common Issues:
1. **CUDA not available**: Ensure PyTorch is installed with CUDA support
2. **Models not loading**: Check Volume mount and file paths
3. **Slow startup**: Use memory snapshots for faster cold starts
4. **Out of memory**: Switch to A100 GPU or reduce batch size

### Debug Commands:
```bash
# Check GPU status
modal run app.py test-gpu

# List volume contents
modal run app.py list-models

# View logs
modal logs sd-webui-controlnet
```

## Security Notes

- Change the default Gradio password in `run_webui()`
- Consider adding IP restrictions for production use
- Models in Volume are persistent - manage access carefully

## Support

For issues:
- Check Modal documentation: https://modal.com/docs
- SD WebUI issues: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- ControlNet issues: https://github.com/Mikubill/sd-webui-controlnet