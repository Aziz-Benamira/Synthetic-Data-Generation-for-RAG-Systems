# Ollama GPU Configuration for RTX 5060 (8GB VRAM)
# Set this before running Ollama

$env:CUDA_VISIBLE_DEVICES = "0"  # Use first discrete GPU (RTX 5060)
$env:OLLAMA_NUM_GPU = "1"         # Use 1 GPU
$env:OLLAMA_MAX_LOADED_MODELS = "2"  # Load max 2 models at once (4.7GB + 4.4GB = 9.1GB fits in 8GB with offloading)

Write-Host "✅ Ollama GPU settings configured for RTX 5060:"
Write-Host "   CUDA_VISIBLE_DEVICES: $env:CUDA_VISIBLE_DEVICES"
Write-Host "   OLLAMA_NUM_GPU: $env:OLLAMA_NUM_GPU"
Write-Host "   OLLAMA_MAX_LOADED_MODELS: $env:OLLAMA_MAX_LOADED_MODELS"
Write-Host ""
Write-Host "Run 'ollama ps' after starting Ollama to verify GPU usage"
