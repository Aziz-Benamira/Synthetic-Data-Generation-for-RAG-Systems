# 🤖 Quantized LLM Installation Guide - ENSTA Cluster

## 📋 Overview

Installing two quantized models with **sequential execution** only:

1. **LLaMA 3.3 70B Instruct (Q4_K_M)** - ~38 GB on disk, ~36-38 GB VRAM
2. **DeepSeek R1 32B (Q4_K_M)** - ~18 GB on disk, ~18-20 GB VRAM

**Total disk space needed:** ~56 GB
**Your available space:** 15 TB ✅

---

## 🔧 Installation Methods

### **Method 1: Interactive (Recommended for first-time)**

Get GPU node access and install interactively:

```bash
# 1. Request GPU node (4 hours for download + compilation)
srun --partition=ENSTA-l40s --gres=gpu:1 --mem=48G --cpus-per-task=16 --time=04:00:00 --pty bash

# 2. Activate environment
source ~/envs/agentic_ai/bin/activate

# 3. Install HuggingFace CLI (for downloading models)
pip install huggingface-hub

# 4. Run installation
cd ~/projects/Agentic_AI
bash cluster_utils/install_models.sh

# 5. When done, exit
exit
```

### **Method 2: Submit SLURM Job (Fire and forget)**

Submit job and let it run in background:

```bash
# Make sure you're on login node
cd ~/projects/Agentic_AI

# Submit installation job
sbatch cluster_utils/slurm_install_models.sh

# Check status
squeue -u $USER

# View progress
tail -f logs/install_models_*.out
```

---

## 📦 What Gets Installed

### Directory Structure:
```
~/models/
├── llama.cpp/                          # GGUF inference engine
│   ├── llama-cli                       # CLI executable
│   └── llama-server                    # Server executable
├── llama-3.3-70b-instruct/
│   └── Llama-3.3-70B-Instruct-Q4_K_M.gguf    # ~38 GB
├── deepseek-r1-32b/
│   └── DeepSeek-R1-Q4_K_M.gguf               # ~18 GB
└── model_info.txt                      # Reference info
```

### Storage Usage After Installation:
- llama.cpp: ~500 MB (compiled)
- LLaMA 3.3 70B: ~38 GB
- DeepSeek R1 32B: ~18 GB
- **Total: ~56.5 GB** (0.4% of your available space!)

---

## 🧪 Testing Models

### **Method 1: Interactive Testing**

```bash
# Get GPU access
srun --partition=ENSTA-l40s --gres=gpu:1 --mem=48G --time=01:00:00 --pty bash

# Go to models directory
cd ~/models

# Test LLaMA 3.3 70B
./llama.cpp/llama-cli \
    -m llama-3.3-70b-instruct/Llama-3.3-70B-Instruct-Q4_K_M.gguf \
    -p "What is quantum entanglement? Explain briefly." \
    -n 100 \
    -ngl 99

# Clear GPU memory (important!)
nvidia-smi

# Test DeepSeek R1 32B
./llama.cpp/llama-cli \
    -m deepseek-r1-32b/DeepSeek-R1-Q4_K_M.gguf \
    -p "Evaluate: Is quantum computing practical today?" \
    -n 150 \
    -ngl 99

# Exit
exit
```

### **Method 2: Automated Sequential Testing**

```bash
# Submit test job
cd ~/projects/Agentic_AI
sbatch cluster_utils/slurm_test_models.sh

# Check results
tail -f logs/test_models_*.out

# View output files
cat ~/models/test_llama33_output.txt
cat ~/models/test_deepseek_output.txt
```

---

## 🎯 Model Usage in Your Project

### **Key Parameters for llama-cli:**

```bash
# Basic usage
./llama.cpp/llama-cli \
    -m <model_path> \              # Path to GGUF file
    -p "<prompt>" \                # Your prompt
    -n <tokens> \                  # Max tokens to generate
    -ngl 99 \                      # GPU layers (99 = all)
    --temp <temperature> \         # 0.1-1.0 (creativity)
    --ctx-size <context> \         # Context window (default 512)
    --repeat-penalty 1.1           # Avoid repetition
```

### **Recommended Settings:**

**For Question Generator (LLaMA 3.3 70B):**
```bash
--temp 0.7          # Creative but controlled
-n 200              # Enough for question + context
--ctx-size 2048     # Large context for chunks
```

**For Critic Agent (DeepSeek R1 32B):**
```bash
--temp 0.3          # More deterministic
-n 300              # Detailed evaluation
--ctx-size 4096     # Large context for Q+A+chunk
```

---

## 🔄 Sequential Execution (CRITICAL)

### **Why Sequential?**
- L40S has 44 GB VRAM
- LLaMA 3.3 70B uses ~36-38 GB
- DeepSeek R1 uses ~18-20 GB
- **Cannot load both simultaneously!**

### **Proper Workflow:**

```python
# Python wrapper example
import subprocess
import time

def run_model(model_path, prompt, max_tokens):
    """Run model and return output"""
    cmd = [
        "./llama.cpp/llama-cli",
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "-ngl", "99"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def clear_vram():
    """Clear GPU memory between models"""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2)  # Wait for cleanup

# Generate Q&A
question = run_model(
    "~/models/llama-3.3-70b-instruct/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
    "Generate a question...",
    200
)

# CRITICAL: Clear VRAM before next model
clear_vram()

# Evaluate with critic
evaluation = run_model(
    "~/models/deepseek-r1-32b/DeepSeek-R1-Q4_K_M.gguf",
    f"Evaluate this: {question}",
    300
)
```

---

## 📊 VRAM Monitoring

```bash
# Real-time VRAM usage
watch -n 1 nvidia-smi

# Check specific info
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# Log VRAM during execution
nvidia-smi dmon -s mu -c 100 > vram_log.txt
```

---

## 🐛 Troubleshooting

### **Installation Issues:**

**Problem: Download fails**
```bash
# Retry with wget manually
cd ~/models/llama-3.3-70b-instruct
wget -c https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf
```

**Problem: Compilation fails**
```bash
# Check CUDA availability
nvcc --version

# Rebuild
cd ~/models/llama.cpp
make clean
LLAMA_CUDA=1 make -j$(nproc)
```

### **Runtime Issues:**

**Problem: CUDA out of memory**
- Ensure no other model is loaded
- Check: `nvidia-smi` for other processes
- Kill: `pkill -9 llama-cli`

**Problem: Model not found**
```bash
# Verify paths
ls -lh ~/models/llama-3.3-70b-instruct/
ls -lh ~/models/deepseek-r1-32b/
```

---

## 🚀 Next Steps

1. **Install models** (choose Method 1 or 2 above)
2. **Test individually** to verify they work
3. **Test sequentially** with the provided script
4. **Integrate into your pipeline** (I'll help with this!)

---

## 📞 Quick Commands Reference

```bash
# Install (interactive)
srun --partition=ENSTA-l40s --gres=gpu:1 --mem=48G --cpus-per-task=16 --time=04:00:00 --pty bash
bash ~/projects/Agentic_AI/cluster_utils/install_models.sh

# Test (interactive)
srun --partition=ENSTA-l40s --gres=gpu:1 --mem=48G --time=01:00:00 --pty bash
bash ~/projects/Agentic_AI/cluster_utils/test_models_sequential.sh

# Or submit jobs
sbatch ~/projects/Agentic_AI/cluster_utils/slurm_install_models.sh
sbatch ~/projects/Agentic_AI/cluster_utils/slurm_test_models.sh

# Check status
squeue -u $USER
```

Ready to start? Let me know which method you prefer! 🎉
