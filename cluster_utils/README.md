# ENSTA GPU Cluster Utilities

Scripts and tools for running Agentic AI on the ENSTA mesogip.r2.enst.fr cluster.

## 🖥️ Cluster Info

- **Hostname**: mesogip.r2.enst.fr
- **GPUs Available**:
  - 2x L40s nodes (ENSTA-l40s partition) - ~48GB VRAM each
  - 1x H100 node (ENSTA-h100 partition) - ~80GB VRAM
- **Job Scheduler**: SLURM

## 📁 Files

- `test_gpu.py` - Simple GPU availability test
- `slurm_gpu_test.sh` - SLURM job to test GPU
- `slurm_run_pipeline.sh` - Run any Python script on L40s
- `slurm_h100.sh` - Run intensive tasks on H100
- `setup_environment.sh` - Initial environment setup

## 🚀 Quick Start

### 1. Test GPU Access (Interactive)

```bash
# Request interactive session on L40s
srun --partition=ENSTA-l40s --gres=gpu:1 --mem=16G --pty bash

# Once on compute node, test GPU
python cluster_utils/test_gpu.py
exit
```

### 2. Test GPU via SLURM Job

```bash
# Create logs directory
mkdir -p logs

# Submit job
sbatch cluster_utils/slurm_gpu_test.sh

# Check job status
squeue -u $USER

# View output once complete
cat logs/gpu_test_*.out
```

### 3. Run Your Pipeline

```bash
# Run a specific test script
sbatch cluster_utils/slurm_run_pipeline.sh test_ollama_local.py

# Check job status
squeue -u $USER

# View logs
tail -f logs/pipeline_*.out
```

## 🔧 Common SLURM Commands

```bash
# View job queue
squeue -u $USER

# Cancel a job
scancel <job_id>

# View job details
scontrol show job <job_id>

# View cluster info
sinfo

# View partition info
sinfo -p ENSTA-l40s
sinfo -p ENSTA-h100

# Interactive session (L40s)
srun --partition=ENSTA-l40s --gres=gpu:1 --mem=16G --pty bash

# Interactive session (H100)
srun --partition=ENSTA-h100 --gres=gpu:1 --mem=32G --pty bash
```

## 📊 Resource Guidelines

### For L40s (~48GB VRAM):
- **Light tasks**: 1 GPU, 8GB RAM, 4 CPUs
- **Medium tasks**: 1 GPU, 16-32GB RAM, 8 CPUs
- **Heavy tasks**: 1 GPU, 32-48GB RAM, 16 CPUs

### For H100 (~80GB VRAM):
- Use for large models, intensive processing, or batch jobs
- Request more memory and CPUs as needed

## 🤖 Ollama on Cluster

### Option 1: Install Ollama on Cluster (Recommended)
**Pros:**
- Direct GPU access for models
- Faster inference on L40s/H100
- No network latency
- Can run multiple models simultaneously

**Cons:**
- Need to download models (~4-8GB each)
- Takes cluster storage space

**How to do it:**
```bash
# On compute node (interactive session)
srun --partition=ENSTA-l40s --gres=gpu:1 --mem=16G --pty bash

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download models
ollama pull mistral:latest
ollama pull llama3:8b

# Start Ollama server (in background)
ollama serve &

# Test
ollama run mistral "Hello"
```

### Option 2: Use Remote Ollama + OpenRouter/API
**Pros:**
- No storage needed on cluster
- Models already set up on your Windows machine

**Cons:**
- Network latency (SSH tunnel needed)
- More complex setup
- Potential bandwidth issues

**Not recommended** for cluster use - defeats the purpose of having powerful GPUs.

### Recommended Approach:
✅ **Install Ollama on cluster** and use the L40s/H100 GPUs directly.
- Mistral 7B: ~4.5GB VRAM
- Llama3 8B: ~4.7GB VRAM
- Total: ~10GB VRAM (fits easily on L40s with 48GB!)

## 📝 Customizing SLURM Scripts

Edit the `#SBATCH` directives:

```bash
#SBATCH --partition=ENSTA-l40s    # or ENSTA-h100
#SBATCH --gres=gpu:1              # Number of GPUs (1-2 for L40s)
#SBATCH --mem=32G                 # RAM request
#SBATCH --time=02:00:00           # Max time (HH:MM:SS)
#SBATCH --cpus-per-task=8         # Number of CPUs
```

## 🐛 Troubleshooting

**Job pending forever?**
```bash
squeue -u $USER  # Check job status
sinfo            # Check if nodes are available
```

**GPU not visible?**
```bash
# Make sure you're on a compute node, not login node
echo $CUDA_VISIBLE_DEVICES  # Should show GPU numbers
nvidia-smi                   # Should show GPU info
```

**Ollama not found?**
```bash
# Make sure Ollama is installed on compute nodes
# You may need to install it in your user directory
```

## 📚 Next Steps

1. Test GPU access (start with test_gpu.py)
2. Install Ollama on cluster
3. Download required models (mistral, llama3)
4. Test your existing pipeline scripts
5. Scale up to larger datasets!
