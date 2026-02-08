# 🎓 GPU Cluster Guide for Beginners

**Your Setup:**
- Cluster: `mesogip.r2.enst.fr`
- Your username: `ensta-ben-amira`
- Your home directory: `/home/ensta/ensta-ben-amira/`

---

## 📚 **Basic Concepts**

### **1. What is a Cluster?**
- A cluster = Multiple powerful computers connected together
- Each computer is called a **"node"**
- Some nodes have powerful GPUs (L40S with 44GB RAM, H100 with 80GB RAM)

### **2. Types of Nodes:**

```
LOGIN NODE (where you are now)
  ├── No GPU access
  ├── Used for: editing files, submitting jobs, light tasks
  └── Command: Already here when you SSH!

COMPUTE NODES (where the GPUs are)
  ├── ENSTA-l40s: 2 nodes with L40S GPUs (44GB each)
  ├── ENSTA-h100: 1 node with H100 GPU (80GB)
  └── Used for: Running your GPU code
```

### **3. How to Access GPUs?**

**Option A: Interactive** (like using your own computer)
```bash
# Request access to a GPU node
srun --partition=ENSTA-l40s --gres=gpu:1 --mem=16G --pty bash

# Now you're on a compute node! You can run commands:
python your_script.py

# Exit when done:
exit
```

**Option B: Submit a Job** (fire and forget)
```bash
# Create a script that runs your code
sbatch my_job.sh

# SLURM runs it when GPU is available
# You can log out - it keeps running!
```

---

## 🗄️ **Your Storage**

```
/home/ensta/ensta-ben-amira/  ← YOUR HOME DIRECTORY
  ├── Available space: 15 TB (huge!)
  ├── Shared storage: accessible from ALL nodes
  │
  ├── envs/
  │   └── agentic_ai/  ← Your Python virtual environment
  │       ├── bin/python  ← Your Python installation
  │       └── lib/  ← Your installed packages (PyTorch, etc.)
  │
  └── projects/
      └── Agentic_AI/  ← Your project
          ├── src/  ← Your code
          ├── data/  ← PDFs will go here
          ├── cluster_utils/  ← Helper scripts
          └── logs/  ← Job outputs
```

**Important:** Anything you put in `/home/ensta/ensta-ben-amira/` is accessible everywhere!

---

## ✅ **What We've Done So Far**

### **Step 1: Created Virtual Environment** ✅
```bash
Location: ~/envs/agentic_ai
Activation: source ~/envs/agentic_ai/bin/activate
```

A virtual environment is like a separate Python installation just for you. It won't interfere with system Python or other users.

### **Step 2: Installed PyTorch with CUDA** ✅
```bash
Installed: PyTorch 2.5.1 with CUDA 12.1 support
Size: ~2.5 GB
GPU Access: ✅ Working!
```

### **Step 3: Tested GPU** ✅
```bash
GPU: NVIDIA L40S
Memory: 44 GB
Status: Working perfectly!
```

---

## 🚀 **Next Steps**

### **Step 4: Install Your Project Dependencies**

Let's install the packages your project needs:

```bash
# Activate virtual environment
source ~/envs/agentic_ai/bin/activate

# Go to your project
cd ~/projects/Agentic_AI

# Install requirements
pip install -r requirements.txt
```

### **Step 5: Install Ollama + Models**

Your project uses Ollama (Mistral 7B, Llama3 8B). We need to install them on the cluster.

**Why install on cluster?**
- ✅ Direct GPU access (much faster!)
- ✅ No network latency
- ✅ Can use powerful L40S/H100 GPUs
- ❌ Takes ~10GB storage (but you have 15TB!)

**How to install:**

```bash
# Get interactive GPU session
srun --partition=ENSTA-l40s --gres=gpu:1 --mem=16G --time=01:00:00 --pty bash

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download models (takes ~10-15 minutes)
ollama pull mistral:latest    # ~4.5 GB
ollama pull llama3:8b          # ~4.7 GB

# Test
ollama run mistral "Hello, test message!"

# Exit
exit
```

---

## 📝 **Common Commands**

### **File Management:**
```bash
# See where you are
pwd

# List files
ls -lh

# Check disk space
df -h ~

# Check folder size
du -sh ~/projects
```

### **Virtual Environment:**
```bash
# Activate (do this every time you login!)
source ~/envs/agentic_ai/bin/activate

# Deactivate
deactivate

# Check Python location
which python
```

### **SLURM (Job Scheduler):**
```bash
# Check your running jobs
squeue -u $USER

# Cancel a job
scancel <job_id>

# View job output
cat logs/gpu_test_*.out

# Interactive GPU session
srun --partition=ENSTA-l40s --gres=gpu:1 --mem=16G --pty bash
```

### **GPU Information:**
```bash
# View available partitions
sinfo

# View GPU details (when on compute node)
nvidia-smi
```

---

## 🎯 **Typical Workflow**

### **For Development (interactive):**
```bash
# 1. Login to cluster
ssh ensta-ben-amira@mesogip.r2.enst.fr

# 2. Activate environment
source ~/envs/agentic_ai/bin/activate

# 3. Get GPU access
srun --partition=ENSTA-l40s --gres=gpu:1 --mem=16G --pty bash

# 4. Go to project
cd ~/projects/Agentic_AI

# 5. Run your code
python test_pipeline.py

# 6. Exit when done
exit
```

### **For Long Jobs (submit and forget):**
```bash
# 1. Login and activate environment
ssh ensta-ben-amira@mesogip.r2.enst.fr
source ~/envs/agentic_ai/bin/activate

# 2. Submit job
cd ~/projects/Agentic_AI
sbatch cluster_utils/slurm_run_pipeline.sh test_pipeline.py

# 3. Check status
squeue -u $USER

# 4. View output later
tail -f logs/pipeline_*.out
```

---

## ⚠️ **Important Notes**

1. **Always activate your virtual environment:**
   ```bash
   source ~/envs/agentic_ai/bin/activate
   ```

2. **Don't run GPU code on login node:**
   - Use `srun` to get on a compute node first!

3. **Check your jobs:**
   ```bash
   squeue -u $USER
   ```

4. **Storage:**
   - You have 15 TB available
   - Currently using only 71 MB
   - Plenty of space for models!

5. **GPU Resources:**
   - L40S: 44 GB VRAM (perfect for your models!)
   - Your models need ~10 GB total
   - Can easily run multiple at once

---

## 🆘 **Troubleshooting**

**"Command not found" error:**
```bash
# Make sure virtual environment is activated!
source ~/envs/agentic_ai/bin/activate
```

**"CUDA not available":**
```bash
# Make sure you're on a COMPUTE NODE with GPU
srun --partition=ENSTA-l40s --gres=gpu:1 --pty bash
nvidia-smi  # Should show GPU info
```

**"Job pending forever":**
```bash
# Check queue
squeue

# Maybe all GPUs are busy - wait or use H100 partition
srun --partition=ENSTA-h100 --gres=gpu:1 --pty bash
```

**"Permission denied":**
```bash
# Make sure scripts are executable
chmod +x script.sh
```

---

## 📖 **Learning Resources**

- **SLURM Basics:** https://slurm.schedmd.com/quickstart.html
- **Cluster Computing:** Ask me any questions!
- **Your Project:** See SYSTEM_PRESENTATION.md

---

## ✅ **Status Checklist**

- [x] Virtual environment created
- [x] PyTorch installed with CUDA
- [x] GPU access tested and working
- [ ] Project dependencies installed (next step!)
- [ ] Ollama + models installed
- [ ] Test run on GPU

---

**Questions? Just ask!** I'm here to help you understand everything step by step.
