#!/bin/bash
#SBATCH --job-name=mi201_vl_demo
#SBATCH --output="/home/ensta/ensta-ghozzi/AI RAG/logs/mi201_vl_demo_%j.log"
#SBATCH --error="/home/ensta/ensta-ghozzi/AI RAG/logs/mi201_vl_demo_%j.err"
#SBATCH --partition=ENSTA-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00

echo "======================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "======================================================"

source ~/envs/agentic_ai/bin/activate
cd "/home/ensta/ensta-ghozzi/AI RAG"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo ""
echo "Multimodal VL Demo — MI201_2022_poly (first 20 figures)"
echo ""

srun python3 -u scripts/run_multimodal_demo.py \
    --pdf "data/pdfs/MI201_2022_poly.pdf" \
    --output "output/multimodal_mi201" \
    --vl-model "/home/ensta/data/Qwen2-VL-7B-Instruct" \
    --max-new-tokens 512 \
    --limit 20

echo ""
echo "======================================================"
echo "Terminé: $(date)"
echo "======================================================"
