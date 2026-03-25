#!/bin/bash
#SBATCH --job-name=annotate_difficulty
#SBATCH --output="/home/ensta/ensta-ghozzi/AI RAG/logs/annotate_difficulty_%j.log"
#SBATCH --error="/home/ensta/ensta-ghozzi/AI RAG/logs/annotate_difficulty_%j.err"
#SBATCH --partition=ENSTA-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00

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
echo "Annotation difficulté — 84 entrées Gold Dataset"
echo ""

srun python3 -u scripts/annotate_difficulty.py \
    --gold  "output/gold_dataset_v4_full.jsonl" \
    --chunks "/home/ensta/ensta-ben-amira/projects/Agentic_AI/experiments/critic_v2_baseline/data/chunks_mi201.json" \
    --output "output/gold_dataset_v4_full_with_difficulty.jsonl"

echo ""
echo "======================================================"
echo "Terminé: $(date)"
echo "======================================================"
