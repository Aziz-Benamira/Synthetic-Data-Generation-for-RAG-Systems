#!/bin/bash
#SBATCH --job-name=multimodal_qa
#SBATCH --output="/home/ensta/ensta-ghozzi/AI RAG/logs/multimodal_qa_%j.log"
#SBATCH --error="/home/ensta/ensta-ghozzi/AI RAG/logs/multimodal_qa_%j.err"
#SBATCH --partition=ENSTA-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

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
echo "Multimodal QA Generation — figure chunks from Attention paper"
echo ""

srun python3 -u scripts/run_multimodal_qa_generation.py \
    --chunks "output/multimodal_attention/figure_chunks.json" \
    --output "output/multimodal_attention/gold_dataset_figures.jsonl"

echo ""
echo "======================================================"
echo "Terminé: $(date)"
echo "======================================================"
