#!/bin/bash
#SBATCH --job-name=rag_eval_v2
#SBATCH --output=logs/rag_eval_v2_%j.out
#SBATCH --error=logs/rag_eval_v2_%j.err
#SBATCH --partition=ENSTA-l40s
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem=40G

echo "============================================================"
echo "RAG Evaluation v2 — Maloe metrics + Aziz/Seif Critic Agent"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "============================================================"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

source /home/ensta/ensta-hidouri/miniconda3/etc/profile.d/conda.sh
conda activate rageval
echo "Environment: rageval"

# Install additional metric packages if not present (Maloe's metrics)
pip install rouge-score --quiet 2>/dev/null || true
pip install bert-score --quiet 2>/dev/null || true

cd /home/ensta/ensta-hidouri/Synthetic-Data-Generation-for-RAG-Systems/RAGEval/RAGEval-main/rageval/qar_generation

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo ""
echo "Running RAG evaluation v2 (full — with critic agent)..."
python evaluate_rag_v2.py

echo ""
echo "============================================================"
echo "Results: output/academic/en/eval_results_v2.json"
echo "============================================================"
