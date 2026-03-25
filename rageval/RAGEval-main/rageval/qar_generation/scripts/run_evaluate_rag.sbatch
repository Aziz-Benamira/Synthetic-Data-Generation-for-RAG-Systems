#!/bin/bash
#SBATCH --job-name=rag_eval
#SBATCH --output=logs/rag_eval_%j.out
#SBATCH --error=logs/rag_eval_%j.err
#SBATCH --partition=ENSTA-h100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G

echo "================================================"
echo "RAG Evaluation - Academic MI201"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "================================================"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

source /home/ensta/ensta-hidouri/miniconda3/etc/profile.d/conda.sh
conda activate rageval
echo "Environment activated"

cd /home/ensta/ensta-hidouri/Synthetic-Data-Generation-for-RAG-Systems/RAGEval/RAGEval-main/rageval/qar_generation

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Running RAG evaluation with answer generation..."
python evaluate_rag.py

echo ""
echo "================================================"
echo "Evaluation completed"
echo "Results: output/academic/en/eval_results.json"
echo "================================================"
