#!/bin/bash
#SBATCH --job-name=rag_eval_test
#SBATCH --output=/home/ensta/ensta-ben-amira/projects/Agentic_AI/logs/rag_eval_test_%j.log
#SBATCH --error=/home/ensta/ensta-ben-amira/projects/Agentic_AI/logs/rag_eval_test_%j.err
#SBATCH --partition=ENSTA-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00

echo "======================================================"
echo "RAG EVAL TEST — 3 QA pairs only"
echo "Job ID: $SLURM_JOB_ID | Date: $(date)"
echo "======================================================"

source ~/envs/agentic_ai/bin/activate
cd /home/ensta/ensta-ben-amira/projects/Agentic_AI

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Env vars critiques pour noeuds HPC sans internet
# HF hub + sentence_transformers + BERTScore tentent des appels réseau et bloquent
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export ANONYMIZED_TELEMETRY=False  # ChromaDB telemetry
export CHROMA_TELEMETRY=False
export PYTHONUNBUFFERED=1           # Flush stdout immédiatement

# Quick test: 5 QA, no LLM judge (faster), n_ctx=4096 (proven working with DeepSeek-R1-32B)
srun python3 -u evaluation/run_evaluation.py \
    --top_k 5 \
    --limit 5 \
    --no-judge \
    --output-dir evaluation/results/test_run \
    --n-ctx 4096

echo ""
echo "Test terminé: $(date)"
