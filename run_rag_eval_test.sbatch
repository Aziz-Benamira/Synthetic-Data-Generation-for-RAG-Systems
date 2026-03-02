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
#SBATCH --time=00:30:00

echo "======================================================"
echo "RAG EVAL TEST — 3 QA pairs only"
echo "Job ID: $SLURM_JOB_ID | Date: $(date)"
echo "======================================================"

source ~/envs/agentic_ai/bin/activate
cd /home/ensta/ensta-ben-amira/projects/Agentic_AI

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Quick test: 5 QA, no LLM judge (faster), n_ctx=8192 pour ne pas tronquer les chunks
srun python3 evaluation/run_evaluation.py \
    --top_k 5 \
    --limit 5 \
    --no-judge \
    --output-dir evaluation/results/test_run \
    --n-ctx 8192

echo ""
echo "Test terminé: $(date)"
