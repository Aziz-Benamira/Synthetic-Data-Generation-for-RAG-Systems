#!/bin/bash
#SBATCH --job-name=rag_eval
#SBATCH --output=/home/ensta/ensta-ben-amira/projects/Agentic_AI/logs/rag_eval_%j.log
#SBATCH --error=/home/ensta/ensta-ben-amira/projects/Agentic_AI/logs/rag_eval_%j.err
#SBATCH --partition=ENSTA-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

echo "======================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "======================================================"

source ~/envs/agentic_ai/bin/activate
cd /home/ensta/ensta-ben-amira/projects/Agentic_AI

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║     RAG EVALUATION — MI201 Gold Dataset          ║"
echo "║     RAG LLM  : DeepSeek-R1-Distill-Qwen-32B (IQ3_M) ║"
echo "║     Retriever: SemanticChunker + ChromaDB        ║"
echo "║     Embedding: BAAI/bge-m3 (multilingual SOTA)   ║"
echo "║     85 QA pairs, top-5 retrieval                 ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Full evaluation: 85 QA pairs, with LLM-as-Judge
srun python3 evaluation/run_evaluation.py \
    --top_k 5 \
    --n-ctx 4096

echo ""
echo "======================================================"
echo "Terminé: $(date)"
echo "======================================================"
