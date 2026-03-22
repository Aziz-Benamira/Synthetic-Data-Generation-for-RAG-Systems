#!/bin/bash
#SBATCH --job-name=pipeline_v4_full
#SBATCH --output="/home/ensta/ensta-ghozzi/AI RAG/logs/pipeline_v4_full_%j.log"
#SBATCH --error="/home/ensta/ensta-ghozzi/AI RAG/logs/pipeline_v4_full_%j.err"
#SBATCH --partition=ENSTA-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

echo "======================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "======================================================"

source ~/envs/agentic_ai/bin/activate
cd "/home/ensta/ensta-ghozzi/AI RAG"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "Lancement Pipeline V4 — 100 chunks complets..."
echo ""

srun python3 run_pipeline_v4_full.py

echo ""
echo "======================================================"
echo "Terminé: $(date)"
echo "======================================================"
