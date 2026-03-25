#!/bin/bash
#SBATCH --job-name=sog_experiment
#SBATCH --output=/home/ensta/ensta-ben-amira/projects/sog_experiment/experiment/output/sog_exp_%j.log
#SBATCH --error=/home/ensta/ensta-ben-amira/projects/sog_experiment/experiment/output/sog_exp_%j.err
#SBATCH --partition=ENSTA-l40s
#SBATCH --exclude=ensta-l40s01.r2.enst.fr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

echo "========================================================"
echo "  SoG + Pipeline Experiment"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Date   : $(date)"
echo "========================================================"

source ~/envs/agentic_ai/bin/activate
cd /home/ensta/ensta-ben-amira/projects/sog_experiment/experiment

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

export PYTHONUNBUFFERED=1

echo "╔══════════════════════════════════════════════════╗"
echo "║  Graphe   : Tipler (990 nodes, 8026 edges)     ║"
echo "║  Chunks   : 160 (physique Tipler)               ║"
echo "║  Modes    : graph_only + combined               ║"
echo "║  Max QA   : 10 par mode                         ║"
echo "║  LLM      : DeepSeek R1-32B (IQ3_M)            ║"
echo "║  Embedder : bge-small-en-v1.5                   ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Lancer l'expérimentation : graph_only + combined, 10 QA chacun
srun python3 -u run_experiment.py \
    --mode both \
    --max-qa 10

echo ""
echo "========================================================"
echo "  Terminé : $(date)"
echo "========================================================"
