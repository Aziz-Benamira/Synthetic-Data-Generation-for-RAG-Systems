#!/bin/bash
#SBATCH --job-name=full_sog_pipeline
#SBATCH --output=/home/ensta/ensta-ben-amira/projects/sog_experiment/experiment/output/full_sog_%j.log
#SBATCH --error=/home/ensta/ensta-ben-amira/projects/sog_experiment/experiment/output/full_sog_%j.err
#SBATCH --partition=ENSTA-l40s
#SBATCH --exclude=ensta-l40s01.r2.enst.fr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

echo "════════════════════════════════════════════════════════"
echo "  Full SoG Pipeline"
echo "  QuestionGeneratorV3 → AnswerGeneratorV3SoG → CriticV4"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Date   : $(date)"
echo "════════════════════════════════════════════════════════"

source ~/envs/agentic_ai/bin/activate
cd /home/ensta/ensta-ben-amira/projects/sog_experiment/experiment

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

export PYTHONUNBUFFERED=1

mkdir -p output/full_pipeline_sog

echo "╔══════════════════════════════════════════════════╗"
echo "║  Graphe   : Tipler (990 nodes, 8026 edges)     ║"
echo "║  Chunks   : 160 (physique Tipler)               ║"
echo "║  Mode     : combined (chunk + graph context)    ║"
echo "║  Max QA   : 10                                  ║"
echo "║  LLM      : DeepSeek R1-32B (IQ3_M)            ║"
echo "║  Embedder : bge-small-en-v1.5                   ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Run full pipeline in combined mode ───────────────────────────────────────
echo "=== MODE : combined (graphe + contenu chunk) ==="
srun python3 -u run_full_pipeline_sog.py \
    --mode combined \
    --max-chunks 10 \
    --n-ctx 16384 \
    --output-dir output/full_pipeline_sog

echo ""
echo "=== MODE : graph_only (graphe uniquement) ==="
srun python3 -u run_full_pipeline_sog.py \
    --mode graph_only \
    --max-chunks 10 \
    --n-ctx 16384 \
    --output-dir output/full_pipeline_sog

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Terminé : $(date)"
echo "════════════════════════════════════════════════════════"
