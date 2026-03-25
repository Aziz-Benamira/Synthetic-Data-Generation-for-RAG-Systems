#!/bin/bash
#SBATCH --job-name=mi201_sog
#SBATCH --partition=ENSTA-l40s
#SBATCH --gres=gpu:1
#SBATCH --exclude=ensta-l40s01.r2.enst.fr
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/mi201_sog_%j.out
#SBATCH --error=logs/mi201_sog_%j.err

# ─── environment ────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate agentic_ai

export HF_HOME=/home/ensta/ensta-ben-amira/.cache/huggingface
export TRANSFORMERS_OFFLINE=0

cd /home/ensta/ensta-ben-amira/projects/sog_experiment/experiment
mkdir -p logs output/mi201_sog

echo "================================================================"
echo "  MI201 SoG Pipeline — multi-hop combined mode"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $(hostname)"
echo "  Date   : $(date)"
echo "================================================================"

CHUNKS="data/chunks_mi201.json"
GRAPH="data/MI201_2022_poly_context_graph.json"
OUTPUT="output/mi201_sog"
MODEL="/home/ensta/ensta-ben-amira/models/deepseek-r1-distill-qwen-32b/DeepSeek-R1-Distill-Qwen-32B-IQ3_M.gguf"

# ─── Run: combined mode with multi-hop BFS (depth=2, top_k=3, top_w=3) ────
echo ""
echo ">>> MODE: combined + multihop (depth=2, top_k=3, top_w=3)"
python3 -u run_full_pipeline_sog.py \
    --chunks-path "$CHUNKS" \
    --graph-path  "$GRAPH"  \
    --output-dir  "$OUTPUT" \
    --mode        combined  \
    --max-chunks  10        \
    --n-ctx       16384     \
    --multihop              \
    --sog-depth   2         \
    --sog-top-k   3         \
    --sog-top-w   3         \
    --model-path  "$MODEL"

echo ""
echo "================================================================"
echo "  DONE — $(date)"
echo "================================================================"
