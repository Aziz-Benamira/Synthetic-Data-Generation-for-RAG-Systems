#!/bin/bash
#SBATCH --job-name=academic_refs
#SBATCH --output=logs/academic_refs_%j.out
#SBATCH --error=logs/academic_refs_%j.err
#SBATCH --partition=ENSTA-h100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G

echo "================================================"
echo "RAGEval - Add Verbatim French References (Step 2/4)"
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

echo "Extracting verbatim French references (1 item per LLM call)..."
python add_references_academic.py

echo ""
echo "================================================"
echo "Step 2 completed: References extracted"
echo "Next: python combine_dataset.py"
echo "================================================"
