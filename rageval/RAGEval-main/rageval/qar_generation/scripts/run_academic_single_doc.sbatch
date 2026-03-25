#!/bin/bash
#SBATCH --job-name=academic_single
#SBATCH --output=logs/academic_single_%j.out
#SBATCH --error=logs/academic_single_%j.err
#SBATCH --partition=ENSTA-h100
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G

echo "================================================"
echo "RAGEval - Academic QA Generation (Step 1/4)"
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

echo "Generating QA pairs for all chapters..."
python code/academic/en/qra_pipeline_single_doc.py

echo ""
echo "================================================"
echo "Step 1 completed: QA pairs generated"
echo "Output: output/academic/en/config/"
echo "Next: sbatch scripts/run_add_references.sbatch"
echo "================================================"
