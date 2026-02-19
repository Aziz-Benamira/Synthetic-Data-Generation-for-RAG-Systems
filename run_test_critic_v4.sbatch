#!/bin/bash
#SBATCH --job-name=critic_v4_test
#SBATCH --output=/home/ensta/ensta-ben-amira/projects/Agentic_AI/logs/critic_v4_test_%j.log
#SBATCH --error=/home/ensta/ensta-ben-amira/projects/Agentic_AI/logs/critic_v4_test_%j.err
#SBATCH --partition=ENSTA-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

echo "======================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Date:   $(date)"
echo "======================================================"

# Activer l'environnement Python
source ~/envs/agentic_ai/bin/activate

# Répertoire du projet
cd /home/ensta/ensta-ben-amira/projects/Agentic_AI

# Vérifier le GPU
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "Lancement du test CriticV4..."
echo ""

# Lancer le test
srun python3 test_critic_v4.py

echo ""
echo "======================================================"
echo "Terminé: $(date)"
echo "======================================================"
