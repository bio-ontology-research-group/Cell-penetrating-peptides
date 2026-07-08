#!/bin/bash
#SBATCH --job-name=neural_baselines
#SBATCH --output=logs/neural_baselines_%j.out
#SBATCH --error=logs/neural_baselines_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=96G
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=4

# R1.2 neural-encoder baselines on the manuscript datasets, via the benchmark
# harness (EncoderNNSystem). sapbert + krissbert REUSE their cached concept
# embeddings (no re-embed); biobert + pubmedbert embed the pool once (GPU).
set -euo pipefail
mkdir -p logs
cd /ibex/user/castilmg/CPP_KG/Cell-penetrating-peptides

MODELS=${MODELS:-"sapbert krissbert biobert=dmis-lab/biobert-base-cased-v1.1 pubmedbert=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"}

echo "Job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)}  |  models: $MODELS"
module load cuda 2>/dev/null || true
PY=/usr/bin/python3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

$PY -c "import torch; print('CUDA available:', torch.cuda.is_available())"
$PY scripts/benchmark/score_encoders_manuscript.py --models $MODELS
echo ">> done"
