#!/bin/bash
#SBATCH --job-name=bert_baselines
#SBATCH --output=logs/bert_baselines_%j.out
#SBATCH --error=logs/bert_baselines_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=4

# R1.2 contextual-encoder baselines (BioBERT + a second BERT-family encoder).
# Reuses the SapBERT dense-retrieval pipeline with the encoder swapped; embeds
# the full ChEBI (~708k) and CLO (~39k) synonym pools once (cached), then scores
# single-answer accuracy on the 4 benchmarks. Set MODELS to the slug=hf_id pairs.
set -euo pipefail
mkdir -p logs

WORKDIR=/ibex/user/castilmg/CPP_KG/Cell-penetrating-peptides
cd "$WORKDIR"

# slug=hf_model_id pairs, space-separated. Override by exporting MODELS before sbatch.
MODELS=${MODELS:-"biobert=dmis-lab/biobert-base-cased-v1.1"}

echo "====================================================="
echo "Job:    ${SLURM_JOB_ID:-local}   Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Models: $MODELS"
echo "====================================================="

module load cuda 2>/dev/null || true

# Use the interpreter verified to carry torch(cu)+transformers+faiss (~/.local).
PY=/usr/bin/python3
# Models are pre-cached on the login node; run offline so a compute node without
# internet still loads them from ~/.cache/huggingface.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

$PY -c "import torch; print('CUDA available:', torch.cuda.is_available())"

$PY validation/baseline_bert_encoders.py --models $MODELS

echo ">> done"
