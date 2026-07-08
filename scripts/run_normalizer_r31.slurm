#!/bin/bash
#SBATCH --job-name=normalizer_r31
#SBATCH --output=logs/normalizer_r31_%j.out
#SBATCH --error=logs/normalizer_r31_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=4
#
# R3.1 LLM-error-rate re-run.
# Re-runs the ontology normalizer over the FULL preprocessed dataset with the
# current code, which persists a per-term `<prefix>_match_type` column. From
# that column we tabulate the hallucination-guard trigger rate (the R3.1
# numbers) via scripts/r31_llm_rate.py.
#
# IMPORTANT: outputs go to data/r31/ (a SIDE location) so this does NOT overwrite
# the verified deposit CSV
# (data/Natural_CPP3_download_annotated_preprocessed_Ontology_Normalization.csv),
# whose KG counts are locked to Table 1. The LLM stage is not bit-for-bit
# deterministic, so keep the canonical CSV as-is and use these side outputs for
# the R3.1 statistics only.

set -euo pipefail
mkdir -p logs

# ── Python env ────────────────────────────────────────────────────────────────
# Use the documented conda env (has numpy/pandas/torch/transformers/openai/…).
# `uv run` was used previously but resolves a bare ephemeral env (no numpy).
set +u
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cpp_kg
set -u

# ── Config ────────────────────────────────────────────────────────────────────
RAG_MODEL=${RAG_MODEL:-"gpt-oss:20b"}

# Real repo location (the stock run_normalizer.slurm has a stale WORKDIR).
WORKDIR=/ibex/user/$USER/CPP_KG/Cell-penetrating-peptides
OLLAMA_MODELS_SCRATCH="/ibex/user/$USER/ollama_models_scratch"
SINGULARITY_INSTANCE=ollama_r31
PORT_FILE="$WORKDIR/ollama_port_r31.txt"
OLLAMA_LOG="$WORKDIR/logs/ollama_r31_${SLURM_JOB_ID}.log"

INPUT_CSV="data/Natural_CPP3_download_annotated_preprocessed.csv"
OUTDIR="data/r31"

echo "====================================================="
echo "Job:      $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Model:    $RAG_MODEL"
echo "Input:    $INPUT_CSV"
echo "Outdir:   $OUTDIR"
echo "====================================================="

# ── Cleanup on exit (always runs) ────────────────────────────────────────────
cleanup() {
    echo "Stopping Ollama instance …"
    singularity instance stop "$SINGULARITY_INSTANCE" 2>/dev/null || true
    rm -f "$PORT_FILE"
}
trap cleanup EXIT

# ── Singularity + Ollama ─────────────────────────────────────────────────────
module load singularity
mkdir -p "$OLLAMA_MODELS_SCRATCH"

PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "$PORT" > "$PORT_FILE"
export SINGULARITYENV_OLLAMA_HOST=127.0.0.1:$PORT
export SINGULARITYENV_OLLAMA_MODELS=$OLLAMA_MODELS_SCRATCH
echo "Ollama port: $PORT"

SIF="$OLLAMA_MODELS_SCRATCH/ollama_latest.sif"
if [ ! -f "$SIF" ]; then
    echo "Pulling Ollama Singularity image …"
    singularity pull --name "$SIF" docker://ollama/ollama
fi

singularity instance start --nv -B "/ibex/user:/ibex/user" "$SIF" "$SINGULARITY_INSTANCE"
nohup singularity exec instance://$SINGULARITY_INSTANCE bash -c "ollama serve" \
    > "$OLLAMA_LOG" 2>&1 &

echo "Waiting for Ollama REST API on port $PORT …"
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/api/tags" > /dev/null 2>&1; then
        echo "Ollama ready after ${i}s"; break
    fi
    sleep 1
done

echo "Pulling model: $RAG_MODEL"
singularity exec instance://$SINGULARITY_INSTANCE ollama pull "$RAG_MODEL"

# ── Run the normalizer (both columns) to the side outputs ────────────────────
cd "$WORKDIR"
mkdir -p "$OUTDIR"

echo "=== CLO (Cell Line) $(date) ==="
python scripts/Ontology_normalizer.py \
    --input        "$INPUT_CSV" \
    --column       "Cell Line" \
    --ontology     "clo" \
    --obo          "data/Ontology/clo.obo" \
    --output       "$OUTDIR/full_CLO_Ontology_Normalization_r31.csv" \
    --rag-backend  ollama \
    --rag-model    "$RAG_MODEL" \
    --rag-url      "http://127.0.0.1:$PORT/v1"
echo "Finished CLO: $(date)"

echo "=== ChEBI (Cargo) $(date) ==="
python scripts/Ontology_normalizer.py \
    --input        "$INPUT_CSV" \
    --column       "Cargo" \
    --ontology     "chebi" \
    --obo          "data/Ontology/chebi.obo" \
    --output       "$OUTDIR/full_CHEBI_Ontology_Normalization_r31.csv" \
    --rag-backend  ollama \
    --rag-model    "$RAG_MODEL" \
    --rag-url      "http://127.0.0.1:$PORT/v1"
echo "Finished ChEBI: $(date)"

# ── Tabulate the R3.1 LLM-error / hallucination-guard rate ───────────────────
python scripts/r31_llm_rate.py \
    --clo   "$OUTDIR/full_CLO_Ontology_Normalization_r31.csv" \
    --chebi "$OUTDIR/full_CHEBI_Ontology_Normalization_r31.csv" \
    --out   "$OUTDIR/r31_llm_rate.json"

echo "All done: $(date). See $OUTDIR/r31_llm_rate.json"
