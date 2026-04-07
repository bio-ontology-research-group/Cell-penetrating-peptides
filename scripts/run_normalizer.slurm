#!/bin/bash
#SBATCH --job-name=normalizer
#SBATCH --output=logs/normalizer_%j.out
#SBATCH --error=logs/normalizer_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=4

set -euo pipefail
mkdir -p logs

# ── Config ────────────────────────────────────────────────────────────────────
RAG_MODEL=${RAG_MODEL:-"gpt-oss:20b"}



WORKDIR=/ibex/user/$USER/Cell-penetrating-peptides
OLLAMA_MODELS_SCRATCH="/ibex/user/$USER/ollama_models_scratch"
SINGULARITY_INSTANCE=ollama
PORT_FILE="$WORKDIR/ollama_port.txt"
OLLAMA_LOG="$WORKDIR/logs/ollama_${SLURM_JOB_ID}.log"

echo "====================================================="
echo "Job:      $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Ontology: clo"
echo "Model:    $RAG_MODEL"
echo "====================================================="

# ── Cleanup on exit (always runs) ────────────────────────────────────────────
cleanup() {
    echo "Stopping Ollama instance …"
    singularity instance stop "$SINGULARITY_INSTANCE" 2>/dev/null || true
    rm -f "$PORT_FILE"
}
trap cleanup EXIT

# ── CUDA + Singularity ───────────────────────────────────────────────────────
module load singularity

# ── Start Ollama ─────────────────────────────────────────────────────────────
mkdir -p "$OLLAMA_MODELS_SCRATCH"

# Pick a free port and expose it to the Singularity environment
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "$PORT" > "$PORT_FILE"
export SINGULARITYENV_OLLAMA_HOST=127.0.0.1:$PORT
export SINGULARITYENV_OLLAMA_MODELS=$OLLAMA_MODELS_SCRATCH

echo "Ollama port: $PORT"

# Pull image if not present
SIF="$OLLAMA_MODELS_SCRATCH/ollama_latest.sif"
if [ ! -f "$SIF" ]; then
    echo "Pulling Ollama Singularity image …"
    singularity pull --name "$SIF" docker://ollama/ollama
fi

singularity instance start --nv -B "/ibex/user:/ibex/user" "$SIF" "$SINGULARITY_INSTANCE"
nohup singularity exec instance://$SINGULARITY_INSTANCE bash -c "ollama serve" \
    > "$OLLAMA_LOG" 2>&1 &

# Wait until the REST API is up (max 60 s)
echo "Waiting for Ollama REST API on port $PORT …"
for i in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/api/tags" > /dev/null 2>&1; then
        echo "Ollama ready after ${i}s"
        break
    fi
    sleep 1
done

# Pull the model into the instance
echo "Pulling model: $RAG_MODEL"
singularity exec instance://$SINGULARITY_INSTANCE ollama pull "$RAG_MODEL"

# ── Run Ontology_normalizer ───────────────────────────────────────────────────────────
cd "$WORKDIR"

uv run scripts/Ontology_normalizer.py \
    --input        "data/Natural_CPP3_download_annotated_preprocessed.csv" \
    --column       "Cell Line" \
    --ontology     "clo" \
    --obo          "data/Ontology/clo.obo" \
    --output       "data/Natural_CPP3_download_annotated_preprocessed_clo.csv" \
    --rag-backend  ollama \
    --rag-model    "$RAG_MODEL" \
    --rag-url      "http://127.0.0.1:$PORT/v1"

echo "Finished CLO: $(date)"

# ── Config ────────────────────────────────────────────────────────────────────

echo "====================================================="
echo "Job:      $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Ontology: chebi"
echo "Model:    $RAG_MODEL"
echo "Started:  $(date)"
echo "====================================================="

# ── Run Ontology_normalizer on CHEBI too (same Ollama instance) ───────────────────────
uv run scripts/Ontology_normalizer.py \
    --input        "data/Natural_CPP3_download_annotated_preprocessed.csv" \
    --column       "Cargo" \
    --ontology     "chebi" \
    --obo          "data/Ontology/chebi.obo" \
    --output       "data/Natural_CPP3_download_annotated_preprocessed_chebi.csv" \
    --rag-backend  ollama \
    --rag-model    "$RAG_MODEL" \
    --rag-url      "http://127.0.0.1:$PORT/v1"
echo "Finished CHEBI: $(date)"