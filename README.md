# CPP Knowledge Graph

A knowledge graph of cell-penetrating peptides (CPPs) built from curated
experimental data and linked to biomedical ontologies (SIO, GO, ChEBI, CLO).
The pipeline preprocesses annotated CSV data, normalizes ontology terms, and
constructs an OWL/RDF knowledge graph that can be queried via SPARQL.

Web interface: <https://cppkg.bio2vec.net/>

## Repository structure

```
scripts/
  Preprocess_dataset.py          Clean and preprocess the raw annotated CSV
  Ontology_normalizer.py         Map free-text terms to CLO / ChEBI IDs (exact -> SapBERT -> Graph-RAG)
  build_and_extend_ontology.py   Construct the OWL/RDF knowledge graph
  verify_paper_numbers.py        Re-derive and assert every quantitative claim in the paper
  sample_ground_truth.py         Draw the internal ground-truth sample (300 terms/ontology)
  build_single_annotator_gold.py Build gold sheets from the annotator CSVs
  compute_iaa.py                 Inter-annotator agreement
  r31_llm_rate.py                Tabulate the LLM hallucination-guard rate
  figure_sequence_diversity.py   Sequence-diversity / net-charge figure
  run_*.slurm                    SLURM jobs (GPU + Ollama) for normalization and baselines
  benchmark/                     Neural-encoder entity-linking benchmark

validation/
  Evaluate_ontology_normalizer.py  Per-stage TP/FP/FN of the normalizer (ablation table)
  evaluate_internal_gt.py          Internal ground-truth evaluation (mappable + abstention)
  evaluate_single_answer.py        Single-answer baseline comparison vs our pipeline
  baseline_ols.py                  OLS (EBI) lexical/dictionary baseline
  baseline_bioportal.py            BioPortal Annotator baseline
  baseline_cellosaurus.py          Cellosaurus dictionary baseline (CLO)
  baseline_bert_encoders.py        Pretrained-encoder dense-NN baselines
  Competency_Questions.py          Competency + federated SPARQL queries
  validate_shex.py                 ShEx shape validation of the KG
  mechanisms_shapes.shex           ShEx shape definitions

data/
  Natural_CPP3_download_annotated.csv                Raw annotated dataset (CPPsite3 upstream)
  Natural_CPP3_download_annotated_preprocessed.csv   Preprocessed dataset
  Natural_CPP3_download_annotated_preprocessed_Ontology_Normalization.csv
                                                     Normalized dataset (KG source)
  CRAFT.csv, biosamples.csv                          Public benchmark corpora (with gold IDs)
  CRAFT_Ontology_Normalization.csv                   Pipeline predictions on CRAFT (ChEBI)
  biosamples_Ontology_Normalization.csv              Pipeline predictions on biosamples (CLO)
  Ground_Truth_CHEBI_v2.csv, Ground_Truth_CLO_v2.csv Internal ground-truth gold (300 terms/ontology)
  Ground_Truth_CHEBI_Ontology_Normalization.csv      Internal gold + pipeline predictions (ChEBI)
  Ground_Truth_CLO_Ontology_Normalization.csv        Internal gold + pipeline predictions (CLO)
  ground_truth_v2/               Annotator sheets (GT_*_annotatorA_filled.csv)
  baselines/                     Baseline prediction CSVs (OLS / BioPortal / Cellosaurus / neural)
  r31/                           Full-dataset normalization + LLM hallucination-rate stats
  triplets/                      CPP -> cargo/cell/location/mechanism tables (KG build input)
  intermediate/                  Gene/inhibitor -> GO mechanism inputs (chebi_to_go, gene_to_go, mech_metadata)
  Ontology/
    CPP_KG.ttl                   Final knowledge graph (Turtle)
    CPP_KG_materialized.ttl      HORST-materialized graph (inferred triples)
    sio.owl, chebi.obo, clo.obo  Ontology inputs (large; fetched, not tracked)
    biosyn_cache_*.pkl           Cached SapBERT/BioSyn ontology embeddings (large; not tracked)
  void.ttl                       VoID dataset description

environment.yml                  Conda environment specification
```

## Prerequisites

- Python 3.10
- Conda (recommended) or pip
- Java runtime (required by mowl / OWL API)
- OBO files for CLO and ChEBI (used by the ontology normalizer)
- SIO ontology (`Ontology/sio.owl`)

## Installation

```bash
conda env create -f environment.yml
conda activate cpp_kg
```

## Reproducing the results

Run each step from the repository root.

### 1. Preprocess the raw annotated CSV

```bash
python scripts/Preprocess_dataset.py
```

Reads `data/Natural_CPP3_download_annotated.csv` and produces
`data/Natural_CPP3_download_annotated_preprocessed.csv`.

### 2. Normalize ontology terms

```bash
python scripts/Ontology_normalizer.py --input data/Natural_CPP3_download_annotated_preprocessed.csv \
--column       "Cargo" \ #"Cell Line"
--ontology     "chebi" \#"clo"
--obo          "data/Ontology/chebi.obo" \""data/Ontology/clo.obo"
--rag-backend  ollama \
--rag-model    "gpt-oss:20b" \
--rag-url      "http://127.0.0.1:$PORT/v1"
```

Requires CLO and ChEBI OBO files. It is recommended to set up an Ollama instance beforehand (`run_normalizer.slurm`). Otherwise, set the OpenRouter API key and OpenRouter model as environment variables or in your configuration file.
Produces
`data/Natural_CPP3_download_annotated_preprocessed_Ontology_Normalization.csv`.

### 3. Build the knowledge graph

```bash
python scripts/build_and_extend_ontology.py
```

Requires `data/Ontology/sio.owl`. Produces `data/Ontology/CPP_KG.ttl`.

### 4. Validate the knowledge graph

```bash
python validation/validate_shex.py --ttl data/Ontology/CPP_KG.ttl --shex validation/mechanisms_shapes.shex
```

Validates `data/Ontology/CPP_KG.ttl` against the ShEx shapes in
`validation/mechanisms_shapes.shex`.

### 5. Evaluate the ontology normalizer

```bash
python validation/Evaluate_ontology_normalizer.py
```
To reproduce Table 3 results. First, to reproduce the `data/Ground_Truth_CHEBI_Ontology_Normalization.csv`, `data/Ground_Truth_CLO_Ontology_Normalization.csv`, `data/CRAFT.csv` and `data/biosamples.csv`, run `scripts/Ontology_normalizer.py` on `data/Ground_Truth_CHEBI.csv`, `data/Ground_Truth_CLO.csv`, `data/CRAFT.csv` and `data/biosamples.csv`. Results will be saved in `data/evaluation_report.csv`.

### 6. Run Competency Questions

```bash
python validation/Competency_Questions.py --federated
```
To reproduce the results in Table 4, it runs 3 local and 3 federated SPARQL Queries against the knowledge graph. 

### 7. Verify all numerical claims in the paper

```bash
python scripts/verify_paper_numbers.py
```

Checks every quantitative statement (counts, percentages) against the
data files and reports PASS/FAIL for each. All checks must pass before
submission.

## Expected key numbers

All values below are asserted by `scripts/verify_paper_numbers.py`
(source of truth = `data/Ontology/CPP_KG.ttl`).

| Metric | Value |
|--------|-------|
| Downloaded sequences | 5,288 |
| Entries after preprocessing | 10,799 |
| Distinct sequences | 2,708 |
| KG CPP individuals | 2,642 |
| KG CPP-Complexes | 4,132 |
| KG experiments | 4,598 |
| KG unique entities (total) | 15,015 |
| Total RDF triples (asserted `CPP_KG.ttl`) | 244,572 |

Ontology-normalization accuracy (full Graph-RAG pipeline, deterministic rag
stage at `temperature=0`, `seed=42`):

| Benchmark | Accuracy |
|-----------|----------|
| CRAFT:ChEBI (public) | 0.87 |
| Biosamples:CLO (public) | 0.73 |
| Internal ground truth: ChEBI | 0.37 |
| Internal ground truth: CLO | 0.71 |

## License

This work is licensed under
[Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use this knowledge graph or code in your research, please cite:

> *Citation details to be added upon publication.*

## Contact

For questions or issues, please open a GitHub issue in this repository.
