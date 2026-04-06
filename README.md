# CPP Knowledge Graph

A knowledge graph of cell-penetrating peptides (CPPs) built from curated
experimental data and linked to biomedical ontologies (SIO, GO, ChEBI, CLO).
The pipeline preprocesses annotated CSV data, normalises ontology terms, and
constructs an OWL/RDF knowledge graph that can be queried via SPARQL.

Web interface: <https://cppkg.bio2vec.net/>

## Repository structure

```
scripts/
  Preprocess_dataset.py          Clean and preprocess raw annotated CSV
  Ontology_normalizer.py         Map free-text terms to CLO / ChEBI identifiers
  build_and_extend_ontology.py   Construct the OWL/RDF knowledge graph

validation/
  validate_shex.py               ShEx shape validation of the KG
  Evaluate_ontology_normalizer.py  Evaluate normaliser against ground truth
  mechanisms_shapes.shex         ShEx shape definitions

data/
  Natural_CPP3_download_annotated.csv                Raw annotated dataset
  Natural_CPP3_download_annotated_preprocessed.csv   Preprocessed dataset
  Natural_CPP3_download_annotated_preprocessed_Ontology_Normalization.csv
                                                     Dataset after ontology normalisation
  Ground_Truth_CHEBI.csv         ChEBI ground truth for evaluation
  Ground_Truth_CLO.csv           CLO ground truth for evaluation
  biosamples.csv                 biosamples corpus (NER support)
  CRAFT.csv                      CRAFT corpus (NER support)
  CPP_KG.ttl                     Final knowledge graph in Turtle format
  void.ttl                       VoID dataset description

Ontology/
  sio.owl                        Semanticscience Integrated Ontology (input)

environment.yml                  Conda environment specification
```

## Prerequisites

- Python 3.10
- Conda (recommended) or pip
- Java runtime (required by mowl / OWL API)
- OBO files for CLO and ChEBI (used by the ontology normaliser)
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

### 2. Normalise ontology terms

```bash
python scripts/Ontology_normalizer.py
```

Requires CLO and ChEBI OBO files. Produces
`data/Natural_CPP3_download_annotated_preprocessed_Ontology_Normalization.csv`.

### 3. Build the knowledge graph

```bash
python scripts/build_and_extend_ontology.py
```

Requires `Ontology/sio.owl`. Produces `data/CPP_KG.ttl`.

### 4. Validate the knowledge graph

```bash
python validation/validate_shex.py
```

Validates `data/CPP_KG.ttl` against the ShEx shapes in
`validation/mechanisms_shapes.shex`.

### 5. Evaluate the ontology normaliser

```bash
python validation/Evaluate_ontology_normalizer.py
```

Compares normalised terms against `data/Ground_Truth_CHEBI.csv` and
`data/Ground_Truth_CLO.csv`.

### 6. Verify all numerical claims in the paper

```bash
python scripts/verify_paper_numbers.py
```

Checks every quantitative statement (counts, percentages) against the
data files and reports PASS/FAIL for each. All checks must pass before
submission.

## Expected key numbers

| Metric | Value |
|--------|-------|
| Downloaded sequences | 5,288 |
| Entries after preprocessing | 10,799 |
| Distinct sequences | 2,708 |
| KG CPP individuals | 2,637 |
| KG experiments | 4,408 |
| Total RDF triples | 159,810 |

## License

This work is licensed under
[Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use this knowledge graph or code in your research, please cite:

> *Citation details to be added upon publication.*

## Contact

For questions or issues, please open a GitHub issue in this repository.
