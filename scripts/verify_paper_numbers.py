#!/usr/bin/env python3
"""
verify_paper_numbers.py
=======================
Reproduce and verify every quantitative claim made in the paper.
Run from the repository root:

    python scripts/verify_paper_numbers.py

All assertions compare computed values against the numbers stated in
the manuscript.  If any assertion fails the script exits with code 1.
"""
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

DATA = Path("data")

RAW_CSV = DATA / "Natural_CPP3_download_annotated.csv"
PREPROCESSED_CSV = DATA / "Natural_CPP3_download_annotated_preprocessed.csv"
NORMALIZED_CSV = DATA / "Natural_CPP3_download_annotated_preprocessed_Ontology_Normalization.csv"
KG_TTL = DATA / "Ontology" / "CPP_KG.ttl"

ok = True


def check(name: str, actual, expected, tol=0.0):
    global ok
    if isinstance(expected, float):
        match = abs(actual - expected) <= tol
    else:
        match = actual == expected
    status = "PASS" if match else "FAIL"
    if not match:
        ok = False
    print(f"  [{status}] {name}: got {actual}, expected {expected}")


print("=" * 70)
print("Paper number verification")
print("=" * 70)

# --- Raw data ---
df_raw = pd.read_csv(RAW_CSV)
print("\n--- Data Collection (Methods) ---")
check("Downloaded sequences", len(df_raw), 5288)
check("Unique raw sequences", df_raw["Sequence"].nunique(), 3048)

# --- Preprocessed data ---
df = pd.read_csv(PREPROCESSED_CSV)
print("\n--- After Preprocessing (Methods / Filtration) ---")
check("Final dataset entries", len(df), 10799)
check("Distinct sequences", df["Sequence"].nunique(), 2708)

# --- Source reference column (editor comment E6.3) ---
# Every entry carries a `reference` pointing to its exact CPPsite3 source
# (PubMed URL and/or patent identifier). Built by scripts/add_reference_column.py.
if "reference" in df.columns:
    _ref = df["reference"].fillna("").astype(str).str.strip()
    check("Entries with a source reference", int((_ref != "").sum()), 10754)
    check("Entries with a PubMed reference",
          int(_ref.str.contains("pubmed.ncbi", regex=False).sum()), 10508)
    check("Entries with a patent reference",
          int(_ref.str.contains("Patent:", regex=False).sum()), 585)
else:
    print("  [SKIP] 'reference' column absent (run scripts/add_reference_column.py)")

# --- Sequence-diversity figure (Data Records, Fig. sequence_diversity) ---
_seq = df["Sequence"].dropna().astype(str).str.strip().drop_duplicates()
_seq = _seq[_seq.str.len() > 0]
_len = _seq.str.len()
_charge = _seq.apply(lambda s: sum(s.count(a) for a in "KR") - sum(s.count(a) for a in "DE"))
check("Seq diversity: distinct sequences", len(_seq), 2708)
check("Seq diversity: median length", int(_len.median()), 16)
check("Seq diversity: median net charge", int(_charge.median()), 5)
check("Seq diversity: cationic %", round(100 * (_charge > 0).mean(), 1), 88.0, tol=0.05)

# --- Ontology-normalised data ---
df3 = pd.read_csv(NORMALIZED_CSV)
print("\n--- After Ontology Normalization ---")
check("Normalized entries", len(df3), 10799)

# --- Missingness (Table: Annotation Integrity Summary) ---
print("\n--- Missingness Table ---")
miss = {
    "Main Uptake Mechanism": (7699, 71.29),
    "Subcategory Uptake Mechanism": (9701, 89.83),
    "Subcellular Localization Category": (5642, 52.25),
    "Cargo Type": (205, 1.90),
    "Cell Line": (174, 1.61),
}
for col, (exp_count, exp_pct) in miss.items():
    m = int(df3[col].isna().sum())
    pct = round(m / len(df3) * 100, 2)
    check(f"{col} missing count", m, exp_count)
    check(f"{col} missing %", pct, exp_pct, tol=0.01)

# --- Mapping coverage (Technical Validation, R1.3) ---
# Reviewer 1 asked for CLO mapping success rate and the share of entries lacking an
# uptake-mechanism annotation. The manuscript ("Mapping coverage" paragraph) states:
# cell-line value present 98.4% (10,625/10,799), cargo value present 98.1% (10,590/10,799),
# and only 3,100/10,799 (28.7%) carry an uptake-mechanism annotation (so 71.3% lack one).
print("\n--- Mapping Coverage (R1.3) ---")
n_all = len(df)
cell_present = int(df["Cell Line"].fillna("").astype(str).str.strip().ne("").sum())
cargo_present = int(df["Cargo"].fillna("").astype(str).str.strip().ne("").sum())
mech_present = int(df["Main Uptake Mechanism"].fillna("").astype(str).str.strip().ne("").sum())
check("Cell-line value present count", cell_present, 10625)
check("Cell-line value present %", round(cell_present / n_all * 100, 1), 98.4, tol=0.05)
check("Cargo value present count", cargo_present, 10590)
check("Cargo value present %", round(cargo_present / n_all * 100, 1), 98.1, tol=0.05)
check("Uptake-mechanism present count", mech_present, 3100)
check("Uptake-mechanism present %", round(mech_present / n_all * 100, 1), 28.7, tol=0.05)

# --- LLM hallucination rate (Methods, R3.1) ---
# The Methods (Graph-RAG stage) states the guard rejected out-of-ontology identifiers
# for 0.3% of ChEBI terms (3/994) and 0.0% of CLO terms (0/728). Source: data/r31/r31_llm_rate.json.
import json
_r31 = DATA / "r31" / "r31_llm_rate.json"
if _r31.exists():
    print("\n--- LLM Hallucination Rate (R3.1) ---")
    r31 = json.loads(_r31.read_text())
    ch, cl = r31["chebi"], r31["clo"]
    check("ChEBI hallucinated terms", ch["llm_hallucinated"], 3)
    check("ChEBI terms passed to LLM", ch["llm_asked"], 994)
    check("ChEBI hallucination %", round(ch["llm_hallucinated"] / ch["llm_asked"] * 100, 1), 0.3, tol=0.05)
    check("CLO hallucinated terms", cl["llm_hallucinated"], 0)
    check("CLO terms passed to LLM", cl["llm_asked"], 728)
    check("CLO hallucination %", round(cl["llm_hallucinated"] / cl["llm_asked"] * 100, 1), 0.0, tol=0.05)
else:
    print("  [SKIP] data/r31/r31_llm_rate.json absent (run scripts/run_normalizer_r31.slurm)")

# --- Main Uptake Mechanism distribution ---
print("\n--- Uptake Mechanism Distribution ---")
mech = df3["Main Uptake Mechanism"].dropna()
cats_mech = Counter()
for v in mech:
    for p in str(v).split(", "):
        cats_mech[p.strip()] += 1
total_mech = sum(cats_mech.values())
endo = cats_mech["Endocytosis"]
direct = cats_mech["Direct penetration"]
check("Endocytosis %", round(endo / total_mech * 100, 2), 75.17, tol=0.01)
check("Direct penetration %", round(direct / total_mech * 100, 2), 24.83, tol=0.01)

# --- Subcategory Uptake Mechanism ---
print("\n--- Subcategory Uptake Mechanism ---")
sub = df3["Subcategory Uptake Mechanism"].dropna()
cats_sub = Counter()
for v in sub:
    for p in str(v).split(", "):
        cats_sub[p.strip()] += 1
total_sub = sum(cats_sub.values())
expected_sub = {
    "Macropinocytosis": 36.80,
    "Clathrin-mediated endocytosis": 31.32,
    "Caveolae-mediated endocytosis": 11.66,
    "Clathrin and caveolae independent": 18.48,
    "Phagocytosis": 1.74,
}
for name, exp_pct in expected_sub.items():
    actual_pct = round(cats_sub[name] / total_sub * 100, 2)
    check(f"{name} %", actual_pct, exp_pct, tol=0.01)

# --- Subcellular Localization ---
print("\n--- Subcellular Localization ---")
loc = df3["Subcellular Localization Category"].dropna()
cats_loc = Counter()
for v in loc:
    for p in str(v).split(", "):
        cats_loc[p.strip()] += 1
total_loc = sum(cats_loc.values())
expected_loc = {
    "Cytoplasm": 60.52,
    "Nucleus": 28.68,
    "Vesicles": 4.45,
    "Mitochondria": 3.29,
    "Endosomes": 3.06,
}
for name, exp_pct in expected_loc.items():
    actual_pct = round(cats_loc[name] / total_loc * 100, 2)
    check(f"{name} %", actual_pct, exp_pct, tol=0.01)

# --- Knowledge graph individual counts (manuscript Table 1) ---
# These are the numbers that previously drifted out of sync with the README.
# Counted as distinct subject IRIs in the asserted graph (CPP_KG.ttl).
print("\n--- Knowledge Graph (Table 1, asserted CPP_KG.ttl) ---")
if KG_TTL.exists():
    ttl = KG_TTL.read_text(encoding="utf-8")
    n_cpp = len(set(re.findall(r"dataset/CPP_\d+>", ttl)))
    n_complex = len(set(re.findall(r"dataset/cpp_complex_\d+>", ttl)))
    n_exp = len(set(re.findall(r"dataset/experiment_\d+>", ttl)))
    check("KG CPP individuals", n_cpp, 2642)
    check("KG CPP-Complexes", n_complex, 4132)
    check("KG experiments", n_exp, 4598)
else:
    print(f"  [SKIP] {KG_TTL} not present (build the KG to verify Table 1 counts)")

# --- Baseline-comparison table: our pipeline accuracy on public benchmarks ---
# (Table tab:baselines. The OLS baseline rows depend on a live service and are NOT
#  asserted here; these two are reproducible from the committed normalization CSVs.)
print("\n--- Baseline comparison (Table baselines, our pipeline accuracy) ---")


def _canon(raw, prefix):
    if pd.isna(raw):
        return None
    s = str(raw).strip().replace("_", ":")
    m = re.search(rf"{prefix}:?(\d+)", s, flags=re.I)
    return f"{prefix}:{m.group(1)}" if m else None


def _acc(csv, gold_col, pred_col, prefix):
    d = pd.read_csv(DATA / csv)
    g = [_canon(x, prefix) for x in d[gold_col]]
    p = [_canon(x, prefix) for x in d[pred_col]]
    return round(sum(a is not None and a == b for a, b in zip(g, p)) / len(d), 2)


craft_n = DATA / "CRAFT_Ontology_Normalization.csv"
bios_n = DATA / "biosamples_Ontology_Normalization.csv"
if craft_n.exists() and bios_n.exists():
    # Deterministic rag stage (temperature=0, seed=42): CRAFT 3966/4548=0.872,
    # biosamples 1544/2121=0.728. Reproducible from the deposited CSVs.
    check("Pipeline acc CRAFT:ChEBI (full)",
          _acc("CRAFT_Ontology_Normalization.csv", "gold_id", "rag_curie", "CHEBI"), 0.87, tol=0.005)
    check("Pipeline acc Biosamples:CLO (full)",
          _acc("biosamples_Ontology_Normalization.csv", "CLO_ID", "rag_curie", "CLO"), 0.73, tol=0.005)
else:
    print("  [SKIP] benchmark normalization CSVs not present")


def _nil_prf(csv, gold_col, pred_col, prefix):
    """NIL-aware single-answer scoring over ALL internal-GT terms (mappable + the
    terms the annotator marked unmappable). Returns Precision / Recall / F1 /
    NIL-aware Accuracy. FN = a MAPPABLE term (non-empty gold) left unmapped; a
    wrong id on a mappable term and any id on an unmappable term are false
    positives; a correct abstention on an unmappable term is a true negative."""
    d = pd.read_csv(DATA / csv)
    tp = fp = fn = tn = 0
    for g_raw, p_raw in zip(d[gold_col], d[pred_col]):
        g, p = _canon(g_raw, prefix), _canon(p_raw, prefix)
        if g is not None:
            if p is None:   fn += 1
            elif p == g:    tp += 1
            else:           fp += 1
        else:
            if p is None:   tn += 1
            else:           fp += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(d) if len(d) else 0.0
    return dict(prec=prec, rec=rec, f1=f1, acc=acc)


# --- Internal ground-truth (v2), NIL-aware F1 (Table baselines / internal
#     comparison). Reproducible from the committed norm CSVs. FN = mappable term
#     left unmapped; mapping an unmappable term is a false positive. ---
gt_chebi = DATA / "Ground_Truth_CHEBI_Ontology_Normalization.csv"
gt_clo = DATA / "Ground_Truth_CLO_Ontology_Normalization.csv"
CH = "Ground_Truth_CHEBI_Ontology_Normalization.csv"
CL = "Ground_Truth_CLO_Ontology_Normalization.csv"
if gt_chebi.exists() and gt_clo.exists():
    ch = _nil_prf(CH, "Cargo_CHEBI_id", "rag_curie", "CHEBI")
    cl = _nil_prf(CL, "CLO_id", "rag_curie", "CLO")
    check("Pipeline F1 Internal:ChEBI (NIL-aware)", ch["f1"], 0.53, tol=0.01)
    check("Pipeline acc Internal:ChEBI (NIL-aware)", ch["acc"], 0.36, tol=0.01)
    check("Pipeline F1 Internal:CLO (NIL-aware)", cl["f1"], 0.64, tol=0.01)
    check("Pipeline acc Internal:CLO (NIL-aware)", cl["acc"], 0.47, tol=0.01)
    # CLO precision/recall trade-off cited in the internal comparison: the abstaining
    # exact-match stage has higher accuracy on CLO but lower F1 than the full pipeline.
    ex_cl = _nil_prf(CL, "CLO_id", "exact_curie", "CLO")
    check("Exact-stage F1 Internal:CLO (NIL-aware)", ex_cl["f1"], 0.54, tol=0.01)
    check("Exact-stage acc Internal:CLO (NIL-aware)", ex_cl["acc"], 0.58, tol=0.01)
    # Ablation-table F1 column (tab:ablation_results). Internal SapBERT (Semantic) stage:
    check("SapBERT-stage F1 Internal:ChEBI",
          _nil_prf(CH, "Cargo_CHEBI_id", "biosyn_curie", "CHEBI")["f1"], 0.48, tol=0.01)
    check("SapBERT-stage F1 Internal:CLO",
          _nil_prf(CL, "CLO_id", "biosyn_curie", "CLO")["f1"], 0.58, tol=0.01)
    # Public deterministic stages (no unmappable terms, TN = 0):
    if craft_n.exists() and bios_n.exists():
        check("Graph-RAG F1 CRAFT:ChEBI (ablation)",
              _nil_prf("CRAFT_Ontology_Normalization.csv", "gold_id", "rag_curie", "CHEBI")["f1"], 0.93, tol=0.01)
        check("Graph-RAG F1 Biosamples:CLO (ablation)",
              _nil_prf("biosamples_Ontology_Normalization.csv", "CLO_ID", "rag_curie", "CLO")["f1"], 0.84, tol=0.01)
        check("SapBERT F1 CRAFT:ChEBI (ablation)",
              _nil_prf("CRAFT_Ontology_Normalization.csv", "gold_id", "biosyn_curie", "CHEBI")["f1"], 0.92, tol=0.01)
        check("SapBERT F1 Biosamples:CLO (ablation)",
              _nil_prf("biosamples_Ontology_Normalization.csv", "CLO_ID", "biosyn_curie", "CLO")["f1"], 0.79, tol=0.01)
else:
    print("  [SKIP] internal ground-truth normalization CSVs not present")

# --- Neural-encoder baselines (Table baselines). All four encoders are scored inside
#     the pipeline's OWN dense retrieval: validation/baseline_bert_encoders.py builds
#     BERTNormalizer with only the encoder swapped, so SapBERT in this role is
#     identical to the Semantic Mapping stage asserted above (0.92 / 0.79 / 0.48 /
#     0.58) and the two tables agree by construction. The encoders never abstain
#     (coverage 1.0), so precision = TP/N, and on the internal sets N = 300 because any
#     identifier returned for an unmappable term is a false positive; F1 = 2P/(P+1).
#     Reproduced by scripts/run_bert_baselines.slurm. ---
print("\n--- Neural-encoder baselines (Table baselines) ---")


def _enc_f1(model, dataset, denom):
    p = DATA / "baselines" / f"preds_{model}_{dataset}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    pfx = "CHEBI" if "ChEBI" in dataset else "CLO"
    tp = sum(_canon(r["pred"], pfx) == _canon(r["gold"], pfx) for _, r in d.iterrows())
    P = tp / denom
    return 2 * P / (P + 1)


ENC_EXPECTED = {
    "biobert":    {"CRAFT_ChEBI": (4548, 0.83), "Biosamples_CLO": (2121, 0.13),
                   "Internal_ChEBI": (300, 0.11), "Internal_CLO": (300, 0.22)},
    "pubmedbert": {"CRAFT_ChEBI": (4548, 0.82), "Biosamples_CLO": (2121, 0.07),
                   "Internal_ChEBI": (300, 0.06), "Internal_CLO": (300, 0.17)},
    "krissbert":  {"CRAFT_ChEBI": (4548, 0.85), "Biosamples_CLO": (2121, 0.58),
                   "Internal_ChEBI": (300, 0.24), "Internal_CLO": (300, 0.49)},
}
if (DATA / "baselines" / "preds_krissbert_CRAFT_ChEBI.csv").exists():
    for _m, _exp in ENC_EXPECTED.items():
        for _ds, (_denom, _want) in _exp.items():
            check(f"{_m} F1 {_ds}", _enc_f1(_m, _ds, _denom), _want, tol=0.015)
else:
    print(f"  [SKIP] {NEURAL_CSV} not present (run score_encoders_manuscript.py)")

print("\n" + "=" * 70)
if ok:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED")
    sys.exit(1)
