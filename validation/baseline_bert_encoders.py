#!/usr/bin/env python
"""
Contextual-encoder entity-linking baselines for reviewer comment R1.2
("compare with BioBERT / ERNIE-Bio").

Fair BERT-family comparison: this reuses the EXACT dense-retrieval pipeline of
the reported SapBERT "Semantic Mapping" stage (embed every ontology synonym ->
FAISS inner-product index -> nearest concept via synonym marginalization), and
swaps ONLY the encoder. The encoder is therefore the single independent
variable, so the row is directly comparable to the SapBERT (Semantic Mapping)
row already in Table tab:ablation_results / tab:baselines.

Datasets & scoring match validation/evaluate_single_answer.py exactly:
single-answer accuracy over the MAPPABLE terms (those with a gold id), with id
canonicalization via the same `canon` regex.

    CRAFT:ChEBI      data/CRAFT.csv                 entity_text -> gold_id      (row-wise, all mentions)
    Biosamples:CLO   data/biosamples.csv            Cell Line   -> CLO_ID       (row-wise, all mentions)
    Internal:ChEBI   data/Ground_Truth_CHEBI_v2.csv Cargo       -> Cargo_CHEBI_id  (dedup)
    Internal:CLO     data/Ground_Truth_CLO_v2.csv   Cell Line   -> CLO_id          (dedup)

Encoders are given as `slug=hf_model_id` pairs on the command line, e.g.
    --models biobert=dmis-lab/biobert-base-cased-v1.1

GPU strongly recommended (ChEBI = ~708k synonym strings). Concept embeddings are
cached per (ontology, model, obo-mtime) via the BERTNormalizer cache, so re-runs
are fast.

Output:
    data/baselines/baseline_bert_encoders.csv           (model,dataset,ontology,N,accuracy,coverage)
    data/baselines/preds_<model>_<dataset>.csv          (term,gold,pred) for auditing
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from Ontology_normalizer import BERTNormalizer, preprocess  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "data" / "baselines"
OBO = {"chebi": DATA / "Ontology" / "chebi.obo", "clo": DATA / "Ontology" / "clo.obo"}


def canon(raw, pfx):
    """Canonicalize an id to '<PFX>:<digits>' (same rule as evaluate_single_answer.py)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    m = re.search(rf"{pfx}[:_]?(\d+)", str(raw), flags=re.I)
    return f"{pfx}:{m.group(1)}" if m else None


# (title, csv, term_col, gold_col, pfx, ontology, dedup)
DATASETS = [
    ("CRAFT:ChEBI",    DATA / "CRAFT.csv",                  "entity_text", "gold_id",       "CHEBI", "chebi", False),
    ("Biosamples:CLO", DATA / "biosamples.csv",             "Cell Line",   "CLO_ID",        "CLO",   "clo",   False),
    ("Internal:ChEBI", DATA / "Ground_Truth_CHEBI_v2.csv",  "Cargo",       "Cargo_CHEBI_id","CHEBI", "chebi", True),
    ("Internal:CLO",   DATA / "Ground_Truth_CLO_v2.csv",    "Cell Line",   "CLO_id",        "CLO",   "clo",   True),
]


def load_dataset(csv, term_col, gold_col, pfx, dedup):
    """Return (terms, gold_curies) over MAPPABLE rows only, matching the scorer."""
    d = pd.read_csv(csv)
    d.columns = [c.strip() for c in d.columns]
    if dedup:
        d = d.drop_duplicates(term_col)
    d["_g"] = d[gold_col].map(lambda x: canon(x, pfx))
    d = d[d["_g"].notna()].reset_index(drop=True)
    terms = [str(t).strip() for t in d[term_col].tolist()]
    return terms, d["_g"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="slug=hf_model_id pairs, e.g. biobert=dmis-lab/biobert-base-cased-v1.1")
    args = ap.parse_args()
    models = dict(m.split("=", 1) for m in args.models)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for slug, model_id in models.items():
        # A normalizer is built per (model, ontology); reuse it across datasets.
        normalizers = {}
        for title, csv, term_col, gold_col, pfx, onto, dedup in DATASETS:
            if onto not in normalizers:
                # Swap the encoder for this ontology, then build the index (cached).
                BERTNormalizer.MODELS[onto] = model_id
                print(f"\n>>> {slug} ({model_id}) on {onto} ...", flush=True)
                normalizers[onto] = BERTNormalizer(str(OBO[onto]), onto, llm_backend=None)
            nrm = normalizers[onto]

            terms, gold = load_dataset(csv, term_col, gold_col, pfx, dedup)
            preds_raw = nrm.run(terms)                       # aligned list of dicts
            preds = [canon(r.get("curie"), pfx) for r in preds_raw]

            n = len(gold)
            hit = sum(p == g for p, g in zip(preds, gold))
            cov = sum(p is not None for p in preds)
            acc = hit / n if n else 0.0
            print(f"    {title:<16} N={n:<5} acc={acc:.3f} cov={cov/n:.3f}", flush=True)
            rows.append({"model": slug, "model_id": model_id, "dataset": title,
                         "ontology": onto, "N": n, "accuracy": round(acc, 4),
                         "coverage": round(cov / n, 4)})
            pd.DataFrame({"term": terms, "gold": gold, "pred": preds}).to_csv(
                OUT / f"preds_{slug}_{title.replace(':', '_')}.csv", index=False)

    res = pd.DataFrame(rows)
    # Model-specific filename so concurrent jobs (e.g. biobert+pubmedbert vs krissbert)
    # do not clobber each other's summary; merge_bert_encoder_baselines() combines them.
    out_csv = OUT / f"baseline_bert_encoders_{'_'.join(models)}.csv"
    res.to_csv(out_csv, index=False)
    print("\n=== summary ===")
    print(res.to_string(index=False))
    print(f"\n>> wrote {OUT / 'baseline_bert_encoders.csv'}")


if __name__ == "__main__":
    main()
