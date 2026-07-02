#!/usr/bin/env python
"""
Score our normalization pipeline against external baselines on the public benchmarks
(manuscript revision, R3.5 / R1.2).

Uniform rule: a method is correct on an entity if the gold identifier is among the
identifiers it returns. OLS and our pipeline return a single identifier (so this is
ordinary top-1 accuracy); the Cellosaurus and BioPortal Annotator baselines may return
several candidate classes (semicolon-separated), scored by set membership.

  accuracy  = correct / N
  precision = correct / entities for which the method returned >=1 identifier
  coverage  = entities with >=1 identifier / N

Baselines:
  OLS (EBI)            : ols_exact (strict dictionary), ols_top (rank-1 lexical)
  Cellosaurus          : cello_exact, cello_relaxed   (CLO only)
  BioPortal Annotator  : bio_pref (PREF match), bio_all (any match)
Our pipeline (from the benchmark normalization CSVs):
  ours_exact (exact_curie), ours_sapbert (biosyn_curie), ours_full (rag_curie)

Usage:
    python scripts/evaluate_baselines.py
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BL = ROOT / "revision" / "baselines"

BENCH = {
    "CRAFT (ChEBI)": dict(prefix="CHEBI", gold="gold_id",
                          ours=ROOT / "data" / "CRAFT_Ontology_Normalization.csv",
                          ols=BL / "baseline_ols_CRAFT.csv",
                          bioportal=BL / "baseline_bioportal_CRAFT.csv",
                          cello=None),
    "biosamples (CLO)": dict(prefix="CLO", gold="CLO_ID",
                             ours=ROOT / "data" / "biosamples_Ontology_Normalization.csv",
                             ols=BL / "baseline_ols_biosamples.csv",
                             bioportal=BL / "baseline_bioportal_biosamples.csv",
                             cello=BL / "baseline_cellosaurus_biosamples.csv"),
}


def canon(raw, prefix):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if not s or s.upper() in {"NIL", "NAN", "NONE"}:
        return None
    s = s.replace("_", ":")
    m = re.search(rf"{prefix}:?(\d+)", s, flags=re.I)
    return f"{prefix}:{m.group(1)}" if m else None


def to_set(cell, prefix):
    """Parse a cell (single id or ';'-joined ids) into a set of canonical ids."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return set()
    return {c for c in (canon(p, prefix) for p in str(cell).split(";")) if c}


def score(gold, pred_sets):
    n = len(gold)
    made = sum(1 for p in pred_sets if p)
    correct = sum(1 for g, p in zip(gold, pred_sets) if g in p)
    return dict(N=n, predicted=made, correct=correct,
                accuracy=correct / n if n else 0.0,
                precision=correct / made if made else float("nan"),
                coverage=made / n if n else 0.0)


def add(results, rows, bench, label, gold, sets):
    r = score(gold, sets)
    results.append((label, r))
    rows.append({"benchmark": bench, "method": label, **r})


def main():
    all_rows = []
    for bname, cfg in BENCH.items():
        pfx = cfg["prefix"]
        ours = pd.read_csv(cfg["ours"])
        gold = [canon(x, pfx) for x in ours[cfg["gold"]]]
        results = []

        # baselines (each scored against its own file's gold column / row order)
        if cfg["ols"].exists():
            ols = pd.read_csv(cfg["ols"])
            g = [canon(x, pfx) for x in ols[cfg["gold"]]]
            add(results, all_rows, bname, "OLS dictionary (exact)", g,
                [to_set(x, pfx) for x in ols["ols_exact"]])
            add(results, all_rows, bname, "OLS lexical (rank-1)", g,
                [to_set(x, pfx) for x in ols["ols_top"]])
        if cfg["bioportal"] and cfg["bioportal"].exists():
            bp = pd.read_csv(cfg["bioportal"])
            g = [canon(x, pfx) for x in bp[cfg["gold"]]]
            add(results, all_rows, bname, "BioPortal Annotator (PREF)", g,
                [to_set(x, pfx) for x in bp["bio_pref"]])
            add(results, all_rows, bname, "BioPortal Annotator (any)", g,
                [to_set(x, pfx) for x in bp["bio_all"]])
        if cfg["cello"] and cfg["cello"].exists():
            ce = pd.read_csv(cfg["cello"])
            g = [canon(x, pfx) for x in ce[cfg["gold"]]]
            add(results, all_rows, bname, "Cellosaurus (dictionary)", g,
                [to_set(x, pfx) for x in ce["cello_exact"]])
            add(results, all_rows, bname, "Cellosaurus (relaxed)", g,
                [to_set(x, pfx) for x in ce["cello_relaxed"]])

        # our pipeline (single-id columns)
        for label, col in [("Our exact-match stage", "exact_curie"),
                           ("Our SapBERT stage", "biosyn_curie"),
                           ("Our full pipeline", "rag_curie")]:
            add(results, all_rows, bname, label, gold,
                [to_set(x, pfx) for x in ours[col]])

        n = results[-1][1]["N"]
        print(f"\n=== {bname}  (N={n}) ===")
        print(f"{'method':<28}{'acc':>7}{'prec':>7}{'cov':>7}{'pred':>8}")
        for label, r in results:
            p = r["precision"]
            print(f"{label:<28}{r['accuracy']:>7.3f}{p:>7.3f}{r['coverage']:>7.3f}{r['predicted']:>8}")

    if all_rows:
        out = BL / "baseline_comparison.csv"
        pd.DataFrame(all_rows).to_csv(out, index=False)
        print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
