#!/usr/bin/env python
"""
Internal ground-truth evaluation of the ontology-normalization pipeline
(manuscript revision, R3.4 / R1.2 / R3.5).

The gold terms (data/Ground_Truth_<ONTO>_v2.csv, 300/ontology, built by
scripts/build_single_annotator_gold.py) are a subset of the full dataset, which
was already normalized end-to-end in the R3.1 run (data/r31/full_<ONTO>_...r31.csv,
with per-term exact_curie / biosyn_curie / rag_curie). Rather than re-running the
LLM, we attach those predictions to the gold terms by surface form, write the
canonical
    data/Ground_Truth_<ONTO>_Ontology_Normalization.csv
(gold columns + prediction columns; drop-in for Evaluate_ontology_normalizer.py),
and report:

  1. Per-stage precision / recall / F1 / accuracy over ALL gold terms with
     NIL-aware scoring: mappable gold (gold != NIL) is scored TP/FP/FN as usual,
     while for an UNMAPPABLE gold term (the annotator found no adequate class) a
     stage that emits any identifier is charged a FALSE POSITIVE and a stage that
     abstains earns a true negative. Recall is over the mappable terms; precision
     and accuracy include the unmappable terms.
  2. An abstention breakdown on the NIL gold subset: how often each stage abstains
     (true negative) vs. force-maps (false positive) — ties to R3.1/R3.2.

Usage:
    python validation/evaluate_internal_gt.py
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NIL = "NIL"

TARGETS = {
    "CHEBI": dict(term_col="Cargo", gold_col="Cargo_CHEBI_id",
                  gold="data/Ground_Truth_CHEBI_v2.csv",
                  r31="data/r31/full_CHEBI_Ontology_Normalization_r31.csv",
                  out="data/Ground_Truth_CHEBI_Ontology_Normalization.csv"),
    "CLO":   dict(term_col="Cell Line", gold_col="CLO_id",
                  gold="data/Ground_Truth_CLO_v2.csv",
                  r31="data/r31/full_CLO_Ontology_Normalization_r31.csv",
                  out="data/Ground_Truth_CLO_Ontology_Normalization.csv"),
}
PRED_COLS = [("exact_curie", "Exact"), ("biosyn_curie", "Semantic (SapBERT/BioSyn)"),
             ("rag_curie", "Graph-RAG (full)")]


def norm_id(raw):
    """Canonical CLO:###/CHEBI:### or None (empty / NIL)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if not s or s.upper() == NIL:
        return None
    s = re.sub(r"^(CLO|CHEBI)[_:](\d+)$", r"\1:\2", s, flags=re.I)
    return s or None


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> None:
    for onto, cfg in TARGETS.items():
        tc, gc = cfg["term_col"], cfg["gold_col"]
        gold = pd.read_csv(ROOT / cfg["gold"], dtype=str)
        r31 = pd.read_csv(ROOT / cfg["r31"], dtype=str)

        # one prediction row per surface form (deterministic within a run)
        pred_cols = [c for c, _ in PRED_COLS]
        r31[tc] = r31[tc].astype(str).str.strip()
        preds = r31.drop_duplicates(tc).set_index(tc)[pred_cols]

        gold[tc] = gold[tc].astype(str).str.strip()
        merged = gold.merge(preds, left_on=tc, right_index=True, how="left")
        out_path = ROOT / cfg["out"]
        merged.to_csv(out_path, index=False)

        gold_norm = merged[gc].map(norm_id)
        mappable = gold_norm.notna()
        n_map = int(mappable.sum())
        n_nil = int((~mappable).sum())

        print("\n" + "=" * 74)
        print(f"{onto}  —  internal GT (single-annotator A, interim)   "
              f"[{len(merged)} terms: {n_map} mappable, {n_nil} NIL]")
        print(f"  wrote {out_path.relative_to(ROOT)}")
        print("-" * 74)
        print(f"  {'Stage':<28}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}"
              f"{'Prec':>8}{'Rec':>8}{'F1':>8}{'Acc':>8}   (NIL-aware, all terms)")
        for col, label in PRED_COLS:
            tp = fp = fn = tn = 0
            for gid, pred in zip(gold_norm, merged[col]):
                p = norm_id(pred)
                if gid is not None:              # mappable gold term
                    if p is None:    fn += 1     # abstained on a mappable term
                    elif p == gid:   tp += 1     # correct mapping
                    else:            fp += 1     # wrong mapping
                else:                            # unmappable (NIL) gold term
                    if p is None:    tn += 1     # correct abstention
                    else:            fp += 1     # mapped an unmappable term -> false positive
            P, R, F = prf(tp, fp, fn)
            acc = (tp + tn) / len(merged) if len(merged) else 0.0
            print(f"  {label:<28}{tp:>5}{fp:>5}{fn:>5}{tn:>5}"
                  f"{P:>8.3f}{R:>8.3f}{F:>8.3f}{acc:>8.3f}")

        if n_nil:
            print("-" * 74)
            print(f"  Abstention on {n_nil} NIL gold terms "
                  f"(annotator: no adequate class):")
            for col, label in PRED_COLS:
                sub = merged.loc[~mappable, col].map(norm_id)
                abstained = int(sub.isna().sum())
                mapped = int(sub.notna().sum())
                print(f"    {label:<28} abstained={abstained:>4}  "
                      f"force-mapped={mapped:>4}  "
                      f"(abstention {100*abstained/n_nil:.1f}%)")
    print()


if __name__ == "__main__":
    main()
