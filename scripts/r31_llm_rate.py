#!/usr/bin/env python
"""
Tabulate the R3.1 LLM error / hallucination-guard rate from a normalizer re-run.

Reviewer 3 (R3.1) asked how often the LLM (Graph-RAG) stage hallucinated or
mis-assigned an ontology term. The normalizer's Graph-RAG stage validates every
LLM-proposed CURIE against the real ChEBI/CLO class set and, when the LLM returns
a class that is not a genuine ontology term (or is not among the retrieved
candidates), rejects it and falls back to the embedding top-1. That decision is
recorded per term in the `RAG_match_type` column:

  * rag_selected / rag_graph_selected  -> LLM output ACCEPTED (a real, in-candidate
                                          class): the LLM was used and trusted.
  * rag_unconfident                    -> LLM output REJECTED by the guard
                                          (hallucinated / out-of-candidate CURIE)
                                          -> fell back to embedding top-1.
  * NIL                                -> no candidates / blank input (abstained).

The "hallucination / mis-assignment rate" reported to R3.1 is therefore
rag_unconfident / (terms the LLM was asked to resolve).

Run over the side-output CSVs produced by scripts/run_normalizer_r31.slurm:

    python scripts/r31_llm_rate.py \
        --clo   data/r31/full_CLO_Ontology_Normalization_r31.csv \
        --chebi data/r31/full_CHEBI_Ontology_Normalization_r31.csv \
        --out   data/r31/r31_llm_rate.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

# match_type buckets, grouped for R3.1 reporting.
LLM_ACCEPTED     = {"rag_selected", "rag_graph_selected"}   # LLM output adopted
LLM_HALLUCINATED = {"rag_hallucinated"}                     # invalid CURIE -> guard fired
LLM_ABSTAINED    = {"rag_abstained"}                        # LLM said NONE/empty
# Legacy label (pre-2026-07-02 runs conflated hallucination + abstention here).
LLM_UNCONFIDENT  = {"rag_unconfident"}
NIL_NO_CAND      = {"NIL"}


def _rag_match_type_col(df: pd.DataFrame) -> str:
    cands = [c for c in df.columns if c.lower().endswith("match_type")
             and c.lower().startswith("rag")]
    if not cands:
        cands = [c for c in df.columns if c.lower().endswith("match_type")]
    if not cands:
        raise SystemExit(
            "No *_match_type column found. Re-run the normalizer with current "
            "code (Ontology_normalizer.py:1099 writes it)."
        )
    return cands[0]


def _term_col(df: pd.DataFrame) -> str | None:
    for c in ("Cargo", "Cell Line", "Cell_Line", "term"):
        if c in df.columns:
            return c
    return None


def summarize(csv_path: str, ontology: str) -> dict:
    df = pd.read_csv(csv_path, dtype=str)
    mt_col = _rag_match_type_col(df)
    term_col = _term_col(df)

    # Per distinct non-blank surface form (type-level), which is what the RAG
    # stage actually decides on; fall back to row-level if no term column.
    if term_col is not None:
        sub = df[[term_col, mt_col]].copy()
        sub = sub[sub[term_col].astype(str).str.strip().ne("")]
        sub = sub.drop_duplicates(subset=[term_col])
        unit = "distinct_term"
    else:
        sub = df[[mt_col]].copy()
        unit = "row"

    counts = sub[mt_col].fillna("NIL").value_counts().to_dict()
    accepted     = sum(v for k, v in counts.items() if k in LLM_ACCEPTED)
    hallucinated = sum(v for k, v in counts.items() if k in LLM_HALLUCINATED)
    abstained    = sum(v for k, v in counts.items() if k in LLM_ABSTAINED)
    unconfident  = sum(v for k, v in counts.items() if k in LLM_UNCONFIDENT)
    nil_no_cand  = sum(v for k, v in counts.items() if k in NIL_NO_CAND)

    # Terms the LLM was actually asked to resolve (i.e. reached the RAG stage;
    # excludes exact-stage hits like exact_pref / prefix_truncated and NIL).
    llm_asked = accepted + hallucinated + abstained + unconfident
    denom = llm_asked or 1

    # Precise hallucination rate (needs the 2026-07-02 split). If the CSV still
    # uses the legacy conflated `rag_unconfident`, `unconfident` > 0 and the
    # split is unavailable — `fallback_rate` is then the reportable upper bound.
    split_available = unconfident == 0 and (hallucinated + abstained) > 0
    fallback_total = hallucinated + abstained + unconfident

    return {
        "ontology": ontology,
        "csv": str(csv_path),
        "match_type_column": mt_col,
        "counting_unit": unit,
        "raw_match_type_counts": counts,
        "llm_asked": llm_asked,
        "llm_accepted": accepted,
        "llm_hallucinated": hallucinated,
        "llm_abstained": abstained,
        "llm_unconfident_legacy": unconfident,
        "nil_no_candidates": nil_no_cand,
        "split_available": split_available,
        "hallucination_pct": round(100 * hallucinated / denom, 2),
        "abstention_pct": round(100 * abstained / denom, 2),
        "fallback_upper_bound_pct": round(100 * fallback_total / denom, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clo", help="CLO (Cell Line) normalization CSV")
    ap.add_argument("--chebi", help="ChEBI (Cargo) normalization CSV")
    ap.add_argument("--out", default="data/r31/r31_llm_rate.json")
    args = ap.parse_args()

    report = {}
    if args.chebi and Path(args.chebi).exists():
        report["chebi"] = summarize(args.chebi, "ChEBI")
    if args.clo and Path(args.clo).exists():
        report["clo"] = summarize(args.clo, "CLO")
    if not report:
        raise SystemExit("Provide at least one existing --chebi/--clo CSV.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print("\n=== R3.1 LLM hallucination / abstention rate ===")
    for onto, r in report.items():
        print(f"\n[{r['ontology']}]  (unit: {r['counting_unit']})")
        print(f"  LLM asked to resolve : {r['llm_asked']}")
        print(f"  accepted (valid)     : {r['llm_accepted']}")
        if r["split_available"]:
            print(f"  hallucinated (guard) : {r['llm_hallucinated']} "
                  f"({r['hallucination_pct']}%)")
            print(f"  abstained (NONE)     : {r['llm_abstained']} "
                  f"({r['abstention_pct']}%)")
        else:
            print(f"  fallback (legacy)    : {r['llm_unconfident_legacy']} "
                  f"({r['fallback_upper_bound_pct']}% upper bound; "
                  f"hallucination+abstention conflated — re-run for the split)")
        print(f"  NIL (no candidates)  : {r['nil_no_candidates']}")
        print(f"  raw match_type counts: {r['raw_match_type_counts']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
