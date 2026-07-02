#!/usr/bin/env python
"""
Correct spurious ontology assignments for entries whose SOURCE field is empty
(manuscript revision, R3.1 / R3.2).

The normalization pipeline (scripts/Ontology_normalizer.py) never abstains: its
SapBERT and Graph-RAG stages always return the nearest ontology concept, with no
confidence threshold. As a result, entries whose free-text `Cargo` or `Cell Line`
field is empty were still assigned an identifier -- the nearest embedding neighbour
of the empty string -- rather than being left unmapped. In the released
normalization table this produced:

  * 209 entries with an empty `Cargo`     -> CHEBI:49997 ("radon(0)")
  * 174 entries with an empty `Cell Line`  -> a single placeholder CLO class

These are not real annotations; they are artefacts of the no-abstention design.
This script sets those assignments to NIL (empty) so that:

  * blank-cargo rows are dropped from the knowledge graph (the builder skips rows
    whose `RAG_curie_CheBI` is empty), and
  * blank-cell-line rows keep their other content but lose the spurious cell-line
    link (the builder only adds a cell line when `RAG_curie_CLO` is non-empty).

The original file is backed up to `*.pre_blankfix.csv`. The script is idempotent.

Usage:
    python scripts/fix_blank_mappings.py            # apply in place (with backup)
    python scripts/fix_blank_mappings.py --dry-run  # report only
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "Natural_CPP3_download_annotated_preprocessed_Ontology_Normalization.csv"
BACKUP = CSV.with_name(CSV.stem + ".pre_blankfix.csv")

# source free-text field -> the mapping columns it drives
GROUPS = {
    "Cargo":     ["Biosyn_curie_CheBI", "Biosyn_label_ChebI", "Biosyn_score_CheBI",
                  "RAG_curie_CheBI", "RAG_label_CheBI", "RAG_score_CheBI"],
    "Cell Line": ["Biosyn_curie_CLO", "Biosyn_label_CLO", "Biosyn_score_CLO",
                  "RAG_curie_CLO", "RAG_label_CLO", "RAG_score_CLO"],
}


def is_blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "") | \
        (s.astype(str).str.upper().isin(["NIL", "NAN", "NONE"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(CSV, dtype=str)
    n = len(df)
    print(f"Loaded {CSV.relative_to(ROOT)} ({n} rows)")

    total_fixed = 0
    for field, cols in GROUPS.items():
        blank = is_blank(df[field])
        curie_col = [c for c in cols if c.startswith("RAG_curie")][0]
        # rows currently carrying a spurious (non-empty) id despite a blank source
        spurious = blank & ~is_blank(df[curie_col])
        cnt = int(spurious.sum())
        assigned = df.loc[spurious, curie_col].value_counts().to_dict()
        print(f"  {field:10s}: {int(blank.sum())} blank, {cnt} force-mapped -> {assigned}")
        if not args.dry_run and cnt:
            present = [c for c in cols if c in df.columns]
            df.loc[spurious, present] = pd.NA
        total_fixed += cnt

    if args.dry_run:
        print(f"[dry-run] would clear {total_fixed} spurious assignments; no file written.")
        return

    if not BACKUP.exists():
        pd.read_csv(CSV, dtype=str).to_csv(BACKUP, index=False)
        print(f"  backed up original -> {BACKUP.relative_to(ROOT)}")
    df.to_csv(CSV, index=False)
    print(f"Cleared {total_fixed} spurious assignments; wrote {CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
