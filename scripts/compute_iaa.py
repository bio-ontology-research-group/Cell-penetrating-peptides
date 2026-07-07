#!/usr/bin/env python
"""
Inter-annotator agreement + consensus gold standard for the ontology-normalization
ground-truth set (manuscript revision, comment R3.4).

Run AFTER both annotators have independently filled their blank sheets
(GT_<ONTO>_annotator{A,B}_filled.csv). It reports:

  * raw percent agreement on the exact ontology ID (incl. NIL),
  * Cohen's kappa on the binary mappable-vs-NIL decision,
  * exact-ID agreement restricted to items both annotators judged mappable,

writes the disagreements to GT_<ONTO>_disagreements.csv for adjudication, and -
once an adjudicated column is supplied - emits the consensus gold standard in the
canonical format consumed by validation/Evaluate_ontology_normalizer.py.

Usage:
    python scripts/compute_iaa.py                 # IAA + disagreement sheets
    python scripts/compute_iaa.py --build-gold    # also build consensus GT from adjudication
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The revision/ folder may live inside the repo or as a sibling of it.
GTDIR = next((p for p in (ROOT / "revision" / "ground_truth_v2",
                          ROOT.parent / "revision" / "ground_truth_v2")
              if p.exists()), ROOT / "revision" / "ground_truth_v2")

TARGETS = {
    "CHEBI": dict(term_col="Cargo", id_col="Cargo_CHEBI_id", label_col="Cargo_CHEBI_label",
                  prefix="CHEBI", gold_out="Ground_Truth_CHEBI_v2.csv"),
    "CLO":   dict(term_col="Cell Line", id_col="CLO_id", label_col="CLO_label",
                  prefix="CLO", gold_out="Ground_Truth_CLO_v2.csv"),
}
NIL = "NIL"


def canon_id(raw, prefix: str) -> str:
    """Canonicalize to PREFIX:digits, or NIL for empty / explicit NIL."""
    if raw is None or (isinstance(raw, float)):
        return NIL
    s = str(raw).strip()
    if s == "" or s.upper() == NIL:
        return NIL
    s = s.replace("_", ":")
    m = re.search(rf"{prefix}:?(\d+)", s, flags=re.I)
    return f"{prefix}:{m.group(1)}" if m else NIL


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two equal-length label lists."""
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-gold", action="store_true",
                    help="build consensus GT from an 'adjudicated_id' column")
    args = ap.parse_args()

    for onto, cfg in TARGETS.items():
        fa = GTDIR / f"GT_{onto}_annotatorA_filled.csv"
        fb = GTDIR / f"GT_{onto}_annotatorB_filled.csv"
        if not (fa.exists() and fb.exists()):
            print(f"[{onto}] waiting for filled sheets: "
                  f"{fa.name} / {fb.name} (not found) - skipping.")
            continue

        # join on the term column (the natural key): annotator sheets may differ in
        # column layout - e.g. one annotator's copy has no sample_id - but both always
        # carry the term and the *_id / *_label columns.
        tc = cfg["term_col"]
        A = pd.read_csv(fa)
        B = pd.read_csv(fb)
        A[tc] = A[tc].astype(str).str.strip()
        B[tc] = B[tc].astype(str).str.strip()
        A = A.drop_duplicates(tc).set_index(tc)
        B = B.drop_duplicates(tc).set_index(tc)
        terms = [t for t in A.index if t in B.index]
        ida = [canon_id(A.loc[t, cfg["id_col"]], cfg["prefix"]) for t in terms]
        idb = [canon_id(B.loc[t, cfg["id_col"]], cfg["prefix"]) for t in terms]

        n = len(terms)
        if n == 0:
            print(f"[{onto}] no shared terms between the two sheets - check the files.")
            continue
        exact_agree = sum(x == y for x, y in zip(ida, idb)) / n
        binA = ["NIL" if x == NIL else "MAP" for x in ida]
        binB = ["NIL" if x == NIL else "MAP" for x in idb]
        kappa_nil = cohen_kappa(binA, binB)
        both_map = [(x, y) for x, y in zip(ida, idb) if x != NIL and y != NIL]
        id_agree_mapped = (sum(x == y for x, y in both_map) / len(both_map)
                           if both_map else float("nan"))

        print(f"\n=== {onto}  (n={n} shared terms) ===")
        print(f"  raw exact-ID agreement (incl. NIL) : {exact_agree:.3f}")
        print(f"  Cohen's kappa, mappable-vs-NIL     : {kappa_nil:.3f}")
        print(f"  exact-ID agreement | both mapped   : {id_agree_mapped:.3f} "
              f"(n={len(both_map)})")

        dis = pd.DataFrame({
            tc: terms,
            "annotatorA_id": ida, "annotatorB_id": idb,
            "annotatorA_label": [A.loc[t, cfg["label_col"]] for t in terms],
            "annotatorB_label": [B.loc[t, cfg["label_col"]] for t in terms],
            "adjudicated_id": "", "adjudicated_label": "",
        })
        dis = dis[[a != b for a, b in zip(ida, idb)]]
        out = GTDIR / f"GT_{onto}_disagreements.csv"
        dis.to_csv(out, index=False)
        print(f"  -> {len(dis)} disagreements written to {out.relative_to(ROOT)}")

        if args.build_gold:
            adj = GTDIR / f"GT_{onto}_disagreements_adjudicated.csv"
            # where annotators agree, gold = agreed id; else use adjudicated
            gold_id, gold_label = {}, {}
            for term, x, y in zip(terms, ida, idb):
                if x == y:
                    gold_id[term] = x
                    gold_label[term] = A.loc[term, cfg["label_col"]]
            if adj.exists():
                adf = pd.read_csv(adj).set_index(cfg["term_col"])
                for term, row in adf.iterrows():
                    gold_id[term] = canon_id(row.get("adjudicated_id"), cfg["prefix"])
                    gold_label[term] = row.get("adjudicated_label", "")
            else:
                print(f"  ! {adj.name} not found - gold will omit unresolved disagreements")
            rows = [{cfg["term_col"]: t, cfg["label_col"]: gold_label.get(t, ""),
                     cfg["id_col"]: gid} for t, gid in gold_id.items()]
            gdf = pd.DataFrame(rows)
            gpath = ROOT / "data" / cfg["gold_out"]
            gdf.to_csv(gpath, index=False)
            print(f"  -> consensus gold ({len(gdf)} terms) -> {gpath.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
