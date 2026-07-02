#!/usr/bin/env python
"""
Reproducible, frequency-stratified sampler for the ontology-normalization
ground-truth evaluation set (manuscript revision, comment R3.4).

It draws a fixed-size random sample of *distinct* free-text terms for each
target ontology directly from the preprocessed dataset, stratified by how often
each term occurs. The output is a set of blank double-annotation sheets: one per
ontology per annotator, containing only the surface form (and light context),
with empty columns for the gold ontology ID. Annotators fill these in
independently and blind to the pipeline's predictions; agreement is computed
afterwards with compute_iaa.py.

Design rationale (see revision/ground_truth_protocol.md):
  * Sampling unit = distinct surface form (type), not row (token): we evaluate
    the normalizer's ability to map the *vocabulary* it encounters.
  * Stratify by occurrence-frequency band and allocate proportional to the number
    of distinct terms per band, with a per-band floor. This deliberately gives the
    long tail of rare terms - where normalization errors concentrate - adequate
    weight, while the floor guarantees the small high-frequency head is covered.
  * Fixed RNG seed -> the exact sample is reproducible from this script alone.

Usage:
    python scripts/sample_ground_truth.py            # default n=300, seed=42
    python scripts/sample_ground_truth.py --n 300 --seed 42
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from urllib.parse import quote_plus
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data" / "Natural_CPP3_download_annotated_preprocessed.csv"
OUTDIR = ROOT / "revision" / "ground_truth_v2"

# (free-text column, optional context column, output-term column name, id/label names, v1 file)
TARGETS = {
    "CHEBI": dict(text_col="Cargo", ctx_col=None,
                  term_col="Cargo", label_col="Cargo_CHEBI_label", id_col="Cargo_CHEBI_id",
                  v1="data/Ground_Truth_CHEBI.csv", v1_term="Cargo"),
    "CLO":   dict(text_col="Cell Line", ctx_col=None,
                  term_col="Cell Line", label_col="CLO_label", id_col="CLO_id",
                  v1="data/Ground_Truth_CLO.csv", v1_term="Cell Line"),
}

# frequency bands (closed integer intervals on per-term occurrence count)
BANDS = [("singleton", 1, 1), ("rare", 2, 5), ("common", 6, 20), ("frequent", 21, 10**9)]
FLOOR = 30          # minimum distinct terms drawn from any non-empty band
ANNOTATORS = ["A", "B"]

# Canonical per-ontology seeds. These are the *recovered* effective seeds of the
# sample originally distributed to the annotators (the first draw used a salted,
# non-reproducible process hash; these fixed values reproduce that exact draw and are
# now frozen so the released sample never changes again). Recovered by brute force
# (scripts/recover check) against the distributed sheets; do not change.
CANON_SEED = {"CHEBI": 764, "CLO": 1021}


def band_of(freq: int) -> str:
    for name, lo, hi in BANDS:
        if lo <= freq <= hi:
            return name
    raise ValueError(freq)


def allocate(band_sizes: dict[str, int], n: int, floor: int) -> dict[str, int]:
    """Proportional-to-distinct-terms allocation with a per-band floor, summing to n."""
    bands = [b for b in band_sizes if band_sizes[b] > 0]
    total = sum(band_sizes[b] for b in bands)
    # start from proportional, then lift each band to the floor (capped at its size)
    alloc = {b: max(min(floor, band_sizes[b]),
                    int(round(n * band_sizes[b] / total))) for b in bands}
    alloc = {b: min(alloc[b], band_sizes[b]) for b in bands}      # never exceed available
    # repair rounding so the total equals n exactly
    def fix():
        diff = n - sum(alloc.values())
        order = sorted(bands, key=lambda b: band_sizes[b], reverse=(diff > 0))
        i = 0
        d = diff
        while d != 0 and order:
            b = order[i % len(order)]
            if d > 0 and alloc[b] < band_sizes[b]:
                alloc[b] += 1; d -= 1
            elif d < 0 and alloc[b] > min(floor, band_sizes[b]):
                alloc[b] -= 1; d += 1
            i += 1
            if i > 10000:
                break
        return d
    fix()
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="sample size per ontology")
    args = ap.parse_args()

    df = pd.read_csv(PREP, low_memory=False)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    manifest = {"canonical_seeds": CANON_SEED, "n_per_ontology": args.n,
                "source": str(PREP.relative_to(ROOT)), "bands": [b[0] for b in BANDS],
                "floor": FLOOR, "ontologies": {}}

    for onto, cfg in TARGETS.items():
        # fixed canonical seed per ontology (frozen to match the distributed sample)
        rng = np.random.default_rng(CANON_SEED[onto])
        s = df[cfg["text_col"]].dropna().astype(str).str.strip()
        s = s[s != ""]
        vc = s.value_counts()                                   # distinct term -> freq
        frame = pd.DataFrame({"term": vc.index, "freq": vc.values})
        frame["band"] = frame["freq"].map(band_of)

        band_sizes = frame.groupby("band").size().to_dict()
        alloc = allocate(band_sizes, args.n, FLOOR)

        picks = []
        for band, k in alloc.items():
            pool = frame[frame["band"] == band].sort_values("term").reset_index(drop=True)
            idx = rng.choice(len(pool), size=k, replace=False)
            picks.append(pool.iloc[sorted(idx)])
        sample = pd.concat(picks).sort_values(["band", "term"]).reset_index(drop=True)

        # light context for cargo (the categorical Cargo Type most frequently seen with the term)
        ctx_map = {}
        if cfg["ctx_col"] and cfg["ctx_col"] in df.columns:
            g = df.dropna(subset=[cfg["text_col"]]).copy()
            g[cfg["text_col"]] = g[cfg["text_col"]].astype(str).str.strip()
            for term, sub in g[g[cfg["text_col"]].isin(sample["term"])].groupby(cfg["text_col"]):
                ctx_map[term] = sub[cfg["ctx_col"]].dropna().astype(str).mode().iloc[0] \
                    if sub[cfg["ctx_col"]].notna().any() else ""

        # overlap with the v1 convenience set (for transparency)
        v1_terms = set(pd.read_csv(ROOT / cfg["v1"])[cfg["v1_term"]].astype(str).str.strip())
        overlap = int(sample["term"].isin(v1_terms).sum())

        # blank annotation sheet (one per annotator)
        for ann in ANNOTATORS:
            sheet = pd.DataFrame({
                "sample_id": [f"{onto}_{i:03d}" for i in range(len(sample))],
                "band": sample["band"].values,
                "frequency": sample["freq"].values,
                cfg["term_col"]: sample["term"].values,
            })
            # ready-made BioPortal search link (scoped to the target ontology) per term
            sheet["bioportal_search_url"] = sheet[cfg["term_col"]].astype(str).map(
                lambda t: f"https://bioportal.bioontology.org/search?q={quote_plus(t)}"
                          f"&ontologies={onto}")
            if ctx_map:
                sheet["context_cargo_type"] = sheet[cfg["term_col"]].map(ctx_map).fillna("")
            sheet[cfg["id_col"]] = ""          # annotator fills: CHEBI:xxxxx / CLO:xxxxxxx  (or NIL)
            sheet[cfg["label_col"]] = ""       # annotator fills: ontology preferred label
            sheet["is_NIL"] = ""               # yes if no adequate ontology concept exists
            sheet["confidence_1to3"] = ""      # 1 low .. 3 high (optional)
            sheet["notes"] = ""
            out = OUTDIR / f"GT_{onto}_annotator{ann}_blank.csv"
            sheet.to_csv(out, index=False)

        manifest["ontologies"][onto] = {
            "seed": CANON_SEED[onto],
            "distinct_terms_total": int(len(frame)),
            "band_sizes": {b: int(band_sizes.get(b, 0)) for b, *_ in BANDS},
            "allocation": {b: int(alloc.get(b, 0)) for b, *_ in BANDS},
            "sampled": int(len(sample)),
            "overlap_with_v1_120": overlap,
            "blank_sheets": [f"GT_{onto}_annotator{a}_blank.csv" for a in ANNOTATORS],
        }
        print(f"[{onto}] distinct={len(frame)} sample={len(sample)} "
              f"alloc={manifest['ontologies'][onto]['allocation']} overlap_v1={overlap}")

    (OUTDIR / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote blank annotation sheets + manifest to {OUTDIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
