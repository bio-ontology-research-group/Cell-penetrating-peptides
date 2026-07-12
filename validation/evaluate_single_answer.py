#!/usr/bin/env python
"""
True single-answer scoring of the external baselines, for the R3.5 comparison.

A set-membership rule (a hit if the gold id is anywhere in the returned candidate
set) equals top-1 accuracy for single-valued methods (OLS, our pipeline), but the
BioPortal Annotator (bio_pref, up to 4 concepts) and Cellosaurus (cello_exact, up
to 5) return several candidates, so that rule overstates them and is not
comparable to methods that commit to one identifier. This script therefore scores
every method as a SINGLE answer: its first valid candidate must equal the gold id.
This single-answer rule is the only baseline scoring the repository keeps.

Scoring is NIL-aware: the internal ground truth keeps the terms the annotator
judged UNMAPPABLE (no adequate ontology class). For such a term the correct
behaviour is to abstain — a method that returns any identifier is charged a false
positive (reported in the `FP` column), and one that abstains is credited. So
`single` accuracy = (correct mappings + correct abstentions) / all terms. The
public benchmarks (CRAFT, biosamples) are all mappable, so this is a no-op there.

For the internal ground truth we additionally report an "accepted-set" score:
the annotator recorded alternative acceptable ids in the `notes` column (mostly
the second entity of multi-entity cargo strings). accepted = gold + notes ids; a
method is credited if its single answer is in that set. This is reported as a
robustness check, not the headline (still single-annotator; annotator B pending).

Per method we report Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1, NIL-aware
accuracy, the FP count, and coverage. FN counts only a MAPPABLE term (non-empty
gold) that the method left unmapped; a wrong id on a mappable term is an FP, and
an id on an unmappable term is an FP. Every number is auditable. No GPU / network.

Usage:  python validation/evaluate_single_answer.py
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOP = ROOT / "data" / "baselines"                     # public-benchmark baselines
SIB = ROOT / "data" / "baselines"                     # internal-GT baselines
DATA = ROOT / "data"
GTV2 = ROOT / "data" / "ground_truth_v2"              # annotator sheets (notes/alts)


def canon(raw, pfx):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    m = re.search(rf"{pfx}[:_]?(\d+)", str(raw), flags=re.I)
    return f"{pfx}:{m.group(1)}" if m else None


def candidates(cell, pfx):
    """All valid ids in a ';'-separated cell, order preserved, blanks dropped."""
    if pd.isna(cell):
        return []
    return [c for c in (canon(x, pfx) for x in str(cell).split(";")) if c]


def single(cell, pfx):
    c = candidates(cell, pfx)
    return c[0] if c else None            # first valid candidate = the one answer


def all_ids(cell, pfx):
    if pd.isna(cell):
        return set()
    return {f"{pfx}:{m}" for m in re.findall(rf"{pfx}[:_]?(\d+)", str(cell), flags=re.I)}


def score(golds, preds, accepted=None):
    """NIL-aware single-answer scoring. A gold of None marks an UNMAPPABLE term
    (annotator found no adequate class): a method that abstains (pred None) is
    correct (true negative); any identifier it emits is a false positive.
    single_acc = (TP + TN) / n; coverage = fraction that emitted an id."""
    n = len(golds)
    tp = fp = fn = tn = made = acc_hit = 0
    for i, g in enumerate(golds):
        p = preds[i]
        if p is not None:
            made += 1
        if g is not None:                       # mappable gold term
            if p is None:      fn += 1
            elif p == g:       tp += 1
            else:              fp += 1
        else:                                   # unmappable (NIL) gold term
            if p is None:      tn += 1
            else:              fp += 1           # mapped an unmappable term -> FP
        if accepted is not None:
            ok = (p is None and g is None) or (p is not None and p in accepted[i])
            acc_hit += ok
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0   # FN = a mappable term left unmapped
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(n=n, tp=tp, fp=fp, fn=fn, tn=tn,
                prec=prec, rec=rec, f1=f1,
                single=(tp + tn) / n if n else 0.0,
                cover=made / n if n else 0.0,
                accept=(acc_hit / n if accepted is not None else None))


def load(path, term):
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    return d.drop_duplicates(term).set_index(term)


def notes_accepted(filled_path, term, gcol, pfx, terms):
    """accepted[i] = {gold} U {ids in notes} for each surface term."""
    f = pd.read_csv(filled_path)
    f.columns = [c.strip() for c in f.columns]
    acc = {}
    for _, r in f.iterrows():
        t = str(r[term]).strip()
        g = canon(r.get(gcol), pfx)
        if g is None:
            continue
        acc[t] = {g} | all_ids(r.get("notes"), pfx)
    return [acc.get(t, set()) for t in terms]


def bench(title, gold_map, method_preds, accepted=None):
    print(f"\n=== {title} ===")
    print(f"  {'method':<26}{'Prec':>7}{'Rec':>7}{'F1':>7}{'Acc':>7}{'FP':>6}{'Cov':>7}"
          + ("  accept" if accepted else ""))
    for name, preds in method_preds:
        r = score(gold_map, preds, accepted)
        line = (f"  {name:<26}{r['prec']:>7.3f}{r['rec']:>7.3f}{r['f1']:>7.3f}"
                f"{r['single']:>7.3f}{r['fp']:>6}{r['cover']:>7.3f}")
        if accepted is not None and r["accept"] is not None:
            line += f"{r['accept']:>8.3f}"
        print(line + f"   (n={r['n']})")


def public(title, path, gcol_name, pfx, methods):
    """Row-wise single-answer scoring over ALL mappable rows of a self-contained
    baseline file (public benchmarks: one gold + prediction columns per mention;
    NO dedup — CRAFT/biosamples score every mention)."""
    d = pd.read_csv(path); d.columns = [c.strip() for c in d.columns]
    gcol = [c for c in d.columns if c.lower() == gcol_name.lower()][0]
    d["_g"] = d[gcol].map(lambda x: canon(x, pfx))
    dm = d[d["_g"].notna()].reset_index(drop=True)
    gold = dm["_g"].tolist()
    preds = [(name, [single(v, pfx) for v in dm[col]]) for name, col in methods]
    bench(title, gold, preds)


def main():
    # ---------- Public: CRAFT ChEBI (row-wise, all mentions) ----------
    public("CRAFT:ChEBI  (public, OLS file)", TOP / "baseline_ols_CRAFT.csv", "gold_id", "CHEBI",
           [("OLS dictionary (exact)", "ols_exact"), ("OLS lexical (rank-1)", "ols_top")])
    public("CRAFT:ChEBI  (public, BioPortal file)", TOP / "baseline_bioportal_CRAFT.csv", "gold_id", "CHEBI",
           [("BioPortal (1-answer)", "bio_pref")])

    # ---------- Public: Biosamples CLO (row-wise, all mentions) ----------
    public("Biosamples:CLO  (public, OLS file)", TOP / "baseline_ols_biosamples.csv", "CLO_ID", "CLO",
           [("OLS dictionary (exact)", "ols_exact"), ("OLS lexical (rank-1)", "ols_top")])
    public("Biosamples:CLO  (public, BioPortal file)", TOP / "baseline_bioportal_biosamples.csv", "CLO_ID", "CLO",
           [("BioPortal (1-answer)", "bio_pref")])
    public("Biosamples:CLO  (public, Cellosaurus file)", TOP / "baseline_cellosaurus_biosamples.csv", "CLO_ID", "CLO",
           [("Cellosaurus (1-answer)", "cello_exact")])

    # ---------- Internal: ChEBI ----------
    v2 = pd.read_csv(DATA / "Ground_Truth_CHEBI_v2.csv"); v2.columns = [c.strip() for c in v2.columns]
    v2 = v2.drop_duplicates("Cargo")
    gmap = {str(r["Cargo"]).strip(): canon(r["Cargo_CHEBI_id"], "CHEBI") for _, r in v2.iterrows()}
    terms = list(gmap.keys())              # keep unmappable (NIL) terms; gold is None for them
    gold = [gmap[t] for t in terms]
    acc = notes_accepted(GTV2 / "GT_CHEBI_annotatorA_filled.csv", "Cargo", "Cargo_CHEBI_id", "CHEBI", terms)
    nrm = load(DATA / "Ground_Truth_CHEBI_Ontology_Normalization.csv", "Cargo")
    bpf = load(SIB / "baseline_bioportal_GT_CHEBI.csv", "Cargo")
    olf = load(SIB / "baseline_ols_GT_CHEBI.csv", "Cargo")
    rag_ch = [single(nrm.loc[t, "rag_curie"], "CHEBI") if t in nrm.index else None for t in terms]
    preds = [
        ("Our full pipeline (rag)", rag_ch),
        ("Our exact-match",         [single(nrm.loc[t, "exact_curie"], "CHEBI") if t in nrm.index else None for t in terms]),
        ("BioPortal (1-answer)",    [single(bpf.loc[t, "bio_pref"], "CHEBI") if t in bpf.index else None for t in terms]),
        ("OLS lexical (rank-1)",    [single(olf.loc[t, "ols_top"], "CHEBI") if t in olf.index else None for t in terms]),
        ("OLS dictionary (exact)",  [single(olf.loc[t, "ols_exact"], "CHEBI") if t in olf.index else None for t in terms]),
    ]
    bench("Internal:ChEBI", gold, preds, accepted=acc)
    gold_ch, acc_ch, terms_ch, bpf_ch = gold, acc, terms, bpf

    # ---------- Internal: CLO ----------
    v2 = pd.read_csv(DATA / "Ground_Truth_CLO_v2.csv"); v2.columns = [c.strip() for c in v2.columns]
    v2 = v2.drop_duplicates("Cell Line")
    gmap = {str(r["Cell Line"]).strip(): canon(r["CLO_id"], "CLO") for _, r in v2.iterrows()}
    terms = list(gmap.keys())              # keep unmappable (NIL) terms; gold is None for them
    gold = [gmap[t] for t in terms]
    acc = notes_accepted(GTV2 / "GT_CLO_annotatorA_filled.csv", "Cell Line", "CLO_id", "CLO", terms)
    nrm = load(DATA / "Ground_Truth_CLO_Ontology_Normalization.csv", "Cell Line")
    bpf = load(SIB / "baseline_bioportal_GT_CLO.csv", "Cell Line")
    olf = load(SIB / "baseline_ols_GT_CLO.csv", "Cell Line")
    cef = load(SIB / "baseline_cellosaurus_GT_CLO.csv", "Cell Line")
    preds = [
        ("Our full pipeline (rag)", [single(nrm.loc[t, "rag_curie"], "CLO") if t in nrm.index else None for t in terms]),
        ("Our exact-match",         [single(nrm.loc[t, "exact_curie"], "CLO") if t in nrm.index else None for t in terms]),
        ("BioPortal (1-answer)",    [single(bpf.loc[t, "bio_pref"], "CLO") if t in bpf.index else None for t in terms]),
        ("Cellosaurus (1-answer)",  [single(cef.loc[t, "cello_exact"], "CLO") if t in cef.index else None for t in terms]),
        ("OLS lexical (rank-1)",    [single(olf.loc[t, "ols_top"], "CLO") if t in olf.index else None for t in terms]),
        ("OLS dictionary (exact)",  [single(olf.loc[t, "ols_exact"], "CLO") if t in olf.index else None for t in terms]),
    ]
    bench("Internal:CLO", gold, preds, accepted=acc)
    clo_gold, clo_acc, clo_bpf, clo_cef, clo_terms = gold, acc, bpf, cef, terms

    # ---- Guard the exact values reported in Table tab:baselines / the comparison text ----
    # NIL-aware: single accuracy credits a correct abstention on an unmappable term
    # (gold None) and charges a FP for any id emitted on one.
    def acc_of(gold, preds):
        return sum(p == g for g, p in zip(gold, preds)) / len(gold)
    def accepted_of(gold, accepted, preds):
        return sum((p is None and g is None) or (p is not None and p in accepted[i])
                   for i, (g, p) in enumerate(zip(gold, preds))) / len(gold)
    chk = []
    bp_ch = [single(bpf_ch.loc[t, "bio_pref"], "CHEBI") if t in bpf_ch.index else None for t in terms_ch]
    chk.append(("BioPortal single-answer Internal:ChEBI", acc_of(gold_ch, bp_ch), 0.32))
    chk.append(("BioPortal accepted Internal:ChEBI",      accepted_of(gold_ch, acc_ch, bp_ch), 0.36))
    chk.append(("Our pipeline accepted Internal:ChEBI",   accepted_of(gold_ch, acc_ch, rag_ch), 0.39))
    bp_clo = [single(clo_bpf.loc[t, "bio_pref"], "CLO") if t in clo_bpf.index else None for t in clo_terms]
    ce_clo = [single(clo_cef.loc[t, "cello_exact"], "CLO") if t in clo_terms and t in clo_cef.index else None for t in clo_terms]
    chk.append(("BioPortal single-answer Internal:CLO",   acc_of(clo_gold, bp_clo), 0.42))
    chk.append(("Cellosaurus single-answer Internal:CLO", acc_of(clo_gold, ce_clo), 0.46))
    print("\n=== NIL-aware regression guard (tol 0.01) ===")
    bad = 0
    for name, got, exp in chk:
        ok = abs(round(got, 2) - exp) <= 0.01
        bad += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got:.3f}, expected {exp}")
    if bad:
        raise SystemExit(f"{bad} assertion(s) FAILED")


if __name__ == "__main__":
    main()
