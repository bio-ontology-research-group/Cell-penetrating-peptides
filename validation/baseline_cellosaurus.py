#!/usr/bin/env python
"""
Cellosaurus dictionary baseline for CLO cell-line normalization (revision, R3.5).

Cellosaurus is the reference cell-line knowledge resource; every cell line carries an
extensive synonym list and direct cross-references to the Cell Line Ontology (CLO). We
build a name -> CLO dictionary from the Cellosaurus flat-file release and use it as a
strong, domain-specific dictionary baseline that predicts in CLO's own identifier space
(no lossy cross-walk). This is the established-tool comparison a practitioner would use
for cell lines, and it avoids the ranking artefacts of the live search API.

Variants (mirroring the OLS baselines):
  * cello_exact   - exact (normalized) match of the surface form to a Cellosaurus
                    name or synonym.
  * cello_relaxed - additionally strips a trailing "cell(s)"/"cell line" before matching
                    (the same nomenclature tolerance our pipeline's exact stage applies).
A prediction is the SET of CLO ids attached to the matched cell line(s); it is scored
correct if the gold id is among them (a cell line may map to several CLO classes).

Requires the Cellosaurus release at data/intermediate/cellosaurus.txt
(https://ftp.expasy.org/databases/cellosaurus/cellosaurus.txt).

Usage:
    python scripts/baseline_cellosaurus.py
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CELLO = ROOT / "data" / "intermediate" / "cellosaurus.txt"
OUTDIR = ROOT / "data" / "baselines"

# Benchmarks scored against the Cellosaurus name->CLO dictionary (CLO only).
BENCH = {
    "biosamples": dict(csv="data/biosamples.csv", term="Cell Line", gold="CLO_ID",
                       out="baseline_cellosaurus_biosamples.csv"),
    "GT_CLO":     dict(csv="data/Ground_Truth_CLO_v2.csv", term="Cell Line", gold="CLO_id",
                       out="baseline_cellosaurus_GT_CLO.csv"),
}

_SUFFIX = re.compile(r"\s+(cell lines?|cells?)\.?$", flags=re.I)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def relax(s: str) -> str:
    return norm(_SUFFIX.sub("", norm(s)))


def canon_clo(s):
    m = re.search(r"CLO[_:]?(\d+)", str(s), flags=re.I)
    return f"CLO:{m.group(1)}" if m else None


def build_dictionary(path: Path):
    """name(normalized) -> set(CLO ids), built from ID/SY names and DR CLO xrefs."""
    name2clo: dict[str, set] = {}
    names, clo = [], set()

    def flush():
        if clo:
            for nm in names:
                name2clo.setdefault(norm(nm), set()).update(clo)

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("//"):
                flush()
                names, clo = [], set()
            elif line.startswith("ID   "):
                names.append(line[5:].strip())
            elif line.startswith("SY   "):
                names.extend(p.strip() for p in line[5:].strip().split(";") if p.strip())
            elif line.startswith("DR   CLO;"):
                c = canon_clo(line.split(";", 1)[1])
                if c:
                    clo.add(c)
        flush()
    return name2clo


def score(gold, preds):
    n = len(gold)
    made = sum(1 for p in preds if p)
    correct = sum(1 for g, p in zip(gold, preds) if p and g in p)
    return dict(N=n, predicted=made, correct=correct,
                accuracy=round(correct / n, 3) if n else 0.0,
                precision=round(correct / made, 3) if made else float("nan"),
                coverage=round(made / n, 3) if n else 0.0)


def main():
    if not CELLO.exists():
        raise SystemExit(f"Missing {CELLO.relative_to(ROOT)} - download cellosaurus.txt first.")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    d = build_dictionary(CELLO)
    # relaxed index: strip suffix from each key too
    d_relaxed: dict[str, set] = {}
    for k, v in d.items():
        d_relaxed.setdefault(relax(k), set()).update(v)
    print(f"Cellosaurus dictionary: {len(d)} names -> CLO ({sum(len(v) for v in d.values())} pairs)")

    for name, cfg in BENCH.items():
        csv = ROOT / cfg["csv"]
        if not csv.exists():
            print(f"[skip] {name}: {cfg['csv']} not found")
            continue
        df = pd.read_csv(csv)
        gold = [canon_clo(x) for x in df[cfg["gold"]]]
        exact, relaxed = [], []
        for t in df[cfg["term"]].astype(str):
            exact.append(set(d.get(norm(t), set())))
            relaxed.append(set(d_relaxed.get(relax(t), set())))

        out = OUTDIR / cfg["out"]
        pd.DataFrame({cfg["term"]: df[cfg["term"]], cfg["gold"]: df[cfg["gold"]],
                      "cello_exact": [";".join(sorted(s)) for s in exact],
                      "cello_relaxed": [";".join(sorted(s)) for s in relaxed]}).to_csv(out, index=False)
        print(f"\n[{name}] -> wrote {out.relative_to(ROOT)}")
        for label, sets in [("cello_exact", exact), ("cello_relaxed", relaxed)]:
            r = score(gold, sets)
            print(f"  {label:<14} acc={r['accuracy']:.3f} prec={r['precision']:.3f} "
                  f"cov={r['coverage']:.3f} (correct={r['correct']}/{r['N']}, pred={r['predicted']})")


if __name__ == "__main__":
    main()
