#!/usr/bin/env python
"""
BioPortal Annotator baseline for ontology normalization (revision, R3.5).

The BioPortal Annotator (the established tool the reviewer names) performs concept
recognition against a chosen ontology, natively returning ChEBI / CLO classes (no
cross-walk). For each distinct surface form we annotate it restricted to the target
ontology and record two variants:

  * bio_pref - classes matched on a PREFerred label (strict).
  * bio_all  - all matched classes, PREF or SYN (lenient).
A prediction is the SET of returned class ids; scored correct if the gold id is among
them (the Annotator may return several candidate classes for one surface form).

Reads the API key from the environment variable BIOPORTAL_API_KEY (do not hard-code it).
Results are cached per (ontology, surface form) and the run is resumable.

Usage:
    BIOPORTAL_API_KEY=... python scripts/baseline_bioportal.py
    BIOPORTAL_API_KEY=... python scripts/baseline_bioportal.py --benchmark CRAFT
"""
from __future__ import annotations
import argparse, json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "baselines"
CACHE = OUTDIR / "bioportal_cache.json"
API = "https://data.bioontology.org/annotator"

BENCH = {
    "CRAFT":      dict(csv="data/CRAFT.csv", term="entity_text", gold="gold_id", onto="CHEBI"),
    "biosamples": dict(csv="data/biosamples.csv", term="Cell Line", gold="CLO_ID", onto="CLO"),
    # Internal 300-term ground truth (R3.4 single-annotator interim; R1.2/R3.5).
    "GT_CHEBI":   dict(csv="data/Ground_Truth_CHEBI_v2.csv", term="Cargo", gold="Cargo_CHEBI_id", onto="CHEBI"),
    "GT_CLO":     dict(csv="data/Ground_Truth_CLO_v2.csv", term="Cell Line", gold="CLO_id", onto="CLO"),
}


def canon(s, prefix):
    m = re.search(rf"{prefix}[_:]?(\d+)", str(s), flags=re.I)
    return f"{prefix}:{m.group(1)}" if m else None


def annotate(term, onto, key, session, retries=3):
    """Return {'pref': [ids], 'all': [ids]} for the target ontology."""
    params = {"text": term, "ontologies": onto, "apikey": key,
              "longest_only": "true", "whole_word_only": "false"}
    for attempt in range(retries):
        try:
            r = session.get(API, params=params, timeout=30)
            r.raise_for_status()
            anns = r.json()
            break
        except Exception:
            if attempt == retries - 1:
                return {"pref": [], "all": []}
            time.sleep(2.0 * (attempt + 1))
    allids, prefids = [], []
    for a in anns:
        cid = canon(a.get("annotatedClass", {}).get("@id", ""), onto)
        if not cid:
            continue
        allids.append(cid)
        if any(an.get("matchType") == "PREF" for an in a.get("annotations", [])):
            prefids.append(cid)
    return {"pref": sorted(set(prefids)), "all": sorted(set(allids))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=list(BENCH))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    key = os.environ.get("BIOPORTAL_API_KEY")
    if not key:
        raise SystemExit("Set BIOPORTAL_API_KEY in the environment.")
    benches = [args.benchmark] if args.benchmark else list(BENCH)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    lock = threading.Lock()
    session = requests.Session()

    for name in benches:
        cfg = BENCH[name]
        df = pd.read_csv(ROOT / cfg["csv"])
        terms = df[cfg["term"]].dropna().astype(str).str.strip()
        uniq = sorted(set(terms[terms != ""]))
        todo = [t for t in uniq if f"{cfg['onto']}||{t}" not in cache]
        print(f"[{name}] rows={len(df)} unique={len(uniq)} to_query={len(todo)} "
              f"onto={cfg['onto']}", flush=True)

        done = [0]
        def work(t, cfg=cfg):
            res = annotate(t, cfg["onto"], key, session)
            with lock:
                cache[f"{cfg['onto']}||{t}"] = res
                done[0] += 1
                if done[0] % 100 == 0:
                    CACHE.write_text(json.dumps(cache))
                    print(f"  [{name}] {done[0]}/{len(todo)}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))
        CACHE.write_text(json.dumps(cache))

        rows = []
        for _, r in df.iterrows():
            t = str(r[cfg["term"]]).strip()
            c = cache.get(f"{cfg['onto']}||{t}", {"pref": [], "all": []})
            rows.append({cfg["term"]: t, cfg["gold"]: r[cfg["gold"]],
                         "bio_pref": ";".join(c.get("pref", [])),
                         "bio_all": ";".join(c.get("all", []))})
        out = OUTDIR / f"baseline_bioportal_{name}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  -> wrote {out.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
