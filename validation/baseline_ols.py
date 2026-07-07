#!/usr/bin/env python
"""
Lexical/dictionary baseline for ontology normalization via the EBI Ontology Lookup
Service (OLS4), for comparison against our pipeline (manuscript revision, R3.5 / R1.2).

OLS requires no API key. For each distinct surface form in a benchmark we query OLS
(restricted to the target ontology) and record two baseline predictions:

  * ols_exact   - accept the top hit only if its label or one of its synonyms matches
                  the query exactly (case-insensitive); otherwise NIL. This is a strict
                  dictionary-lookup baseline.
  * ols_top     - the rank-1 search hit regardless of exactness (lexical-search baseline).

Results are cached per (ontology, surface form) so the run is resumable and re-running
is free. Predictions are written per benchmark for scoring by evaluate_baselines.py.

Usage:
    python scripts/baseline_ols.py                 # both benchmarks
    python scripts/baseline_ols.py --benchmark CRAFT
"""
from __future__ import annotations
import argparse, json, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "revision" / "baselines"
CACHE = OUTDIR / "ols_cache.json"
OLS = "https://www.ebi.ac.uk/ols4/api/search"

BENCH = {
    "CRAFT":      dict(csv="data/CRAFT.csv", term="entity_text", gold="gold_id", onto="chebi"),
    "biosamples": dict(csv="data/biosamples.csv", term="Cell Line", gold="CLO_ID", onto="clo"),
    # Internal 300-term ground truth (R3.4 single-annotator interim; R1.2/R3.5).
    "GT_CHEBI":   dict(csv="data/Ground_Truth_CHEBI_v2.csv", term="Cargo", gold="Cargo_CHEBI_id", onto="chebi"),
    "GT_CLO":     dict(csv="data/Ground_Truth_CLO_v2.csv", term="Cell Line", gold="CLO_id", onto="clo"),
}


def load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def save_cache(cache: dict):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=0))


def query_ols(term: str, onto: str, session: requests.Session, retries=3) -> dict:
    """Return {'top': obo_id|None, 'top_label': str, 'exact': obo_id|None}."""
    params = {"q": term, "ontology": onto, "rows": 10}
    for attempt in range(retries):
        try:
            r = session.get(OLS, params=params, timeout=20)
            r.raise_for_status()
            docs = r.json().get("response", {}).get("docs", [])
            break
        except Exception:
            if attempt == retries - 1:
                return {"top": None, "top_label": "", "exact": None, "error": True}
            time.sleep(1.5 * (attempt + 1))
    if not docs:
        return {"top": None, "top_label": "", "exact": None}
    top = docs[0]
    q = term.strip().lower()
    exact = None
    for d in docs:
        labels = [d.get("label", "")] + list(d.get("synonym", []) or [])
        if any((l or "").strip().lower() == q for l in labels):
            exact = d.get("obo_id")
            break
    return {"top": top.get("obo_id"), "top_label": top.get("label", ""), "exact": exact}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=list(BENCH), help="default: all")
    ap.add_argument("--workers", type=int, default=10, help="concurrent OLS requests")
    args = ap.parse_args()
    benches = [args.benchmark] if args.benchmark else list(BENCH)

    cache = load_cache()
    lock = threading.Lock()
    session = requests.Session()
    session.headers.update({"User-Agent": "CPP-KG-baseline/1.0 (research; OLS4)"})

    for name in benches:
        cfg = BENCH[name]
        df = pd.read_csv(ROOT / cfg["csv"])
        terms = df[cfg["term"]].dropna().astype(str).str.strip()
        uniq = sorted(set(terms[terms != ""]))
        todo = [t for t in uniq if f"{cfg['onto']}||{t}" not in cache]
        print(f"[{name}] rows={len(df)} unique={len(uniq)} to_query={len(todo)} "
              f"onto={cfg['onto']} workers={args.workers}", flush=True)

        done = [0]
        def work(t, cfg=cfg):
            res = query_ols(t, cfg["onto"], session)
            with lock:
                cache[f"{cfg['onto']}||{t}"] = res
                done[0] += 1
                if done[0] % 100 == 0:
                    save_cache(cache)
                    print(f"  [{name}] {done[0]}/{len(todo)} queried", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(work, todo))
        save_cache(cache)

        # write per-row predictions
        rows = []
        for _, r in df.iterrows():
            t = str(r[cfg["term"]]).strip()
            c = cache.get(f"{cfg['onto']}||{t}", {})
            rows.append({cfg["term"]: t, cfg["gold"]: r[cfg["gold"]],
                         "ols_exact": c.get("exact"), "ols_top": c.get("top")})
        out = OUTDIR / f"baseline_ols_{name}.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  -> wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
