"""
Evaluate normalisation results (TP / FP / FN) for every CSV in results/.

Definition used (per entity / row):
  TP  – a prediction was returned AND it matches the ground-truth identifier
  FP  – a prediction was returned BUT it differs from the ground-truth identifier
  FN  – no prediction was returned (empty / null)

For *_OUR.csv files the predicted column may contain several ids separated by
";"; only the FIRST one is used for evaluation.

For *_exactmatch_others.csv files two methods are evaluated independently:
"exact" (exact_curie) and "biosyn" (biosyn_curie).

ID normalisation: both CLO_XXXXXXX and CLO:XXXXXXX are converted to CLO:XXXXXXX
(same for CHEBI).
"""

import re
import os
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_id(raw):
    """Return a canonical ID string (CLO:XXXXXXX / CHEBI:XXXXXXX) or None."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if not s:
        return None
    # CLO_0001008  →  CLO:0001008
    # CHEBI_52661  →  CHEBI:52661
    s = re.sub(r'^(CLO|CHEBI)_(\d+)$', r'\1:\2', s)
    return s if s else None


def get_first(raw):
    """Return the first ID from a semicolon-separated list."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    first = str(raw).split(";")[0].strip()
    return first if first else None


# ---------------------------------------------------------------------------
# Per-file configuration
# ---------------------------------------------------------------------------
# Each entry:
#   ground_truth_col  – column holding the curated/gold ID
#   predicted         – list of dicts, each with:
#                         col        – column name for the prediction
#                         multi      – True if values are ";"-separated (take first)
#                         label      – name used in the report
# ---------------------------------------------------------------------------

FILE_CONFIG = {
    "Gold_standard_CLO_OUR.csv": {
        "ground_truth_col": "CLO_ID",
        "predicted": [
            {"col": "clo_ids", "multi": True, "label": "Gold_standard_CLO_OUR"},
        ],
    },
    "Gold_standard_CLO_exactmatch_others.csv": {
        "ground_truth_col": "CLO_ID",
        "predicted": [
            {"col": "exact_curie",  "multi": False, "label": "Gold_standard_CLO_exact"},
            {"col": "biosyn_curie", "multi": False, "label": "Gold_standard_CLO_biosyn"},
            {"col": "rag_curie", "multi": False, "label": "Gold_standard_CLO_rag"},
        ],
    },
    
    "Ground_Truth_CHEBI_OUR.csv": {
        "ground_truth_col": "Cargo_CHEBI_id",
        "predicted": [
            {"col": "chebi_ids", "multi": True, "label": "Ground_Truth_CHEBI_OUR"},
        ],
    },
    
    "Ground_Truth_CHEBI_exactmatch_others.csv": {
        "ground_truth_col": "Cargo_CHEBI_id",
        "predicted": [
            {"col": "exact_curie",  "multi": False, "label": "Ground_Truth_CHEBI_exact"},
            {"col": "biosyn_curie", "multi": False, "label": "Ground_Truth_CHEBI_biosyn"},
            {"col": "rag_curie", "multi": False, "label": "Ground_Truth_CHEBI_rag"},
        ],
    },
    "Ground_Truth_CLO_exactmatch_others.csv": {
        "ground_truth_col": "CLO_id",
        "predicted": [
            {"col": "exact_curie",  "multi": False, "label": "Ground_Truth_CLO_exact"},
            {"col": "biosyn_curie", "multi": False, "label": "Ground_Truth_CLO_biosyn"},
            {"col": "rag_curie", "multi": False, "label": "Ground_Truth_CLO_rag"},
        ],
    },
    "Ground_Truth_CLO_OURS.csv": {
        "ground_truth_col": "CLO_id",
        "predicted": [
            {"col": "clo_ids", "multi": True, "label": "Ground_Truth_CLO_OUR"},
        ],
    },
    
    "gold_standard_craft_chebi_OUR.csv": {
        "ground_truth_col": "gold_id",
        "predicted": [
            {"col": "chebi_ids", "multi": True, "label": "gold_standard_craft_chebi_OUR"},
        ],
    },
    "gold_standard_craft_chebi_exactmatch_others.csv": {
        "ground_truth_col": "gold_id",
        "predicted": [
            {"col": "exact_curie",  "multi": False, "label": "gold_standard_craft_chebi_exact"},
            {"col": "biosyn_curie", "multi": False, "label": "gold_standard_craft_chebi_biosyn"},
            {"col": "rag_curie", "multi": False, "label": "gold_standard_craft_chebi_rag"},
        ],
    },
    
}

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def compute_metrics(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {"TP": tp, "FP": fp, "FN": fn,
            "Precision": round(precision, 4),
            "Recall":    round(recall, 4),
            "F1":        round(f1, 4)}


def evaluate_column(df: pd.DataFrame, gt_col: str, pred_col: str,
                    multi: bool) -> dict:
    tp = fp = fn = 0
    for _, row in df.iterrows():
        gold = normalize_id(row[gt_col])
        raw  = row[pred_col] if pred_col in df.columns else None
        pred = normalize_id(get_first(raw) if multi else raw)

        if pred is None:
            fn += 1
        elif pred == gold:
            tp += 1
        else:
            fp += 1

    return compute_metrics(tp, fp, fn)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    records = []

    for filename, cfg in FILE_CONFIG.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.isfile(filepath):
            print(f"[SKIP] {filename} — file not found")
            continue

        df = pd.read_csv(filepath)
        gt_col = cfg["ground_truth_col"]

        if gt_col not in df.columns:
            print(f"[SKIP] {filename} — ground-truth column '{gt_col}' not found")
            continue

        for pred_cfg in cfg["predicted"]:
            label    = pred_cfg["label"]
            pred_col = pred_cfg["col"]
            multi    = pred_cfg["multi"]

            if pred_col not in df.columns:
                print(f"[WARN] {filename} / '{pred_col}' column not found — skipping")
                continue

            metrics = evaluate_column(df, gt_col, pred_col, multi)
            total   = len(df)

            row = {
                "File":      filename,
                "System":    label,
                "N_total":   total,
                **metrics,
            }
            records.append(row)

            print(
                f"{label:<50}  "
                f"N={total:4d}  TP={metrics['TP']:4d}  FP={metrics['FP']:4d}  "
                f"FN={metrics['FN']:4d}  "
                f"P={metrics['Precision']:.4f}  R={metrics['Recall']:.4f}  "
                f"F1={metrics['F1']:.4f}"
            )

    # Save report
    report_path = os.path.join(RESULTS_DIR, "evaluation_report.csv")
    report_df = pd.DataFrame(records)
    report_df.to_csv(report_path, index=False)
    print(f"\nReport saved → {report_path}")


if __name__ == "__main__":
    main()
