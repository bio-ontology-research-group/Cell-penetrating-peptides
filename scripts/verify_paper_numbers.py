#!/usr/bin/env python3
"""
verify_paper_numbers.py
=======================
Reproduce and verify every quantitative claim made in the paper.
Run from the repository root:

    python scripts/verify_paper_numbers.py

All assertions compare computed values against the numbers stated in
the manuscript.  If any assertion fails the script exits with code 1.
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

DATA = Path("data")

RAW_CSV = DATA / "Natural_CPP3_download_annotated.csv"
PREPROCESSED_CSV = DATA / "Natural_CPP3_download_annotated_preprocessed.csv"
NORMALIZED_CSV = DATA / "Natural_CPP3_download_annotated_preprocessed_Ontology_Normalization.csv"

ok = True


def check(name: str, actual, expected, tol=0.0):
    global ok
    if isinstance(expected, float):
        match = abs(actual - expected) <= tol
    else:
        match = actual == expected
    status = "PASS" if match else "FAIL"
    if not match:
        ok = False
    print(f"  [{status}] {name}: got {actual}, expected {expected}")


print("=" * 70)
print("Paper number verification")
print("=" * 70)

# --- Raw data ---
df_raw = pd.read_csv(RAW_CSV)
print("\n--- Data Collection (Methods) ---")
check("Downloaded sequences", len(df_raw), 5288)
check("Unique raw sequences", df_raw["Sequence"].nunique(), 3048)

# --- Preprocessed data ---
df = pd.read_csv(PREPROCESSED_CSV)
print("\n--- After Preprocessing (Methods / Filtration) ---")
check("Final dataset entries", len(df), 10799)
check("Distinct sequences", df["Sequence"].nunique(), 2708)

# --- Ontology-normalised data ---
df3 = pd.read_csv(NORMALIZED_CSV)
print("\n--- After Ontology Normalization ---")
check("Normalized entries", len(df3), 10799)

# --- Missingness (Table: Annotation Integrity Summary) ---
print("\n--- Missingness Table ---")
miss = {
    "Main Uptake Mechanism": (7699, 71.29),
    "Subcategory Uptake Mechanism": (9902, 91.69),
    "Subcellular Localization Category": (5642, 52.25),
    "Cargo Type": (205, 1.90),
    "Cell Line": (174, 1.61),
}
for col, (exp_count, exp_pct) in miss.items():
    m = int(df3[col].isna().sum())
    pct = round(m / len(df3) * 100, 2)
    check(f"{col} missing count", m, exp_count)
    check(f"{col} missing %", pct, exp_pct, tol=0.01)

# --- Main Uptake Mechanism distribution ---
print("\n--- Uptake Mechanism Distribution ---")
mech = df3["Main Uptake Mechanism"].dropna()
cats_mech = Counter()
for v in mech:
    for p in str(v).split(", "):
        cats_mech[p.strip()] += 1
total_mech = sum(cats_mech.values())
endo = cats_mech["Endocytosis"]
direct = cats_mech["Direct penetration"]
check("Endocytosis %", round(endo / total_mech * 100, 2), 76.14, tol=0.01)
check("Direct penetration %", round(direct / total_mech * 100, 2), 23.86, tol=0.01)

# --- Subcategory Uptake Mechanism ---
print("\n--- Subcategory Uptake Mechanism ---")
sub = df3["Subcategory Uptake Mechanism"].dropna()
cats_sub = Counter()
for v in sub:
    for p in str(v).split(", "):
        cats_sub[p.strip()] += 1
total_sub = sum(cats_sub.values())
expected_sub = {
    "Macropinocytosis": 42.19,
    "Clathrin-mediated endocytosis": 37.62,
    "Caveolae-mediated endocytosis": 12.57,
    "Clathrin and caveolae independent": 5.90,
    "Phagocytosis": 1.71,
}
for name, exp_pct in expected_sub.items():
    actual_pct = round(cats_sub[name] / total_sub * 100, 2)
    check(f"{name} %", actual_pct, exp_pct, tol=0.01)

# --- Subcellular Localization ---
print("\n--- Subcellular Localization ---")
loc = df3["Subcellular Localization Category"].dropna()
cats_loc = Counter()
for v in loc:
    for p in str(v).split(", "):
        cats_loc[p.strip()] += 1
total_loc = sum(cats_loc.values())
expected_loc = {
    "Cytoplasm": 60.52,
    "Nucleus": 28.68,
    "Vesicles": 4.45,
    "Mitochondria": 3.29,
    "Endosomes": 3.06,
}
for name, exp_pct in expected_loc.items():
    actual_pct = round(cats_loc[name] / total_loc * 100, 2)
    check(f"{name} %", actual_pct, exp_pct, tol=0.01)

print("\n" + "=" * 70)
if ok:
    print("ALL CHECKS PASSED")
else:
    print("SOME CHECKS FAILED")
    sys.exit(1)
