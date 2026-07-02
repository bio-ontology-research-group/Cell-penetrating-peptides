#!/usr/bin/env python
"""
Sequence-diversity figure for the CPP dataset (manuscript revision, comment R3.6).

Produces a charge-vs-length overview of the distinct CPP sequences, with marginal
length and net-charge distributions, plus an amino-acid composition panel that shows
the expected enrichment of cationic residues. Net charge is the standard CPP
approximation at physiological pH: (#Arg + #Lys) - (#Asp + #Glu). All numbers printed
to stdout are reproducible from the released preprocessed dataset.

Usage:
    python scripts/figure_sequence_diversity.py
"""
from __future__ import annotations
from pathlib import Path
import collections
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "data" / "Natural_CPP3_download_annotated_preprocessed.csv"
OUT = ROOT / "Cell-Penetrating-Peptides-CPPs---Database" / "Fig" / "sequence_diversity.png"
AA = list("ACDEFGHIKLMNPQRSTVWY")
POS, NEG = set("KR"), set("DE")


def net_charge(seq: str) -> int:
    c = collections.Counter(seq)
    return sum(c[a] for a in POS) - sum(c[a] for a in NEG)


def main():
    df = pd.read_csv(PREP, low_memory=False)
    seqs = (df["Sequence"].dropna().astype(str).str.strip().drop_duplicates())
    seqs = seqs[seqs.str.len() > 0]
    d = pd.DataFrame({"seq": seqs})
    d["length"] = d["seq"].str.len()
    d["charge"] = d["seq"].map(net_charge)

    n = len(d)
    cap = int(np.percentile(d["length"], 99))          # x-view cap; outliers reported
    beyond = int((d["length"] > cap).sum())
    print(f"distinct sequences: {n}")
    print(f"length  median={int(d.length.median())}  IQR={int(d.length.quantile(.25))}-"
          f"{int(d.length.quantile(.75))}  range={int(d.length.min())}-{int(d.length.max())}")
    print(f"charge  median={int(d.charge.median())}  IQR={int(d.charge.quantile(.25))}-"
          f"{int(d.charge.quantile(.75))}  range={int(d.charge.min())}-{int(d.charge.max())}")
    print(f"cationic (net charge > 0): {100*(d.charge>0).mean():.1f}%  | "
          f"anionic (<0): {100*(d.charge<0).mean():.1f}%  | "
          f"neutral (0): {100*(d.charge==0).mean():.1f}%")
    print(f"x-view capped at 99th pct length={cap}; {beyond} sequences beyond cap")

    # ---- figure ----
    fig = plt.figure(figsize=(10, 4.6))
    gs = GridSpec(2, 3, width_ratios=[3, 1.1, 2.2], height_ratios=[1, 4],
                  wspace=0.08, hspace=0.08, figure=fig)
    ax = fig.add_subplot(gs[1, 0])
    axtop = fig.add_subplot(gs[0, 0], sharex=ax)
    axright = fig.add_subplot(gs[1, 1], sharey=ax)
    axcomp = fig.add_subplot(gs[:, 2])

    view = d[d["length"] <= cap]
    hb = ax.hexbin(view["length"], view["charge"], gridsize=28, cmap="viridis",
                   mincnt=1, bins="log")
    ax.set_xlabel("Sequence length (residues)")
    ax.set_ylabel("Net charge  (R+K) − (D+E)")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    cb = fig.colorbar(hb, ax=axright, fraction=0.5, pad=0.05)
    cb.set_label("sequences (log)")

    axtop.hist(view["length"], bins=range(0, cap + 2), color="#4C72B0")
    axtop.set_ylabel("count")
    axtop.tick_params(labelbottom=False)
    axtop.set_title(f"CPP sequence diversity (n={n} distinct sequences)", loc="left")

    axright.hist(view["charge"], bins=range(int(view.charge.min()), int(view.charge.max()) + 2),
                 orientation="horizontal", color="#4C72B0")
    axright.set_xlabel("count")
    axright.tick_params(labelleft=False)

    comp = pd.Series({a: d["seq"].str.count(a).sum() for a in AA})
    comp = comp / comp.sum() * 100
    comp = comp.sort_values(ascending=True)
    colors = ["#C44E52" if a in POS else "#55A868" if a in NEG else "#BBBBBB"
              for a in comp.index]
    axcomp.barh(comp.index, comp.values, color=colors)
    axcomp.set_xlabel("amino-acid composition (%)")
    axcomp.set_title("Residue composition", loc="left")
    axcomp.tick_params(labelsize=8)
    # legend
    from matplotlib.patches import Patch
    axcomp.legend(handles=[Patch(color="#C44E52", label="cationic (K,R)"),
                           Patch(color="#55A868", label="anionic (D,E)")],
                  fontsize=7, loc="lower right", frameon=False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
