import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import os

# Load data
data_path = 'data/Natural_CPP3_download_annotated_preprocessed_Ontology_Normalization.csv'
df = pd.read_csv(data_path)

fig_dir = '/home/leechuck/Documents/papers/cpp-database/Fig'
os.makedirs(fig_dir, exist_ok=True)

# ============================================================
# 1. Main Uptake Mechanism Venn (2-circle)
# ============================================================
print("=== Main Uptake Mechanism ===")
print(df['Main Uptake Mechanism'].value_counts(dropna=False))
print()

col = 'Main Uptake Mechanism'
endocytosis_set = set()
direct_set = set()

for idx, val in df[col].dropna().items():
    categories = [c.strip() for c in str(val).split(',')]
    for cat in categories:
        if 'Endocytosis' in cat or 'endocytosis' in cat:
            endocytosis_set.add(idx)
        if 'Direct' in cat or 'direct' in cat:
            direct_set.add(idx)

only_endo = len(endocytosis_set - direct_set)
only_direct = len(direct_set - endocytosis_set)
both = len(endocytosis_set & direct_set)

print(f"Endocytosis only: {only_endo}")
print(f"Direct penetration only: {only_direct}")
print(f"Both: {both}")
print()

fig, ax = plt.subplots(figsize=(6, 5))
v = venn2(
    subsets=(only_direct, only_endo, both),
    set_labels=('Direct\npenetration', 'Endocytosis'),
    ax=ax
)
# Pastel colors
if v.get_patch_by_id('10'):
    v.get_patch_by_id('10').set_color('#FFB3BA')
    v.get_patch_by_id('10').set_alpha(0.7)
if v.get_patch_by_id('01'):
    v.get_patch_by_id('01').set_color('#BAE1FF')
    v.get_patch_by_id('01').set_alpha(0.7)
if v.get_patch_by_id('11'):
    v.get_patch_by_id('11').set_color('#D5BAFF')
    v.get_patch_by_id('11').set_alpha(0.7)

plt.title('Main Uptake Mechanism', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'Main_Uptake_Mechanism_venn.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved Main_Uptake_Mechanism_venn.png")

# ============================================================
# 2. Subcategory Uptake Mechanism Venn (3-circle, top 3)
# ============================================================
print("\n=== Subcategory Uptake Mechanism ===")
print(df['Subcategory Uptake Mechanism'].value_counts(dropna=False))
print()

col2 = 'Subcategory Uptake Mechanism'

macro_set = set()
clathrin_set = set()
caveolae_set = set()

for idx, val in df[col2].dropna().items():
    categories = [c.strip() for c in str(val).split(',')]
    for cat in categories:
        if 'Macropinocytosis' in cat or 'macropinocytosis' in cat:
            macro_set.add(idx)
        if 'Clathrin' in cat or 'clathrin' in cat:
            clathrin_set.add(idx)
        if 'Caveolae' in cat or 'caveolae' in cat:
            caveolae_set.add(idx)

# Compute all 7 regions for venn3
only_macro = len(macro_set - clathrin_set - caveolae_set)
only_clathrin = len(clathrin_set - macro_set - caveolae_set)
only_caveolae = len(caveolae_set - macro_set - clathrin_set)
macro_clathrin = len((macro_set & clathrin_set) - caveolae_set)
macro_caveolae = len((macro_set & caveolae_set) - clathrin_set)
clathrin_caveolae = len((clathrin_set & caveolae_set) - macro_set)
all_three = len(macro_set & clathrin_set & caveolae_set)

print(f"Macropinocytosis only: {only_macro}")
print(f"Clathrin-mediated only: {only_clathrin}")
print(f"Caveolae-mediated only: {only_caveolae}")
print(f"Macro & Clathrin: {macro_clathrin}")
print(f"Macro & Caveolae: {macro_caveolae}")
print(f"Clathrin & Caveolae: {clathrin_caveolae}")
print(f"All three: {all_three}")
print()

fig, ax = plt.subplots(figsize=(7, 6))
v3 = venn3(
    subsets=(only_macro, only_clathrin, macro_clathrin,
             only_caveolae, macro_caveolae, clathrin_caveolae, all_three),
    set_labels=('Macropinocytosis', 'Clathrin-mediated\nendocytosis', 'Caveolae-mediated\nendocytosis'),
    ax=ax
)
# Pastel colors for 3-circle
colors = {
    '100': '#FFB3BA', '010': '#BAE1FF', '001': '#BAFFC9',
    '110': '#D5BAFF', '101': '#FFE4BA', '011': '#BAFFF5',
    '111': '#F0F0F0'
}
for region_id, color in colors.items():
    patch = v3.get_patch_by_id(region_id)
    if patch:
        patch.set_color(color)
        patch.set_alpha(0.7)

plt.title('Subcategory Uptake Mechanism', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'Subcategory_Uptake_Mechanism_venn.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved Subcategory_Uptake_Mechanism_venn.png")

# ============================================================
# 3. Subcellular Localization Category Venn (3-circle)
# ============================================================
print("\n=== Subcellular Localization Category ===")
print(df['Subcellular Localization Category'].value_counts(dropna=False))
print()

col3 = 'Subcellular Localization Category'

cytoplasm_set = set()
nucleus_set = set()
endosomes_set = set()

for idx, val in df[col3].dropna().items():
    categories = [c.strip() for c in str(val).split(',')]
    for cat in categories:
        if 'Cytoplasm' in cat or 'cytoplasm' in cat:
            cytoplasm_set.add(idx)
        if 'Nucleus' in cat or 'nucleus' in cat:
            nucleus_set.add(idx)
        if 'Endosomes' in cat or 'endosomes' in cat or 'Endosome' in cat:
            endosomes_set.add(idx)

# Compute all 7 regions for venn3
only_cyto = len(cytoplasm_set - nucleus_set - endosomes_set)
only_nuc = len(nucleus_set - cytoplasm_set - endosomes_set)
only_endo = len(endosomes_set - cytoplasm_set - nucleus_set)
cyto_nuc = len((cytoplasm_set & nucleus_set) - endosomes_set)
cyto_endo = len((cytoplasm_set & endosomes_set) - nucleus_set)
nuc_endo = len((nucleus_set & endosomes_set) - cytoplasm_set)
all_three_loc = len(cytoplasm_set & nucleus_set & endosomes_set)

print(f"Cytoplasm only: {only_cyto}")
print(f"Nucleus only: {only_nuc}")
print(f"Endosomes only: {only_endo}")
print(f"Cytoplasm & Nucleus: {cyto_nuc}")
print(f"Cytoplasm & Endosomes: {cyto_endo}")
print(f"Nucleus & Endosomes: {nuc_endo}")
print(f"All three: {all_three_loc}")
print()

fig, ax = plt.subplots(figsize=(7, 6))
v3_loc = venn3(
    subsets=(only_cyto, only_nuc, cyto_nuc,
             only_endo, cyto_endo, nuc_endo, all_three_loc),
    set_labels=('Cytoplasm', 'Nucleus', 'Endosomes'),
    ax=ax
)
# Pastel colors
colors_loc = {
    '100': '#FFB3BA', '010': '#BAE1FF', '001': '#BAFFC9',
    '110': '#D5BAFF', '101': '#FFE4BA', '011': '#BAFFF5',
    '111': '#F0F0F0'
}
for region_id, color in colors_loc.items():
    patch = v3_loc.get_patch_by_id(region_id)
    if patch:
        patch.set_color(color)
        patch.set_alpha(0.7)

plt.title('Subcellular Localization Category', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'Subcellular_Localization_Category_venn.png'), dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved Subcellular_Localization_Category_venn.png")

print("\nAll Venn diagrams generated successfully!")