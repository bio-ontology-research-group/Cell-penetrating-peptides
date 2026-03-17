from cProfile import label
import pandas as pd
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import requests
import difflib
from urllib.parse import quote
from collections import Counter
import matplotlib.pyplot as plt
#from matplotlib_venn import venn2, venn3
#import seaborn as sns
from typing import Dict, List
import torch
from transformers import AutoModel, AutoTokenizer
import unicodedata

class SequencePreprocessor:
    """
    Preprocessor for biological sequence datasets with annotations.
    Handles sequence validation, annotation normalization, and filtering.
    """
    
    # Standard 20 amino acids
    STANDARD_AA: Set[str] = set('ACDEFGHIKLMNPQRSTVWY')
    
    # Required annotation columns
    REQUIRED_ANNOTATIONS: List[str] = [
        'Cargo Type',
        'Subcellular Localization',
        'Uptake Mechanism'
    ]
    
    PASTEL_COLORS = [
        "#FFB3BA",  # Pastel Red
        "#FFDFBA",  # Pastel Orange
        "#FFFFBA",  # Pastel Yellow
        "#A7F7B8",  # Pastel Green
        "#BAE1FF",  # Pastel Blue
        "#E0BBE4",  # Pastel Purple
        "#FFC8DD",  # Pastel Pink
        "#D4A5A5",  # Pastel Brown
        "#A8E6CF",  # Pastel Mint
        "#FFD3B6",  # Pastel Peach
        "#BADCA9",  # Pastel Lavender
        "#B5EAD7",  # Pastel Teal
    ]

    def __init__(self, df: pd.DataFrame, sequence_col: str = 'Sequence'):
        """
        Initialize preprocessor with dataset.
        
        Args:
            df: Input DataFrame
            sequence_col: Name of the column containing sequences
        """
        self.df = df.copy()
        self.sequence_col = sequence_col
        self.stats = {
            'initial_count': len(df),
            'cleaned_sequences': 0,
            'invalid_sequences': 0,
            'missing_annotations': 0,
            'rows_after_split': 0,
            'final_count': 0
        }
        
    def clean_cargo_text(self, text):
        # This regex looks for:
        # \s* -> zero or more spaces
        # \(   -> an opening parenthesis
        # .*?  -> any content inside (non-greedy)
        # \)   -> a closing parenthesis
        return re.sub(r'\s*\(.*?\)', '', text)

    def normalize_text(self, s: str) -> str:
        s = "" if s is None else str(s)

        # Unicode normalization for consistent matching
        try:
            s = unicodedata.normalize("NFKC", s)
        except Exception:
            pass

        # Fix common mojibake seen in CSV exports (best-effort, safe)
        try:
            s = s.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            pass

        s = s.replace("–", "-").replace("—", "-")
        s = s.replace("α", "alpha").replace("β", "beta").replace("γ", "gamma")
        s = s.replace("–", "-").replace("—", "-").replace("â€“", "-").replace("â€”", "-")
        s = s.replace("Î±", "alpha").replace("α", "alpha")
        s = s.replace("Î²", "beta").replace("β", "beta")
        s = s.replace("â€“", "-").replace("â€”", "-").replace("â€", "-")
        s = s.replace("â²", "").replace("â", "-")  # last resort: turns 266â283 into 266-283

        s = s.replace("Â®", "").replace("®", "")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def clean_sequence(self, seq: str) -> str:
        """
        Remove all non-letter characters from a sequence.
        Keeps only A–Z and a–z.
        """
        if not isinstance(seq, str):
            return ""
        return re.sub(r"[^A-Za-z]", "", seq.upper())

    def add_sequence_ids(
        self,
        id_col: str = "CPP_ID",
        prefix: str = "CPP_00",
        width: int = 4,
    ) -> "SequencePreprocessor":
        """
        Create a unique identifier per unique peptide sequence and attach it to the DataFrame.
        IDs look like CPP_00#### by default (e.g., CPP_000001).
        """
        if self.sequence_col not in self.df.columns:
            raise ValueError(f"Sequence column '{self.sequence_col}' not found in DataFrame")

        seq_series = self.df[self.sequence_col].fillna("").astype(str)
        unique_values = [seq for seq in seq_series.drop_duplicates().tolist() if seq]
        seq_to_id = {
            seq: f"{prefix}{i:0{width}d}"
            for i, seq in enumerate(unique_values, start=1)
        }
        self.df[id_col] = seq_series.map(seq_to_id)
        return self
        
    def validate_sequence(self, seq: str) -> bool:
        """
        Check if sequence contains only standard amino acids.
        
        Args:
            seq: Amino acid sequence string
            
        Returns:
            True if valid, False otherwise
        """
        if pd.isna(seq) or not isinstance(seq, str) or len(seq) == 0:
            return False
        
        # Convert to uppercase and check each character
        seq_upper = seq.upper()
        return all(aa in self.STANDARD_AA for aa in seq_upper)
    
    def split_annotation_value(self, value, column_name: str = None) -> List[str]:
        """
        Split annotation value by delimiters based on column type.
        - For 'cargo_type': only splits on 'and', 'or'
        - For other columns (like 'cell_line'): splits on 'and', 'or', ',', ';'
        
        Args:
            value: Annotation value to split
            column_name: Name of the column being processed
            
        Returns:
            List of individual values
        """
        if pd.isna(value) or value is None:
            return [None]
        
        if not isinstance(value, str):
            return [value]
        
        # Determine which delimiters to use based on column
        if column_name == 'Cargo Type':
            # Only split on 'and' and 'or' for cargo_type
            value = value.replace(' and ', '|')
            value = value.replace(' or ', '|')
            value = value.replace(' And ', '|')
            value = value.replace(' Or ', '|')
            value = value.replace(' AND ', '|')
            value = value.replace(' OR ', '|')
        else:
            # Split on all delimiters for other columns
            value = value.replace(' and ', '|')
            value = value.replace(' or ', '|')
            value = value.replace(' And ', '|')
            value = value.replace(' Or ', '|')
            value = value.replace(' AND ', '|')
            value = value.replace(' OR ', '|')
            value = value.replace(',', '|')
            value = value.replace(';', '|')
        
        # Split and clean
        values = [v.strip() for v in value.split('|') if v.strip()]
        
        # Return None if empty list, otherwise return the list
        return values if values else [None]
    
    def split_multi_value_annotations(self, columns: List[str] = None) -> 'SequencePreprocessor':
        """
        Split rows with multiple values in specified columns.
        Creates one row per combination of values.
        
        Args:
            columns: List of column names to split. 
                    Defaults to ['cell_line', 'cargo_type']
        
        Returns:
            Self for method chaining
        """
        if columns is None:
            columns = ['cell_line', 'cargo_type']
        
        # Check if columns exist
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            print(f"Warning: Columns not found and will be skipped: {missing_cols}")
            columns = [col for col in columns if col in self.df.columns]
        
        if not columns:
            print("No columns to split")
            return self
        
        rows_before = len(self.df)
        
        # Create list to store expanded rows
        expanded_rows = []
        
        for idx, row in self.df.iterrows():
            # Split each specified column
            split_values = {}
            for col in columns:
                split_values[col] = self.split_annotation_value(row[col], column_name=col)
            
            # Create all combinations
            import itertools
            combinations = list(itertools.product(*[split_values[col] for col in columns]))
            
            # Create a row for each combination
            for combo in combinations:
                new_row = row.copy()
                for i, col in enumerate(columns):
                    new_row[col] = combo[i]
                expanded_rows.append(new_row)
        
        # Create new DataFrame
        self.df = pd.DataFrame(expanded_rows).reset_index(drop=True)
        
        rows_after = len(self.df)
        self.stats['rows_after_split'] = rows_after
        
        print(f"Split multi-value annotations in columns: {columns}")
        print(f"Rows before split: {rows_before}, after split: {rows_after} (added {rows_after - rows_before} rows)")
        
        return self
    
    def normalize_annotation(self, value):
        """
        Normalize annotation values (handle NA, empty, None).
        
        Args:
            value: Annotation value to normalize
            
        Returns:
            None for missing values, otherwise the value
        """
        if pd.isna(value):
            return None
        if isinstance(value, str):
            # Strip whitespace
            value = value.strip()
            # Check for empty or common NA representations
            if value == '' or value.lower() in ['na', 'n/a', 'nan', 'none', 'null']:
                return None
        return value
    
    def clean_sequences(self) -> 'SequencePreprocessor':
        """
        Remove non-letter characters from all sequences.
        
        Returns:
            Self for method chaining
        """
        if self.sequence_col not in self.df.columns:
            raise ValueError(f"Sequence column '{self.sequence_col}' not found in DataFrame")
        
        # Apply cleaning
        original_sequences = self.df[self.sequence_col].copy()
        self.df[self.sequence_col] = self.df[self.sequence_col].apply(self.clean_sequence)
        
        # Count how many sequences were modified
        modified = (original_sequences != self.df[self.sequence_col]).sum()
        self.stats['cleaned_sequences'] = modified
        
        print(f"Cleaned {modified} sequences (removed non-letter characters)")
        return self
    
    def filter_valid_sequences(self) -> 'SequencePreprocessor':
        """
        Remove sequences with non-standard amino acids.
        
        Returns:
            Self for method chaining
        """
        if self.sequence_col not in self.df.columns:
            raise ValueError(f"Sequence column '{self.sequence_col}' not found in DataFrame")
        
        # Create mask for valid sequences
        valid_mask = self.df[self.sequence_col].apply(self.validate_sequence)
        
        self.stats['invalid_sequences'] = (~valid_mask).sum()
        self.df = self.df[valid_mask].copy()
        
        print(f"Removed {self.stats['invalid_sequences']} sequences with non-standard amino acids")
        return self
    
    def normalize_annotations(self) -> 'SequencePreprocessor':
        """
        Normalize all annotation columns in the dataset.
        
        Returns:
            Self for method chaining
        """
        annotation_cols = [col for col in self.df.columns if col != self.sequence_col]
        
        for col in annotation_cols:
            self.df[col] = self.df[col].apply(self.normalize_annotation)
        
        print(f"Normalized annotations for {len(annotation_cols)} columns")
        return self
    
    def filter_required_annotations(self) -> 'SequencePreprocessor':
        """
        Remove sequences where ALL THREE required annotations are missing.
        Keeps sequences that have at least one of the required annotations.
        
        Returns:
            Self for method chaining
        """
        # Check if required columns exist
        missing_cols = [col for col in self.REQUIRED_ANNOTATIONS if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Required columns not found in DataFrame: {missing_cols}")
        
        # Create mask for rows where ALL THREE annotations are missing
        all_missing_mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        for col in self.REQUIRED_ANNOTATIONS:
            all_missing_mask &= self.df[col].isna()
        
        # Keep rows where NOT all three are missing (i.e., at least one is present)
        valid_mask = ~all_missing_mask
        
        self.stats['missing_annotations'] = all_missing_mask.sum()
        self.df = self.df[valid_mask].copy()
        
        print(f"Removed {self.stats['missing_annotations']} sequences where ALL THREE required annotations were missing")
        return self
    
    def preprocess(self) -> pd.DataFrame:
        """
        Run complete preprocessing pipeline.
        
        Returns:
            Preprocessed DataFrame
        """
        print("Starting preprocessing...")
        print(f"Initial dataset size: {self.stats['initial_count']} sequences\n")
        
        # Step 1a: Clean sequences (remove non-letter characters)
        self.clean_sequences()
        
        # Step 1b: Filter sequences with non-standard amino acids
        self.filter_valid_sequences()
        
        # Step 1b: Normalize annotations
        self.normalize_annotations()
        
        ### Step 4a: Filter sequences where ALL THREE required annotations are missing
        self.filter_required_annotations()
        self.df.reset_index(drop=True, inplace=True)
        self.stats['final_count'] = len(self.df)
        print(f"Dataset size after sanitization: {self.stats['final_count']} sequences")
        print(f"Retention rate: {self.stats['final_count']/self.stats['initial_count']*100:.2f}%")

        # Step 2: Split multi-value annotations (cell_line, cargo_type)
        self.split_multi_value_annotations(columns=['Cell Line', 'Cargo Type'])
        
        # Step 3: Standardize Uptake Mechanism, Subcellular Localization Category, Subcategory Uptake Mechanism, and map to Ontologies
        self.df['Main Uptake Mechanism'] = self.df['Uptake Mechanism'].apply(self.categorize_uptake)
        self.df['Main Uptake Mechanism ID'] = self.df['Main Uptake Mechanism'].apply(self.map_uptake_main_to_go)
        
        self.df['Subcategory Uptake Mechanism'] = self.df['Uptake Mechanism'].apply(self.classify_subcategory_uptake)
        self.df['Subcategory Uptake Mechanism ID'] = self.df['Subcategory Uptake Mechanism'].apply(self.map_uptake_subcategory_to_go)
        
        self.df['Subcellular Localization Category'] = self.df['Subcellular Localization'].apply(self.classify_localization)
        self.df['Subcellular Delivery ID'] = self.df['Subcellular Localization Category'].apply(self.map_sub_cell_delivery_to_go)
        
        self.df['Assay ID'] = self.df['In-vivo Model'].apply(self.map_assay_to_go)
        
        # Step 4: Remove duplicates based on sequence and key annotations (keep first occurrence)
        dedupe_cols = [
            self.sequence_col,
            "Cargo Type",
            "Cell Line",
            "Main Uptake Mechanism",
            "Subcellular Localization Category",
            "Subcategory Uptake Mechanism",
        ]
        
        n_antes = len(self.df)
        self.df = self.df.drop_duplicates(subset=dedupe_cols, keep="first")
        n_despues = len(self.df)
        n_eliminados = n_antes - n_despues
        print(f"--- Filter report ---")
        print(f"Dataset size: {n_antes}")
        print(f"Entries duplicated and removed: {n_eliminados}")
        print(f"Unique entries remaining: {n_despues}")
        self.df.reset_index(drop=True, inplace=True)
        
        self.stats['final_count'] = len(self.df)
        
        print(f"\nPreprocessing complete!")
        print(f"Final dataset size: {self.stats['final_count']} sequences")

        return self.df
    
    @staticmethod
    def categorize_uptake(mech_text):
        if pd.isna(mech_text):
            return pd.NA

        text = str(mech_text).lower()

        # -----------------------
        # Keywords for each group
        # -----------------------
        endocytosis_keywords = [
            "endocytosis", "macropinocytosis", "phagocytosis", "pinocytosis",
            "caveolae", "caveolin", "clathrin", "lipid raft", "raft",
            "receptor-mediated", "rme", "transcytosis", "adsorptive",
            "ldl-mediated", "tfr1", "nrp-1", "hs-dependent"
        ]

        direct_keywords = [
            "direct translocation", "direct penetration", "direct membrane",
            "membrane translocation", "translocation", "membrane damage",
            "membrane disruption", "energy-independent pathway",
            "non-endocytic", "non-dependent energy", "passive cell penetration"
        ]

        found_endo = any(kw in text for kw in endocytosis_keywords)
        found_direct = any(kw in text for kw in direct_keywords)

        # Decide output
        if found_endo and found_direct:
            return "Direct penetration, Endocytosis"
        elif found_endo:
            return "Endocytosis"
        elif found_direct:
            return "Direct penetration"
        else:
            return pd.NA
    
    @staticmethod
    def classify_localization(text):
        if pd.isna(text):
            return pd.NA
        t = str(text).lower()
        categories = []
        # -----------------------------
        # Keyword groups for each label
        # -----------------------------
        cytoplasm_kw = [
            "cytoplasm", "cytosol", "intramembrane", "perinuclear", 
            "around the nucleus", "centrosmal", "periplasm"
        ]
        endosomes_kw = [
            "endosome", "endosomal", "endocytic", "late endosomal"
        ]
        vesicles_kw = [
            "vesicle", "vacuole", "acrosome", "multivesicular"
        ]
        mitochondria_kw = [
            "mitochondria", "mitochondrial", "chloroplast"
        ]
        nucleus_kw = [
            "nucleus", "nuclei", "nuclear", "nucleolar", "nucleoi"
        ]

        # -----------------------------
        # Matching logic
        # -----------------------------
        if any(kw in t for kw in cytoplasm_kw):
            categories.append("Cytoplasm")
        if any(kw in t for kw in endosomes_kw):
            categories.append("Endosomes")
        if any(kw in t for kw in vesicles_kw):
            categories.append("Vesicles")
        if any(kw in t for kw in mitochondria_kw):
            categories.append("Mitochondria")
        if any(kw in t for kw in nucleus_kw):
            categories.append("Nucleus")
        # If nothing found, return NA
        if not categories:
            return pd.NA
        return ", ".join(categories)

    @staticmethod
    def classify_subcategory_uptake(text: str):
        if pd.isna(text):
            return pd.NA
        t = str(text).lower()
        cats = []
        # -----------------------------
        # Macropinocytosis
        # -----------------------------
        # covers: "macropinocytosis", "fluid phase pinocytosis", etc.
        if "macropinocyt" in t or "fluid phase pinocytosis" in t:
            cats.append("Macropinocytosis")
        # -----------------------------
        # Phagocytosis
        # -----------------------------
        if "phagocytosis" in t or "phagocytic" in t:
            cats.append("Phagocytosis")
        # -----------------------------
        # Caveolae-mediated endocytosis
        # (includes caveolin-mediated)
        # -----------------------------
        caveolae_terms = ["caveolae-mediated", "caveolae mediated",
                            "caveolin-mediated", "caveolin mediated"]
        if any(term in t for term in caveolae_terms):
            cats.append("Caveolae-mediated endocytosis")
        # -----------------------------
        # Clathrin-mediated endocytosis
        # -----------------------------
        # We consider clathrin-mediated / clathrin-dependent mentions
        # (avoid classifying "independent" cases here; those go below)
        if "clathrin" in t and "independent" not in t:
            # e.g. "clathrin-mediated endocytosis",
            #      "clathrin dependent endocytosis"
            cats.append("Clathrin-mediated endocytosis")
        # -----------------------------
        # Clathrin and caveolae independent
        # -----------------------------
        if (
            "independent" in t
            and (
                "clathrin" in t
                or "caveolin" in t
                or "caveolae" in t
            )
        ):
            cats.append("Clathrin and caveolae independent")

        # -----------------------------
        # If nothing matched, return NA
        # -----------------------------
        if not cats:
            return pd.NA

        # Deduplicate while preserving order
        cats_unique = []
        for c in cats:
            if c not in cats_unique:
                cats_unique.append(c)

        return ", ".join(cats_unique)
    
    @staticmethod
    def map_uptake_subcategory_to_go(text: str):
        if pd.isna(text):
            return pd.NA
        mapping = {
            "Clathrin-mediated endocytosis": "http://purl.obolibrary.org/obo/GO_0072583",
            "Macropinocytosis": "http://purl.obolibrary.org/obo/GO_0044351",
            "Phagocytosis": "http://purl.obolibrary.org/obo/GO_0006909",
            "Caveolae-mediated endocytosis": "http://purl.obolibrary.org/obo/GO_0072584",
            "Clathrin- and caveolae-independent endocytosis": "http://purl.obolibrary.org/obo/GO_0160294",
        }
        parts = [p.strip() for p in str(text).split(",") if p.strip()]
        go_terms = []
        for p in parts:
            go_term = mapping.get(p)
            if go_term and go_term not in go_terms:
                go_terms.append(go_term)
        if not go_terms:
            return pd.NA
        return ", ".join(go_terms)

    @staticmethod
    def map_uptake_main_to_go(text: str):
        if pd.isna(text):
            return pd.NA
        mapping = {
            "Direct penetration": "http://purl.obolibrary.org/obo/GO_0022857",
            "Endocytosis": "http://purl.obolibrary.org/obo/GO_0006897",
        }
        parts = [p.strip() for p in str(text).split(",") if p.strip()]
        go_terms = []
        for p in parts:
            go_term = mapping.get(p)
            if go_term and go_term not in go_terms:
                go_terms.append(go_term)
        if not go_terms:
            return pd.NA
        return ", ".join(go_terms)

    @staticmethod
    def map_sub_cell_delivery_to_go(text: str):
        if pd.isna(text):
            return pd.NA
        mapping = {
            "Cytoplasm": "http://purl.obolibrary.org/obo/GO_0005737",
            "Nucleus": "http://purl.obolibrary.org/obo/GO_0005634",
            "Vesicles": "http://purl.obolibrary.org/obo/GO_0031982",
            "Mitochondria": "http://purl.obolibrary.org/obo/GO_0005739",
            "Endosomes": "http://purl.obolibrary.org/obo/GO_0005768",
        }
        parts = [p.strip() for p in str(text).split(",") if p.strip()]
        go_terms = []
        for p in parts:
            go_term = mapping.get(p)
            if go_term and go_term not in go_terms:
                go_terms.append(go_term)
        if not go_terms:
            return pd.NA
        return ", ".join(go_terms)
    
    @staticmethod
    def map_assay_to_go(text: str):
        if pd.isna(text):
            return pd.NA
        mapping = {
            "In vivo": "http://www.bioassayontology.org/bao#BAO_0020009",
            "In vitro": "http://www.bioassayontology.org/bao#BAO_0020008",
            "In situ": "http://www.bioassayontology.org/bao#BAO_0000128",
            "Ex vivo": "http://www.bioassayontology.org/bao#BAO_0020006",
        }
        parts = [p.strip() for p in str(text).split(",") if p.strip()]
        go_terms = []
        for p in parts:
            go_term = mapping.get(p)
            if go_term and go_term not in go_terms:
                go_terms.append(go_term)
        if not go_terms:
            return pd.NA
        return ", ".join(go_terms)
    
    def _count_annotation_values(self, series: pd.Series) -> pd.Series:
        """
        Count annotation values, splitting multi-label cells on commas.
        """
        counter: Counter = Counter()
        for raw in series.dropna():
            for part in str(raw).split(","):
                label = part.strip()
                if label:
                    counter[label] += 1
        return pd.Series(counter).sort_values(ascending=False)

    def get_stats(self) -> dict:
        """
        Get preprocessing statistics.
        
        Returns:
            Dictionary with preprocessing statistics
        """
        return self.stats

    def validate_dataset(
        self,
        min_len: int = 3,
        max_len: int = 200,
        max_single_aa_fraction: float = 0.8,
        missingness_cols: Optional[List[str]] = None,
        report_top_aa: int = 5,
        verbose: bool = True,
    ) -> dict:
        """
        Compute validation metrics for sequence-level checks and annotation missingness.

        Sequence-level validations:
        - Length bounds
        - Composition sanity (max single AA fraction)

        Annotation integrity:
        - Missingness report for specified columns
        """
        if self.sequence_col not in self.df.columns:
            raise ValueError(f"Sequence column '{self.sequence_col}' not found in DataFrame")

        if missingness_cols is None:
            missingness_cols = [
                "Main Uptake Mechanism",
                "Subcategory Uptake Mechanism",
                "Subcellular Localization Category",
                "Cargo Category",
                "CLO_iri",
            ]

        seq_series = self.df[self.sequence_col].fillna("").astype(str)
        lengths = seq_series.str.len()

        # Length bounds
        too_short_mask = lengths < min_len
        too_long_mask = lengths > max_len

        # Composition sanity: max single AA fraction per sequence
        def _max_single_aa_fraction(seq: str) -> float:
            if not seq:
                return 0.0
            counts = Counter(seq)
            return max(counts.values()) / len(seq)

        max_aa_fraction = seq_series.apply(_max_single_aa_fraction)
        high_single_aa_mask = max_aa_fraction > max_single_aa_fraction

        # AA composition summary (overall)
        all_aa = "".join(seq_series.tolist())
        aa_counts = Counter(all_aa)
        aa_total = sum(aa_counts.values()) if aa_counts else 0
        aa_freq = {
            aa: (count / aa_total if aa_total else 0.0)
            for aa, count in aa_counts.most_common(report_top_aa)
        }

        # Missingness report
        missingness = {}
        for col in missingness_cols:
            if col not in self.df.columns:
                missingness[col] = {
                    "missing_count": None,
                    "missing_fraction": None,
                    "note": "column_missing",
                }
                continue
            missing_count = self.df[col].isna().sum()
            missingness[col] = {
                "missing_count": int(missing_count),
                "missing_fraction": float(missing_count / len(self.df)) if len(self.df) else 0.0,
            }

        # Complete annotation assessment
        completeness_cols = [
            "CLO_label",
            "Cargo Category",
            "Main Uptake Mechanism",
            "Subcategory Uptake Mechanism",
            "Subcellular Localization Category",
        ]
        present_cols = [c for c in completeness_cols if c in self.df.columns]
        missing_cols_list = [c for c in completeness_cols if c not in self.df.columns]

        if present_cols:
            complete_mask = self.df[present_cols].notna().all(axis=1)
            complete_entries = int(complete_mask.sum())
            complete_fraction = float(complete_mask.mean()) if len(self.df) else 0.0

            complete_df = self.df.loc[complete_mask]
            unique_seqs = int(complete_df[self.sequence_col].nunique()) if self.sequence_col in complete_df.columns else 0
        else:
            complete_entries = 0
            complete_fraction = 0.0
            unique_seqs = 0

        completeness = {
            "columns_checked": present_cols,
            "columns_not_found": missing_cols_list,
            "complete_entries": complete_entries,
            "complete_fraction": complete_fraction,
            "unique_sequences_with_complete_annotations": unique_seqs,
        }

        report = {
            "sequence_length": {
                "min_len": int(min_len),
                "max_len": int(max_len),
                "too_short_count": int(too_short_mask.sum()),
                "too_long_count": int(too_long_mask.sum()),
                "too_short_fraction": float(too_short_mask.mean()) if len(self.df) else 0.0,
                "too_long_fraction": float(too_long_mask.mean()) if len(self.df) else 0.0,
            },
            "sequence_composition": {
                "max_single_aa_fraction_threshold": float(max_single_aa_fraction),
                "high_single_aa_count": int(high_single_aa_mask.sum()),
                "high_single_aa_fraction": float(high_single_aa_mask.mean()) if len(self.df) else 0.0,
                "top_aa_frequencies": aa_freq,
            },
            "annotation_missingness": missingness,
            "annotation_completeness": completeness,
        }

        if verbose:
            print("--- Validation report ---")
            print("Sequence length bounds:")
            print(
                f"  < {min_len} aa: {report['sequence_length']['too_short_count']} "
                f"({report['sequence_length']['too_short_fraction']:.2%})"
            )
            print(
                f"  > {max_len} aa: {report['sequence_length']['too_long_count']} "
                f"({report['sequence_length']['too_long_fraction']:.2%})"
            )
            print("Composition sanity:")
            print(
                f"  > {max_single_aa_fraction:.2f} single-AA fraction: "
                f"{report['sequence_composition']['high_single_aa_count']} "
                f"({report['sequence_composition']['high_single_aa_fraction']:.2%})"
            )
            if aa_freq:
                print("  Top AA frequencies:")
                for aa, freq in aa_freq.items():
                    print(f"    {aa}: {freq:.2%}")
            print("Annotation missingness:")
            for col, stats in missingness.items():
                if stats.get("note") == "column_missing":
                    print(f"  {col}: column missing")
                else:
                    print(
                        f"  {col}: {stats['missing_count']} "
                        f"({stats['missing_fraction']:.2%})"
                    )
            print("Annotation completeness:")
            print(f"  Columns checked: {', '.join(present_cols)}")
            if missing_cols_list:
                print(f"  Columns not found: {', '.join(missing_cols_list)}")
            print(
                f"  Entries with all annotations: {complete_entries} "
                f"({complete_fraction:.2%})"
            )
            print(
                f"  Unique sequences with complete annotations: {unique_seqs}"
            )

        return report

if __name__ == "__main__":
    
    # Load your dataset
    df = pd.read_csv('data/Natural_CPP3_download_annotated.csv')  # or however you load it

    # Create preprocessor instance
    preprocessor = SequencePreprocessor(df, sequence_col='Sequence')

    # Run preprocessing
    cleaned_df = preprocessor.preprocess()

    # Save cleaned data
    cleaned_df.to_csv('data/Natural_CPP3_download_annotated_final_cleaned.csv', index=False)

    #df = pd.read_csv('data/Natural_CPP3_download_annotated_cleaned.csv')  # latest starting point
    
    preprocessor.add_sequence_ids(id_col="CPP_ID", prefix="https://w3id.org/cpp/dataset/mechanisms/CPP_00", width=4)
    
    cleaned_df = preprocessor.df
    cleaned_df.to_csv('data/preprocessed_data.csv', index=False)
    
    # Get detailed statistics
    stats = preprocessor.get_stats()
    print(stats)
    
