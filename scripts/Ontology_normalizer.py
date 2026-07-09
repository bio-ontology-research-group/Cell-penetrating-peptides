"""
ontology_normalizer.py
======================
Normalize biomedical entity strings to CLO (cell lines) or CHEBI (chemicals)
using four methods:
    1. Exact + synonym match   (lexical dictionary lookup)
    2. Biosyn                  (BERT-based synonym marginalization, HuggingFace)
    3. Graph RAG                     (Graph retrieval + LLM reranking)

Usage
-----
    python normalize.py \
        --input      annotations.csv \
        --column     entity_text \
        --ontology   chebi \                 # chebi | clo
        --obo        ontologies/chebi.obo \  # or .owl for CLO
        --output     results.csv
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Dict, Optional

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

TOP_K_GRAPH = 15   # max nodes returned by traverse_graph

GRAPH_LLM_PROMPT = """\
You are an expert in biomedical terminology and ontologies. \
Your task is to map entity mentions to the {ontology} ontology.

For each term provided, find the most appropriate ontology concept. \
Use the provided Graph RAG context (semantic matches + ontology neighbours) \
to make the best possible mapping.

Terms to map:
{terms_list}

Respond with a JSON object where each key is the original term \
and each value has "curie" and "label".

Example format:
{{
    "HeLa": {{
        "curie": "CLO:0003684",
        "label": "HeLa cell"
    }},
    "Jurkat cells": {{
        "curie": "CLO:0007043",
        "label": "JURKAT cell"
    }}
}}

CRITICAL rules:
- The curie and label must exist exactly in the {ontology} ontology.
- If no candidate is a reasonable match for a term, set both fields to "NONE".
- Prefer SEMANTIC MATCHES; use GRAPH CONTEXT to resolve ambiguity or find \
more specific children.
- Output ONLY the JSON object — no explanation, no markdown fences.

## GRAPH RAG CONTEXT FROM {ontology} ONTOLOGY
{enriched_context}"""

# ──────────────────────────────────────────────────────────────────────────────
# Shared LLM prompt (used by both BERTNormalizer graph-LLM path and RAGNormalizer)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_llm_json(raw: str) -> dict:
    """
    Parse the LLM response as JSON, with two fallback layers:
        1. Strip markdown fences (``` json ... ```) and try json.loads.
        2. If that fails (e.g. truncated response), call _recover_partial_json
            to salvage complete entries before the cut-off point.
    """
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        recovered = _recover_partial_json(raw)
        if recovered:
            log.warning("LLM response was truncated — recovered %d/%s entries",
                        len(recovered), "?")
        return recovered

def _recover_partial_json(raw: str) -> dict:
    """
    Salvage complete key-value pairs from a truncated JSON object.

    When the LLM hits max_tokens mid-response the closing braces are missing.
    This finds the last complete entry (ends with `}`) and closes the outer
    object so json.loads can parse whatever arrived before the cut.
    """
    # Find the last occurrence of a complete nested object: ..."}\n  "term": {
    last_complete = raw.rfind("},")      # last entry that was followed by a comma
    if last_complete == -1:
        last_complete = raw.rfind("}")   # single / last complete entry
    if last_complete == -1:
        return {}
    truncated = raw[: last_complete + 1].rstrip().rstrip(",") + "\n}"
    try:
        return json.loads(truncated)
    except json.JSONDecodeError:
        return {}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Lowercase + collapse whitespace + normalise underscores."""
    return " ".join(str(text).lower().replace("_", " ").split())

def traverse_graph(
    seed_nodes: List[str],
    graph: Dict[str, Dict],
    max_nodes: int = TOP_K_GRAPH,
) -> Dict[str, Dict]:
    """
    BFS from seed_nodes through ontology edges (parents, children, siblings).

    Parameters
    ----------
    seed_nodes : list of CURIE strings returned by FAISS retrieval.
    graph      : {curie: {"label": str, "parents": [curie, …],
                                        "children": [curie, …]}}
                    Built by BERTNormalizer._load_concepts.
    max_nodes  : hard cap on total nodes returned (default TOP_K_GRAPH=15).

    Returns
    -------
    {curie: {"label": str, "relation": str}}
        relation is one of "seed", "parent", "child", "sibling".
    """
    from collections import deque

    related: Dict[str, Dict] = {}
    queue: deque = deque()

    for curie in seed_nodes:
        if curie in graph and curie not in related:
            related[curie] = {"label": graph[curie]["label"], "relation": "seed"}
            queue.append(curie)

    while queue and len(related) < max_nodes:
        current = queue.popleft()
        node = graph.get(current, {})

        # ── parents ──────────────────────────────────────────────────────────
        for parent in node.get("parents", []):
            if parent not in related and len(related) < max_nodes:
                related[parent] = {
                    "label":    graph.get(parent, {}).get("label", ""),
                    "relation": "parent",
                }
                queue.append(parent)

        # ── children ─────────────────────────────────────────────────────────
        for child in node.get("children", []):
            if child not in related and len(related) < max_nodes:
                related[child] = {
                    "label":    graph.get(child, {}).get("label", ""),
                    "relation": "child",
                }
                queue.append(child)

        # ── siblings (other children of each parent) ──────────────────────────
        for parent in node.get("parents", []):
            for sibling in graph.get(parent, {}).get("children", []):
                if sibling not in related and len(related) < max_nodes:
                    related[sibling] = {
                        "label":    graph.get(sibling, {}).get("label", ""),
                        "relation": "sibling",
                    }
                    # siblings are lower-priority; don't push to queue

    return related

def create_enriched_context(
    graph: Dict[str, Dict],
    term_batch: List[str],
    semantic_matches_dict: Dict[str, List],
) -> Dict[str, str]:
    """
    Build a formatted LLM context block for every term in term_batch.

    Parameters
    ----------
    graph                : BERTNormalizer.graph — {curie: {label, parents, children}}
    term_batch           : raw entity strings (may contain duplicates; keyed by preprocess())
    semantic_matches_dict: {preprocessed_text: [(curie, label, score), ...]}
                            top-k FAISS results per unique text, from _retrieve_topk

    Returns
    -------
    {preprocessed_text: formatted_context_string}
    """
    result: Dict[str, str] = {}
    for text in term_batch:
        key = preprocess(text)
        if key in result:          # skip duplicates
            continue
        candidates = semantic_matches_dict.get(key, [])
        if not candidates:
            result[key] = "SEMANTIC MATCHES:\n  (none)\n\nGRAPH CONTEXT:\n  (none)"
            continue

        seed_curies = [c for c, _, _ in candidates]
        score_map   = {c: s for c, _, s in candidates}
        related     = traverse_graph(seed_curies, graph, max_nodes=TOP_K_GRAPH)

        sem_lines: List[str]   = []
        graph_lines: List[str] = []
        for curie, meta in related.items():
            label    = meta["label"]
            relation = meta["relation"]
            if relation == "seed":
                sem_lines.append(
                    f"  \u2022 {curie}  |  {label}  |  similarity={score_map.get(curie, 0.0):.4f}"
                )
            else:
                graph_lines.append(
                    f"  \u2022 {curie}  |  {label}  |  [{relation}]"
                )

        block  = "SEMANTIC MATCHES (embedding similarity):\n"
        block += "\n".join(sem_lines) if sem_lines else "  (none)"
        block += "\n\nGRAPH CONTEXT (ontology neighbours):\n"
        block += "\n".join(graph_lines) if graph_lines else "  (none)"
        result[key] = block

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Method 1 — Exact + Synonym Match
# ──────────────────────────────────────────────────────────────────────────────

class ExactSynonymNormalizer:
    """
    Builds a dictionary of every preferred label and synonym in the ontology,
    then does direct lookup (case-insensitive, whitespace-normalized).

    If no exact match is found, falls back to token-sort fuzzy match via
    RapidFuzz (threshold configurable, default 0.92).
    """

    def __init__(self, obo_path: str, fuzzy_threshold: float = 0.92, ontology: str = None):
        import pronto
        from rapidfuzz import fuzz, process

        self.fuzz = fuzz
        self.process = process
        self.threshold = fuzzy_threshold
        self._lookup: Dict[str, Dict] = {}   # norm_string -> {curie, label, match_type}

        log.info("ExactSynonym: loading ontology from %s …", obo_path)
        try:
            onto = pronto.Ontology(obo_path)
        except Exception as e:
            raise RuntimeError(f"Could not parse ontology file: {e}") from e

        # Build a set of accepted CURIE prefixes — only the target ontology.
        # Rejects imported/external terms (UBERON, RO, BFO, etc.).
        _accepted_prefix = f"{ontology.upper()}:" if ontology else None

        added = 0
        skipped = 0
        terms = list(onto.terms())
        for term in terms:
            curie = term.id
            if _accepted_prefix and not curie.startswith(_accepted_prefix):
                skipped += 1
                continue
            pref  = term.name or ""
            entry_pref = {"curie": curie, "label": pref, "match_type": "exact_pref"}
            entry_syn  = {"curie": curie, "label": pref, "match_type": "exact_syn"}
            # preferred label
            if pref:
                self._lookup[preprocess(pref)] = entry_pref
                added += 1
                # CLO labels end in " cell" — also index without that suffix
                for suffix in (" cell", " cells"):
                    stripped = preprocess(pref)
                    if stripped.endswith(suffix):
                        bare = stripped[: -len(suffix)].strip()
                        if bare and bare not in self._lookup:
                            self._lookup[bare] = entry_pref
                            added += 1
            # synonyms
            for syn in term.synonyms:
                key = preprocess(syn.description)
                if key and key not in self._lookup:
                    self._lookup[key] = entry_syn
                    added += 1
                for suffix in (" cell", " cells"):
                    if key.endswith(suffix):
                        bare = key[: -len(suffix)].strip()
                        if bare and bare not in self._lookup:
                            self._lookup[bare] = entry_syn
                            added += 1

        self._keys = list(self._lookup.keys())
        log.info("ExactSynonym: indexed %d label/synonym strings from %d terms",
                    added, len(terms))

    def predict(self, text: str) -> dict:
        key = preprocess(text)

        def _hit(k, match_type="exact_pref"):
            if k in self._lookup:
                return {**self._lookup[k], "score": 1.0, "method": "exact_synonym"}
            k_cell = k + " cell"
            if k_cell in self._lookup:
                return {**self._lookup[k_cell], "score": 1.0, "method": "exact_synonym"}
            return None

        # 1. Exact hit (with and without " cell")
        hit = _hit(key)
        if hit:
            return hit

        # 2. Truncate augmented names — try progressively shorter prefixes split on
        #    whitespace and underscores (e.g. "22Rv1_10nM_treatment" → "22Rv1")
        parts = key.replace("_", " ").split()
        for length in range(len(parts) - 1, 0, -1):
            prefix = " ".join(parts[:length])
            hit = _hit(prefix)
            if hit:
                return {**hit, "match_type": "prefix_truncated"}

        # 3. Fuzzy fallback
        match = self.process.extractOne(
            key, self._keys,
            scorer=self.fuzz.token_sort_ratio,
            score_cutoff=self.threshold * 100,
        )
        if match:
            matched_key, score, _ = match
            return {
                **self._lookup[matched_key],
                "score": round(score / 100, 4),
                "match_type": "fuzzy",
                "method": "exact_synonym",
            }

        return {"curie": None, "label": None, "score": 0.0,
                "match_type": "NIL", "method": "exact_synonym"}

    def run(self, texts: List[str]) -> List[Dict]:
        return [self.predict(t) for t in tqdm(texts, desc="ExactSynonym")]

# ──────────────────────────────────────────────────────────────────────────────
# Method 2 — BERT Normalizer (HuggingFace)
# ──────────────────────────────────────────────────────────────────────────────

class BERTNormalizer:
    """
    Loads a pretrained BERT model from HuggingFace and performs entity linking
    via dense nearest-neighbour search over a FAISS index of ontology concept
    embeddings (synonym marginalization at query time).

    HuggingFace models:
        chemicals  -> dmis-lab/biosyn-sapbert-bc5cdr-chemical
        cell lines -> cambridgeltl/SapBERT-from-PubMedBERT-fulltext
    """

    MODELS = {
        "chebi": "dmis-lab/biosyn-sapbert-bc5cdr-chemical",
        "clo":   "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    }

    def __init__(self, obo_path: str, ontology: str,
                    batch_size: int = 128, topk: int = 5,
                    llm_backend: str = None,
                    llm_model:   str = None,
                    llm_url:     str = None):
        try:
            import faiss
        except ImportError:
            raise RuntimeError("faiss not found. Install with: pip install faiss-cpu")

        from transformers import AutoModel, AutoTokenizer
        import os

        self.faiss = faiss
        self.batch_size = batch_size
        self.topk = topk
        self.ontology = ontology
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── optional LLM client for graph-augmented prediction ─────────────────
        self._llm_client = None
        self._llm_model  = llm_model
        if llm_model is not None:
            try:
                from openai import OpenAI
                _BACKENDS = {
                    "openrouter": "https://openrouter.ai/api/v1",
                    "ollama":     "http://localhost:11434/v1",
                }
                _backend = llm_backend or "openrouter"
                _url = llm_url or _BACKENDS.get(_backend, _BACKENDS["openrouter"])
                _key = ("ollama" if _backend == "ollama"
                        else os.environ.get("OPENROUTER_API_KEY", ""))
                self._llm_client = OpenAI(base_url=_url, api_key=_key)
                log.info("BioSyn: LLM enabled — backend=%s  model=%s", _backend, llm_model)
            except ImportError:
                log.warning(
                    "BioSyn: openai not installed — LLM path disabled, "
                    "falling back to top-1 marginalization."
                )

        model_name = self.MODELS.get(ontology, self.MODELS["chebi"])
        log.info("BioSyn: loading model '%s' on %s …", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        concept_embeddings = self._load_or_build_cache(obo_path, model_name)

        # IndexFlatIP: exact inner-product search on L2-normalised vectors = cosine similarity.
        log.info("BioSyn: building FAISS inner-product index …")
        d = concept_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(concept_embeddings)
        log.info("BioSyn: index ready — %d vectors, dim=%d", self.index.ntotal, d)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _cache_path(self, obo_path: str, model_name: str) -> Path:
        """
        Deterministic cache filename derived from the OBO file modification
        time and the model name.  Stored next to the OBO file so the cache
        is naturally co-located with the data it represents.

        Format: <obo_stem>_<ontology>_<model_slug>_<mtime_hex>.pkl
        """
        import hashlib
        obo   = Path(obo_path)
        mtime = str(obo.stat().st_mtime).encode()
        key   = hashlib.md5(mtime + model_name.encode()).hexdigest()[:8]
        slug  = model_name.replace("/", "_").replace("-", "_")
        return obo.parent / f"biosyn_cache_{self.ontology}_{slug}_{key}.pkl"

    def _load_or_build_cache(self, obo_path: str, model_name: str) -> np.ndarray:
        """
        Return L2-normalised concept embeddings, loading from disk cache when
        available or encoding from scratch and saving to cache otherwise.

        The cache stores: _curies, _labels, _strings, graph, embeddings.
        The FAISS index itself is NOT cached — it rebuilds from the embedding
        array in <1 s, which avoids faiss serialisation complexity.
        """
        import pickle
        cache_file = self._cache_path(obo_path, model_name)

        if cache_file.exists():
            log.info("BioSyn: loading concept pool + embeddings from cache: %s", cache_file)
            with open(cache_file, "rb") as fh:
                cached = pickle.load(fh)
            self._curies = cached["curies"]
            self._labels = cached["labels"]
            self._strings = cached["strings"]
            self.graph   = cached["graph"]
            embeddings   = cached["embeddings"]
            log.info(
                "BioSyn: cache loaded — %d strings, %d graph nodes",
                len(self._strings), len(self.graph),
            )
            return embeddings

        # Cache miss — build from scratch
        log.info("BioSyn: cache miss — building concept pool from %s …", obo_path)
        self._curies, self._labels, self._strings = self._load_concepts(obo_path)

        log.info("BioSyn: encoding %d concept strings …", len(self._strings))
        embeddings = self._encode(self._strings)
        self.faiss.normalize_L2(embeddings)

        log.info("BioSyn: saving cache to %s …", cache_file)
        with open(cache_file, "wb") as fh:
            pickle.dump({
                "curies":     self._curies,
                "labels":     self._labels,
                "strings":    self._strings,
                "graph":      self.graph,
                "embeddings": embeddings,
            }, fh, protocol=pickle.HIGHEST_PROTOCOL)

        return embeddings

    def _load_concepts(self, obo_path: str):
        import pronto

        # Only keep terms whose ID starts with the target ontology prefix
        # (e.g. "CLO:" or "CHEBI:"). The merged OWL files import CL, PATO,
        # UBERON, GO etc., which would otherwise pollute the concept pool.
        onto_prefix = self.ontology.upper() + ":"

        curies, labels, strings = [], [], []
        graph: Dict[str, Dict] = {}   # curie -> {label, parents, children}

        onto = pronto.Ontology(obo_path)

        # First pass: collect all in-ontology terms and their labels
        for term in onto.terms():
            if not term.id.startswith(onto_prefix):
                continue
            pref = term.name or ""
            graph[term.id] = {"label": pref, "parents": [], "children": []}
            all_syns = [pref] + [s.description for s in term.synonyms]
            for s in all_syns:
                if s.strip():
                    curies.append(term.id)
                    labels.append(pref)
                    strings.append(preprocess(s))

        # Second pass: fill parent / child edges (restrict to in-ontology nodes)
        for term in onto.terms():
            if term.id not in graph:
                continue
            for sup in term.superclasses(distance=1, with_self=False):
                if sup.id in graph:
                    graph[term.id]["parents"].append(sup.id)
                    graph[sup.id]["children"].append(term.id)

        log.info(
            "BioSyn: concept pool filtered to '%s' prefix — %d strings, %d graph nodes",
            onto_prefix, len(strings), len(graph),
        )
        self.graph = graph   # expose for RAGNormalizer
        return curies, labels, strings

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Mean-pool the last hidden state (standard BioSyn strategy)."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size),
                        desc="BioSyn encode", leave=False):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=64, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            # CLS token embedding
            emb = out.last_hidden_state[:, 0, :].cpu().float().numpy()
            all_embeddings.append(emb)
        return np.vstack(all_embeddings).astype("float32")

    # ── synonym marginalization ───────────────────────────────────────────────

    def _marginalize(self, query_emb: np.ndarray) -> dict:
        """
        Retrieve top-k candidate synonym rows, then aggregate scores
        per CURIE by taking the max similarity (synonym marginalization).
        Returns the best-scoring CURIE.
        """
        q = query_emb.copy()
        self.faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, self.topk * 3)

        # Aggregate per CURIE
        curie_best: Dict[str, float] = {}
        curie_label: Dict[str, str]  = {}
        for score, idx in zip(scores[0], idxs[0]):
            curie = self._curies[idx]
            if curie not in curie_best or score > curie_best[curie]:
                curie_best[curie]  = float(score)
                curie_label[curie] = self._labels[idx]

        best_curie = max(curie_best, key=curie_best.get)
        return {
            "curie":      best_curie,
            "label":      curie_label[best_curie],
            "score":      round(curie_best[best_curie], 4),
            "match_type": "dense_nn",
            "method":     "biosyn",
        }

    def _retrieve_topk(self, query_emb: np.ndarray, k: int = 10) -> List[tuple]:
        """
        Return the top-k deduplicated CURIEs by synonym marginalization.

        Parameters
        ----------
        query_emb : shape (1, d) float32 row vector from emb_map
        k         : number of unique CURIEs after deduplication

        Returns
        -------
        [(curie, label, score), ...] sorted descending by cosine similarity.
        """
        q = query_emb.copy()
        self.faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, k * 3)

        curie_best:  Dict[str, float] = {}
        curie_label: Dict[str, str]   = {}
        for score, idx in zip(scores[0], idxs[0]):
            curie = self._curies[idx]
            if curie not in curie_best or score > curie_best[curie]:
                curie_best[curie]  = float(score)
                curie_label[curie] = self._labels[idx]

        ranked = sorted(curie_best.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(c, curie_label[c], s) for c, s in ranked]

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, texts: List[str]) -> List[Dict]:
        # Encode all unique strings at once for efficiency
        unique = list(dict.fromkeys(preprocess(t) for t in texts))
        embeddings = self._encode(unique)
        emb_map = {t: embeddings[i][None] for i, t in enumerate(unique)}

        # ── no LLM: original top-1 path (unchanged behaviour) ─────────────────
        if self._llm_client is None:
            results = []
            for text in tqdm(texts, desc="BioSyn predict"):
                emb = emb_map[preprocess(text)]
                results.append(self._marginalize(emb))
            return results

        # ── LLM path: top-10 FAISS + graph traversal + LLM reranking ──────────
        TOP_K_RETRIEVE = 10

        # Step A: retrieve top-10 for each unique text
        semantic_matches_dict: Dict[str, List] = {
            key: self._retrieve_topk(emb, k=TOP_K_RETRIEVE)
            for key, emb in emb_map.items()
        }

        # Step B: build enriched context blocks for all texts (batch graph traversal)
        context_map = create_enriched_context(self.graph, texts, semantic_matches_dict)

        # Step C: batch LLM call — one request for all texts, JSON response
        BATCH_SIZE_LLM = 20   # terms per LLM call (keep prompt manageable)
        results_map: Dict[str, dict] = {}

        unique_texts = list(dict.fromkeys(texts))   # preserve order, deduplicate
        for batch_start in tqdm(
            range(0, len(unique_texts), BATCH_SIZE_LLM),
            desc="BioSyn graph-LLM batches",
        ):
            batch = unique_texts[batch_start : batch_start + BATCH_SIZE_LLM]

            # build combined context: one block per term
            enriched_parts: List[str] = []
            for text in batch:
                key  = preprocess(text)
                block = context_map.get(key, "  (none)")
                enriched_parts.append(f'### "{text}"\n{block}')
            enriched_context = "\n\n".join(enriched_parts)
            terms_list       = "\n".join(f'- "{t}"' for t in batch)

            user_msg = GRAPH_LLM_PROMPT.format(
                ontology        = self.ontology.upper(),
                terms_list      = terms_list,
                enriched_context= enriched_context,
            )

            llm_json: dict = {}
            try:
                response = self._llm_client.chat.completions.create(
                    model      = self._llm_model,
                    max_tokens = 4096,
                    temperature= 0,        # greedy decoding — deterministic, reproducible
                    seed       = 42,
                    messages   = [{"role": "user", "content": user_msg}],
                )
                raw      = (response.choices[0].message.content or "").strip()
                llm_json = _parse_llm_json(raw)
            except Exception as exc:
                log.warning("BioSyn LLM: batch call failed: %s — top-1 fallback", exc)

            for text in batch:
                key        = preprocess(text)
                candidates = semantic_matches_dict.get(key, [])

                if not candidates:
                    results_map[text] = {"curie": None, "label": None, "score": 0.0,
                                            "match_type": "NIL", "method": "biosyn"}
                    continue

                entry  = llm_json.get(text, {})
                answer = (entry.get("curie") or entry.get("id") or "").strip()

                if not answer or answer.upper() == "NONE" or not llm_json:
                    top_curie, top_label, top_score = candidates[0]
                    results_map[text] = {"curie": top_curie, "label": top_label,
                                            "score": round(top_score, 4),
                                            "match_type": "biosyn_graph_unconfident",
                                            "method": "biosyn"}
                    continue

                curie_map = {c: (lbl, sc) for c, lbl, sc in candidates}
                if answer in curie_map:
                    lbl, sc = curie_map[answer]
                    results_map[text] = {"curie": answer, "label": lbl,
                                            "score": round(sc, 4),
                                            "match_type": "biosyn_graph_llm",
                                            "method": "biosyn"}
                elif answer in self.graph:
                    lbl = self.graph[answer]["label"]
                    results_map[text] = {"curie": answer, "label": lbl, "score": 0.0,
                                            "match_type": "biosyn_graph_selected",
                                            "method": "biosyn"}
                else:
                    log.debug("BioSyn LLM: out-of-candidate CURIE '%s' for '%s'", answer, text)
                    top_curie, top_label, top_score = candidates[0]
                    results_map[text] = {"curie": top_curie, "label": top_label,
                                            "score": round(top_score, 4),
                                            "match_type": "biosyn_graph_unconfident",
                                            "method": "biosyn"}

        return [results_map[t] for t in texts]


# ──────────────────────────────────────────────────────────────────────────────
# Method 3 — RAG (Retrieval-Augmented Generation)
# ──────────────────────────────────────────────────────────────────────────────

class RAGNormalizer:
    """
    Retrieval-Augmented Generation normalizer.

    Step 1 — Retrieve: reuses BERTNormalizer's FAISS index to find the top-k
    most semantically similar ontology concepts (same dense-NN search as method 3).

    Step 2 — Generate: feeds the candidates as context to an LLM via OpenRouter
    (default: stepfun/step-3.5-flash:free), which selects the best-matching CURIE.

    Requires OPENROUTER_API_KEY set in the environment.

    Parameters
    ----------
    biosyn : BERTNormalizer
        A pre-initialised BERTNormalizer whose FAISS index and encoder will be
        reused. Pass the same instance used for method 3 to avoid loading the
        model twice.
    topk : int
        Number of candidate concepts to retrieve and show to the LLM.
    model : str
        Claude model ID.
    """

    # RAGNormalizer uses the shared GRAPH_LLM_PROMPT (batch JSON mode).

    # Supported backends and their default base URLs.
    BACKENDS = {
        "openrouter": "https://openrouter.ai/api/v1",
        "ollama":     "http://localhost:11434/v1",
    }

    def __init__(self, biosyn: "BERTNormalizer", topk: int = 10,
                    model: str = "stepfun/step-3.5-flash:free",
                    backend: str = "openrouter",
                    base_url: str = None):
        """
        Parameters
        ----------
        backend  : "openrouter" (default) or "ollama".
                    Ignored when base_url is set explicitly.
        base_url : Override the backend URL (e.g. a remote Ollama host).
        model    : Model name as understood by the chosen backend.
                    OpenRouter  → "stepfun/step-3.5-flash:free"
                    Ollama      → any model you have pulled, e.g. "llama3.2"
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "openai package not found. Install with: pip install openai"
            )

        import os
        self._biosyn       = biosyn
        self.topk          = topk
        self.model         = model
        self._biosyn_cache: Dict[str, dict] = {}   # filled by run(); used as fallback

        url = base_url or self.BACKENDS.get(backend, self.BACKENDS["openrouter"])
        # Ollama does not require a real key; OpenRouter reads OPENROUTER_API_KEY.
        api_key = "ollama" if backend == "ollama" else os.environ.get("OPENROUTER_API_KEY", "")
        self._client = OpenAI(base_url=url, api_key=api_key)
        log.info("RAG: backend=%s  model=%s  topk=%d", backend, model, topk)

    # ── retrieval ──────────────────────────────────────────────────────────────

    def _retrieve(self, text: str):
        """Return top-k (curie, label, score) sorted by cosine similarity."""
        emb = self._biosyn._encode([preprocess(text)])
        self._biosyn.faiss.normalize_L2(emb)
        scores, idxs = self._biosyn.index.search(emb, self.topk * 3)

        curie_best:  Dict[str, float] = {}
        curie_label: Dict[str, str]   = {}
        for score, idx in zip(scores[0], idxs[0]):
            curie = self._biosyn._curies[idx]
            if curie not in curie_best or score > curie_best[curie]:
                curie_best[curie]  = float(score)
                curie_label[curie] = self._biosyn._labels[idx]

        ranked = sorted(curie_best.items(), key=lambda x: x[1], reverse=True)[: self.topk]
        return [(c, curie_label[c], s) for c, s in ranked]

    # ── graph-enhanced context ──────────────────────────────────────────────────

    def _build_context(self, candidates) -> str:
        """
        Expand the top semantic matches with ontology-graph neighbours and
        format them into the two-section prompt block fed to the LLM.
        """
        seed_curies = [c for c, _, _ in candidates]
        score_map   = {c: s for c, _, s in candidates}

        graph   = getattr(self._biosyn, "graph", {})
        related = traverse_graph(seed_curies, graph, max_nodes=TOP_K_GRAPH)

        sem_lines   = []
        graph_lines = []
        for curie, meta in related.items():
            label    = meta["label"]
            relation = meta["relation"]
            if relation == "seed":
                score = score_map.get(curie, 0.0)
                sem_lines.append(
                    f"  • {curie}  |  {label}  |  similarity={score:.4f}"
                )
            else:
                graph_lines.append(
                    f"  • {curie}  |  {label}  |  [{relation}]"
                )

        block = "SEMANTIC MATCHES (embedding similarity):\n"
        block += "\n".join(sem_lines) if sem_lines else "  (none)"
        block += "\n\nGRAPH CONTEXT (ontology neighbours):\n"
        block += "\n".join(graph_lines) if graph_lines else "  (none)"
        return block

    # ── generation (batch JSON) ────────────────────────────────────────────────

    def _fallback(self, text: str, candidates: list,
                  match_type: str = "rag_unconfident") -> dict:
        """
        When the LLM is unconfident, prefer BioSyn's answer if available
        (it already ran and its result is cached in _biosyn_cache).
        Only fall back to raw top-1 FAISS when no BioSyn result exists.

        `match_type` records *why* we fell back so R3.1 can separate a genuine
        LLM hallucination from a legitimate abstention:
          * rag_abstained    — LLM returned "NONE"/empty (correctly declined)
          * rag_hallucinated — LLM returned a CURIE that is not a real / in-scope
                               ontology class (caught by the validity guard)
          * rag_unconfident  — legacy/unspecified (kept as default)
        The chosen identifier is unchanged; only the label differs.
        """
        biosyn_result = self._biosyn_cache.get(text)
        if biosyn_result and biosyn_result.get("curie"):
            return {**biosyn_result, "match_type": match_type, "method": "rag"}
        top_curie, top_label, top_score = candidates[0]
        return {"curie": top_curie, "label": top_label,
                "score": round(top_score, 4),
                "match_type": match_type, "method": "rag"}

    def _resolve(self, text: str, answer_entry: dict, candidates: list) -> dict:
        """Map one LLM JSON entry back to a result dict."""
        curie_map = {c: (lbl, sc) for c, lbl, sc in candidates}
        graph     = getattr(self._biosyn, "graph", {})

        answer = (answer_entry.get("curie") or "").strip()
        label  = (answer_entry.get("label") or "").strip()

        if not answer or answer.upper() == "NONE":
            return self._fallback(text, candidates, match_type="rag_abstained")

        if answer in curie_map:
            lbl, sc = curie_map[answer]
            return {"curie": answer, "label": lbl, "score": round(sc, 4),
                    "match_type": "rag_selected", "method": "rag"}

        if answer in graph:
            lbl = label or graph[answer]["label"]
            return {"curie": answer, "label": lbl, "score": 0.0,
                    "match_type": "rag_graph_selected", "method": "rag"}

        log.debug("RAG: hallucinated CURIE '%s' for '%s'", answer, text)
        top_curie, top_label, top_score = candidates[0]
        return {"curie": top_curie, "label": top_label,
                "score": round(top_score, 4),
                "match_type": "rag_hallucinated", "method": "rag"}

    def run(self, texts: List[str]) -> List[Dict]:
        BATCH_SIZE_LLM = 20

        # Pre-retrieve candidates — batch-encode all unique strings at once
        # (same strategy as BERTNormalizer.run to avoid per-string GPU overhead)
        unique     = list(dict.fromkeys(preprocess(t) for t in texts))
        embeddings = self._biosyn._encode(unique)
        emb_map    = {key: embeddings[i][None] for i, key in enumerate(unique)}

        candidates_map: Dict[str, list] = {}
        for key, emb in emb_map.items():
            self._biosyn.faiss.normalize_L2(emb)
            candidates_map[key] = self._biosyn._retrieve_topk(emb, k=self.topk)

        # Build per-text context using the shared graph traversal helper
        context_map = create_enriched_context(
            getattr(self._biosyn, "graph", {}), texts, candidates_map
        )

        results_map: Dict[str, dict] = {}
        unique_texts = list(dict.fromkeys(texts))

        for batch_start in tqdm(
            range(0, len(unique_texts), BATCH_SIZE_LLM),
            desc="RAG LLM batches",
        ):
            batch = unique_texts[batch_start : batch_start + BATCH_SIZE_LLM]

            enriched_parts: List[str] = []
            for text in batch:
                key   = preprocess(text)
                block = context_map.get(key, "  (none)")
                enriched_parts.append(f'### "{text}"\n{block}')

            user_msg = GRAPH_LLM_PROMPT.format(
                ontology        = self._biosyn.ontology.upper(),
                terms_list      = "\n".join(f'- "{t}"' for t in batch),
                enriched_context= "\n\n".join(enriched_parts),
            )

            llm_json: dict = {}
            try:
                response = self._client.chat.completions.create(
                    model      = self.model,
                    max_tokens = 4096,
                    temperature= 0,        # greedy decoding — deterministic, reproducible
                    seed       = 42,
                    messages   = [{"role": "user", "content": user_msg}],
                )
                raw      = (response.choices[0].message.content or "").strip()
                llm_json = _parse_llm_json(raw)
            except Exception as exc:
                log.warning("RAG: batch LLM call failed: %s — top-1 fallback", exc)

            for text in batch:
                key        = preprocess(text)
                candidates = candidates_map.get(key, [])
                if not candidates:
                    results_map[text] = {"curie": None, "label": None, "score": 0.0,
                                         "match_type": "NIL", "method": "rag"}
                    continue
                entry = llm_json.get(text, {})
                results_map[text] = self._resolve(text, entry, candidates)

        return [results_map[t] for t in texts]


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────

AVAILABLE_METHODS = ["exact", "biosyn", "rag"]


def run_pipeline(
    input_path: str,
    column: str,
    ontology: str,
    obo_path: str,
    output_path: str,
    methods: List[str],
    fuzzy_threshold: float,
    biosyn_batch: int,
    rag_backend: str = "openrouter",
    rag_model: str = None,
    rag_url: str = None,
):
    # ── load input ────────────────────────────────────────────────────────────
    log.info("Loading input: %s", input_path)
    df = pd.read_csv(input_path)
    if column not in df.columns:
        log.error("Column '%s' not found. Available: %s", column, list(df.columns))
        sys.exit(1)

    texts = df[column].fillna("").tolist()
    log.info("Loaded %d rows, column='%s'", len(df), column)

    # ── enforce canonical order: exact → biosyn → rag ────────────────────────
    # If rag is requested, biosyn must also run (index reuse + fallback).
    _ORDER = ["exact", "biosyn", "rag"]
    methods = [m for m in _ORDER if m in methods]
    if "rag" in methods and "biosyn" not in methods:
        log.info("RAG requires BioSyn — auto-adding 'biosyn' before 'rag'.")
        methods = [m for m in _ORDER if m in methods + ["biosyn"]]
    log.info("Execution order: %s", " → ".join(methods))

    # ── run each method — collect raw results ────────────────────────────────
    # Raw results are stored here; cascade logic is applied afterwards.
    _NIL: Dict = {"curie": None, "label": None, "score": 0.0, "match_type": "NIL"}
    raw: Dict[str, List[Dict]] = {}
    biosyn_normalizer = None

    for method in methods:
        t0 = time.time()
        log.info("=" * 60)
        log.info("Running method: %s", method.upper())

        if method == "exact":
            normalizer = ExactSynonymNormalizer(obo_path, fuzzy_threshold, ontology=ontology)
            raw["exact"] = normalizer.run(texts)

        elif method == "biosyn":
            biosyn_normalizer = BERTNormalizer(
                obo_path, ontology,
                batch_size = biosyn_batch,
            )
            raw["biosyn"] = biosyn_normalizer.run(texts)

        elif method == "rag":
            if biosyn_normalizer is None:
                log.info("RAG: BioSyn index not cached — building now …")
                biosyn_normalizer = BERTNormalizer(obo_path, ontology, batch_size=biosyn_batch)
            _defaults = {"openrouter": "stepfun/step-3.5-flash:free", "ollama": "llama3.2"}
            normalizer = RAGNormalizer(
                biosyn_normalizer,
                backend  = rag_backend,
                model    = rag_model or _defaults.get(rag_backend, "llama3.2"),
                base_url = rag_url,
            )
            # Pass BioSyn raw results so RAG falls back to them when unconfident.
            if "biosyn" in raw:
                normalizer._biosyn_cache = {texts[i]: r for i, r in enumerate(raw["biosyn"])}
            raw["rag"] = normalizer.run(texts)

        else:
            log.warning("Unknown method '%s', skipping.", method)
            continue

        elapsed = time.time() - t0
        log.info("%s finished in %.1fs (%.0f ms/query)",
                 method, elapsed, 1000 * elapsed / max(len(texts), 1))

    # ── cascade: exact beats biosyn beats rag ────────────────────────────────
    # Priority for each output column set:
    #   biosyn_* : exact match (if found) → biosyn cosine similarity
    #   rag_*    : exact match (if found) → rag LLM+graph (which already falls
    #              back to biosyn internally when unconfident)
    exact_raw = raw.get("exact", [_NIL] * len(texts))

    _FULL_EXACT = {"exact_pref", "exact_syn", "prefix_truncated"}

    def _cascade(primary_raw: List[Dict], method_name: str) -> List[Dict]:
        out = []
        for i, r in enumerate(primary_raw):
            e = exact_raw[i]
            if e.get("match_type") in _FULL_EXACT:
                out.append({**e, "method": method_name})
            else:
                out.append(r)
        return out

    cascaded: Dict[str, List[Dict]] = {}
    if "exact" in raw:
        cascaded["exact"] = raw["exact"]
    if "biosyn" in raw:
        cascaded["biosyn"] = _cascade(raw["biosyn"], "biosyn")
    if "rag" in raw:
        cascaded["rag"] = _cascade(raw["rag"], "rag")

    # ── guard: never emit a mapping for a blank / empty input term ────────────
    # SapBERT and RAG never abstain, so an empty input string would otherwise be
    # mapped to the nearest embedding neighbour (a spurious identifier, e.g. an
    # empty Cargo -> "radon(0)"). Force such inputs to NIL so they are treated as
    # unmapped downstream and excluded from the knowledge graph.
    _blank = [not str(t).strip() for t in texts]
    if any(_blank):
        log.info("Forcing NIL for %d blank/empty input term(s).", sum(_blank))
        for results in cascaded.values():
            for i, is_blank in enumerate(_blank):
                if is_blank:
                    results[i] = dict(_NIL)

    # ── write cascaded results to dataframe ───────────────────────────────────
    # CURIEs are output in OBO standard format (e.g. "CLO:0001008").
    # If your gold standard uses underscores ("CLO_0001008") normalise at
    # evaluation time: df['col'].str.replace(':', '_', n=1)
    for prefix, results in cascaded.items():
        df[f"{prefix}_curie"]      = [r.get("curie")      for r in results]
        df[f"{prefix}_label"]      = [r.get("label")      for r in results]
        df[f"{prefix}_score"]      = [r.get("score")      for r in results]
        df[f"{prefix}_match_type"] = [r.get("match_type") for r in results]

        nil_count = sum(1 for r in results if r.get("curie") is None)
        log.info("%s (cascaded) hit rate: %.1f%%  (%d NIL / %d total)",
                 prefix, 100 * (1 - nil_count / max(len(texts), 1)),
                    nil_count, len(texts))

    # ── save output ───────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("=" * 60)
    log.info("Output saved: %s  (%d rows, %d columns)", output_path, len(df), len(df.columns))

    # ── quick summary table ───────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────")
    print(f"{'Method':<12} {'Hit Rate':>10}  {'Avg Score':>10}  {'NIL count':>10}")
    print("-" * 50)
    for prefix in cascaded:
        if f"{prefix}_curie" not in df.columns:
            continue
        hits       = df[f"{prefix}_curie"].notna().sum()
        hit_rate   = 100 * hits / len(df)
        avg_score  = df[f"{prefix}_score"].mean()
        nil_count  = len(df) - hits
        print(f"{method:<12} {hit_rate:>9.1f}%  {avg_score:>10.3f}  {nil_count:>10}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Normalize biomedical entities to CLO/CHEBI using three methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",    required=True,  help="Input CSV file path")
    p.add_argument("--column",   required=True,  help="Column name containing entity strings")
    p.add_argument("--ontology", required=True,  choices=["chebi", "clo"],
                    help="Target ontology")
    p.add_argument("--obo",      required=True,
                    help="Path to ontology file (.obo for CHEBI, .owl or .obo for CLO)")
    p.add_argument("--output",   default="results/normalized.csv",
                    help="Output CSV file path")
    p.add_argument("--methods",  nargs="+", default=AVAILABLE_METHODS,
                    choices=AVAILABLE_METHODS,
                    help="Which methods to run (space-separated)")
    p.add_argument("--fuzzy-threshold",  type=float, default=0.92,
                    help="Fuzzy match threshold for exact method (0–1)")
    p.add_argument("--biosyn-batch",     type=int, default=256,
                    help="Batch size for BioSyn encoding")
    p.add_argument("--rag-backend",  default="openrouter",
                    choices=["openrouter", "ollama"],
                    help="LLM backend for RAG method")
    p.add_argument("--rag-model",    default=None,
                    help="Model name for RAG (default: step-3.5-flash for openrouter, "
                        "llama3.2 for ollama)")
    p.add_argument("--rag-url",      default=None,
                    help="Override RAG backend base URL (e.g. remote Ollama host)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_path      = args.input,
        column          = args.column,
        ontology        = args.ontology,
        obo_path        = args.obo,
        output_path     = args.output,
        methods         = args.methods,
        fuzzy_threshold = args.fuzzy_threshold,
        biosyn_batch    = args.biosyn_batch,
        rag_backend     = args.rag_backend,
        rag_model       = args.rag_model,
        rag_url         = args.rag_url,
    )

"""uv run python Normalizer.py \
    --input Gold_standard_CLO.csv \
    --column "Cell Line" \
    --ontology clo \
    --obo /ontology/clo.obo \
    --output Gold_standard_CLO_annotated.csv"""