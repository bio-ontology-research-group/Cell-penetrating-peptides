# -*- coding: utf-8 -*-
"""
CPP Mechanisms Knowledge Graph Dashboard - Flask Version
Run: python3 app.py
"""

import threading
from pathlib import Path
from typing import Optional
from urllib.parse import quote, unquote

import pandas as pd
import requests
from flask import Flask, Response, jsonify, redirect, render_template, request
import traceback
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.url_map.merge_slashes = False

# Primary source: GitHub raw TTL (single request, CDN-backed).
GITHUB_TTL_URL = "https://raw.githubusercontent.com/bio-ontology-research-group/Cell-penetrating-peptides/main/data/Ontology/CPP_KG_materialized.ttl"
# Fallback: Zenodo record JSON API for the canonical versioned TTL file.
# The API endpoint exposes a `files` array with download links.
ZENODO_API_URL = "https://zenodo.org/api/records/21031596"
_HERE = Path(__file__).parent
LOCAL_TTL_CANDIDATES = [
    _HERE / "../data/Ontology/CPP_KG_materialized.ttl",
    _HERE / "../data/Ontology/CPP_KG.ttl",
]

ENTITY_URI_PREFIXES = [
    "https://cppkg.bio2vec.net/dataset/",
]

# ---------------------------------------------------------------------------
# Graph singleton
# ---------------------------------------------------------------------------

_graph: Optional[Graph] = None
_graph_error: Optional[str] = None
_graph_lock = threading.Lock()
_sparql_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Query result cache  (single-flight + background pre-warm)
# ---------------------------------------------------------------------------
# The graph is immutable after loading, so results never change.
# We cache the serialisable Python payload; jsonify() wraps it at call-time.
#
# Single-flight guarantee: if two requests race on a cold key, only one
# thread runs fn(); the other waits on a threading.Event and reads the
# stored value.  This prevents redundant SPARQL query storms on startup.

_cache: dict = {}
_cache_lock   = threading.Lock()
_cache_events: dict = {}          # key → Event, set when value is ready

# Disk-persistence for the query cache so restarts are instant.
_CACHE_FILE = _HERE / ".query_cache.pkl"


def _load_cache_from_disk() -> bool:
    """Load a previously saved cache from disk into _cache.

    Returns True if any entries were loaded, False otherwise.
    """
    import pickle
    if not _CACHE_FILE.exists():
        return False
    try:
        with open(_CACHE_FILE, "rb") as fh:
            data = pickle.load(fh)
        if not isinstance(data, dict):
            return False
        with _cache_lock:
            _cache.update(data)
        print(f"[Cache] Loaded {len(data)} entries from disk.")
        return True
    except Exception as exc:
        print(f"[Cache] Disk cache unreadable ({exc}), will recompute.")
        return False


def _save_cache_to_disk() -> None:
    """Persist the current in-memory cache to disk."""
    import pickle
    with _cache_lock:
        snapshot = dict(_cache)
    try:
        with open(_CACHE_FILE, "wb") as fh:
            pickle.dump(snapshot, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Cache] Saved {len(snapshot)} entries to disk.")
    except Exception as exc:
        print(f"[Cache] Failed to save cache to disk: {exc}")


def _get_cached(key, fn):
    """Return the cached payload for *key*, computing via *fn* on a miss.

    Concurrent callers for the same cold key will block until the first
    caller finishes and stores the result — no duplicate computation.
    """
    # Fast path — no lock needed for a hit after the entry is stored.
    with _cache_lock:
        if key in _cache:
            return _cache[key]
        if key in _cache_events:
            event    = _cache_events[key]
            is_owner = False
        else:
            event              = threading.Event()
            _cache_events[key] = event
            is_owner           = True

    if not is_owner:
        # Wait for the owning thread to finish (2-minute safety timeout).
        event.wait(timeout=120)
        with _cache_lock:
            if key in _cache:
                return _cache[key]
        # Timeout or owner failed — fall through and compute ourselves.
        return fn()

    # This thread owns the computation.
    try:
        result = fn()
        with _cache_lock:
            _cache[key] = result
        return result
    finally:
        # Always unblock waiters, even if fn() raised.
        with _cache_lock:
            _cache_events.pop(key, None)
        event.set()


def _load_graph() -> Graph:
    g = Graph()
    # First try: local files (offline / dev fallback).
    for ttl_path in LOCAL_TTL_CANDIDATES:
        if ttl_path.exists():
            g.parse(str(ttl_path), format="turtle")
            print(f"Loaded TTL from local file: {ttl_path}")
            return g
    # Second try: GitHub raw (single request, CDN-backed — fastest).
    try:
        r = requests.get(GITHUB_TTL_URL, timeout=30)
        r.raise_for_status()
        g.parse(data=r.text, format='turtle')
        print(f"Loaded TTL from GitHub: {GITHUB_TTL_URL}")
        return g
    except Exception as exc:
        print(f"GitHub TTL fetch failed ({exc}), trying Zenodo...")

    # Third try: Zenodo record API (canonical versioned copy).
    try:
        r = requests.get(ZENODO_API_URL, timeout=30)
        r.raise_for_status()
        info = r.json()
        files = info.get('files', []) if isinstance(info, dict) else []
        ttl_link = None
        for f in files:
            fname = (f.get('key') or f.get('filename') or '').lower()
            links = f.get('links', {}) or {}
            candidate = links.get('download') or links.get('self') or links.get('content')
            if fname.endswith('.ttl') and candidate:
                ttl_link = candidate
                break
        if ttl_link:
            rr = requests.get(ttl_link, timeout=60)
            rr.raise_for_status()
            g.parse(data=rr.text, format='turtle')
            print(f"Loaded TTL from Zenodo: {ttl_link}")
            return g
    except Exception as exc:
        print(f"Zenodo TTL fetch failed ({exc}), trying local files...")

    
    

    raise RuntimeError(
        'Failed to load TTL from GitHub, Zenodo, and no local TTL found. '
        'Please place the file at data/Ontology/CPP_KG.ttl.'
    )


def get_graph() -> Graph:
    global _graph, _graph_error
    if _graph is not None:
        return _graph
    with _graph_lock:
        if _graph is None:
            try:
                _graph = _load_graph()
                _graph_error = None
                # Restore persisted cache first; missing keys will be filled
                # by the warmer (which skips keys already present).
                _load_cache_from_disk()
                t = threading.Thread(target=_warm_cache, args=(_graph,),
                                     daemon=True, name="cache-warmer")
                t.start()
            except Exception as exc:
                _graph_error = str(exc)
                raise
    return _graph


# ---------------------------------------------------------------------------
# SPARQL helpers
# ---------------------------------------------------------------------------

def sparql_to_records(g: Graph, query: str):
    with _sparql_lock:
        results = g.query(query)
        columns = [str(v) for v in results.vars]
        rows = [[str(cell) if cell is not None else "" for cell in row] for row in results]
    return columns, rows


def _log_sparql_query(context: str, query: str, **meta):
    meta_parts = [f"{key}={value}" for key, value in meta.items() if value not in (None, "")]
    meta_text = f" [{', '.join(meta_parts)}]" if meta_parts else ""
    print(f"\n[SPARQL DEBUG] {context}{meta_text}\n{query.strip()}\n")


def _escape_sparql_literal(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
             .replace('"', '\\"')
             .replace("\n", "\\n")
             .replace("\r", "\\r")
    )


def _is_uri(val: str) -> bool:
    """Return True if val is a safe absolute URI suitable for SPARQL angle-bracket syntax."""
    return (
        (val.startswith("http://") or val.startswith("https://"))
        and ">" not in val
        and "<" not in val
        and " " not in val
        and "\n" not in val
        and "\r" not in val
    )


def _uri_fragment(uri: str) -> str:
    """Extract the local name from any URI."""
    for sep in ("#", "/"):
        idx = uri.rfind(sep)
        if 0 <= idx < len(uri) - 1:
            return uri[idx + 1:]
    return uri


# ---------------------------------------------------------------------------
# Static data catalogs
# ---------------------------------------------------------------------------

Q_CLASS_COUNTS = """
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?class (COUNT(?instance) AS ?count) (SAMPLE(?lbl) AS ?label)
WHERE {
    ?instance rdf:type ?class .
    OPTIONAL { ?class rdfs:label ?lbl . }
}
GROUP BY ?class
ORDER BY DESC(?count)
"""

_SIO   = "http://semanticscience.org/resource/"
_CPP   = "https://cppkg.bio2vec.net/dataset/"
_OWL   = "http://www.w3.org/2002/07/owl#"
_RDF   = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

ASSOCIATION_CATALOG = [
    {
        "name": "Positively Regulates",
        "description": "Gene interaction with the Uptake mechanism.",
        "query": (
            "PREFIX rdf: <{rdf}> "
            "PREFIX sio: <{sio}> "
            "SELECT (COUNT(DISTINCT ?role) AS ?count) WHERE {{ "
            "  ?gene rdf:type <{sio}SIO_010035> . "
            "  ?gene <{sio}SIO_000008> ?role . "
            "  ?role rdf:type <{sio}SIO_000804> . "
            "  ?role <{sio}SIO_000356> ?proc . "
            "}}"
        ).format(rdf=_RDF, sio=_SIO),
    },
    {
        "name": "Negatively Regulates",
        "description": "Inhibitor interaction with the Uptake mechanism.",
        "query": (
            "PREFIX rdf: <{rdf}> "
            "PREFIX sio: <{sio}> "
            "SELECT (COUNT(DISTINCT ?role) AS ?count) WHERE {{ "
            "  ?inhibitor rdf:type <{sio}SIO_010435> . "
            "  ?inhibitor <{sio}SIO_000008> ?role . "
            "  ?role rdf:type <{sio}SIO_000803> . "
            "  ?role <{sio}SIO_000356> ?proc . "
            "}}"
        ).format(rdf=_RDF, sio=_SIO),
    },
    {
        "name": "Cargo Role: Is Realized In",
        "description": "Functional role of the Cargo realized in the Uptake Mechanism.",
        # HORST-materialized: sio:SIO_000356 is now explicit on the role individual.
        "query": (
            "PREFIX rdf: <{rdf}> "
            "PREFIX sio: <{sio}> "
            "SELECT (COUNT(DISTINCT ?role) AS ?count) WHERE {{ "
            "  ?role rdf:type <{cpp}CargoRole> . "
            "  ?role <{sio}SIO_000356> ?proc . "
            "}}"
        ).format(rdf=_RDF, cpp=_CPP, sio=_SIO),
    },
    {
        "name": "CPP Role: Is Realized In",
        "description": "Functional role of the Cell-penetrating peptide realized in the Uptake Mechanism.",
        # HORST-materialized: sio:SIO_000356 is now explicit on the role individual.
        "query": (
            "PREFIX rdf: <{rdf}> "
            "PREFIX sio: <{sio}> "
            "SELECT (COUNT(DISTINCT ?role) AS ?count) WHERE {{ "
            "  ?role rdf:type <{cpp}CellPenetratingPeptideRole> . "
            "  ?role <{sio}SIO_000356> ?proc . "
            "}}"
        ).format(rdf=_RDF, cpp=_CPP, sio=_SIO),
    },
    {
        "name": "CPP-Complex: Is Participant In",
        "description": "CPP-Complex interaction with a specific the uptake mechanism.",
        # Uses full URI for CPP-Complex to avoid rdflib SPARQL lexer issues with hyphens.
        "query": (
            "PREFIX rdf: <{rdf}> "
            "SELECT (COUNT(DISTINCT ?s) AS ?count) WHERE {{ "
            "  ?s rdf:type <{cpp}CPP-Complex> . "
            "  ?s <{sio}SIO_000062> ?o . "
            "}}"
        ).format(rdf=_RDF, cpp=_CPP, sio=_SIO),
    },
    {
        "name": "Cell Line: Is Participant In",
        "description": "Cell Line interaction with a specific the uptake mechanism.",
        "query": (
            "PREFIX rdf: <{rdf}> "
            "SELECT (COUNT(DISTINCT ?s) AS ?count) WHERE {{ "
            "  ?s rdf:type <{sio}SIO_010054> . "
            "  ?s <{sio}SIO_000062> ?o . "
            "}}"
        ).format(rdf=_RDF, sio=_SIO),
    },
    {
        "name": "Subcellular Delivery Localization",
        "description": "CPP-Complex interaction with Subcellular Entity.",
        # HORST-materialized: sio:SIO_000061 is now explicit on the CPP-Complex individual.
        "query": (
            "PREFIX rdf: <{rdf}> "
            "PREFIX sio: <{sio}> "
            "SELECT (COUNT(DISTINCT ?s) AS ?count) WHERE {{ "
            "  ?s rdf:type <{cpp}CPP-Complex> . "
            "  ?s <{sio}SIO_000061> ?loc . "
            "}}"
        ).format(rdf=_RDF, cpp=_CPP, sio=_SIO),
    },
]

CLASS_CATALOG = {
    "http://semanticscience.org/resource/SIO_010035": {
        "name": "Gene",
        "description": "Upregulator genes that positively regulates the endocytic uptake mechanism.",
    },
    "http://semanticscience.org/resource/SIO_010435": {
        "name": "Inhibitor",
        "description": "A chemical compound used experimentally to selectively block an endocytic uptake mechanism.",
    },
    "https://cppkg.bio2vec.net/schema#UptakeMechanism": {
        "name": "Uptake Mechanism",
        "description": "A cellular route or process - endocytic or non-endocytic - through which the CPP-complex crosses the cell membrane.",
    },
    "https://cppkg.bio2vec.net/dataset/CPP-Complex": {
        "name": "CPP-Complex",
        "description": "A molecular assembly formed between a cell-penetrating peptide and its associated cargo molecule.",
    },
    "https://cppkg.bio2vec.net/dataset/CellPenetratingPeptide": {
        "name": "Cell-Penetrating Peptide",
        "description": "A short amino acid sequence with intrinsic ability to traverse biological membranes.",
    },
    "https://cppkg.bio2vec.net/dataset/Cargo": {
        "name": "Cargo",
        "description": "A therapeutic or reporter molecule transported into cells through its association with a CPP.",
    },
    "http://semanticscience.org/resource/SIO_010054": {
        "name": "Cell Line",
        "description": "The cultured cell population serving as the experimental host model for studying CPP uptake.",
    },
    "http://semanticscience.org/resource/SIO_001400": {
        "name": "Subcellular Entity",
        "description": "An organelle or intracellular compartment in which internalized CPPs or cargo are observed to be delivered.",
    },
    "http://semanticscience.org/resource/SIO_000994": {
        "name": "Experiment",
        "description": "The experimental validation for the CPP-complex internalization activity in a cell model.",
    },
    "http://semanticscience.org/resource/SIO_000148": {
        "name": "Document",
        "description": "The scientific article or report that evidences the experimental validation.",
    },
}

# ---------------------------------------------------------------------------
# Per-field SPARQL pattern builders
# Each builder accepts an already-escaped term and returns a WHERE fragment
# with unique variable names (no regex rewriting required).
# ---------------------------------------------------------------------------

def _pattern_sequence(val: str) -> str:
    # Sequence lives directly on the peptide — filter immediately after binding it.
    esc = _escape_sparql_literal(val)
    return (
        f'    ?peptide cppS:sequence ?seq_sq .\n'
        f'    FILTER(CONTAINS(LCASE(STR(?seq_sq)), LCASE("{esc}")))\n'
    )


def _pattern_mechanism(val: str) -> str:
    # HORST-materialized graph: sio:SIO_000356 is now explicit on the CPPRole individual
    # (was hidden inside a blank-node OWL restriction).
    if _is_uri(val):
        return (
            f'    ?cppRole_mec sio:SIO_000356 <{val}> ;\n'
            f'                 rdf:type cpp:CellPenetratingPeptideRole .\n'
            f'    ?peptide sio:SIO_000008 ?cppRole_mec .\n'
        )
    # Text path: filter on mechanism label, then walk to peptide via explicit triple.
    esc = _escape_sparql_literal(val)
    return (
        f'    ?mech_mec rdfs:label ?mechLbl_mec .\n'
        f'    FILTER(CONTAINS(LCASE(STR(?mechLbl_mec)), LCASE("{esc}")))\n'
        f'    ?cppRole_mec sio:SIO_000356 ?mech_mec ;\n'
        f'                 rdf:type cpp:CellPenetratingPeptideRole .\n'
        f'    ?peptide sio:SIO_000008 ?cppRole_mec .\n'
    )


def _pattern_subcell(val: str) -> str:
    # HORST-materialized graph: sio:SIO_000061 is now explicit on the CPP-Complex individual
    # (was hidden inside a blank-node OWL restriction).
    if _is_uri(val):
        return (
            f'    ?cplx_sub sio:SIO_000061 <{val}> .\n'
            f'    ?peptide sio:SIO_000313 ?cplx_sub ;\n'
            f'             rdf:type cpp:CellPenetratingPeptide .\n'
        )
    # Text path: filter subcellular label, then walk to peptide via explicit triple.
    esc = _escape_sparql_literal(val)
    return (
        f'    ?subcellNode_sub rdfs:label ?subcellLbl_sub .\n'
        f'    FILTER(CONTAINS(LCASE(STR(?subcellLbl_sub)), LCASE("{esc}")))\n'
        f'    ?cplx_sub sio:SIO_000061 ?subcellNode_sub .\n'
        f'    ?peptide sio:SIO_000313 ?cplx_sub ;\n'
        f'             rdf:type cpp:CellPenetratingPeptide .\n'
    )


def _pattern_cell_line(val: str) -> str:
    # URI path: anchor on the known cell-line URI inside the experiment, walk to peptide.
    if _is_uri(val):
        return (
            f'    ?experiment_cl sio:SIO_000132 <{val}> .\n'
            f'    ?cplx_cl sio:SIO_000062 ?experiment_cl .\n'
            f'    ?peptide sio:SIO_000313 ?cplx_cl ;\n'
            f'             rdf:type cpp:CellPenetratingPeptide .\n'
        )
    # Text path: filter cell-line label first, then follow the participation chain.
    esc = _escape_sparql_literal(val)
    return (
        f'    ?cell_line_cl rdfs:label ?cell_line_label_cl .\n'
        f'    FILTER(CONTAINS(LCASE(STR(?cell_line_label_cl)), LCASE("{esc}")))\n'
        f'    ?experiment_cl sio:SIO_000132 ?cell_line_cl .\n'
        f'    ?cplx_cl sio:SIO_000062 ?experiment_cl .\n'
        f'    ?peptide sio:SIO_000313 ?cplx_cl .\n'
    )


def _pattern_cargo(val: str) -> str:
    # Both cargo and peptide share SIO_000313 → cpp_complex, so without the type
    # constraint ?peptide would match the cargo itself.  The cpp:CellPenetratingPeptide
    # type filter ensures only peptide individuals are returned.
    if _is_uri(val):
        return (
            f'    <{val}> sio:SIO_000313 ?cplx_cg .\n'
            f'    ?peptide sio:SIO_000313 ?cplx_cg ;\n'
            f'             rdf:type cpp:CellPenetratingPeptide .\n'
        )
    esc = _escape_sparql_literal(val)
    return (
        f'    ?cargoNode_cg cppS:cargoType ?cargoType_cg .\n'
        f'    FILTER(CONTAINS(LCASE(STR(?cargoType_cg)), LCASE("{esc}")))\n'
        f'    ?cargoNode_cg sio:SIO_000313 ?cplx_cg .\n'
        f'    ?peptide sio:SIO_000313 ?cplx_cg ;\n'
        f'             rdf:type cpp:CellPenetratingPeptide .\n'
    )


def _pattern_doc_id(val: str) -> str:
    # Filter on document URI/literal first, then walk back to peptide.
    esc = _escape_sparql_literal(val)
    return (
        f'    ?experiment_doc sio:SIO_000557 ?document_doc .\n'
        f'    FILTER(CONTAINS(LCASE(STR(?document_doc)), LCASE("{esc}")))\n'
        f'    ?cplx_doc sio:SIO_000062 ?experiment_doc .\n'
        f'    ?peptide sio:SIO_000313 ?cplx_doc ;\n'
        f'             rdf:type cpp:CellPenetratingPeptide .\n'
    )

_FIELD_BUILDERS = {
    "sequence":  _pattern_sequence,
    "mechanism": _pattern_mechanism,
    "subcell":   _pattern_subcell,
    "cell_line": _pattern_cell_line,
    "cargo":     _pattern_cargo,
    "doc_id":    _pattern_doc_id,
}

# ---------------------------------------------------------------------------
# Module-level compute functions  (shared by endpoints + cache pre-warmer)
# ---------------------------------------------------------------------------

# Classes whose members are declared as rdfs:subClassOf (not rdf:type).
_SUBCLASS_COUNTED_URIS = [
    "https://cppkg.bio2vec.net/schema#UptakeMechanism",
    "http://semanticscience.org/resource/SIO_001400",
]
_SUBCLASS_BROWSE_URIS = set(_SUBCLASS_COUNTED_URIS)


def _compute_metrics(g: Graph) -> dict:
    subjects = set(g.subjects())
    objects  = set(g.objects())
    return {
        "total_triples":    len(g),
        "total_nodes":      len(subjects | objects),
        "total_predicates": len(set(g.predicates())),
    }


def _compute_classes(g: Graph) -> list:
    cols, rows = sparql_to_records(g, Q_CLASS_COUNTS)
    df = pd.DataFrame(rows, columns=cols)
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    count_map = {r["class"]: (int(r["count"]) if pd.notna(r["count"]) else 0)
                 for _, r in df.iterrows()}

    for parent_uri in _SUBCLASS_COUNTED_URIS:
        q = (
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            "SELECT (COUNT(DISTINCT ?sub) AS ?count) WHERE {\n"
            "    ?sub rdfs:subClassOf <%s> .\n"
            "}" % parent_uri
        )
        with _sparql_lock:
            res = list(g.query(q))
        if res and res[0][0] is not None:
            count_map[parent_uri] = int(str(res[0][0]))

    result = [
        {"name": e["name"], "description": e["description"],
         "uri": uri, "count": count_map.get(uri, 0)}
        for uri, e in CLASS_CATALOG.items()
    ]
    result.sort(key=lambda x: x["count"], reverse=True)
    return result


def _compute_associations(g: Graph) -> list:
    rows = []
    for entry in ASSOCIATION_CATALOG:
        _log_sparql_query("/api/associations", entry["query"],
                          association=entry["name"])
        try:
            with _sparql_lock:
                res = list(g.query(entry["query"]))
            count = int(str(res[0][0])) if res and res[0][0] is not None else 0
        except Exception as exc:
            traceback.print_exc()
            rows.append({"name": entry["name"], "description": entry["description"],
                         "count": 0, "error": str(exc)})
            continue
        rows.append({"name": entry["name"], "description": entry["description"],
                     "count": count})
    return rows


def _compute_browse_page(g: Graph, class_uri: str,
                         page: int, per_page: int, filter_text: str) -> dict:
    pred = "rdfs:subClassOf" if class_uri in _SUBCLASS_BROWSE_URIS else "rdf:type"
    flt  = ""
    if filter_text:
        esc = _escape_sparql_literal(filter_text)
        flt = ('FILTER(CONTAINS(LCASE(COALESCE(STR(?label), STR(?entity))), '
               'LCASE("{}")))'.format(esc))

    count_q = """
    PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT (COUNT(DISTINCT ?entity) AS ?total) WHERE {{
      ?entity {pred} <{cls}> .
      OPTIONAL {{ ?entity rdfs:label ?label . }}
      {flt}
    }}
    """.format(pred=pred, cls=class_uri, flt=flt)
    _log_sparql_query("/api/browse count", count_q, class_uri=class_uri,
                      page=page, per_page=per_page, filter=filter_text)
    with _sparql_lock:
        count_res = list(g.query(count_q))
    total = int(str(count_res[0][0])) if count_res and count_res[0][0] else 0

    offset = (page - 1) * per_page
    page_q = """
    PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX cppS: <https://cppkg.bio2vec.net/schema#>
    SELECT DISTINCT ?entity (SAMPLE(?lbl) AS ?label) (SAMPLE(?seq) AS ?sequence) WHERE {{
      ?entity {pred} <{cls}> .
      OPTIONAL {{ ?entity rdfs:label ?lbl . }}
      OPTIONAL {{ ?entity cppS:sequence ?seq . }}
      {flt}
    }}
    GROUP BY ?entity
    ORDER BY ?label ?entity
    LIMIT {lim} OFFSET {off}
    """.format(pred=pred, cls=class_uri, flt=flt, lim=per_page, off=offset)
    _log_sparql_query("/api/browse page", page_q, class_uri=class_uri,
                      page=page, per_page=per_page, filter=filter_text,
                      offset=offset)
    _, rows = sparql_to_records(g, page_q)

    items = [
        {"uri": r[0], "local_id": _uri_fragment(r[0]),
         "label": r[1] or _uri_fragment(r[0]), "extra": r[2]}
        for r in rows
    ]
    return {"items": items, "total": total, "page": page,
            "per_page": per_page, "pages": max(1, (total + per_page - 1) // per_page)}


def _warm_cache(g: Graph) -> None:
    """Pre-populate the cache immediately after graph load.

    Runs in a daemon thread so the server stays responsive.  By the time the
    first browser tab opens, every home-page query and browse page-1 for each
    class will already be cached.
    """
    import time
    t0 = time.time()
    print("[Cache] Pre-warming …")

    for label, key, fn in [
        ("metrics",      "metrics",      lambda: _compute_metrics(g)),
        ("classes",      "classes",      lambda: _compute_classes(g)),
        ("associations", "associations", lambda: _compute_associations(g)),
    ]:
        try:
            _get_cached(key, fn)
            print(f"[Cache]   {label} ✓")
        except Exception as exc:
            print(f"[Cache]   {label} FAILED: {exc}")

    for uri in CLASS_CATALOG:
        key = ("browse", uri, 1, 50, "")
        try:
            _get_cached(key, lambda u=uri: _compute_browse_page(g, u, 1, 50, ""))
            print(f"[Cache]   browse/{uri.rsplit('/', 1)[-1]} ✓")
        except Exception as exc:
            print(f"[Cache]   browse/{uri} FAILED: {exc}")

    print(f"[Cache] Pre-warm done in {time.time() - t0:.1f}s")
    _save_cache_to_disk()


def _compose_candidate_query(filters: dict):
    """Build the candidate SPARQL query from a validated filters dict.

    Returns (candidate_query_str, candidate_where_str).
    Returns (None, None) when no active filter terms are present.
    Raises ValueError for unknown field keys.
    """
    parts = []
    for key, raw_term in filters.items():
        term = (raw_term or "").strip()
        if not term:
            continue
        builder = _FIELD_BUILDERS.get(key)
        if builder is None:
            raise ValueError(f"Unknown search field: {key}")
        parts.append(builder(term))

    if not parts:
        return None, None

    candidate_where = "\n".join(parts)
    candidate_query = CANDIDATE_QUERY.replace(_CANDIDATE_WHERE_TOKEN, candidate_where)
    return candidate_query, candidate_where

MAX_MATCHED_PEPTIDES = 250

_CANDIDATE_WHERE_TOKEN = "##CANDIDATE_WHERE##"

CANDIDATE_QUERY = (
    "PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "PREFIX owl:  <http://www.w3.org/2002/07/owl#>\n"
    "PREFIX sio:  <http://semanticscience.org/resource/>\n"
    "PREFIX cpp:  <https://cppkg.bio2vec.net/dataset/>\n"
    "PREFIX cppS: <https://cppkg.bio2vec.net/schema#>\n"
    "SELECT DISTINCT ?peptide\n"
    "WHERE {\n"
    "    "+ _CANDIDATE_WHERE_TOKEN + "\n"
    "}\n"
    "LIMIT " + str(MAX_MATCHED_PEPTIDES) + "\n"
)


# ---------------------------------------------------------------------------
# Autocomplete queries — return (uri, label) pairs filtered by typed text.
# Used by the frontend dropdowns for mechanism / subcell / cell_line / cargo.
# ---------------------------------------------------------------------------

_AUTOCOMPLETE_QUERIES = {
    "mechanism": (
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX owl:  <http://www.w3.org/2002/07/owl#>\n"
        "PREFIX sio:  <http://semanticscience.org/resource/>\n"
        "PREFIX cppS: <https://cppkg.bio2vec.net/schema#>\n"
        "SELECT ?uri (SAMPLE(?rawLabel) AS ?label) WHERE {{\n"
        "  ?r owl:onProperty sio:SIO_000356 .\n"
        "  ?r owl:someValuesFrom ?uri .\n"
        "  ?uri rdfs:subClassOf cppS:UptakeMechanism .\n"
        "  ?uri rdfs:label ?rawLabel .\n"
        "{filter}"
        "}} GROUP BY ?uri ORDER BY LCASE(STR(?label)) LIMIT 250\n"
    ),
    "subcell": (
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX owl:  <http://www.w3.org/2002/07/owl#>\n"
        "PREFIX sio:  <http://semanticscience.org/resource/>\n"
        "SELECT ?uri (SAMPLE(?rawLabel) AS ?label) WHERE {{\n"
        "  ?r owl:onProperty sio:SIO_000061 .\n"
        "  ?r owl:someValuesFrom ?uri .\n"
        "  ?uri rdfs:label ?rawLabel .\n"
        "{filter}"
        "}} GROUP BY ?uri ORDER BY LCASE(STR(?label)) LIMIT 250\n"
    ),
    "cell_line": (
        "PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX sio:  <http://semanticscience.org/resource/>\n"
        "SELECT ?uri (SAMPLE(?rawLabel) AS ?label) WHERE {{\n"
        "  ?uri rdf:type sio:SIO_010054 .\n"
        "  ?uri rdfs:label ?rawLabel .\n"
        "{filter}"
        "}} GROUP BY ?uri ORDER BY LCASE(STR(?label))\n"
    ),
    "cargo": (
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX sio:  <http://semanticscience.org/resource/>\n"
        "PREFIX cppS: <https://cppkg.bio2vec.net/schema#>\n"
        "SELECT ?uri (SAMPLE(?rawLabel) AS ?label) WHERE {{\n"
        "  ?cplx sio:SIO_000369 ?uri .\n"
        "  ?uri cppS:cargoType ?rawLabel .\n"
        "  FILTER(STRSTARTS(STR(?uri), \"http://purl.obolibrary.org/obo/CHEBI\"))\n"
        "{filter}"
        "}} GROUP BY ?uri ORDER BY LCASE(STR(?label))\n"
    ),
}

_AUTOCOMPLETE_FILTER_VARS = {
    "mechanism": "rawLabel",
    "subcell": "rawLabel",
    "cell_line": "rawLabel",
    "cargo": "rawLabel",
}


# ---------------------------------------------------------------------------
# Routes - existing
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    try:
        g = get_graph()
        return jsonify({"status": "ready", "triples": len(g)})
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/metrics")
def api_metrics():
    try:
        g = get_graph()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    try:
        return jsonify(_get_cached("metrics", lambda: _compute_metrics(g)))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/classes")
def api_classes():
    try:
        g = get_graph()
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    try:
        return jsonify(_get_cached("classes", lambda: _compute_classes(g)))
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"items": [], "error": str(exc)}), 200


@app.route("/api/associations")
def api_associations():
    try:
        g = get_graph()
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    try:
        return jsonify(_get_cached("associations", lambda: _compute_associations(g)))
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json() or {}
    filters = data.get("filters") if isinstance(data.get("filters"), dict) else None

    if not filters or not any((str(v) or "").strip() for v in filters.values()):
        return jsonify({"error": "At least one filter must be provided."}), 400

    for k in list(filters.keys()):
        if k not in _FIELD_BUILDERS:
            return jsonify({"error": f"Unknown search field: {k}"}), 400

    try:
        candidate_query, _ = _compose_candidate_query(filters)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not candidate_query:
        return jsonify({"rows": [], "columns": [], "warnings": []})

    try:
        g = get_graph()
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    # Cache key: sorted filter items so order doesn't matter.
    cache_key = ("search", tuple(sorted(
        (k, (str(v) or "").strip()) for k, v in filters.items()
        if (str(v) or "").strip()
    )))

    def _compute():
        try:
            with _sparql_lock:
                candidate_rows = list(g.query(candidate_query))
        except Exception as qexc:
            traceback.print_exc()
            return {"_error": "Candidate query failed", "debug": str(qexc)}

        peptides = [str(getattr(r, 'peptide', r[0])) for r in candidate_rows]

        if not peptides:
            return {"items": [], "matched_peptides": 0, "warnings": [],
                    "debug": {"candidate_query": candidate_query}}

        values_block = " ".join("<{}>".format(uri) for uri in peptides)
        basics_q = (
            "PREFIX cppS: <https://cppkg.bio2vec.net/schema#>\n"
            "SELECT ?peptide ?pepName ?sequence WHERE {\n"
            "  VALUES ?peptide { " + values_block + " }\n"
            "  OPTIONAL { ?peptide cppS:peptideName ?pepName . }\n"
            "  OPTIONAL { ?peptide cppS:sequence ?sequence . }\n"
            "}\n"
        )
        try:
            with _sparql_lock:
                basics_rows = list(g.query(basics_q))
        except Exception as qexc:
            traceback.print_exc()
            return {"_error": "Basics query failed", "debug": str(qexc)}

        basics = {str(r[0]): (str(r[1]) if r[1] else "", str(r[2]) if r[2] else "")
                  for r in basics_rows}

        items = []
        for uri in peptides:
            pep_name, sequence = basics.get(uri, ("", ""))
            local_id = _uri_fragment(uri)
            items.append({
                "uri":      uri,
                "local_id": local_id,
                "label":    pep_name or local_id,
                "extra":    sequence,
            })

        warnings = []
        if len(peptides) >= MAX_MATCHED_PEPTIDES:
            warnings.append(
                f"Matched at least {MAX_MATCHED_PEPTIDES} peptides. "
                "Refine the search term for faster results."
            )
        return {
            "items":            items,
            "matched_peptides": len(items),
            "warnings":         warnings,
            "debug":            {"candidate_query": candidate_query},
        }

    try:
        result = _get_cached(cache_key, _compute)
        if "_error" in result:
            return jsonify({"error": result["_error"], "debug": result.get("debug")}), 500
        return jsonify(result)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/search/preview", methods=["POST"])
def api_search_preview():
    """Compose and return the SPARQL queries without executing them."""
    data = request.get_json() or {}
    filters = data.get("filters") if isinstance(data.get("filters"), dict) else None

    if not filters or not any((str(v) or "").strip() for v in filters.values()):
        return jsonify({"error": "At least one filter must be provided."}), 400

    for k in list(filters.keys()):
        if k not in _FIELD_BUILDERS:
            return jsonify({"error": f"Unknown search field: {k}"}), 400

    try:
        candidate_query, _ = _compose_candidate_query(filters)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"candidate_query": candidate_query or ""})


@app.route("/api/autocomplete/<field>")
def api_autocomplete(field):
    """Return available (uri, label, id) options for the given dropdown field."""
    if field not in _AUTOCOMPLETE_QUERIES:
        return jsonify([]), 400
    q_term = request.args.get("q", "").strip()
    filter_clause = ""
    if q_term:
        esc = _escape_sparql_literal(q_term)
        filter_var = _AUTOCOMPLETE_FILTER_VARS.get(field, "label")
        filter_clause = f'  FILTER(CONTAINS(LCASE(STR(?{filter_var})), LCASE("{esc}")))\n'
    query = _AUTOCOMPLETE_QUERIES[field].format(filter=filter_clause)
    try:
        g = get_graph()
        _, rows = sparql_to_records(g, query)
        results = []
        for row in rows:
            uri, label = row[0], row[1]
            if not uri or not label:
                continue
            local = _uri_fragment(uri).replace("_", ":")
            results.append({"uri": uri, "label": label, "id": local})
        return jsonify(results)
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/sparql", methods=["POST"])
def api_sparql():
    data = request.get_json()
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Query is required."}), 400

    try:
        g = get_graph()
        cols, rows = sparql_to_records(g, query)
        return jsonify({"columns": cols, "rows": rows})
    except SystemExit:
        return jsonify({"error": "Query timed out. Simplify your query or add a LIMIT clause."}), 504
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Routes - new: Linked Data
# ---------------------------------------------------------------------------

@app.route("/robots.txt")
def robots_txt():
    host = request.host_url.rstrip("/")
    content = "User-agent: *\nAllow: /\nSitemap: {}/sitemap.xml\n".format(host)
    return Response(content, mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap_xml():
    try:
        g = get_graph()
        q = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT DISTINCT ?entity WHERE {
          ?entity rdf:type ?type .
          #FILTER(STRSTARTS(STR(?entity), "https://cppkg.bio2vec.net/dataset/"))
        }
        LIMIT 5000
        """
        base = request.host_url.rstrip("/")
        urls = []
        with _sparql_lock:
            sitemap_rows = list(g.query(q))
        for row in sitemap_rows:
            frag = _uri_fragment(str(row[0]))
            if frag:
                urls.append("  <url><loc>{}/dataset/{}</loc></url>".format(base, frag))
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            + "\n".join(urls)
            + "\n</urlset>"
        )
        return Response(xml, mimetype="application/xml")
    except Exception:
        return Response(
            '<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"/>',
            mimetype="application/xml",
        )


@app.route("/dataset/")
def dataset_void():
    accept = request.headers.get("Accept", "text/html")
    if "text/turtle" in accept or "application/rdf+xml" in accept:
        void = _build_void_turtle()
        return Response(void, mimetype="text/turtle")
    return render_template("index.html")


@app.route("/dataset/<path:entity_id>")
def dataset_entity(entity_id):
    entity_id = _fix_collapsed_scheme(unquote(entity_id))
    accept = request.headers.get("Accept", "text/html")
    if "text/turtle" in accept:
        return _entity_as_turtle(entity_id)
    if "application/ld+json" in accept:
        return _entity_as_jsonld(entity_id)
    # 303 See Other for browsers.
    # Reconstruct the full entity URI and percent-encode it so the hash router
    # receives the complete URI without slashes fragmenting the section/sub split.
    if entity_id.startswith("http"):
        entity_uri = entity_id
    else:
        entity_uri = ENTITY_URI_PREFIXES[0] + entity_id
    resp = redirect("/dataset/#entity/{}".format(quote(entity_uri, safe="")), 303)
    return resp


# ---------------------------------------------------------------------------
# Routes - new: API
# ---------------------------------------------------------------------------

def _fix_collapsed_scheme(entity_id: str) -> str:
    """Reverse proxies often collapse https:// → https:/ in URL paths.
    Restore the correct double-slash so URI lookups succeed."""
    if entity_id.startswith("https:/") and not entity_id.startswith("https://"):
        return "https://" + entity_id[7:]
    if entity_id.startswith("http:/") and not entity_id.startswith("http://"):
        return "http://" + entity_id[6:]
    return entity_id


@app.route("/api/entity/<path:entity_id>")
def api_entity(entity_id):
    entity_id = _fix_collapsed_scheme(unquote(entity_id))
    try:
        g = get_graph()
        uri = _resolve_entity_uri(g, entity_id)
        if uri is None:
            return jsonify({"error": "Entity not found: {}".format(entity_id)}), 404

        # Flat property list: only named URIs and literals as objects.
        # Blank nodes (OWL restrictions, anonymous classes) are excluded entirely.
        props_q = """
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX owl:  <http://www.w3.org/2002/07/owl#>
        SELECT ?pred (SAMPLE(?predLbl) AS ?predLabel) ?obj (SAMPLE(?objLbl) AS ?objLabel) WHERE {{
          <{uri}> ?pred ?obj .
          FILTER(!isBlank(?obj))
          FILTER(?pred != owl:sameAs)
          FILTER(?pred != rdf:type)
          FILTER(?obj NOT IN (owl:NamedIndividual, owl:Class, owl:Ontology))
          OPTIONAL {{ ?pred rdfs:label ?predLbl . }}
          OPTIONAL {{ ?obj  rdfs:label ?objLbl  . }}
        }}
        GROUP BY ?pred ?obj
        ORDER BY ?predLabel ?pred ?objLabel ?obj
        LIMIT 400
        """.format(uri=uri)
        _, prop_rows = sparql_to_records(g, props_q)

        # Build structured property list with human-readable display values.
        _IDENTIFIER_PREDS = {
            "http://www.w3.org/2000/01/rdf-schema#label",
            "http://www.w3.org/2000/01/rdf-schema#comment",
            "http://www.w3.org/2004/02/skos/core#prefLabel",
            "http://www.w3.org/2004/02/skos/core#altLabel",
            "http://purl.org/dc/elements/1.1/title",
            "http://purl.org/dc/terms/title",
            "http://purl.org/dc/elements/1.1/identifier",
            "http://purl.org/dc/terms/identifier",
            "http://semanticscience.org/resource/SIO_000116",  # has name
        }
        _RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

        def _prop_order(pred):
            if pred in _IDENTIFIER_PREDS:
                return 0   # identifiers first
            return 1       # domain metadata

        properties = []
        for pred, pred_label, obj, obj_label in prop_rows:
            pred_display = pred_label or _uri_fragment(pred)
            if obj.startswith("http"):
                obj_display = obj_label or _uri_fragment(obj)
                obj_type = "uri"
            else:
                obj_display = obj
                obj_type = "literal"
            properties.append({
                "predicate":       pred,
                "predicate_label": pred_display,
                "object":          obj,
                "object_label":    obj_display,
                "object_type":     obj_type,
            })
        properties.sort(key=lambda p: (_prop_order(p["predicate"]), p["predicate_label"], p["object_label"]))

        graph_data = _get_neighborhood(g, uri)

        return jsonify({
            "uri":        uri,
            "entity_id":  entity_id,
            "properties": properties,
            "graph":      graph_data,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/browse")
def api_browse():
    class_uri   = request.args.get("class", "").strip()
    filter_text = request.args.get("filter", "").strip()
    try:
        page     = max(1, int(request.args.get("page", 1)))
        per_page = min(100, max(10, int(request.args.get("per_page", 50))))
    except (ValueError, TypeError):
        page, per_page = 1, 50

    if not class_uri:
        return jsonify({"error": "class parameter required"}), 400

    try:
        g = get_graph()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    cache_key = ("browse", class_uri, page, per_page, filter_text)
    try:
        return jsonify(_get_cached(
            cache_key,
            lambda: _compute_browse_page(g, class_uri, page, per_page, filter_text),
        ))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500



@app.route("/api/smiles/svg")
def api_smiles_svg():
    smiles = request.args.get("smi", "").strip()
    if not smiles:
        return Response("Missing smi", status=400)
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return Response("Invalid SMILES", status=400)
        drawer = rdMolDraw2D.MolDraw2DSVG(400, 220)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return Response(svg, mimetype="image/svg+xml",
                        headers={"Cache-Control": "public, max-age=86400"})
    except Exception as exc:
        return Response(str(exc), status=500)


@app.route("/api/download/ttl")
def api_download_ttl():
    return redirect("https://zenodo.org/records/21031596/files/CPP_KG.ttl?download=1", code=302)


@app.route("/api/download/jsonld")
def api_download_jsonld():
    try:
        g = get_graph()
        try:
            jld = g.serialize(format="json-ld", indent=2)
        except Exception:
            return jsonify({"error": "JSON-LD serialization requires pyld. Run: pip3 install pyld"}), 500
        return Response(
            jld,
            mimetype="application/ld+json",
            headers={"Content-Disposition": "attachment; filename=CPP_KG.jsonld"},
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _resolve_entity_uri(g: Graph, entity_id: str) -> Optional[str]:
    """Find the full URI for an entity given its local fragment."""
    for prefix in ENTITY_URI_PREFIXES:
        uri = prefix + entity_id
        ref = URIRef(uri)
        if next(g.triples((ref, None, None)), None) is not None:
            return uri
        if next(g.triples((None, None, ref)), None) is not None:
            return uri
    # Try treating entity_id as a full URI
    if entity_id.startswith("http"):
        ref = URIRef(entity_id)
        if (next(g.triples((ref, None, None)), None) is not None or
                next(g.triples((None, None, ref)), None) is not None):
            return entity_id
    return None


def _get_neighborhood(g: Graph, uri: str, max_edges: int = 60):
    """Build a Cytoscape-ready node/edge payload for the immediate neighborhood."""
    nodes = {}
    edges = []
    center = URIRef(uri)

    def get_label(ref):
        lbl = next(g.objects(ref, RDFS.label), None)
        return str(lbl) if lbl else _uri_fragment(str(ref))

    def get_type(ref):
        rdf_type = next(g.objects(ref, RDF.type), None)
        return _uri_fragment(str(rdf_type)) if rdf_type else "other"

    def get_predicate_label(ref):
        lbl = next(g.objects(ref, RDFS.label), None)
        return str(lbl) if lbl else _uri_fragment(str(ref))

    def add_node(u_str, is_focus=False):
        if u_str not in nodes:
            ref = URIRef(u_str)
            nodes[u_str] = {
                "data": {
                    "id":       u_str,
                    "label":    get_label(ref),
                    "entityId": _uri_fragment(u_str),
                    "type":     get_type(ref),
                    "focus":    is_focus,
                }
            }

    add_node(uri, is_focus=True)
    edge_count = 0

    _OWL = "http://www.w3.org/2002/07/owl#"
    _RDF_TYPE = RDF.type

    # Outgoing edges
    for pred, obj in g.predicate_objects(center):
        if edge_count >= max_edges:
            break
        if not isinstance(obj, URIRef):
            continue
        if str(obj).startswith(_OWL):
            continue
        pred_label = get_predicate_label(pred)[:30]
        obj_str = str(obj)
        add_node(obj_str)
        edges.append({
            "data": {
                "id":     "e{}".format(edge_count),
                "source": uri,
                "target": obj_str,
                "label":  pred_label,
            }
        })
        edge_count += 1

    # Incoming edges (capped)
    for subj, pred in list(g.subject_predicates(center))[:20]:
        if edge_count >= max_edges:
            break
        pred_label = get_predicate_label(pred)[:30]
        subj_str = str(subj)
        add_node(subj_str)
        edges.append({
            "data": {
                "id":     "e{}".format(edge_count),
                "source": subj_str,
                "target": uri,
                "label":  pred_label,
            }
        })
        edge_count += 1

    return {"nodes": list(nodes.values()), "edges": edges}


def _build_void_turtle() -> str:
    return """@prefix void:    <http://rdfs.org/ns/void#> .
@prefix dcterms: <http://purl.org/dc/terms/> .
@prefix foaf:    <http://xmlns.com/foaf/0.1/> .
@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .

<https://cppkg.bio2vec.net/dataset/>
    a void:Dataset ;
    dcterms:title "CPP Mechanisms Knowledge Graph"@en ;
    dcterms:description "A structured RDF Knowledge Graph encoding endocytic and non-endocytic uptake mechanisms for Cell-Penetrating Peptides."@en ;
    dcterms:license <https://creativecommons.org/licenses/by/4.0/> ;
    void:dataDump <https://mariacastillo982.github.io/cpp-mechanisms/data/mechanisms.ttl> ;
    void:uriSpace "https://cppkg.bio2vec.net/dataset/" .
"""


def _entity_as_turtle(entity_id: str) -> Response:
    try:
        g = get_graph()
        uri = _resolve_entity_uri(g, entity_id)
        if not uri:
            return Response("Entity not found", status=404, mimetype="text/plain")
        sub_g = Graph()
        for prefix, ns in g.namespaces():
            sub_g.bind(prefix, ns)
        ref = URIRef(uri)
        for pred, obj in g.predicate_objects(ref):
            sub_g.add((ref, pred, obj))
        ttl = sub_g.serialize(format="turtle")
        return Response(
            ttl,
            mimetype="text/turtle",
            headers={"Content-Disposition": 'inline; filename="{}.ttl"'.format(entity_id)},
        )
    except Exception as exc:
        return Response(str(exc), status=500, mimetype="text/plain")


def _entity_as_jsonld(entity_id: str) -> Response:
    try:
        g = get_graph()
        uri = _resolve_entity_uri(g, entity_id)
        if not uri:
            return Response("Entity not found", status=404, mimetype="text/plain")
        sub_g = Graph()
        ref = URIRef(uri)
        for pred, obj in g.predicate_objects(ref):
            sub_g.add((ref, pred, obj))
        try:
            jld = sub_g.serialize(format="json-ld", indent=2)
        except Exception:
            return Response("JSON-LD requires pyld: pip3 install pyld", status=500, mimetype="text/plain")
        return Response(
            jld,
            mimetype="application/ld+json",
            headers={"Content-Disposition": 'inline; filename="{}.jsonld"'.format(entity_id)},
        )
    except Exception as exc:
        return Response(str(exc), status=500, mimetype="text/plain")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def _preload():
        try:
            get_graph()
            print("Knowledge Graph loaded successfully.")
        except Exception as exc:
            print("Failed to pre-load graph: {}".format(exc))

    t = threading.Thread(target=_preload, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
