# -*- coding: utf-8 -*-
"""
CPP Mechanisms Knowledge Graph Dashboard - Flask Version
Run: python3 app.py
"""

import threading
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import pandas as pd
import requests
import re
from flask import Flask, Response, jsonify, redirect, render_template, request
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Primary source: GitHub raw TTL (single request, CDN-backed).
GITHUB_TTL_URL = "https://raw.githubusercontent.com/bio-ontology-research-group/Cell-penetrating-peptides/main/data/Ontology/CPP_KG.ttl"
# Fallback: Zenodo record JSON API for the canonical versioned TTL file.
# The API endpoint exposes a `files` array with download links.
ZENODO_API_URL = "https://zenodo.org/api/records/19427198"
_HERE = Path(__file__).parent
LOCAL_TTL_CANDIDATES = [
    _HERE / "../data/Ontology/CPP_KG.ttl",
    _HERE / "../data/CPP_KG.ttl",
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
            except Exception as exc:
                _graph_error = str(exc)
                raise
    return _graph


# ---------------------------------------------------------------------------
# SPARQL helpers
# ---------------------------------------------------------------------------

def sparql_to_records(g: Graph, query: str):
    results = g.query(query)
    columns = [str(v) for v in results.vars]
    rows = [[str(cell) if cell is not None else "" for cell in row] for row in results]
    return columns, rows


def _escape_sparql_literal(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
             .replace('"', '\\"')
             .replace("\n", "\\n")
             .replace("\r", "\\r")
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

ASSOCIATION_CATALOG = [
    {
        "name": "Positively Regulates",
        "description": "Gene interaction with the Uptake mechanism.",
        "query": "PREFIX sio: <http://semanticscience.org/resource/> SELECT (COUNT(*) AS ?count) WHERE { ?s sio:SIO_001401 ?o . }",
    },
    {
        "name": "Negatively Regulates",
        "description": "Inhibitor interaction with the Uptake mechanism.",
        "query": "PREFIX sio: <http://semanticscience.org/resource/> SELECT (COUNT(*) AS ?count) WHERE { ?s sio:SIO_001402 ?o . }",
    },
    {
        "name": "Cargo Role: Is Realized In",
        "description": "Functional role of the Cargo realized in the Uptake Mechanism.",
        "query": "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX sio: <http://semanticscience.org/resource/> PREFIX cpp: <https://cppkg.bio2vec.net/dataset/> SELECT (COUNT(*) AS ?count) WHERE { ?s rdf:type cpp:CargoRole . ?s sio:SIO_000356 ?o . }",
    },
    {
        "name": "CPP Role: Is Realized In",
        "description": "Functional role of the Cell-penetrating peptide realized in the Uptake Mechanism.",
        "query": "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX sio: <http://semanticscience.org/resource/> PREFIX cpp: <https://cppkg.bio2vec.net/dataset/> SELECT (COUNT(*) AS ?count) WHERE { ?s rdf:type cpp:CellPenetratingPeptideRole . ?s sio:SIO_000356 ?o . }",
    },
    {
        "name": "CPP-Complex: Is Participant In",
        "description": "CPP-Complex interaction with a specific the uptake mechanism.",
        "query": "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX sio: <http://semanticscience.org/resource/> PREFIX cpp: <https://cppkg.bio2vec.net/dataset/> SELECT (COUNT(*) AS ?count) WHERE { ?s rdf:type cpp:CPP-Complex . ?s sio:SIO_000062 ?o . }",
    },
    {
        "name": "Cell Line: Is Participant In",
        "description": "Cell Line interaction with a specific the uptake mechanism.",
        "query": "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX sio: <http://semanticscience.org/resource/> SELECT (COUNT(*) AS ?count) WHERE { ?s rdf:type sio:SIO_010054 . ?s sio:SIO_000062 ?o . }",
    },
    {
        "name": "Subcellular Delivery Localization",
        "description": "CPP-Complex interaction with Subcellular Entity.",
        "query": "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX sio: <http://semanticscience.org/resource/> PREFIX cpp: <https://cppkg.bio2vec.net/dataset/> SELECT (COUNT(*) AS ?count) WHERE { ?s rdf:type cpp:CPP-Complex . ?s sio:SIO_000061 ?o . }",
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

CANDIDATE_PATTERNS = {
    "sequence": (
        "    ?peptide cppS:sequence ?sequence .\n"
        '    FILTER(CONTAINS(LCASE(STR(?sequence)), LCASE("{term}")))\n'
    ),
    "cpp_id": (
        '    BIND(REPLACE(STR(?peptide), ".*/(.*)$", "$1") AS ?cppId)\n'
        '    FILTER(LCASE(STR(?cppId)) = LCASE("{term}"))\n'
    ),
    "mechanism": (
        "    ?peptide sio:SIO_000313 ?complex .\n"
        "    ?complex sio:SIO_000062 ?mech .\n"
        "    ?mech rdfs:label ?mechanismLabel .\n"
        '    FILTER(CONTAINS(LCASE(STR(?mechanismLabel)), LCASE("{term}")))\n'
    ),
    "subcell": (
        "    ?peptide sio:SIO_000313 ?complex .\n"
        "    ?complex sio:SIO_000061 ?subcellNode .\n"
        "    ?subcellNode rdfs:label ?subcellLabel .\n"
        '    FILTER(CONTAINS(LCASE(STR(?subcellLabel)), LCASE("{term}")))\n'
    ),
    "cell_line": (
        "    ?peptide sio:SIO_000313 ?complex .\n"
        "    ?complex sio:SIO_000062 ?mech .\n"
        "    ?cellLineNode rdf:type sio:SIO_010054 .\n"
        "    ?cellLineNode sio:SIO_000062 ?mech .\n"
        "    ?cellLineNode rdfs:label ?cellLineLabel .\n"
        '    FILTER(CONTAINS(LCASE(STR(?cellLineLabel)), LCASE("{term}")))\n'
    ),
    "cargo": (
        "    ?peptide sio:SIO_000313 ?complex .\n"
        "    ?complex sio:SIO_000369 ?cargoNode .\n"
        "    OPTIONAL { ?cargoNode cppS:cargoType ?cargoType . }\n"
        '    FILTER(CONTAINS(LCASE(STR(?cargoType)), LCASE("{term}")))\n'
    ),
    "doc_id": (
        "    ?peptide sio:SIO_000313 ?complex .\n"
        "    ?complex sio:SIO_000062 ?mech .\n"
        "    ?exp rdf:type sio:SIO_000994 .\n"
        "    ?exp sio:SIO_000053 ?mech .\n"
        "    ?exp sio:SIO_000557 ?docNode .\n"
        '    BIND(REPLACE(STR(?docNode), ".*identifiers.org/", "") AS ?docId)\n'
        '    FILTER(CONTAINS(LCASE(STR(?docId)), LCASE("{term}")))\n'
    ),
}

MAX_MATCHED_PEPTIDES = 250
MAX_RESULT_ROWS = 5000

_CANDIDATE_WHERE_TOKEN = "##CANDIDATE_WHERE##"
_VALUES_TOKEN = "##PEPTIDE_VALUES##"

CANDIDATE_QUERY = (
    "PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "PREFIX sio:  <http://semanticscience.org/resource/>\n"
    "PREFIX cpp:  <https://cppkg.bio2vec.net/dataset/>\n"
    "PREFIX cppS: <https://cppkg.bio2vec.net/schema#>\n"
    "SELECT DISTINCT ?peptide\n"
    "WHERE {\n"
    "    ?peptide rdf:type cpp:CellPenetratingPeptide .\n"
    "    " + _CANDIDATE_WHERE_TOKEN + "\n"
    "}\n"
    "LIMIT " + str(MAX_MATCHED_PEPTIDES) + "\n"
)

DETAILS_QUERY = (
    "PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
    "PREFIX sio:  <http://semanticscience.org/resource/>\n"
    "PREFIX cpp:  <https://cppkg.bio2vec.net/dataset/>\n"
    "PREFIX cppS: <https://cppkg.bio2vec.net/schema#>\n"
    "SELECT DISTINCT\n"
    "    ?cppId ?pepName ?sequence\n"
    "    ?cargoId ?cargoType\n"
    "    ?mechanismLabel\n"
    "    ?cellLineLabel\n"
    "    ?subcellLabel\n"
    "    ?docId\n"
    "    ?efficiency ?model\n"
    "WHERE {\n"
    "    VALUES ?peptide { " + _VALUES_TOKEN + " }\n"
    '    BIND(REPLACE(STR(?peptide), ".*/(.*)$", "$1") AS ?cppId)\n'
    "    OPTIONAL { ?peptide cppS:peptideName ?pepName . }\n"
    "    OPTIONAL { ?peptide cppS:sequence ?sequence . }\n"
    "    OPTIONAL {\n"
    "        ?peptide sio:SIO_000313 ?complex .\n"
    "        ?complex rdf:type cpp:CPP-Complex .\n"
    "        OPTIONAL {\n"
    "            ?complex sio:SIO_000369 ?cargoNode .\n"
    '            BIND(REPLACE(STR(?cargoNode), ".*/(.*)$", "$1") AS ?cargoId)\n'
    "            OPTIONAL { ?cargoNode cppS:cargoType ?cargoType . }\n"
    "        }\n"
    "        OPTIONAL {\n"
    "            ?complex sio:SIO_000062 ?mech .\n"
    "            ?mech rdf:type cppS:UptakeMechanism .\n"
    "            ?mech rdfs:label ?mechanismLabel .\n"
    '            FILTER(!REGEX(STR(?mechanismLabel), "^GO_[0-9]+$"))\n'
    "            OPTIONAL {\n"
    "                ?cellLineNode rdf:type sio:SIO_010054 .\n"
    "                ?cellLineNode sio:SIO_000062 ?mech .\n"
    "                ?cellLineNode rdfs:label ?cellLineLabel .\n"
    "            }\n"
    "            OPTIONAL {\n"
    "                ?exp rdf:type sio:SIO_000994 .\n"
    "                ?exp sio:SIO_000053 ?mech .\n"
    "                OPTIONAL {\n"
    "                    ?exp sio:SIO_000557 ?docNode .\n"
    '                    BIND(REPLACE(STR(?docNode), ".*identifiers.org/", "") AS ?docId)\n'
    "                }\n"
    "                OPTIONAL { ?exp cppS:uptakeEfficiency ?efficiency . }\n"
    "                OPTIONAL { ?exp cppS:inVivoModel ?model . }\n"
    "            }\n"
    "        }\n"
    "        OPTIONAL {\n"
    "            ?complex sio:SIO_000061 ?subcellNode .\n"
    "            ?subcellNode rdfs:label ?subcellLabel .\n"
    "        }\n"
    "    }\n"
    "}\n"
    "ORDER BY ?cppId ?mechanismLabel\n"
    "LIMIT " + str(MAX_RESULT_ROWS) + "\n"
)

COLUMN_LABELS = {
    "cppId":          "CPP ID",
    "pepName":        "Peptide Name",
    "sequence":       "Sequence",
    "cargoId":        "Cargo (ChEBI)",
    "cargoType":      "Cargo Type",
    "mechanismLabel": "Uptake Mechanism",
    "cellLineLabel":  "Cell Line",
    "subcellLabel":   "Subcellular Delivery",
    "docId":          "PubMed / Patent ID",
    "efficiency":     "Uptake Efficiency",
    "model":          "In Vivo / In Vitro",
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
        subjects = set(g.subjects())
        objects = set(g.objects())
        total_nodes = len(subjects | objects)
        return jsonify({
            "total_triples":    len(g),
            "total_nodes":      total_nodes,
            "total_predicates": len(set(g.predicates())),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/classes")
def api_classes():
    try:
        g = get_graph()
        cols, rows = sparql_to_records(g, Q_CLASS_COUNTS)
        df = pd.DataFrame(rows, columns=cols)
        df["count"] = pd.to_numeric(df["count"], errors="coerce")

        result = []
        for _, r in df.iterrows():
            entry = CLASS_CATALOG.get(r["class"])
            if entry:
                result.append({
                    "name":        entry["name"],
                    "description": entry["description"],
                    "uri":         r["class"],
                    "count":       int(r["count"]) if pd.notna(r["count"]) else 0,
                })
        result.sort(key=lambda x: x["count"], reverse=True)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/associations")
def api_associations():
    try:
        g = get_graph()
        rows = []
        for entry in ASSOCIATION_CATALOG:
            result = list(g.query(entry["query"]))
            count = int(str(result[0][0])) if result and result[0][0] is not None else 0
            rows.append({
                "name":        entry["name"],
                "description": entry["description"],
                "count":       count,
            })
        return jsonify(rows)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json() or {}
    filters = data.get("filters") if isinstance(data.get("filters"), dict) else None

    if not filters or not any((str(v) or "").strip() for v in filters.values()):
        return jsonify({"error": "At least one filter must be provided."}), 400

    # Validate filter keys
    for k in list(filters.keys()):
        if k not in CANDIDATE_PATTERNS:
            return jsonify({"error": f"Unknown search field: {k}"}), 400

    try:
        g = get_graph()

        # Build candidate WHERE by concatenating all active patterns (AND logic)
        # Variables inside each pattern are suffixed with the filter key to avoid name clashes,
        # except for ?peptide which remains shared.
        parts = []
        for key, raw_term in filters.items():
            term = (raw_term or "").strip()
            if not term:
                continue
            esc = _escape_sparql_literal(term)
            pattern = CANDIDATE_PATTERNS[key].format(term=esc)

            def _suf(m):
                v = m.group(1)
                if v == 'peptide':
                    return '?peptide'
                return f'?{v}_{key}'

            # Suffix variables in the pattern to keep them local to that filter
            pattern = re.sub(r"\?([A-Za-z_][A-Za-z0-9_]*)", _suf, pattern)
            parts.append(pattern)

        if not parts:
            return jsonify({"rows": [], "columns": [], "warnings": []})

        candidate_where = "\n".join(parts)
        candidate_query = CANDIDATE_QUERY.replace(_CANDIDATE_WHERE_TOKEN, candidate_where)
        candidate_rows = list(g.query(candidate_query))
        peptides = [str(getattr(r, 'peptide', r[0])) for r in candidate_rows]

        if not peptides:
            return jsonify({
                "columns": [], "rows": [], "matched_peptides": 0, "warnings": [],
                "debug": {
                    "candidate_query": candidate_query,
                    "details_query": None,
                    "step1_matched_peptides": [],
                    "step2_values_block": None,
                    "step3_detail_rows": [],
                },
            })

        values_block = " ".join("<{}>".format(uri) for uri in peptides)
        details_query = DETAILS_QUERY.replace(_VALUES_TOKEN, values_block)
        vars_, rows = sparql_to_records(g, details_query)

        display_cols = [COLUMN_LABELS.get(c, c) for c in vars_]

        warnings = []
        if len(peptides) >= MAX_MATCHED_PEPTIDES:
            warnings.append(f"Matched at least {MAX_MATCHED_PEPTIDES} peptides. Refine the search term for faster results.")
        if len(rows) >= MAX_RESULT_ROWS:
            warnings.append(f"Result set reached the {MAX_RESULT_ROWS}-row limit. Refine your term for a narrower search.")

        # Sample the first detail row to show what fields were bound vs None
        sample_row = dict(zip(vars_, rows[0])) if rows else {}

        return jsonify({
            "columns": display_cols,
            "rows":    rows,
            "matched_peptides": len(peptides),
            "warnings": warnings,
            "debug": {
                "candidate_query":          candidate_query,
                "details_query":            details_query,
                "step1_matched_peptides":   peptides,
                "step2_values_block":       values_block,
                "step3_total_detail_rows":  len(rows),
                "step3_sample_first_row":   sample_row,
            },
        })
    except Exception as exc:
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
          FILTER(STRSTARTS(STR(?entity), "https://cppkg.bio2vec.net/dataset/"))
        }
        LIMIT 5000
        """
        base = request.host_url.rstrip("/")
        urls = []
        for row in g.query(q):
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
    return redirect("/#download")


@app.route("/dataset/<path:entity_id>")
def dataset_entity(entity_id):
    entity_id = unquote(entity_id)
    accept = request.headers.get("Accept", "text/html")
    if "text/turtle" in accept:
        return _entity_as_turtle(entity_id)
    if "application/ld+json" in accept:
        return _entity_as_jsonld(entity_id)
    # 303 See Other for browsers
    resp = redirect("/#entity/{}".format(entity_id), 303)
    return resp


# ---------------------------------------------------------------------------
# Routes - new: API
# ---------------------------------------------------------------------------

@app.route("/api/entity/<path:entity_id>")
def api_entity(entity_id):
    entity_id = unquote(entity_id)
    try:
        g = get_graph()
        uri = _resolve_entity_uri(g, entity_id)
        if uri is None:
            return jsonify({"error": "Entity not found: {}".format(entity_id)}), 404

        # Outgoing properties
        out_q = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?pred ?obj ?objLabel WHERE {{
          <{uri}> ?pred ?obj .
          OPTIONAL {{ ?obj rdfs:label ?objLabel . }}
        }}
        ORDER BY ?pred ?obj
        LIMIT 400
        """.format(uri=uri)
        _, out_rows = sparql_to_records(g, out_q)

        # Incoming links
        in_q = """
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?subj ?subjLabel ?pred WHERE {{
          ?subj ?pred <{uri}> .
          OPTIONAL {{ ?subj rdfs:label ?subjLabel . }}
        }}
        ORDER BY ?pred ?subj
        LIMIT 150
        """.format(uri=uri)
        _, in_rows = sparql_to_records(g, in_q)

        graph_data = _get_neighborhood(g, uri)

        return jsonify({
            "uri":       uri,
            "entity_id": entity_id,
            "outgoing":  out_rows,
            "incoming":  in_rows,
            "graph":     graph_data,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/browse")
def api_browse():
    class_uri = request.args.get("class", "").strip()
    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = min(100, max(10, int(request.args.get("per_page", 50))))
    except (ValueError, TypeError):
        page, per_page = 1, 50
    filter_text = request.args.get("filter", "").strip()

    if not class_uri:
        return jsonify({"error": "class parameter required"}), 400

    try:
        g = get_graph()

        filter_clause = ""
        if filter_text:
            esc = _escape_sparql_literal(filter_text)
            filter_clause = 'FILTER(CONTAINS(LCASE(COALESCE(STR(?label), STR(?entity))), LCASE("{}")))'.format(esc)

        # Count total
        count_q = """
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT (COUNT(DISTINCT ?entity) AS ?total) WHERE {{
          ?entity rdf:type <{cls}> .
          OPTIONAL {{ ?entity rdfs:label ?label . }}
          {flt}
        }}
        """.format(cls=class_uri, flt=filter_clause)
        count_res = list(g.query(count_q))
        total = int(str(count_res[0][0])) if count_res and count_res[0][0] else 0

        # Fetch page  (rdflib 7 supports OFFSET)
        offset = (page - 1) * per_page
        q = """
        PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX cppS: <https://cppkg.bio2vec.net/schema#>
        SELECT DISTINCT ?entity (SAMPLE(?lbl) AS ?label) (SAMPLE(?seq) AS ?sequence) WHERE {{
          ?entity rdf:type <{cls}> .
          OPTIONAL {{ ?entity rdfs:label ?lbl . }}
          OPTIONAL {{ ?entity cppS:sequence ?seq . }}
          {flt}
        }}
        GROUP BY ?entity
        ORDER BY ?label ?entity
        LIMIT {lim} OFFSET {off}
        """.format(cls=class_uri, flt=filter_clause, lim=per_page, off=offset)
        _, rows = sparql_to_records(g, q)

        items = []
        for row in rows:
            ent_uri, label, seq = row[0], row[1], row[2]
            local_id = _uri_fragment(ent_uri)
            items.append({
                "uri":      ent_uri,
                "local_id": local_id,
                "label":    label or local_id,
                "extra":    seq,
            })

        pages = max(1, (total + per_page - 1) // per_page)
        return jsonify({
            "items":    items,
            "total":    total,
            "page":     page,
            "per_page": per_page,
            "pages":    pages,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500



@app.route("/api/download/ttl")
def api_download_ttl():
    return redirect("https://zenodo.org/records/19427198/files/CPP_KG.ttl?download=1", code=302)


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

    # Outgoing edges
    for pred, obj in g.predicate_objects(center):
        if edge_count >= max_edges:
            break
        if not isinstance(obj, URIRef):
            continue
        pred_label = _uri_fragment(str(pred))[:30]
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
        pred_label = _uri_fragment(str(pred))[:30]
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

    app.run(host="0.0.0.0", port=5001, debug=False)
