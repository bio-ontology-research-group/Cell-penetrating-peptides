#!/usr/bin/env python3
"""
CPP Knowledge-Graph Evaluator
==============================
Evaluates the CPP KG against six SPARQL
queries split into two groups:

  GROUP 1 – Competency Questions (CQ1–CQ3)
      Tests Hierarchical Subsumption (rdfs:subClassOf*) and Transitive Closure
      (SPARQL + / * property paths) entirely on the local graph.

  GROUP 2 – Federated Queries (FQ1–FQ3)
      Joins the local graph with external SPARQL endpoints:
        • FQ1 and FQ3 → UniProt  (https://sparql.uniprot.org/sparql)
        • FQ2 → Wikidata (https://query.wikidata.org/sparql)

Usage
-----
    python cpp_kg_evaluator.py                # CQ1-3 only (fast, offline)
    python cpp_kg_evaluator.py --federated    # CQ1-3 + FQ1-3 (requires internet)
    python cpp_kg_evaluator.py --query FQ2   # run a single named query

Ontology namespace key
----------------------
  SIO_000004  material entity         SIO_000006  process
  SIO_000053  has proper part         SIO_000062  is participant in
  SIO_000061  is located in (Trans.)  SIO_000093  is proper part of
  SIO_000313  is component part of    SIO_000355  realizes
  SIO_000356  is realized in          SIO_000369  has component part
  SIO_000557  is described by         SIO_001401  positively regulates
  SIO_001402  negatively regulates    SIO_010035  gene
  SIO_010054  cell line               SIO_010295  up-regulation process
  SIO_010296  down-regulation process SIO_000994  experiment
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path

from rdflib import Graph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KG_PATH = Path("data/Ontology/CPP_KG.ttl")

UNIPROT_ENDPOINT = "https://sparql.uniprot.org/sparql"
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

# ---------------------------------------------------------------------------
# Common PREFIX block reused across all queries
# ---------------------------------------------------------------------------

PREFIXES = """\
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:   <http://www.w3.org/2002/07/owl#>
PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>
PREFIX skos:  <http://www.w3.org/2004/02/skos/core#>
PREFIX dct:   <http://purl.org/dc/terms/>
PREFIX sio:   <http://semanticscience.org/resource/>
PREFIX obo:   <http://purl.obolibrary.org/obo/>
PREFIX up:    <http://purl.uniprot.org/core/>
PREFIX taxon: <http://purl.uniprot.org/taxonomy/>
PREFIX wdt:   <http://www.wikidata.org/prop/direct/>
PREFIX wd:    <http://www.wikidata.org/entity/>
PREFIX cpp:   <https://cppkg.bio2vec.net/dataset/>
PREFIX cppS:  <https://w3id.org/cpp/schema#>
"""

# ===========================================================================
# GROUP 1 – Competency Questions: Hierarchical Subsumption & Transitive Closure
# ===========================================================================

CQ1_LABEL = (
    "CQ1 | Hierarchical Subsumption — CPP uses Endocytosis (GO:0006897) as an uptake mechanism"
)
CQ1 = (
    PREFIXES
    + """
SELECT DISTINCT ?peptide
WHERE {
  ?peptide a cpp:CellPenetratingPeptide ;
           sio:SIO_000008 ?cpp_role .
  ?cpp_role sio:SIO_000356 ?mechanism .
  ?mechanism rdfs:subClassOf* obo:GO_0006897 .
}

ORDER BY ?peptide
"""
)

CQ2_LABEL = (
    "CQ2 | Hierarchical Subsumption (role chain) — genes whose activator role "
    "is realized in upregulation of macropinocytosis GO:0044351"
)
CQ2 = (
    PREFIXES
    + """
SELECT DISTINCT ?gene ?geneLabel ?activatorRole ?upregulation
WHERE {
  # ── Compound property path (transitive role-realization chain) ──────────
  # Gene → (has attribute) → ActivatorRole
  #      → (is realized in) → UpregulationProcess
  #      → (positively regulates) → GO:0044351 (macropinocytosis)
  ?gene a sio:SIO_010035 .
  ?gene (sio:SIO_000008 / sio:SIO_000356 / sio:SIO_001401) obo:GO_0044351 .

  # ── Retrieve intermediate nodes for inspection ──────────────────────────
  ?gene sio:SIO_000008 ?activatorRole .
  ?activatorRole sio:SIO_000356 ?upregulation .
  # Filter: only upregulation processes that target GO:0044351
  ?upregulation sio:SIO_001401 obo:GO_0044351 .

  OPTIONAL { ?gene rdfs:label ?geneLabel . }
}
ORDER BY ?geneLabel
"""
)

CQ3_LABEL = (
    "CQ3 | Transitive Closure — CPP-complexes reaching nucleus (GO:0005634) "
)
CQ3 = (
    PREFIXES
    + """
SELECT DISTINCT ?complex ?location ?locationLabel
                ?mechanism 
WHERE {
    ?complex a cpp:CPP-Complex .

    # ── Transitive Closure: is-located-in (SIO_000061, owl:TransitiveProperty) ──
    # Anchoring at obo:GO_0005634 (nucleus) retrieves every complex that
    # reaches the nucleus through any chain of subcellular location steps.
    ?complex sio:SIO_000061+ obo:GO_0005634 .
    BIND(obo:GO_0005634 AS ?location)
    OPTIONAL { obo:GO_0005634 rdfs:label ?locationLabel . }

    # ── Uptake mechanism (optional context) ────────────────────────────────
    OPTIONAL {
        ?complex sio:SIO_000062 ?mechanism .
        ?mechanism a cppS:UptakeMechanism .
    }
}
ORDER BY ?complex

"""
)

# ===========================================================================
# GROUP 2 – Federated Queries  (local KG  ⟺  external SPARQL endpoints)

# ---------------------------------------------------------------------------
# FQ1 – Local KG ⟺ UniProt
#        Step 1 (local): extract CQ2 genes + construct UniProt "agora" IRIs
#        Step 2 (remote UniProt): for each agora IRI find the encoded protein
#
#  UniProt cross-references Ensembl genes via the "agora" namespace:
#    http://purl.uniprot.org/agora/ENSG00000XXXXXX
#  (verified: ENSG00000006451 → P11233 Ral-A, ENSG00000006740 → Q17R89, …)
# ---------------------------------------------------------------------------
FQ1_LABEL = (
    f"FQ1 | Federated (2-phase) — CQ2 macropinocytosis genes × UniProt proteins  "
    f"[remote: {UNIPROT_ENDPOINT}]"
)
FQ1_LOCAL = (
    PREFIXES
    + """
SELECT DISTINCT ?gene ?agoraURI
WHERE {
    # CQ2 property-path: Gene -[has attribute]-> ActivatorRole
    #                        -[is realized in]-> UpregulationProcess
    #                        -[positively regulates]-> GO:0044351 (macropinocytosis)
    ?gene a sio:SIO_010035 .
    ?gene (sio:SIO_000008 / sio:SIO_000356 / sio:SIO_001401) obo:GO_0044351 .

    # Build the UniProt agora cross-reference IRI from the Ensembl accession
    BIND(REPLACE(STR(?gene), "http://identifiers.org/ensembl/", "") AS ?ensemblAcc)
    BIND(IRI(CONCAT("http://purl.uniprot.org/agora/", ?ensemblAcc)) AS ?agoraURI)
}
ORDER BY ?gene
"""
)
# {values} → "(<gene_iri> <agora_iri>)\n    …" injected at runtime
FQ1_REMOTE = """\
PREFIX up:    <http://purl.uniprot.org/core/>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX taxon: <http://purl.uniprot.org/taxonomy/>

SELECT DISTINCT ?gene ?uniprotEntry ?proteinName
WHERE {{
    VALUES (?gene ?agoraURI) {{
{values}
    }}
    ?uniprotEntry a up:Protein ;
                  up:organism taxon:9606 ;
                  rdfs:seeAlso ?agoraURI .
    OPTIONAL {{ ?uniprotEntry up:recommendedName / up:fullName ?proteinName . }}
}}
ORDER BY ?gene

"""

# ---------------------------------------------------------------------------
# FQ2 – Local KG ⟺ Wikidata
#        Step 1 (local): collect all ChEBI-typed cargo URIs + numeric IDs
#        Step 2 (remote Wikidata): look up English label + average mass
#
#  Wikidata properties used:
#    wdt:P683  = ChEBI ID (numeric string, e.g. "16670")
#    wdt:P2067 = average molecular mass (daltons)
#
#  Note: cargos are typed at the ChEBI class level (peptide, protein, …).
#  Class-level entries have no mass; specific compounds (e.g. CHEBI:141393
#  Ser-Thr, CHEBI:17362 quinoline) do return numeric masses.

# ---------------------------------------------------------------------------
FQ2_LABEL = (
    f"FQ2 | Federated (2-phase) — ChEBI cargos × label + mass (Wikidata P2067)  "
    f"[remote: {WIKIDATA_ENDPOINT}]"
)
FQ2_LOCAL = (
    PREFIXES
    + """
SELECT DISTINCT ?cargo ?chebiNum
WHERE {
    ?complex a cpp:CPP-Complex .
    ?complex sio:SIO_000369 ?cargo .
    ?cargo   a cpp:Cargo .
    FILTER(STRSTARTS(STR(?cargo), "http://purl.obolibrary.org/obo/CHEBI_"))
    # Extract numeric ChEBI ID used by Wikidata's P683
    BIND(REPLACE(STR(?cargo), "http://purl.obolibrary.org/obo/CHEBI_", "") AS ?chebiNum)
}
ORDER BY ?cargo
"""
)
# {values} → '(<cargo_iri> "chebiNum")\n    …' injected at runtime
FQ2_REMOTE = """\
PREFIX wdt:  <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?cargo ?chebiNum ?cargoLabel ?mass
WHERE {{
    VALUES (?cargo ?chebiNum) {{
{values}
    }}
    ?wdItem wdt:P683 ?chebiNum .
    OPTIONAL {{ ?wdItem rdfs:label ?cargoLabel .
               FILTER(LANG(?cargoLabel) = "en") }}
    OPTIONAL {{ ?wdItem wdt:P2067 ?mass . }}
}}
ORDER BY ?cargo

"""

# ---------------------------------------------------------------------------
# FQ3 – Local KG ⟺ UniProt (GO ontology)
#        Step 1 (local): collect all GO cellular-component delivery targets
#        Step 2 (remote UniProt): one rdfs:subClassOf hop → direct GO parents
#
#  UniProt embeds the full Gene Ontology and answers rdfs:subClassOf queries
#  for cellular-component terms (verified for all 5 locations in this KG).
# ---------------------------------------------------------------------------
FQ3_LABEL = (
    f"FQ3 | Federated (2-phase) — Subcellular locations × direct GO parents  "
    f"[remote: {UNIPROT_ENDPOINT}]"
)
FQ3_LOCAL = (
    PREFIXES
    + """
SELECT DISTINCT ?location
WHERE {
    ?complex a cpp:CPP-Complex .
    ?complex sio:SIO_000061 ?location .
    FILTER(STRSTARTS(STR(?location), "http://purl.obolibrary.org/obo/GO_"))
}
ORDER BY ?location
"""
)
# {values} → "<location_iri>\n    …" injected at runtime
FQ3_REMOTE = """\
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?location ?parent ?parentLabel
WHERE {{
    VALUES ?location {{
{values}
    }}
    ?location rdfs:subClassOf ?parent .
    FILTER(!isBlank(?parent))
    ?parent rdfs:label ?parentLabel .
}}
ORDER BY ?location ?parent

"""

# ===========================================================================
# Query registry
# ===========================================================================

LOCAL_QUERIES: dict[str, tuple[str, str]] = {
    "CQ1": (CQ1_LABEL, CQ1),
    "CQ2": (CQ2_LABEL, CQ2),
    "CQ3": (CQ3_LABEL, CQ3),
}

# Each FQ entry: (label, local_query, remote_template, endpoint)
FEDERATED_QUERIES: dict[str, tuple[str, str, str, str]] = {
    "FQ1": (FQ1_LABEL, FQ1_LOCAL, FQ1_REMOTE, UNIPROT_ENDPOINT),
    "FQ2": (FQ2_LABEL, FQ2_LOCAL, FQ2_REMOTE, WIKIDATA_ENDPOINT),
    "FQ3": (FQ3_LABEL, FQ3_LOCAL, FQ3_REMOTE, UNIPROT_ENDPOINT),
}

ALL_QUERIES = {**LOCAL_QUERIES, **FEDERATED_QUERIES}

# ===========================================================================
# Execution helpers
# ===========================================================================

_SEP = "=" * 72


def _header(label: str) -> None:
    print(f"\n{_SEP}")
    print(textwrap.fill(f"  {label}", width=70, subsequent_indent="    "))
    print(_SEP)


def _print_query(query: str) -> None:
    """Pretty-print the SPARQL query (without the long PREFIX block)."""
    body = query.split("\n", query.count("\n") - query.lstrip("PREFIX").count("\n"))
    # strip prefix lines for readability
    lines = [l for l in query.splitlines() if not l.startswith("PREFIX ")]
    print("\n  SPARQL:\n")
    for line in lines:
        if line.strip():
            print("   ", line)
    print()


def run_local(g: Graph, label: str, query: str, row_limit: int = 15) -> None:
    """Execute a query on the in-memory rdflib graph and print results."""
    _header(label)
    _print_query(query)

    t0 = time.perf_counter()
    try:
        results = list(g.query(query))
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ Query error: {exc}")
        return

    elapsed = time.perf_counter() - t0
    print(f"  → {len(results)} row(s) in {elapsed:.2f}s\n")

    if not results:
        print("  (no results)")
        return

    # Print column headers
    vars_ = results[0].labels if hasattr(results[0], "labels") else []
    if vars_:
        header = "  " + " | ".join(f"{v:<30}" for v in vars_)
        print(header)
        print("  " + "-" * (len(header) - 2))

    for i, row in enumerate(results[:row_limit]):
        cells = [
            str(v)[:50] if v is not None else "—" for v in row
        ]
        print("  " + " | ".join(f"{c:<30}" for c in cells))

    if len(results) > row_limit:
        print(f"  … {len(results) - row_limit} more row(s) not shown")


def _sparql_http(endpoint: str, query: str, timeout: int = 60) -> dict:
    """
    Send a SPARQL SELECT query via HTTP POST and return the parsed JSON response.
    POST avoids HTTP 414 (URI Too Long) when VALUES clauses are large.
    """
    import json
    import ssl
    import urllib.parse
    import urllib.request

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    body = urllib.parse.urlencode({"query": query}).encode()
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "CPP-KG-Evaluator/2.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read())


def _build_values_fq1(local_rows) -> str:
    """(<gene_iri> <agora_iri>) rows for FQ1."""
    return "\n".join(
        f"        (<{row.gene}> <{row.agoraURI}>)" for row in local_rows
    )


def _build_values_fq2(local_rows) -> str:
    """(<cargo_iri> "chebiNum") rows for FQ2."""
    return "\n".join(
        f'        (<{row.cargo}> "{row.chebiNum}")' for row in local_rows
    )


def _build_values_fq3(local_rows) -> str:
    """<location_iri> rows for FQ3."""
    return "\n".join(f"        <{row.location}>" for row in local_rows)


# Map each FQ name to the function that formats its VALUES clause
_VALUES_BUILDER = {
    "FQ1": _build_values_fq1,
    "FQ2": _build_values_fq2,
    "FQ3": _build_values_fq3,
}


def run_federated(
    g: Graph,
    name: str,
    label: str,
    local_query: str,
    remote_template: str,
    endpoint: str,
    row_limit: int = 15,
) -> None:
    """
    Two-phase federated execution:
      Phase 1 – run *local_query* on the in-memory rdflib graph to obtain the
                bindings that the remote endpoint needs to look up.
      Phase 2 – inject those bindings as a SPARQL VALUES clause into
                *remote_template* and POST it to *endpoint* via HTTP.

    This avoids the 0-result problem that occurs when the whole query (local KG
    patterns + SERVICE clause) is sent to the remote endpoint, which has no
    knowledge of the local graph.
    """
    _header(label)

    # ── Phase 1: local rdflib query ────────────────────────────────────────
    print("  Phase 1 — local rdflib query …")
    _print_query(local_query)
    t0 = time.perf_counter()
    try:
        local_rows = list(g.query(local_query))
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ Local query error: {exc}")
        return

    elapsed_local = time.perf_counter() - t0
    print(f"  → {len(local_rows)} local binding(s) in {elapsed_local:.2f}s")

    if not local_rows:
        print("  (no local results — remote query skipped)")
        return

    # ── Build VALUES clause and assemble remote query ──────────────────────
    try:
        values_str = _VALUES_BUILDER[name](local_rows)
        remote_query = remote_template.format(values=values_str)
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ Failed to build remote query: {exc}")
        return

    # ── Phase 2: remote HTTP SPARQL query ──────────────────────────────────
    print(f"\n  Phase 2 — remote query → {endpoint}")
    _print_query(remote_query)
    t0 = time.perf_counter()
    try:
        data = _sparql_http(endpoint, remote_query)
    except Exception as exc:  # noqa: BLE001
        print(f"  ✗ Endpoint error: {exc}")
        return

    elapsed_remote = time.perf_counter() - t0
    bindings = data.get("results", {}).get("bindings", [])
    print(f"  → {len(bindings)} row(s) in {elapsed_remote:.2f}s\n")

    if not bindings:
        print("  (no results)")
        return

    vars_ = data.get("head", {}).get("vars", [])
    header = "  " + " | ".join(f"{v:<28}" for v in vars_)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for row in bindings[:row_limit]:
        cells = [row.get(v, {}).get("value", "—")[:45] for v in vars_]
        print("  " + " | ".join(f"{c:<28}" for c in cells))

    if len(bindings) > row_limit:
        print(f"  … {len(bindings) - row_limit} more row(s) not shown")


# ===========================================================================
# Main
# ===========================================================================


def load_graph(path: Path) -> Graph:
    print(f"Loading KG: {path} …")
    if not path.exists():
        sys.exit(f"  ✗ File not found: {path}")
    g = Graph()
    t0 = time.perf_counter()
    g.parse(str(path), format="turtle")
    elapsed = time.perf_counter() - t0
    print(f"  {len(g):,} triples loaded in {elapsed:.1f}s")
    return g


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPP KG Evaluator — SPARQL competency & federated queries"
    )
    parser.add_argument(
        "--federated",
        action="store_true",
        help="Also execute federated queries FQ1–FQ3 (requires internet + SPARQLWrapper)",
    )
    parser.add_argument(
        "--query",
        metavar="NAME",
        help="Run a single named query (CQ1, CQ2, CQ3, FQ1, FQ2, FQ3)",
    )
    parser.add_argument(
        "--kg",
        default=str(KG_PATH),
        metavar="PATH",
        help=f"Path to the KG Turtle file (default: {KG_PATH})",
    )
    args = parser.parse_args()

    kg_path = Path(args.kg)

    # ── Single-query mode ──────────────────────────────────────────────────
    if args.query:
        name = args.query.upper()
        if name not in ALL_QUERIES:
            sys.exit(f"Unknown query '{name}'. Choose from: {', '.join(ALL_QUERIES)}")
        g = load_graph(kg_path)
        if name in LOCAL_QUERIES:
            label, sparql = LOCAL_QUERIES[name]
            run_local(g, label, sparql)
        else:
            label, local_q, remote_t, endpoint = FEDERATED_QUERIES[name]
            run_federated(g, name, label, local_q, remote_t, endpoint)
        return

    # ── Full evaluation mode ───────────────────────────────────────────────
    g = load_graph(kg_path)

    print("\n" + "#" * 72)
    print("  GROUP 1 — Competency Questions: Hierarchical Subsumption & "
            "Transitive Closure")
    print("#" * 72)
    for name, (label, sparql) in LOCAL_QUERIES.items():
        run_local(g, label, sparql)

    if args.federated:
        print("\n" + "#" * 72)
        print("  GROUP 2 — Federated Queries (local KG ⟺ external endpoints)")
        print("#" * 72)
        for name, (label, local_q, remote_t, endpoint) in FEDERATED_QUERIES.items():
            try:
                run_federated(g, name, label, local_q, remote_t, endpoint)
            except Exception as exc:  # noqa: BLE001
                print(f"  ✗ {name} failed unexpectedly: {exc} — continuing with next query")
    else:
        print(
            "\n"
            + _SEP
            + "\n  Federated queries FQ1–FQ3 were skipped.\n"
            "  Re-run with --federated to execute them (requires internet).\n"
            + _SEP
        )


if __name__ == "__main__":
    main()
