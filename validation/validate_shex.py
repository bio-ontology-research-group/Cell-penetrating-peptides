#!/usr/bin/env python3
"""
ShEx Validation Script for the CPP Mechanisms Dataset
======================================================
Validates the RDF Knowledge Graph (Ontology/mechanisms.ttl) against
ShEx shapes defined in mechanisms_shapes.shex using the PyShEx library.

ABox entity types validated:
  CellPenetratingPeptide, Gene, Inhibitor, Cell Line,
  Subcellular Entity, Uptake Mechanism, CPP-Complex,
  Cargo, Experiment, Document

Focus-node collection strategy (all ABox):
  Each entity type → nodes with rdf:type <class IRI>

Key structural constraints enforced:
  CPP          ⊑ ∃sio:SIO_000313.CPP-Complex   (has-component-part)
  Gene         ⊑ ∃sio:SIO_000008.ActivatorRole  (has-attribute)
  Inhibitor    ⊑ ∃sio:SIO_000008.InhibitorRole  (has-attribute)
  CPP-Complex  ⊑ ∃sio:SIO_000369.{CPP,Cargo}   (has-component-part)
  Cargo        ⊑ ∃sio:SIO_000008.CargoRole      (has-attribute)
  Experiment   ⊑ ∃sio:SIO_000557.Document       (is-cited-by)

Dependencies: rdflib, PyShEx
Usage: python validate_shex.py [--ttl FILE] [--shex FILE]
"""

import argparse
import sys
from pathlib import Path

from rdflib import Graph, URIRef, Namespace, RDFS, OWL, RDF
from pyshex.shex_evaluator import ShExEvaluator


# ============================================================================
# Namespaces
# ============================================================================

SIO       = Namespace("http://semanticscience.org/resource/")
MECH      = Namespace("https://w3id.org/cpp/dataset/mechanisms/")
CPPSCHEMA = Namespace("https://w3id.org/cpp/schema#")

# ============================================================================
# Class URIs → shape mapping  (all ABox individuals for mechanisms.ttl)
# (class_iri, shape_name, human_label, collect_by)
#
# collect_by:
#   "subClassOf" — focus nodes selected via rdfs:subClassOf <class_iri>  (TBox)
#   "type"       — focus nodes selected via rdf:type <class_iri>         (ABox)
# ============================================================================

SHAPE_MAP = [
    # Class IRI                            ShEx shape name           Label                 Collect
    (MECH["CellPenetratingPeptide"],       "CPPShape",               "CPP",                "type"),
    (SIO["SIO_010035"],                    "GeneShape",              "Gene",               "type"),
    (SIO["SIO_010435"],                    "InhibitorShape",         "Inhibitor",          "type"),
    (SIO["SIO_010054"],                    "CellLineShape",          "Cell Line",          "type"),
    (SIO["SIO_001400"],                    "SubcellularEntityShape", "Subcellular Entity", "type"),
    (CPPSCHEMA["UptakeMechanism"],         "UptakeMechanismShape",   "Uptake Mechanism",   "type"),
    (MECH["CPP-Complex"],                  "CPPComplexShape",        "CPP-Complex",        "type"),
    (MECH["Cargo"],                        "CargoShape",             "Cargo",              "type"),
    (SIO["SIO_000994"],                    "ExperimentShape",        "Experiment",         "type"),
    (SIO["SIO_000148"],                    "DocumentShape",          "Document",           "type"),
]


def collect_focus_nodes(graph: Graph, sio_class: URIRef,
                        collect_by: str = "subClassOf") -> list:
    """
    Return focus nodes for a given SIO class.

    collect_by="subClassOf" — TBox classes: nodes that are rdfs:subClassOf
                              the given class IRI (plus rdf:type fallback).
    collect_by="type"       — ABox individuals: nodes typed directly via
                              rdf:type (default for mechanisms.ttl).
    """
    nodes = set()
    if collect_by == "subClassOf":
        for s in graph.subjects(RDFS.subClassOf, sio_class):
            if s != sio_class:
                nodes.add(s)
    # Always also collect direct rdf:type matches (covers Documents, Observations,
    # and any TBox class that is also instantiated as an individual)
    for s in graph.subjects(RDF.type, sio_class):
        if s != sio_class:
            nodes.add(s)
    return sorted(nodes, key=str)


def validate_shape(graph: Graph, shex_schema: str,
                   focus_nodes: list, shape_name: str,
                   label: str, max_report: int = 5) -> tuple:
    """
    Validate a list of focus nodes against a single ShEx shape.
    Returns (passed, failed, errors_list).
    """
    if not focus_nodes:
        print(f"  [{label}] No focus nodes found — skipping.")
        return 0, 0, []

    shape_iri = f"http://example.org/shapes/{shape_name}"
    passed = 0
    failed = 0
    errors = []

    evaluator = ShExEvaluator(
        rdf=graph,
        schema=shex_schema,
        focus=focus_nodes,
        start=shape_iri,
    )

    for result in evaluator.evaluate():
        if result.result:
            passed += 1
        else:
            failed += 1
            errors.append((str(result.focus), result.reason))

    print(f"  [{label}] {passed} passed, {failed} failed (out of {len(focus_nodes)} nodes)")

    if errors:
        shown = errors[:max_report]
        for focus, reason in shown:
            short_focus = focus.rsplit("/", 1)[-1] if "/" in focus else focus
            reason_short = reason[:200] if reason else "no reason"
            print(f"    FAIL: {short_focus}")
            print(f"          {reason_short}")
        if len(errors) > max_report:
            print(f"    ... and {len(errors) - max_report} more failures")

    return passed, failed, errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate CPP Knowledge Graph against ShEx shapes"
    )
    parser.add_argument(
        "--ttl", default="Ontology/mechanisms.ttl",
        help="Path to the RDF Turtle file (default: Ontology/mechanisms.ttl)",
    )
    parser.add_argument(
        "--shex", default="mechanisms_shapes.shex",
        help="Path to the ShEx schema file (default: mechanisms_shapes.shex)",
    )
    parser.add_argument(
        "--max-report", type=int, default=5,
        help="Max number of failures to display per shape (default: 5)",
    )
    args = parser.parse_args()

    # Load RDF graph
    ttl_path = Path(args.ttl)
    if not ttl_path.exists():
        print(f"ERROR: RDF file not found: {ttl_path}")
        sys.exit(1)
    print(f"Loading RDF graph from {ttl_path}...")
    graph = Graph()
    graph.parse(str(ttl_path), format="turtle")
    print(f"  {len(graph)} triples loaded.\n")

    # Load ShEx schema — inject a BASE so shape names resolve to full IRIs
    shex_path = Path(args.shex)
    if not shex_path.exists():
        print(f"ERROR: ShEx file not found: {shex_path}")
        sys.exit(1)
    shex_text = shex_path.read_text()
    # Prepend a BASE declaration so <PeptideShape> → http://example.org/shapes/PeptideShape
    shex_schema = f"BASE <http://example.org/shapes/>\n{shex_text}"

    # Validate each entity type
    print("=" * 70)
    print("ShEx VALIDATION RESULTS")
    print("=" * 70)

    total_passed = 0
    total_failed = 0

    for sio_class, shape_name, label, collect_by in SHAPE_MAP:
        focus_nodes = collect_focus_nodes(graph, sio_class, collect_by)
        p, f, _ = validate_shape(
            graph, shex_schema, focus_nodes, shape_name, label, args.max_report
        )
        total_passed += p
        total_failed += f

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total nodes validated: {total_passed + total_failed}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")

    if total_failed == 0:
        print("\n  ALL SHAPES VALIDATED SUCCESSFULLY")
    else:
        print(f"\n  {total_failed} node(s) did not conform to their expected shape.")

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
