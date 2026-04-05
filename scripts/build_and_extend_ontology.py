#!/usr/bin/env python3
"""
Bio-Ontology Construction Script using mowl Library

This script performs three phases:
1. Phase 1: Load Ontology/sio.owl and define:
                - cpp:UptakeMechanism as a subclass of SIO process (SIO_000006)
                - Ensembl genes as SIO gene (SIO_010035) instances, each with an
                activator role (SIO_000804) via hasAttribute (SIO_000008),
                realized in process up-regulation (SIO_010295)
                - GO terms as cpp:UptakeMechanism instances
                - Gene→GO links via SIO_001401 (positively regulates)
2. Phase 2: Prepare annotation data from CSV file
3. Phase 3: Extend ontology with new annotations

Author: Maria Gomez
Dependencies: mowl, pandas
"""
import jpype
import jpype.imports

if not jpype.isJVMStarted():
    jpype.startJVM(classpath=['/Users/hadmin1/miniconda3/envs/fab_2024/lib/python3.10/site-packages/mowl/lib/*'])
# Now you can import Java packages
#from org.orekit.data import DirectoryCrawler 
import pandas as pd
import os
from typing import List, Tuple
from mowl.ontology.extend import insert_annotations

from org.semanticweb.owlapi.apibinding import OWLManager
from org.semanticweb.owlapi.model import (
    IRI, AddAxiom, AddOntologyAnnotation, RemoveOntologyAnnotation,
    OWLOntologyID, SetOntologyID
)
import java.io

# ============================================================================
# CONFIGURATION: Define URIs for predicates
# ============================================================================

# Phase 1 predicates
GENE_TO_GO_RELATION = "http://semanticscience.org/resource/SIO_001401"  # positively regulates
CHEBI_TO_GO_RELATION = "http://semanticscience.org/resource/SIO_001402"  # negatively regulates

# Phase 2 predicates
CPP_CARGO_RELATION = "http://semanticscience.org/resource/SIO_000203"  # is connected to
CPP_LOCATION_RELATION = "http://semanticscience.org/resource/SIO_000356"  # is realized in
CPP_CELL_RELATION = "http://semanticscience.org/resource/SIO_000793"  # measured at
CPP_MECH_RELATION = "http://semanticscience.org/resource/SIO_000062" # is participant in

# SIO URIs used in Phase 1
SIO_GENE             = "http://semanticscience.org/resource/SIO_010035"  # gene class
SIO_HAS_ATTRIBUTE    = "http://semanticscience.org/resource/SIO_000008"  # has attribute
SIO_ACTIVATOR_ROLE   = "http://semanticscience.org/resource/SIO_000804"  # activator role class
SIO_IS_REALIZED_IN   = "http://semanticscience.org/resource/SIO_000356"  # is realized in
SIO_UPREGULATION     = "http://semanticscience.org/resource/SIO_010295"  # up-regulation process class
SIO_PROCESS          = "http://semanticscience.org/resource/SIO_000006"  # process class

# SIO URIs used in Phase 2 (inhibitor regulation)
SIO_INHIBITOR        = "http://semanticscience.org/resource/SIO_010435"  # inhibitor class
SIO_INHIBITOR_ROLE   = "http://semanticscience.org/resource/SIO_000803"  # inhibitor role class
SIO_DOWNREGULATION   = "http://semanticscience.org/resource/SIO_010296"  # process down-regulation class

# SIO URIs used in Phase 3 (CPP-Complex TBox)
SIO_MATERIAL_ENTITY  = "http://semanticscience.org/resource/SIO_000004"  # material entity
SIO_PEPTIDE          = "http://semanticscience.org/resource/SIO_001425"  # peptide
SIO_HAS_COMP_PART    = "http://semanticscience.org/resource/SIO_000369"  # has component part
SIO_IS_COMP_PART_OF  = "http://semanticscience.org/resource/SIO_000313"  # is component part of
SIO_PROCESSUAL_ROLE  = "http://semanticscience.org/resource/SIO_000677"  # processual role
SIO_IS_ATTRIBUTE_OF  = "http://semanticscience.org/resource/SIO_000011"  # is attribute of (inverse of has attribute)
SIO_REALIZES         = "http://semanticscience.org/resource/SIO_000355"  # realizes (inverse of is realized in)
SIO_CELL_LINE        = "http://semanticscience.org/resource/SIO_010054"  # cell line class
SIO_SUBCELLULAR      = "http://semanticscience.org/resource/SIO_001400"  # subcellular entity class
SIO_EXPERIMENT       = "http://semanticscience.org/resource/SIO_000994"  # experiment class
SIO_HAS_PARTICIPANT  = "http://semanticscience.org/resource/SIO_000132"  # has participant
SIO_IS_PART_IN       = "http://semanticscience.org/resource/SIO_000062"  # is participant in
SIO_IS_LOCATED_IN    = "http://semanticscience.org/resource/SIO_000061"  # is located in
SIO_IS_LOCATION_OF   = "http://semanticscience.org/resource/SIO_000145"  # is location of (inverse)
SIO_DOCUMENT         = "http://semanticscience.org/resource/SIO_000148"  # document class
SIO_IS_DESCRIBED_BY  = "http://semanticscience.org/resource/SIO_000557"  # is described by
SIO_DESCRIBES        = "http://semanticscience.org/resource/SIO_000563"  # describes (inverse)
SIO_HAS_PROPER_PART  = "http://semanticscience.org/resource/SIO_000053"  # has proper part
SIO_IS_PROPER_PART   = "http://semanticscience.org/resource/SIO_000093"  # is proper part of

# CPP namespaces
CPP_SCHEMA_NS          = "https://cppkg.bio2vec.net/schema#"
CPP_DATASET_NS   = "https://cppkg.bio2vec.net/dataset/"

# TBox: UptakeMechanism class lives in the schema namespace (not the dataset)
UPTAKE_MECHANISM_CLASS = CPP_SCHEMA_NS + "UptakeMechanism"

SIO_OWL = "Ontology/sio.owl"

# ============================================================================
# FAIR METADATA CONFIGURATION
# ============================================================================

# F1 / F3 — Dataset IRI (globally unique, persistent) + versioned ontology IRI
DATASET_IRI       = "https://cppkg.bio2vec.net"
ONTOLOGY_IRI      = DATASET_IRI                              # ontology describes the dataset
ONTOLOGY_VERSION  = DATASET_IRI + "/2026-03-19"

# F1B — Identifiers.org-registered persistent identifier for the dataset.
# FAIR-Checker F1B strong check requires dct:identifier / schema:identifier
# whose value is traceable to an identifiers.org-registered namespace (e.g. DOI).
# ⚠ Replace the placeholder below with the actual DOI once the dataset is
#   deposited in Zenodo / figshare: https://zenodo.org → "New upload"
#   Then the value becomes e.g. "https://identifiers.org/doi:10.5281/zenodo.XXXXXXX"
DATASET_DOI       = "https://identifiers.org/doi:10.5281/zenodo.19427198"

# SPARQL / service endpoint for dcat:DataService
# Matches the Flask route @app.route("/api/sparql", methods=["POST"])
SPARQL_ENDPOINT     = DATASET_IRI + "/api/sparql"
SERVICE_DESCRIPTION = DATASET_IRI + "/api/sparql"   # SD doc at same IRI

# F2 — Dublin Core Terms properties for rich ontology-level metadata
DC_TERMS   = "http://purl.org/dc/terms/"
RDFS_NS    = "http://www.w3.org/2000/01/rdf-schema#"

# F4 — VOID / DCAT for dataset discoverability
VOID_NS      = "http://rdfs.org/ns/void#"
DCAT_NS      = "http://www.w3.org/ns/dcat#"
SCHEMA_NS    = "http://schema.org/"


def _apply_ontology_metadata(manager, ontology, factory):
    """
    Apply FAIR F1-F4 metadata to the ontology object.

    F1 + F3: Set a globally unique, versioned ontology IRI via SetOntologyID.
    F2:      Add Dublin Core Terms annotations (title, description, creator,
                created, license) so the ontology is self-describing.
    F4:      Add a void:Dataset and dcat:Dataset type annotation so the
                ontology can be discovered and indexed by dataset registries
                (e.g. BioPortal, OLS, LOD Cloud).
             Add a dcat:DataService node with dcat:endpointURL and
                dcat:endpointDescription pointing to the SPARQL API endpoint.
    """
    # F1 / F3 — Globally unique + versioned ontology IRI
    onto_iri    = IRI.create(ONTOLOGY_IRI)
    version_iri = IRI.create(ONTOLOGY_VERSION)
    manager.applyChange(SetOntologyID(ontology,
        OWLOntologyID(onto_iri, version_iri)))

    # ------------------------------------------------------------------ #
    # Strip all existing ontology-level annotations from the loaded SIO  #
    # ontology so that only our authorship and description are retained.  #
    # ------------------------------------------------------------------ #
    for ann in list(ontology.getAnnotations()):
        manager.applyChange(RemoveOntologyAnnotation(ontology, ann))

    # ------------------------------------------------------------------ #
    # Literal-valued annotations                                         #
    # ------------------------------------------------------------------ #
    # F1B (strong): dct:identifier / schema:identifier in ontology header.
    # Prefer the DOI (identifiers.org-registered) when available; fall back
    # to the w3id.org IRI so the property is always present.
    # F2A (strong): dct:title + dct:description present in metadata
    # A1.2: dct:accessRights describes who can access the data
    persistent_id = DATASET_DOI if DATASET_DOI else DATASET_IRI
    literal_meta = [
        # F1B strong
        (DC_TERMS  + "identifier",    persistent_id),
        (SCHEMA_NS + "identifier",    persistent_id),
        # F1B weak: void:uriSpace exposes identifiers.org namespace in metadata
        (VOID_NS   + "uriSpace",      "http://identifiers.org/ensembl/"),
        # F2A strong
        (DC_TERMS  + "title",         "CPP Mechanisms Dataset"),
        (DC_TERMS  + "description",
        "This ontology extends the Semanticscience Integrated Ontology (SIO) "
        "to represent the mechanisms by which cell-penetrating peptides (CPPs) "
        "enter cells. It defines: (1) Ensembl gene activators (SIO_010035) with "
        "activator roles realised in process up-regulation, linked to GO-term "
        "uptake mechanism instances via positive regulation (SIO_001401); "
        "(2) ChEBI chemical inhibitors (SIO_010435) with inhibitor roles realised "
        "in down-regulation, linked to GO terms via negative regulation (SIO_001402); "
        "(3) CPP individuals (CellPenetratingPeptide, subclass of SIO_001425), "
        "Cargo individuals (subclass of SIO_000004), and CPP-Complex individuals "
        "representing peptide-cargo assemblies with has-component-part relations; "
        "(4) Cell line individuals (SIO_010054) from CLO and subcellular entity "
        "individuals (SIO_001400) indicating delivery locations; "
        "(5) Experiment individuals (SIO_000994) linking each CPP-Complex and "
        "cell line as participants of the uptake mechanism, with subcellular "
        "delivery encoded as is-located-in relations, and each experiment "
        "described by a PubMed or patent document (SIO_000148)."),
        (DC_TERMS  + "creator",       "Maria Gomez"),
        (DC_TERMS  + "contributor",   "Robert Hoehndorf"),
        (DC_TERMS  + "created",       "2026-03-19"),
        (DC_TERMS  + "modified",      "2026-03-19"),
        (DC_TERMS  + "subject",
        "cell-penetrating peptides; endocytosis; gene regulation; "
        "chemical inhibition"),
        # A1.2 — access rights description (literal)
        (DC_TERMS  + "accessRights",  "Open access under CC BY 4.0"),
    ]
    for prop_iri_str, value in literal_meta:
        prop       = factory.getOWLAnnotationProperty(IRI.create(prop_iri_str))
        annotation = factory.getOWLAnnotation(prop, factory.getOWLLiteral(value))
        manager.applyChange(AddOntologyAnnotation(ontology, annotation))

    # ------------------------------------------------------------------ #
    # IRI-valued annotations                                              #
    # ------------------------------------------------------------------ #
    # A1.2: dct:license and schema:license MUST be IRI-typed (not literals)
    #       so FAIR-Checker can resolve and validate the access policy.
    #       Using both ensures strong + weak A1.2 checks pass.
    # F2A strong: dcat:accessURL / dcat:downloadURL must be IRI-typed.
    # I3:  Each IRI-typed annotation adds a domain to the authority count.
    #       Domains covered:
    #         - creativecommons.org  (license)
    #         - semanticscience.org  (void:vocabulary → SIO)
    #         - purl.obolibrary.org  (void:vocabulary → GO/ChEBI)
    #         - w3id.org             (ontology + dataset IRIs)
    #       → 4 distinct domains, satisfying the ≥3 threshold.
    iri_meta = [
        # A1.2 — license as IRI (required by FAIR-Checker for access policy check)
        (DC_TERMS  + "license",       "https://creativecommons.org/licenses/by/4.0/"),
        (SCHEMA_NS + "license",       "https://creativecommons.org/licenses/by/4.0/"),
        # F2A — access/download endpoints as IRIs
        (DCAT_NS   + "accessURL",     ONTOLOGY_VERSION),
        (DCAT_NS   + "downloadURL",   ONTOLOGY_VERSION),
        # I3 — void:vocabulary links to shared vocabularies in different domains,
        #      adding semanticscience.org and purl.obolibrary.org to authority count
        (VOID_NS   + "vocabulary",    "http://semanticscience.org/resource/"),
        (VOID_NS   + "vocabulary",    "http://purl.obolibrary.org/obo/"),
        # F4 — VOID / DCAT dataset type as IRI
        (VOID_NS   + "Dataset",       DATASET_IRI),
        (DCAT_NS   + "Dataset",       DATASET_IRI),
    ]
    for prop_iri_str, value in iri_meta:
        prop       = factory.getOWLAnnotationProperty(IRI.create(prop_iri_str))
        annotation = factory.getOWLAnnotation(prop, IRI.create(value))
        manager.applyChange(AddOntologyAnnotation(ontology, annotation))

    # ------------------------------------------------------------------ #
    # dcat:DataService — exposes endpointURL + endpointDescription        #
    # DCAT2 requires these properties on a DataService node, not directly #
    # on the Dataset. The service is linked back via dcat:servesDataset.  #
    # ------------------------------------------------------------------ #
    RDF_NS      = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    service_iri = IRI.create(DATASET_IRI + "/service")
    dataset_iri = IRI.create(DATASET_IRI)
    service_triples = [
        (RDF_NS  + "type",                    IRI.create(DCAT_NS + "DataService")),
        (DCAT_NS + "endpointURL",             IRI.create(SPARQL_ENDPOINT)),
        (DCAT_NS + "endpointDescription",     IRI.create(SERVICE_DESCRIPTION)),
        (DCAT_NS + "servesDataset",           dataset_iri),
        (RDFS_NS + "label",                   factory.getOWLLiteral("CPP KG SPARQL Endpoint")),
    ]
    for prop_iri_str, value in service_triples:
        prop = factory.getOWLAnnotationProperty(IRI.create(prop_iri_str))
        manager.applyChange(AddAxiom(ontology,
            factory.getOWLAnnotationAssertionAxiom(prop, service_iri, value)))


def _annotate_class(manager, ontology, factory, class_iri_str, label, description):
    """Add rdfs:label and rdfs:comment to a named class IRI."""
    subj     = IRI.create(class_iri_str)
    rdfs_lbl = factory.getRDFSLabel()
    rdfs_cmt = factory.getOWLAnnotationProperty(IRI.create(RDFS_NS + "comment"))
    for ann_prop, value in [(rdfs_lbl, label), (rdfs_cmt, description)]:
        manager.applyChange(AddAxiom(ontology,
            factory.getOWLAnnotationAssertionAxiom(
                ann_prop, subj, factory.getOWLLiteral(value))))


def _annotate_individual(manager, ontology, factory, subject_iri_str, label, db_source_iri=None):
    """
    Add F2 / F3 annotations to a named individual.

    F2: rdfs:label with a human-readable local identifier.
    F3: dc:identifier making the IRI-encoded identity explicit as a literal.
        Optionally rdfs:seeAlso pointing to the source database record.
    """
    subj        = IRI.create(subject_iri_str)
    rdfs_label  = factory.getRDFSLabel()
    dc_id_prop  = factory.getOWLAnnotationProperty(IRI.create(DC_TERMS + "identifier"))

    for axiom in [
        factory.getOWLAnnotationAssertionAxiom(
            rdfs_label, subj, factory.getOWLLiteral(label)),
        factory.getOWLAnnotationAssertionAxiom(
            dc_id_prop, subj, factory.getOWLLiteral(subject_iri_str)),
    ]:
        manager.applyChange(AddAxiom(ontology, axiom))

    # F3 — rdfs:seeAlso → source database HTML page (if provided)
    if db_source_iri:
        see_also = factory.getOWLAnnotationProperty(IRI.create(RDFS_NS + "seeAlso"))
        manager.applyChange(AddAxiom(ontology,
            factory.getOWLAnnotationAssertionAxiom(
                see_also, subj, IRI.create(db_source_iri))))


# Annotation property IRIs for CPP-domain features (all under CPP_SCHEMA_NS)
CPP_FEATURE_PROPS = {
    # CellPenetratingPeptide features
    "Sequence":               CPP_SCHEMA_NS + "sequence",
    "dssp":                   CPP_SCHEMA_NS + "dssp",
    "SMILES":                 CPP_SCHEMA_NS + "smiles",
    "Peptide Name":           CPP_SCHEMA_NS + "peptideName",
    "Source of Peptide":      CPP_SCHEMA_NS + "sourceOfPeptide",
    "Chemical Modification":  CPP_SCHEMA_NS + "chemicalModification",
    "N-terminal Modification":CPP_SCHEMA_NS + "nTerminalModification",
    "C terminal Modification":CPP_SCHEMA_NS + "cTerminalModification",
    "Chirality":              CPP_SCHEMA_NS + "chirality",
    "Linearity":              CPP_SCHEMA_NS + "linearity",
    "Ionic State":            CPP_SCHEMA_NS + "ionicState",
    # Experiment features
    "In-vivo Model":          CPP_SCHEMA_NS + "inVivoModel",
    "In-vitro model":         CPP_SCHEMA_NS + "inVitroModel",
    "Uptake Efficiency":      CPP_SCHEMA_NS + "uptakeEfficiency",
    # Cell line feature
    "Cell Line":              CPP_SCHEMA_NS + "cellLine",
    # Cargo feature
    "Cargo Type":             CPP_SCHEMA_NS + "cargoType",
}


def _sanitize_xml(text: str) -> str:
    """
    Remove characters that are illegal in XML 1.0 documents.
    Valid ranges: #x9 | #xA | #xD | #x20–#xD7FF | #xE000–#xFFFD | #x10000–#x10FFFF
    """
    return "".join(
        ch for ch in text
        if ch in ("\t", "\n", "\r")
        or ("\x20" <= ch <= "\ud7ff")
        or ("\ue000" <= ch <= "\ufffd")
        or ("\U00010000" <= ch <= "\U0010ffff")
    )


def _add_features(manager, ontology, factory, subject_iri_str, col_value_pairs):
    """
    Add literal annotation assertions for feature columns to a named individual.
    Skips columns whose value is null, empty, or the string 'nan'.
    Strips invalid XML 1.0 control characters before writing literals.

    Args:
        col_value_pairs: iterable of (csv_column_name, value) tuples.
    """
    subj = IRI.create(subject_iri_str)
    for col, value in col_value_pairs:
        if col not in CPP_FEATURE_PROPS:
            continue
        val_str = _sanitize_xml(str(value).strip())
        if not val_str or val_str.lower() == "nan":
            continue
        prop = factory.getOWLAnnotationProperty(IRI.create(CPP_FEATURE_PROPS[col]))
        manager.applyChange(AddAxiom(ontology,
            factory.getOWLAnnotationAssertionAxiom(
                prop, subj, factory.getOWLLiteral(val_str))))


# ============================================================================
# PHASE 1: LOAD SIO ONTOLOGY AND POPULATE GENE INDIVIDUALS
# ============================================================================

def extend_gene_regulation(gene_to_go_file: str, output_file: str):
    """
    Load Ontology/sio.owl and add Ensembl gene individuals from gene_to_go_file.

    TBox addition:
        cpp:UptakeMechanism  rdfs:subClassOf  sio:SIO_000006  (process)

    ABox additions per unique gene URI:
        gene  rdf:type  sio:SIO_010035          (gene)

    ABox additions per unique GO term URI:
        go_term  rdf:type  cpp:UptakeMechanism

    ABox additions per unique uptake mechanism:
        role          rdf:type  sio:SIO_000804          (activator role)
        role          sio:SIO_000356  upreg_go_proc     (is realized in)

    ABox additions per unique (gene, go_term) pair:
        gene          sio:SIO_000008  role              (has attribute)
        role         sio:SIO_000356  upreg_go_proc     (is realized in)
        upreg_go_proc sio:SIO_001401 go_term           (positively regulates)

    Args:
        gene_to_go_file: Path to triplets/gene_to_go.tsv
        output_file:     Path where the populated ontology will be saved
    """

    print("\n" + "="*70)
    print("PHASE 1: Loading SIO Ontology and Populating Gene Individuals")
    print("="*70)

    # Load base ontology
    print(f"\nLoading base ontology from {SIO_OWL} ...")
    manager  = OWLManager.createOWLOntologyManager()
    ontology = manager.loadOntologyFromOntologyDocument(java.io.File(SIO_OWL))
    factory  = manager.getOWLDataFactory()

    # --- TBox: UptakeMechanism subclass of SIO process ---
    sio_process_cl   = factory.getOWLClass(IRI.create(SIO_PROCESS))
    uptake_mech_cl   = factory.getOWLClass(IRI.create(UPTAKE_MECHANISM_CLASS))
    manager.applyChange(AddAxiom(ontology,
        factory.getOWLSubClassOfAxiom(uptake_mech_cl, sio_process_cl)))
    _annotate_class(manager, ontology, factory, UPTAKE_MECHANISM_CLASS,
        "Uptake Mechanism",
        "A biological process by which a cell-penetrating peptide (CPP) enters "
        "a cell. Subclass of SIO process (SIO_000006). Instances are GO terms "
        "representing specific endocytic or non-endocytic pathways.")

    # --- Reusable OWL entities ---
    gene_class        = factory.getOWLClass(IRI.create(SIO_GENE))
    has_attribute     = factory.getOWLObjectProperty(IRI.create(SIO_HAS_ATTRIBUTE))
    is_attribute_of   = factory.getOWLObjectProperty(IRI.create(SIO_IS_ATTRIBUTE_OF))
    activator_role_cl = factory.getOWLClass(IRI.create(SIO_ACTIVATOR_ROLE))
    is_realized_in    = factory.getOWLObjectProperty(IRI.create(SIO_IS_REALIZED_IN))
    realizes          = factory.getOWLObjectProperty(IRI.create(SIO_REALIZES))
    upregulation_cl   = factory.getOWLClass(IRI.create(SIO_UPREGULATION))
    gene_to_go_prop   = factory.getOWLObjectProperty(IRI.create(GENE_TO_GO_RELATION))

    # F1 / F2 / F3 — Set ontology IRI, version, and DC metadata
    _apply_ontology_metadata(manager, ontology, factory)

    # Read all rows (drop any trailing empty lines)
    df = pd.read_csv(gene_to_go_file, sep='\t', header=None).dropna()

    # --- User-specified positive-regulation GO terms as up-regulation processes ---
    print("Adding user-specified positive regulation GO terms as up-regulation processes...")
    user_pos_reg_terms = {
        'http://purl.obolibrary.org/obo/GO_2000370': 'positive regulation of clathrin-dependent endocytosis',
        'http://purl.obolibrary.org/obo/GO_2001288': 'positive regulation of caveolin-mediated endocytosis',
        'http://purl.obolibrary.org/obo/GO_1905303': 'positive regulation of macropinocytosis',
        'http://purl.obolibrary.org/obo/GO_0050766': 'positive regulation of phagocytosis',
        CPP_DATASET_NS + 'pos_reg_clathrin_caveolae_independent_endocytosis': 'positive regulation of clathrin- and caveolae-independent endocytosis'
    }
    for go_iri_str, go_label in user_pos_reg_terms.items():
        go_local = go_iri_str.split("/")[-1]
        go_ind   = factory.getOWLNamedIndividual(IRI.create(go_iri_str))
        manager.applyChange(AddAxiom(ontology,
            factory.getOWLDeclarationAxiom(go_ind)))
        manager.applyChange(AddAxiom(ontology,
            factory.getOWLClassAssertionAxiom(upregulation_cl, go_ind)))
        db_source = None
        if "obolibrary" in go_iri_str:
            db_source = f"https://amigo.geneontology.org/amigo/term/{go_local.replace('_', ':')}"
        _annotate_individual(manager, ontology, factory, go_iri_str, go_label, db_source_iri=db_source)

    # Map uptake mechanisms to their positive-regulation process.
    # Keys match uptake GO IDs used in triplets/gene_to_go.tsv.
    # Keep an internal IRI for clathrin- and caveolae-independent endocytosis.
    pos_reg_map = {
        # Current uptake GO terms present in gene_to_go.tsv
        'http://purl.obolibrary.org/obo/GO_0072583': 'http://purl.obolibrary.org/obo/GO_2000370',  # clathrin-dependent endocytosis
        'http://purl.obolibrary.org/obo/GO_0072584': 'http://purl.obolibrary.org/obo/GO_2001288',  # caveolin-mediated endocytosis
        'http://purl.obolibrary.org/obo/GO_0044351': 'http://purl.obolibrary.org/obo/GO_1905303',  # macropinocytosis
        'http://purl.obolibrary.org/obo/GO_0006909': 'http://purl.obolibrary.org/obo/GO_0050766',  # phagocytosis
        'http://purl.obolibrary.org/obo/GO_0160294': CPP_DATASET_NS + 'pos_reg_clathrin_caveolae_independent_endocytosis',  # clathrin- and caveolae-independent endocytosis

        # Backward-compatible aliases (if legacy uptake GO IDs appear)
        'http://purl.obolibrary.org/obo/GO_0006903': 'http://purl.obolibrary.org/obo/GO_2000370',
        'http://purl.obolibrary.org/obo/GO_0045334': 'http://purl.obolibrary.org/obo/GO_2001288',
        'http://purl.obolibrary.org/obo/GO_0006905': 'http://purl.obolibrary.org/obo/GO_1905303',
        'http://purl.obolibrary.org/obo/GO_0006906': CPP_DATASET_NS + 'pos_reg_clathrin_caveolae_independent_endocytosis',
        'http://purl.obolibrary.org/obo/GO_0006907': CPP_DATASET_NS + 'pos_reg_clathrin_caveolae_independent_endocytosis',
        'http://purl.obolibrary.org/obo/GO_0006908': CPP_DATASET_NS + 'pos_reg_clathrin_caveolae_independent_endocytosis',
    }

    # Work on unique (gene, mechanism) pairs to avoid duplicate role instances
    df_pairs = df.drop_duplicates()

    # --- Gene individuals ---
    genes = df[0].unique()
    print(f"Adding {len(genes)} gene individuals...")
    for gene_iri_str in genes:
        local_id    = gene_iri_str.split("/")[-1]          # e.g. ENSG00000143226
        gene_ind    = factory.getOWLNamedIndividual(IRI.create(gene_iri_str))

        for ind_to_declare in [gene_ind]:
            manager.applyChange(AddAxiom(ontology, factory.getOWLDeclarationAxiom(ind_to_declare)))

        for axiom in [
            factory.getOWLClassAssertionAxiom(gene_class,        gene_ind),
        ]:
            manager.applyChange(AddAxiom(ontology, axiom))

        # F2 / F3: labels + dc:identifier + rdfs:seeAlso → Ensembl page
        _annotate_individual(manager, ontology, factory, gene_iri_str, local_id,
            db_source_iri=f"https://www.ensembl.org/id/{local_id}")

    # --- UptakeMechanism individuals (unique GO terms) ---
    go_terms = df[1].unique()
    print(f"Adding {len(go_terms)} UptakeMechanism individuals (GO terms)...")
    for go_iri_str in go_terms:
        go_local = go_iri_str.split("/")[-1]               # e.g. GO_0006909
        go_ind   = factory.getOWLNamedIndividual(IRI.create(go_iri_str))
        manager.applyChange(AddAxiom(ontology,
            factory.getOWLDeclarationAxiom(go_ind)))
        manager.applyChange(AddAxiom(ontology,
            factory.getOWLClassAssertionAxiom(uptake_mech_cl, go_ind)))
        # F2 / F3: label + identifier + seeAlso → AmiGO
        _annotate_individual(manager, ontology, factory, go_iri_str, go_local,
            db_source_iri=f"https://amigo.geneontology.org/amigo/term/{go_local.replace('_', ':')}")

    # --- Shared activator role instances: one per uptake mechanism ---
    role_for_mechanism = {}
    for uptake_mech_iri in sorted(df_pairs[1].unique()):
        uptake_local_id = uptake_mech_iri.split("/")[-1]
        role_iri = CPP_DATASET_NS + uptake_local_id + "_activator_role"
        role_ind = factory.getOWLNamedIndividual(IRI.create(role_iri))
        upreg_proc_iri = pos_reg_map.get(uptake_mech_iri)
        if upreg_proc_iri:
            upreg_ind = factory.getOWLNamedIndividual(IRI.create(upreg_proc_iri))
            uptake_mech_ind = factory.getOWLNamedIndividual(IRI.create(uptake_mech_iri))
            role_for_mechanism[uptake_mech_iri] = role_ind
            manager.applyChange(AddAxiom(ontology, factory.getOWLDeclarationAxiom(role_ind)))
            manager.applyChange(AddAxiom(ontology, factory.getOWLClassAssertionAxiom(activator_role_cl, role_ind)))
            manager.applyChange(AddAxiom(ontology,
                factory.getOWLObjectPropertyAssertionAxiom(is_realized_in, role_ind, upreg_ind)))
            manager.applyChange(AddAxiom(ontology,
                factory.getOWLObjectPropertyAssertionAxiom(realizes, upreg_ind, role_ind)))
            manager.applyChange(AddAxiom(ontology,
                factory.getOWLObjectPropertyAssertionAxiom(gene_to_go_prop, upreg_ind, uptake_mech_ind)))
            _annotate_individual(manager, ontology, factory, role_iri,
                f"{uptake_local_id} activator role")
        else:
            print(f"Warning: No positive regulation process mapping for uptake mechanism {uptake_mech_iri}")

    # --- Gene → shared ActivatorRole links (one per unique gene/mechanism pair) ---
    print(f"Linking {len(df_pairs)} gene→shared-activator-role pairs...")
    for _, row in df_pairs.iterrows():
        gene_ind = factory.getOWLNamedIndividual(IRI.create(row[0]))
        uptake_mech_iri = row[1]
        role_ind = role_for_mechanism.get(uptake_mech_iri)
        if role_ind:
            manager.applyChange(AddAxiom(ontology,
                factory.getOWLObjectPropertyAssertionAxiom(has_attribute, gene_ind, role_ind)))
            manager.applyChange(AddAxiom(ontology,
                factory.getOWLObjectPropertyAssertionAxiom(is_attribute_of, role_ind, gene_ind)))

    # Save result
    out_iri = IRI.create(java.io.File(output_file).getAbsoluteFile().toURI())
    manager.saveOntology(ontology, out_iri)
    print(f"Initial ontology saved to {output_file}")
    return ontology

# ============================================================================
# PHASE 2: POPULATE INHIBITOR INDIVIDUALS
# ============================================================================

def extend_inhibitor_regulation(chebi_to_go_file: str, input_file: str, output_file: str):
    """
    Load a previously built ontology and add ChEBI inhibitor individuals from
    chebi_to_go_file, then chain each inhibitor to its associated GO term
    (UptakeMechanism instance) via down-regulation.

    ABox additions per unique ChEBI URI:
        chebi    rdf:type  sio:SIO_010435          (inhibitor)

    ABox additions per unique GO term URI:
        go_term  rdf:type  cpp:UptakeMechanism     (ensured / idempotent)

    ABox additions per unique uptake mechanism:
        role     rdf:type  sio:SIO_000803          (inhibitor role)
        role     sio:SIO_000356  downreg           (is realized in)

    ABox additions per unique (chebi, go_term) pair:
        chebi    sio:SIO_000008  role              (has attribute)
        downreg  sio:SIO_001402  go_term           (negatively regulates)

    Args:
        chebi_to_go_file: Path to triplets/chebi_to_go.tsv
        input_file:       Path to the ontology produced by extend_gene_regulation
        output_file:      Path where the extended ontology will be saved
    """

    print("\n" + "="*70)
    print("PHASE 2: Populating Inhibitor Individuals")
    print("="*70)

    # Load ontology from previous phase
    print(f"\nLoading ontology from {input_file} ...")
    manager  = OWLManager.createOWLOntologyManager()
    ontology = manager.loadOntologyFromOntologyDocument(java.io.File(input_file))
    factory  = manager.getOWLDataFactory()

    # --- Reusable OWL entities ---
    uptake_mech_cl   = factory.getOWLClass(IRI.create(UPTAKE_MECHANISM_CLASS))
    inhibitor_cl     = factory.getOWLClass(IRI.create(SIO_INHIBITOR))
    inhibitor_role_cl = factory.getOWLClass(IRI.create(SIO_INHIBITOR_ROLE))
    downreg_cl       = factory.getOWLClass(IRI.create(SIO_DOWNREGULATION))
    has_attribute    = factory.getOWLObjectProperty(IRI.create(SIO_HAS_ATTRIBUTE))
    is_attribute_of  = factory.getOWLObjectProperty(IRI.create(SIO_IS_ATTRIBUTE_OF))
    is_realized_in   = factory.getOWLObjectProperty(IRI.create(SIO_IS_REALIZED_IN))
    realizes         = factory.getOWLObjectProperty(IRI.create(SIO_REALIZES))
    neg_regulates    = factory.getOWLObjectProperty(IRI.create(CHEBI_TO_GO_RELATION))

    # Read all rows (drop any trailing empty lines)
    df = pd.read_csv(chebi_to_go_file, sep='\t', header=None).dropna()
    # Work on unique (inhibitor, mechanism) pairs to avoid duplicate role instances
    df_pairs = df.drop_duplicates()

    # --- ChEBI inhibitor individuals ---
    chebis = df[0].unique()
    print(f"Adding {len(chebis)} inhibitor individuals (ChEBI)...")
    for chebi_iri_str in chebis:
        local_id    = chebi_iri_str.split("/")[-1]         # e.g. CHEBI_2639
        chebi_ind   = factory.getOWLNamedIndividual(IRI.create(chebi_iri_str))

        for ind_to_declare in [chebi_ind]:
            manager.applyChange(AddAxiom(ontology, factory.getOWLDeclarationAxiom(ind_to_declare)))

        for axiom in [
            factory.getOWLClassAssertionAxiom(inhibitor_cl,      chebi_ind),
        ]:
            manager.applyChange(AddAxiom(ontology, axiom))

        # F2 / F3: labels + dc:identifier + rdfs:seeAlso → EBI ChEBI page
        chebi_accession = local_id.replace("_", ":")       # CHEBI:2639
        _annotate_individual(manager, ontology, factory, chebi_iri_str, chebi_accession,
            db_source_iri=f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId={chebi_accession}")

    # --- User-specified negative regulation GO terms as down-regulation processes ---
    print("Adding user-specified negative regulation GO terms as down-regulation processes...")
    user_go_terms = {
        'http://purl.obolibrary.org/obo/GO_1900186': 'negative regulation of clathrin-dependent endocytosis',
        'http://purl.obolibrary.org/obo/GO_2001287': 'negative regulation of caveolin-mediated endocytosis',
        'http://purl.obolibrary.org/obo/GO_1905302': 'negative regulation of macropinocytosis',
        'http://purl.obolibrary.org/obo/GO_0050765': 'negative regulation of phagocytosis',
        CPP_DATASET_NS + 'neg_reg_clathrin_caveolae_independent_endocytosis': 'negative regulation of clathrin- and caveolae-independent endocytosis'
    }

    for go_iri_str, go_label in user_go_terms.items():
        go_local = go_iri_str.split("/")[-1]
        go_ind   = factory.getOWLNamedIndividual(IRI.create(go_iri_str))
        manager.applyChange(AddAxiom(ontology,
            factory.getOWLDeclarationAxiom(go_ind)))
        manager.applyChange(AddAxiom(ontology,
            factory.getOWLClassAssertionAxiom(downreg_cl, go_ind))) # Changed to downreg_cl
        db_source = None
        if "obolibrary" in go_iri_str:
            db_source = f"https://amigo.geneontology.org/amigo/term/{go_local.replace('_', ':')}"
        _annotate_individual(manager, ontology, factory, go_iri_str, go_label, db_source_iri=db_source)

    # Map uptake mechanisms to their negative-regulation process.
    # The keys here must match the uptake GO IDs used in triplets/chebi_to_go.tsv.
    # We keep the internal IRI for clathrin- and caveolae-independent endocytosis.
    neg_reg_map = {
        # Current uptake GO terms present in chebi_to_go.tsv
        'http://purl.obolibrary.org/obo/GO_0072583': 'http://purl.obolibrary.org/obo/GO_1900186',  # clathrin-dependent endocytosis
        'http://purl.obolibrary.org/obo/GO_0072584': 'http://purl.obolibrary.org/obo/GO_2001287',  # caveolin-mediated endocytosis
        'http://purl.obolibrary.org/obo/GO_0044351': 'http://purl.obolibrary.org/obo/GO_1905302',  # macropinocytosis
        'http://purl.obolibrary.org/obo/GO_0006909': 'http://purl.obolibrary.org/obo/GO_0050765',  # phagocytosis
        'http://purl.obolibrary.org/obo/GO_0160294': CPP_DATASET_NS + 'neg_reg_clathrin_caveolae_independent_endocytosis',  # clathrin- and caveolae-independent endocytosis

        # Backward-compatible aliases (if legacy uptake GO IDs appear)
        'http://purl.obolibrary.org/obo/GO_0006903': 'http://purl.obolibrary.org/obo/GO_1900186',
        'http://purl.obolibrary.org/obo/GO_0006905': 'http://purl.obolibrary.org/obo/GO_1905302',
        'http://purl.obolibrary.org/obo/GO_0045334': 'http://purl.obolibrary.org/obo/GO_2001287',
        'http://purl.obolibrary.org/obo/GO_0006906': CPP_DATASET_NS + 'neg_reg_clathrin_caveolae_independent_endocytosis',
        'http://purl.obolibrary.org/obo/GO_0006907': CPP_DATASET_NS + 'neg_reg_clathrin_caveolae_independent_endocytosis',
        'http://purl.obolibrary.org/obo/GO_0006908': CPP_DATASET_NS + 'neg_reg_clathrin_caveolae_independent_endocytosis',
    }

    # --- Shared inhibitor role instances: one per uptake mechanism ---
    role_for_mechanism = {}
    for uptake_mech_iri in sorted(df_pairs[1].unique()):
        uptake_local_id = uptake_mech_iri.split("/")[-1]
        role_iri = CPP_DATASET_NS + uptake_local_id + "_inhibitor_role"
        role_ind = factory.getOWLNamedIndividual(IRI.create(role_iri))
        downreg_proc_iri = neg_reg_map.get(uptake_mech_iri)
        if downreg_proc_iri:
            downreg_ind = factory.getOWLNamedIndividual(IRI.create(downreg_proc_iri))
            uptake_mech_ind = factory.getOWLNamedIndividual(IRI.create(uptake_mech_iri))
            role_for_mechanism[uptake_mech_iri] = role_ind
            manager.applyChange(AddAxiom(ontology, factory.getOWLDeclarationAxiom(role_ind)))
            manager.applyChange(AddAxiom(ontology, factory.getOWLClassAssertionAxiom(inhibitor_role_cl, role_ind)))
            manager.applyChange(AddAxiom(ontology,
                factory.getOWLObjectPropertyAssertionAxiom(is_realized_in,  role_ind,   downreg_ind)))
            manager.applyChange(AddAxiom(ontology,
                factory.getOWLObjectPropertyAssertionAxiom(realizes,        downreg_ind, role_ind)))
            manager.applyChange(AddAxiom(ontology,
                factory.getOWLObjectPropertyAssertionAxiom(neg_regulates, downreg_ind, uptake_mech_ind)))
            _annotate_individual(manager, ontology, factory, role_iri,
                f"{uptake_local_id} inhibitor role")
        else:
             print(f"Warning: No negative regulation process mapping for uptake mechanism {uptake_mech_iri}")

    # --- Inhibitor → shared InhibitorRole links (one per unique chebi/mechanism pair) ---
    print(f"Linking {len(df_pairs)} inhibitor→shared-inhibitor-role pairs...")
    for _, row in df_pairs.iterrows():
        chebi_ind = factory.getOWLNamedIndividual(IRI.create(row[0]))
        uptake_mech_iri = row[1]
        role_ind = role_for_mechanism.get(uptake_mech_iri)
        if role_ind:
            manager.applyChange(AddAxiom(ontology,
                factory.getOWLObjectPropertyAssertionAxiom(has_attribute, chebi_ind, role_ind)))
            manager.applyChange(AddAxiom(ontology,
                factory.getOWLObjectPropertyAssertionAxiom(is_attribute_of, role_ind, chebi_ind)))

    # Save as OWL/RDF-XML
    out_iri = IRI.create(java.io.File(output_file).getAbsoluteFile().toURI())
    manager.saveOntology(ontology, out_iri)
    print(f"Inhibitor ontology saved to {output_file}")

    # Save as Turtle (.ttl) in the same folder
    ttl_file = output_file.replace(".owl", ".ttl")
    TurtleDocumentFormat = jpype.JClass("org.semanticweb.owlapi.formats.TurtleDocumentFormat")
    ttl_iri = IRI.create(java.io.File(ttl_file).getAbsoluteFile().toURI())
    manager.saveOntology(ontology, TurtleDocumentFormat(), ttl_iri)
    print(f"Turtle serialisation saved to {ttl_file}")

# ============================================================================
# PHASE 3 HELPER: GENERATE TRIPLET TSV FILES FROM CPP CSV
# ============================================================================

def prepare_annotation_files(cpp_csv_file: str, output_dir: str = "triplets"):
    """
    Read the CPP CSV file and write four subject→object TSV files required by
    mowl insert_annotations.  All object columns are converted to full OBO PURLs
    where needed.

    Files produced (tab-separated, no header):
        <output_dir>/cpp_mechanism.tsv  CPP_ID  →  Main Uptake Mechanism ID
        <output_dir>/cpp_cargo.tsv      CPP_ID  →  CHEBI OBO IRI
        <output_dir>/cpp_location.tsv   CPP_ID  →  Subcellular Delivery ID
        <output_dir>/cpp_cell.tsv       CPP_ID  →  CLO OBO IRI

    Returns:
        Tuple of (mechanism_file, cargo_file, location_file, cell_file) paths.
    """
    OBO = "http://purl.obolibrary.org/obo/"

    df = pd.read_csv(cpp_csv_file, usecols=[
        "CPP_ID", "Main Uptake Mechanism ID",
        "RAG_curie_CheBI", "Subcellular Delivery ID", "RAG_curie_CLO"
    ])

    os.makedirs(output_dir, exist_ok=True)

    def _write_tsv(path: str, subj_col: str, obj_col: str,
                    obj_transform=None) -> str:
        pairs = df[[subj_col, obj_col]].dropna()
        pairs = pairs[(pairs[subj_col].str.strip() != "") &
                    (pairs[obj_col].str.strip() != "")]
        pairs = pairs.drop_duplicates()
        if obj_transform:
            pairs = pairs.copy()
            pairs[obj_col] = pairs[obj_col].apply(obj_transform)
        pairs.to_csv(path, sep="\t", header=False, index=False)
        print(f"  Wrote {len(pairs)} pairs to {path}")
        return path

    def _obo_iri(accession: str) -> str:
        """Convert 'PREFIX:XXXXXXX' to full OBO PURL."""
        return OBO + accession.replace(":", "_")

    mech_file     = _write_tsv(f"{output_dir}/cpp_mechanism.tsv",
                                "CPP_ID", "Main Uptake Mechanism ID")
    cargo_file    = _write_tsv(f"{output_dir}/cpp_cargo.tsv",
                                "CPP_ID", "RAG_curie_CheBI", _obo_iri)
    location_file = _write_tsv(f"{output_dir}/cpp_location.tsv",
                                "CPP_ID", "Subcellular Delivery ID")
    cell_file     = _write_tsv(f"{output_dir}/cpp_cell.tsv",
                                "CPP_ID", "RAG_curie_CLO", _obo_iri)

    return mech_file, cargo_file, location_file, cell_file


# ============================================================================
# PHASE 3: EXTEND ONTOLOGY WITH ANNOTATIONS
# ============================================================================

def extend_ontology_with_annotations(ontology: str,
                                    cpp_csv_file: str,
                                    cpp_mech_file: str,
                                    cpp_cargo_file: str,
                                    cpp_location_file: str,
                                    cpp_cell_file: str,
                                    output_file: str = "extended_cpp_ontology.owl"):
    """
    Phase 3a — TBox enrichment (OWLAPI):
        cpp:CPP-Complex          subClassOf  sio:SIO_000004  (material entity)
        cpp:CellPenetratingPeptide subClassOf sio:SIO_001425  (peptide)

        For each unique RAG_curie_CheBI in cpp_csv_file:
            <obo_chebi_iri>      subClassOf  sio:SIO_000004

        For each unique CPP_ID in cpp_csv_file:
            <cpp_iri>            subClassOf  cpp:CellPenetratingPeptide

        For each unique (CPP_ID, RAG_curie_CheBI) pair — named CPP-Complex subclass:
            cpp:<CPP_x>_<CHEBI_y>_Complex  subClassOf  cpp:CPP-Complex
            cpp:<CPP_x>_<CHEBI_y>_Complex  subClassOf  (SIO_000369 some <cpp_iri>)
            cpp:<CPP_x>_<CHEBI_y>_Complex  subClassOf  (SIO_000369 some <obo_chebi_iri>)

    Phase 3b — ABox annotation chains (mowl insert_annotations):
        Inserts mechanism / cargo / location / cell-line triples from TSV files.

    Args:
        ontology:          Path to the input OWL file (Phase 2 output)
        cpp_csv_file:      Path to the CPP CSV (must have CPP_ID, RAG_curie_CheBI columns)
        cpp_mech_file:     Path to triplets/cpp_mechanism.tsv
        cpp_cargo_file:    Path to triplets/cpp_cargo.tsv
        cpp_location_file: Path to triplets/cpp_location.tsv
        cpp_cell_file:     Path to triplets/cpp_cell.tsv
        output_file:       Base name for the chain of output OWL files
    """
    print("\n" + "="*70)
    print("PHASE 3: Extending Ontology with Annotations")
    print("="*70)

    # ------------------------------------------------------------------ #
    # Phase 3a: TBox enrichment with OWLAPI                               #
    # ------------------------------------------------------------------ #
    print("\nStep 3a: Building CPP-Complex TBox from CSV ...")

    manager = OWLManager.createOWLOntologyManager()
    onto    = manager.loadOntologyFromOntologyDocument(java.io.File(ontology))
    factory = manager.getOWLDataFactory()

    # --- TBox classes ---
    mat_entity_cl    = factory.getOWLClass(IRI.create(SIO_MATERIAL_ENTITY))
    peptide_cl       = factory.getOWLClass(IRI.create(SIO_PEPTIDE))
    processual_role_cl = factory.getOWLClass(IRI.create(SIO_PROCESSUAL_ROLE))
    has_comp_part    = factory.getOWLObjectProperty(IRI.create(SIO_HAS_COMP_PART))
    is_comp_part_of  = factory.getOWLObjectProperty(IRI.create(SIO_IS_COMP_PART_OF))
    has_attribute    = factory.getOWLObjectProperty(IRI.create(SIO_HAS_ATTRIBUTE))
    is_attribute_of  = factory.getOWLObjectProperty(IRI.create(SIO_IS_ATTRIBUTE_OF))
    is_realized_in   = factory.getOWLObjectProperty(IRI.create(SIO_IS_REALIZED_IN))
    realizes         = factory.getOWLObjectProperty(IRI.create(SIO_REALIZES))

    cpp_complex_cl   = factory.getOWLClass(IRI.create(CPP_DATASET_NS + "CPP-Complex"))
    cpp_peptide_cl   = factory.getOWLClass(IRI.create(CPP_DATASET_NS + "CellPenetratingPeptide"))
    cargo_cl         = factory.getOWLClass(IRI.create(CPP_DATASET_NS + "Cargo"))
    cpp_role_cl      = factory.getOWLClass(IRI.create(CPP_DATASET_NS + "CellPenetratingPeptideRole"))
    cargo_role_cl    = factory.getOWLClass(IRI.create(CPP_DATASET_NS + "CargoRole"))

    # TBox subclass axioms
    for axiom in [
        factory.getOWLSubClassOfAxiom(cpp_complex_cl,  mat_entity_cl),
        factory.getOWLSubClassOfAxiom(cpp_peptide_cl,  peptide_cl),
        factory.getOWLSubClassOfAxiom(cargo_cl,        mat_entity_cl),
        factory.getOWLSubClassOfAxiom(cpp_role_cl,     processual_role_cl),
        factory.getOWLSubClassOfAxiom(cargo_role_cl,   processual_role_cl),
    ]:
        manager.applyChange(AddAxiom(onto, axiom))

    # Labels and descriptions for each custom subclass
    _class_meta = [
        (CPP_DATASET_NS + "CPP-Complex",
         "CPP-Complex",
         "A molecular assembly composed of a cell-penetrating peptide (CPP) and "
         "its associated cargo molecule. Subclass of SIO material entity "
         "(SIO_000004). Each instance represents a specific peptide-cargo pair "
         "used in an uptake experiment."),
        (CPP_DATASET_NS + "CellPenetratingPeptide",
         "Cell-Penetrating Peptide",
         "A short peptide capable of translocating across cellular membranes and "
         "facilitating intracellular delivery of cargo. Subclass of SIO peptide "
         "(SIO_001425). Instances are individual CPP sequences from the dataset."),
        (CPP_DATASET_NS + "Cargo",
         "Cargo",
         "A molecule delivered intracellularly by a cell-penetrating peptide. "
         "Subclass of SIO material entity (SIO_000004). Instances are ChEBI "
         "chemical entities representing the delivered payload."),
        (CPP_DATASET_NS + "CellPenetratingPeptideRole",
         "Cell-Penetrating Peptide Role",
         "The processual role played by a CPP within a CPP-Complex, realised "
         "during the uptake mechanism. Subclass of SIO processual role "
         "(SIO_000677)."),
        (CPP_DATASET_NS + "CargoRole",
         "Cargo Role",
         "The processual role played by a cargo molecule within a CPP-Complex, "
         "realised during the uptake mechanism. Subclass of SIO processual role "
         "(SIO_000677)."),
    ]
    for class_iri, label, description in _class_meta:
        _annotate_class(manager, onto, factory, class_iri, label, description)

    # Labels and descriptions for each custom annotation property (CPP_SCHEMA_NS)
    _prop_meta = [
        (CPP_SCHEMA_NS + "sequence",
         "sequence",
         "The amino acid sequence of the cell-penetrating peptide."),
        (CPP_SCHEMA_NS + "dssp",
         "DSSP secondary structure",
         "The secondary structure assignment of the peptide derived using the "
         "DSSP algorithm (e.g. H=helix, E=strand, C=coil)."),
        (CPP_SCHEMA_NS + "smiles",
         "SMILES",
         "The Simplified Molecular-Input Line-Entry System (SMILES) string "
         "representing the chemical structure of the peptide."),
        (CPP_SCHEMA_NS + "peptideName",
         "peptide name",
         "The common or reported name of the cell-penetrating peptide."),
        (CPP_SCHEMA_NS + "sourceOfPeptide",
         "source of peptide",
         "The biological or synthetic origin of the peptide "
         "(e.g. protein-derived, synthetic, chimeric)."),
        (CPP_SCHEMA_NS + "chemicalModification",
         "chemical modification",
         "Any chemical modification applied to the peptide "
         "(e.g. PEGylation, lipidation, fluorescent labelling)."),
        (CPP_SCHEMA_NS + "nTerminalModification",
         "N-terminal modification",
         "Chemical modification present at the N-terminus of the peptide "
         "(e.g. acetylation, fluorophore attachment)."),
        (CPP_SCHEMA_NS + "cTerminalModification",
         "C-terminal modification",
         "Chemical modification present at the C-terminus of the peptide "
         "(e.g. amidation, biotin tag)."),
        (CPP_SCHEMA_NS + "chirality",
         "chirality",
         "Stereochemical configuration of the peptide residues "
         "(e.g. L, D, or mixed)."),
        (CPP_SCHEMA_NS + "linearity",
         "linearity",
         "Structural topology of the peptide backbone "
         "(e.g. linear or cyclic)."),
        (CPP_SCHEMA_NS + "ionicState",
         "ionic state",
         "Net charge character of the peptide at physiological pH "
         "(e.g. cationic, anionic, amphoteric, neutral)."),
        (CPP_SCHEMA_NS + "inVivoModel",
         "in vivo model",
         "The animal or organismal model used in the in vivo uptake experiment."),
        (CPP_SCHEMA_NS + "inVitroModel",
         "in vitro model",
         "The cell culture or biochemical system used in the in vitro uptake "
         "experiment."),
        (CPP_SCHEMA_NS + "uptakeEfficiency",
         "uptake efficiency",
         "A qualitative or quantitative measure of how effectively the CPP "
         "and its cargo were internalised by the target cell."),
        (CPP_SCHEMA_NS + "cellLine",
         "cell line",
         "The name of the cell line used in the uptake experiment."),
        (CPP_SCHEMA_NS + "cargoType",
         "cargo type",
         "The functional category of the molecule delivered by the CPP "
         "(e.g. nucleic acid, protein, small molecule, nanoparticle)."),
    ]
    for prop_iri, label, description in _prop_meta:
        # Declare as owl:AnnotationProperty
        ap = factory.getOWLAnnotationProperty(IRI.create(prop_iri))
        manager.applyChange(AddAxiom(onto,
            factory.getOWLDeclarationAxiom(ap)))
        # Add label and comment
        _annotate_class(manager, onto, factory, prop_iri, label, description)

    # Load full set of columns needed for Phase 3a (including feature columns)
    CPP_FEAT_COLS  = ["Sequence", "dssp", "SMILES", "Peptide Name",
                      "Source of Peptide", "Chemical Modification",
                      "N-terminal Modification", "C terminal Modification",
                      "Chirality", "Linearity", "Ionic State"]
    EXP_FEAT_COLS  = ["In-vivo Model", "In-vitro model", "Uptake Efficiency"]

    df_full = pd.read_csv(cpp_csv_file, usecols=[
        "CPP_ID", "RAG_curie_CheBI",
        "Main Uptake Mechanism ID", "Main Uptake Mechanism",
        "Subcategory Uptake Mechanism ID", "Subcategory Uptake Mechanism",
        "RAG_curie_CLO", "Cell Line", "Cargo Type",
    ] + CPP_FEAT_COLS + EXP_FEAT_COLS)
    df = df_full.dropna(subset=["CPP_ID", "RAG_curie_CheBI"])

    # Pre-build feature lookup dicts (first non-null value per key)
    cpp_features  = (df_full.dropna(subset=["CPP_ID"])
                     .groupby("CPP_ID")[CPP_FEAT_COLS].first().to_dict("index"))
    chebi_features = (df_full.dropna(subset=["RAG_curie_CheBI"])
                      .groupby("RAG_curie_CheBI")[["Cargo Type"]].first().to_dict("index"))
    clo_features   = (df_full.dropna(subset=["RAG_curie_CLO"])
                      .groupby("RAG_curie_CLO")[["Cell Line"]].first().to_dict("index"))

    # --- UptakeMechanism individuals from CSV rows ---
    # Rule: use Subcategory Uptake Mechanism ID when available; otherwise Main.
    uptake_mech_cl = factory.getOWLClass(IRI.create(UPTAKE_MECHANISM_CLASS))
    mech_rows = df_full[["Main Uptake Mechanism ID", "Main Uptake Mechanism",
                          "Subcategory Uptake Mechanism ID",
                          "Subcategory Uptake Mechanism"]].copy()
    # Build (iri, label) pairs applying the subcategory-first rule.
    # Both ID and label columns may contain comma-separated values; each
    # position in the IDs list matches the same position in the labels list.
    def _split_pairs(id_cell, lbl_cell):
        """Return list of (iri, label) from potentially comma-separated cells."""
        ids  = [s.strip() for s in str(id_cell).split(",") if s.strip()]
        lbls = [s.strip() for s in str(lbl_cell).split(",") if s.strip()] \
               if pd.notna(lbl_cell) else []
        return [(iri, lbls[i] if i < len(lbls) else iri.split("/")[-1])
                for i, iri in enumerate(ids)]

    def _pick_mechanisms(row):
        sub_id  = row["Subcategory Uptake Mechanism ID"]
        sub_lbl = row["Subcategory Uptake Mechanism"]
        main_id = row["Main Uptake Mechanism ID"]
        main_lbl = row["Main Uptake Mechanism"]
        if pd.notna(sub_id) and str(sub_id).strip():
            return _split_pairs(sub_id, sub_lbl)
        if pd.notna(main_id) and str(main_id).strip():
            return _split_pairs(main_id, main_lbl)
        return []

    # Flatten all rows and deduplicate, keeping first label for each IRI
    seen_mechs = {}
    for pairs_list in mech_rows.apply(_pick_mechanisms, axis=1):
        for iri_str, label in pairs_list:
            if iri_str not in seen_mechs:
                seen_mechs[iri_str] = label

    print(f"  Asserting {len(seen_mechs)} UptakeMechanism individuals from CSV ...")
    for mech_iri, mech_label in seen_mechs.items():
        mech_ind = factory.getOWLNamedIndividual(IRI.create(mech_iri))
        manager.applyChange(AddAxiom(onto,
            factory.getOWLDeclarationAxiom(mech_ind)))
        manager.applyChange(AddAxiom(onto,
            factory.getOWLClassAssertionAxiom(uptake_mech_cl, mech_ind)))
        go_local = mech_iri.split("/")[-1]
        _annotate_individual(manager, onto, factory, mech_iri, mech_label,
            db_source_iri=f"https://amigo.geneontology.org/amigo/term/{go_local.replace('_', ':')}")

    # --- Cell line individuals (SIO_010054) from RAG_curie_CLO column ---
    cell_line_cl = factory.getOWLClass(IRI.create(SIO_CELL_LINE))
    df_clo = pd.read_csv(cpp_csv_file, usecols=["RAG_curie_CLO", "RAG_label_CLO"]).dropna(subset=["RAG_curie_CLO"])
    df_clo = df_clo[df_clo["RAG_curie_CLO"].str.strip() != ""].drop_duplicates(subset=["RAG_curie_CLO"])

    print(f"  Asserting {len(df_clo)} cell line individuals (SIO_010054) from RAG_curie_CLO ...")
    for _, row in df_clo.iterrows():
        clo_acc  = str(row["RAG_curie_CLO"]).strip()                        # e.g. CLO:0003655
        clo_iri  = "http://purl.obolibrary.org/obo/" + clo_acc.replace(":", "_")
        RAG_label_CLO = str(row["RAG_label_CLO"]).strip() if pd.notna(row["RAG_label_CLO"]) else clo_acc
        clo_ind  = factory.getOWLNamedIndividual(IRI.create(clo_iri))
        manager.applyChange(AddAxiom(onto, factory.getOWLDeclarationAxiom(clo_ind)))
        manager.applyChange(AddAxiom(onto,
            factory.getOWLClassAssertionAxiom(cell_line_cl, clo_ind)))
        clo_local = clo_acc.replace(":", "_")
        _annotate_individual(manager, onto, factory, clo_iri, RAG_label_CLO,
            db_source_iri=f"https://www.ebi.ac.uk/ols/ontologies/clo/terms?iri=http://purl.obolibrary.org/obo/{clo_local}")

    # --- Subcellular entity individuals (SIO_001400) from Subcellular Delivery ID ---
    subcell_cl = factory.getOWLClass(IRI.create(SIO_SUBCELLULAR))
    df_sub = pd.read_csv(cpp_csv_file,
                         usecols=["Subcellular Delivery ID", "Subcellular Localization"])
    seen_subcell = {}
    for _, row in df_sub.iterrows():
        id_cell  = row["Subcellular Delivery ID"]
        lbl_cell = row["Subcellular Localization"]
        if pd.isna(id_cell) or not str(id_cell).strip():
            continue
        for iri_str, label in _split_pairs(id_cell, lbl_cell):
            if iri_str not in seen_subcell:
                seen_subcell[iri_str] = label

    print(f"  Asserting {len(seen_subcell)} subcellular entity individuals (SIO_001400) ...")
    for sub_iri, sub_label in seen_subcell.items():
        sub_ind = factory.getOWLNamedIndividual(IRI.create(sub_iri))
        manager.applyChange(AddAxiom(onto, factory.getOWLDeclarationAxiom(sub_ind)))
        manager.applyChange(AddAxiom(onto,
            factory.getOWLClassAssertionAxiom(subcell_cl, sub_ind)))
        go_local = sub_iri.split("/")[-1]
        _annotate_individual(manager, onto, factory, sub_iri, sub_label,
            db_source_iri=f"https://amigo.geneontology.org/amigo/term/{go_local.replace('_', ':')}")

    # Each unique RAG_curie_CheBI → named individual rdf:type cpp:Cargo
    chebi_ind_map = {}
    unique_chebis = df["RAG_curie_CheBI"].unique()
    print(f"  Asserting {len(unique_chebis)} CHEBI individuals as cpp:Cargo instances ...")
    for chebi_acc in unique_chebis:
        obo_iri  = "http://purl.obolibrary.org/obo/" + chebi_acc.replace(":", "_")
        chebi_ind_map[chebi_acc] = obo_iri
        chebi_ind = factory.getOWLNamedIndividual(IRI.create(obo_iri))
        manager.applyChange(AddAxiom(onto, factory.getOWLDeclarationAxiom(chebi_ind)))
        manager.applyChange(AddAxiom(onto,
            factory.getOWLClassAssertionAxiom(cargo_cl, chebi_ind)))
        if chebi_acc in chebi_features:
            _add_features(manager, onto, factory, obo_iri,
                          chebi_features[chebi_acc].items())

    # Each unique CPP_ID → named individual rdf:type cpp:CellPenetratingPeptide
    unique_cpps = df["CPP_ID"].unique()
    print(f"  Asserting {len(unique_cpps)} CPP individuals as cpp:CellPenetratingPeptide instances ...")
    for cpp_iri in unique_cpps:
        cpp_ind = factory.getOWLNamedIndividual(IRI.create(cpp_iri))
        manager.applyChange(AddAxiom(onto, factory.getOWLDeclarationAxiom(cpp_ind)))
        manager.applyChange(AddAxiom(onto,
            factory.getOWLClassAssertionAxiom(cpp_peptide_cl, cpp_ind)))
        if cpp_iri in cpp_features:
            _add_features(manager, onto, factory, cpp_iri,
                          cpp_features[cpp_iri].items())

    # Each unique (CPP_ID, RAG_curie_CheBI) pair → named individual rdf:type cpp:CPP-Complex
    # with ABox property assertions:
    #   complex  sio:SIO_000369  cpp_individual
    #   complex  sio:SIO_000369  chebi_individual
    pairs = df.drop_duplicates(subset=["CPP_ID", "RAG_curie_CheBI"])
    print(f"  Asserting {len(pairs)} CPP-Complex individuals (one per CPP+CHEBI pair) ...")
    for _, row in pairs.iterrows():
        cpp_iri   = row["CPP_ID"]
        chebi_obo = chebi_ind_map[row["RAG_curie_CheBI"]]

        cpp_local   = cpp_iri.split("/")[-1]             # e.g. CPP_001104
        chebi_local = row["RAG_curie_CheBI"].replace(":", "_")  # e.g. CHEBI_38161

        complex_ind = factory.getOWLNamedIndividual(
            IRI.create(CPP_DATASET_NS + f"complex_{cpp_local}_{chebi_local}"))
        cpp_ind   = factory.getOWLNamedIndividual(IRI.create(cpp_iri))
        chebi_ind = factory.getOWLNamedIndividual(IRI.create(chebi_obo))

        # Role individuals — one per (CPP, CHEBI) pair
        cpp_role_ind   = factory.getOWLNamedIndividual(
            IRI.create(CPP_DATASET_NS + f"cpp_role_{cpp_local}_{chebi_local}"))
        cargo_role_ind = factory.getOWLNamedIndividual(
            IRI.create(CPP_DATASET_NS + f"cargo_role_{cpp_local}_{chebi_local}"))

        # Resolve uptake mechanism IRI(s) for this pair (subcategory-first)
        sub_id  = row["Subcategory Uptake Mechanism ID"]
        main_id = row["Main Uptake Mechanism ID"]
        chosen  = sub_id if pd.notna(sub_id) and str(sub_id).strip() else main_id
        mech_inds_for_role = []
        if pd.notna(chosen) and str(chosen).strip():
            mech_inds_for_role = [
                factory.getOWLNamedIndividual(IRI.create(m.strip()))
                for m in str(chosen).split(",") if m.strip()
            ]

        for ind_to_declare in [complex_ind, cpp_role_ind, cargo_role_ind]:
            manager.applyChange(AddAxiom(onto, factory.getOWLDeclarationAxiom(ind_to_declare)))

        for axiom in [
            # CPP-Complex ABox
            factory.getOWLClassAssertionAxiom(cpp_complex_cl,  complex_ind),
            factory.getOWLObjectPropertyAssertionAxiom(has_comp_part,   complex_ind, cpp_ind),
            factory.getOWLObjectPropertyAssertionAxiom(has_comp_part,   complex_ind, chebi_ind),
            factory.getOWLObjectPropertyAssertionAxiom(is_comp_part_of, cpp_ind,     complex_ind),
            factory.getOWLObjectPropertyAssertionAxiom(is_comp_part_of, chebi_ind,   complex_ind),
            # CellPenetratingPeptideRole — type + has_attribute / is_attribute_of
            factory.getOWLClassAssertionAxiom(cpp_role_cl,    cpp_role_ind),
            factory.getOWLObjectPropertyAssertionAxiom(has_attribute,   cpp_ind,      cpp_role_ind),
            factory.getOWLObjectPropertyAssertionAxiom(is_attribute_of, cpp_role_ind, cpp_ind),
            # CargoRole — type + has_attribute / is_attribute_of
            factory.getOWLClassAssertionAxiom(cargo_role_cl,  cargo_role_ind),
            factory.getOWLObjectPropertyAssertionAxiom(has_attribute,   chebi_ind,    cargo_role_ind),
            factory.getOWLObjectPropertyAssertionAxiom(is_attribute_of, cargo_role_ind, chebi_ind),
        ]:
            manager.applyChange(AddAxiom(onto, axiom))

        # Roles are realized in the uptake mechanism instance(s) for this pair
        for mech_ind in mech_inds_for_role:
            for role_ind in [cpp_role_ind, cargo_role_ind]:
                manager.applyChange(AddAxiom(onto,
                    factory.getOWLObjectPropertyAssertionAxiom(is_realized_in, role_ind, mech_ind)))
                manager.applyChange(AddAxiom(onto,
                    factory.getOWLObjectPropertyAssertionAxiom(realizes, mech_ind, role_ind)))

    # --- Experiment individuals (SIO_000994) ---
    # One experiment per CSV row; links CPP-Complex, UptakeMechanism,
    # CellLine, and SubcellularEntity via has_participant / is_participant_in.
    experiment_cl   = factory.getOWLClass(IRI.create(SIO_EXPERIMENT))
    has_participant = factory.getOWLObjectProperty(IRI.create(SIO_HAS_PARTICIPANT))
    is_part_in      = factory.getOWLObjectProperty(IRI.create(SIO_IS_PART_IN))
    is_located_in   = factory.getOWLObjectProperty(IRI.create(SIO_IS_LOCATED_IN))
    is_location_of  = factory.getOWLObjectProperty(IRI.create(SIO_IS_LOCATION_OF))
    document_cl     = factory.getOWLClass(IRI.create(SIO_DOCUMENT))
    is_described_by = factory.getOWLObjectProperty(IRI.create(SIO_IS_DESCRIBED_BY))
    describes       = factory.getOWLObjectProperty(IRI.create(SIO_DESCRIBES))
    has_proper_part = factory.getOWLObjectProperty(IRI.create(SIO_HAS_PROPER_PART))
    is_proper_part  = factory.getOWLObjectProperty(IRI.create(SIO_IS_PROPER_PART))

    df_exp = pd.read_csv(cpp_csv_file, usecols=[
        "id", "CPP_ID", "RAG_curie_CheBI",
        "Main Uptake Mechanism ID", "Subcategory Uptake Mechanism ID",
        "RAG_curie_CLO", "Subcellular Delivery ID",
        "Pubmed ID", "Patent",
    ] + EXP_FEAT_COLS)

    skipped = 0
    print(f"  Building Experiment individuals (rows without CPP-Complex are skipped) ...")
    for _, row in df_exp.iterrows():
        # Skip rows where CPP-Complex cannot be formed
        if pd.isna(row["CPP_ID"]) or pd.isna(row["RAG_curie_CheBI"]):
            skipped += 1
            continue

        cpp_local   = str(row["CPP_ID"]).strip().split("/")[-1]
        chebi_local = str(row["RAG_curie_CheBI"]).strip().replace(":", "_")
        complex_iri = CPP_DATASET_NS + f"complex_{cpp_local}_{chebi_local}"

        exp_iri = CPP_DATASET_NS + f"experiment_{int(row['id'])}"
        exp_ind = factory.getOWLNamedIndividual(IRI.create(exp_iri))
        manager.applyChange(AddAxiom(onto,
            factory.getOWLDeclarationAxiom(exp_ind)))
        manager.applyChange(AddAxiom(onto,
            factory.getOWLClassAssertionAxiom(experiment_cl, exp_ind)))
        _annotate_individual(manager, onto, factory, exp_iri,
                             f"experiment_{int(row['id'])}")
        _add_features(manager, onto, factory, exp_iri,
                      ((col, row[col]) for col in EXP_FEAT_COLS))

        # 2. UptakeMechanism — subcategory-first, split by comma
        #    experiment has_proper_part mechanism / mechanism is_proper_part_of experiment
        sub_id    = row["Subcategory Uptake Mechanism ID"]
        main_id   = row["Main Uptake Mechanism ID"]
        chosen_id = sub_id if pd.notna(sub_id) and str(sub_id).strip() else main_id
        mech_inds = []
        if pd.notna(chosen_id) and str(chosen_id).strip():
            for mech_iri in [s.strip() for s in str(chosen_id).split(",") if s.strip()]:
                mech_ind = factory.getOWLNamedIndividual(IRI.create(mech_iri))
                mech_inds.append(mech_ind)
                manager.applyChange(AddAxiom(onto,
                    factory.getOWLObjectPropertyAssertionAxiom(has_proper_part, exp_ind, mech_ind)))
                manager.applyChange(AddAxiom(onto,
                    factory.getOWLObjectPropertyAssertionAxiom(is_proper_part, mech_ind, exp_ind)))

        # 3. CPP-Complex and cell line are participants of the UptakeMechanism
        participant_iris = [complex_iri]
        if pd.notna(row["RAG_curie_CLO"]) and str(row["RAG_curie_CLO"]).strip():
            RAG_curie_CLO  = str(row["RAG_curie_CLO"]).strip()
            clo_iri = "http://purl.obolibrary.org/obo/" + RAG_curie_CLO.replace(":", "_")
            participant_iris.append(clo_iri)
            if RAG_curie_CLO in clo_features:
                _add_features(manager, onto, factory, clo_iri,
                              clo_features[RAG_curie_CLO].items())

        for p_iri in participant_iris:
            p_ind = factory.getOWLNamedIndividual(IRI.create(p_iri))
            for mech_ind in mech_inds:
                manager.applyChange(AddAxiom(onto,
                    factory.getOWLObjectPropertyAssertionAxiom(has_participant, mech_ind, p_ind)))
                manager.applyChange(AddAxiom(onto,
                    factory.getOWLObjectPropertyAssertionAxiom(is_part_in, p_ind, mech_ind)))

        # 4. Subcellular delivery — CPP-Complex is_located_in SubcellularEntity
        sub_del = row["Subcellular Delivery ID"]
        if pd.notna(sub_del) and str(sub_del).strip():
            complex_ind = factory.getOWLNamedIndividual(IRI.create(complex_iri))
            for loc_iri in [s.strip() for s in str(sub_del).split(",") if s.strip()]:
                loc_ind = factory.getOWLNamedIndividual(IRI.create(loc_iri))
                manager.applyChange(AddAxiom(onto,
                    factory.getOWLObjectPropertyAssertionAxiom(is_located_in, complex_ind, loc_ind)))
                manager.applyChange(AddAxiom(onto,
                    factory.getOWLObjectPropertyAssertionAxiom(is_location_of, loc_ind, complex_ind)))

        # 5. Publication: is_described_by PubMed IRI; fallback to Patent if PubMed == 0 or empty
        pubmed_val  = row["Pubmed ID"]
        patent_val  = row["Patent"]
        pub_iri     = None
        pubmed_valid = pd.notna(pubmed_val) and str(pubmed_val).strip() not in ("", "0", "0.0")
        if pubmed_valid:
            pub_iri = f"https://identifiers.org/pubmed:{int(float(pubmed_val))}"
        elif pd.notna(patent_val) and str(patent_val).strip():
            pub_iri = ("https://patents.google.com/patent/"
                       + str(patent_val).strip().replace(" ", ""))
        if pub_iri:
            pub_ind = factory.getOWLNamedIndividual(IRI.create(pub_iri))
            manager.applyChange(AddAxiom(onto,
                factory.getOWLDeclarationAxiom(pub_ind)))
            manager.applyChange(AddAxiom(onto,
                factory.getOWLClassAssertionAxiom(document_cl, pub_ind)))
            manager.applyChange(AddAxiom(onto,
                factory.getOWLObjectPropertyAssertionAxiom(is_described_by, exp_ind, pub_ind)))
            manager.applyChange(AddAxiom(onto,
                factory.getOWLObjectPropertyAssertionAxiom(describes, pub_ind, exp_ind)))

    print(f"  Skipped {skipped} rows (no CPP-Complex available).")

    # Save TBox-enriched ontology; this becomes the seed for the annotation chain
    tbox_onto = ontology.replace(".owl", "_cpp_complex.owl")
    tbox_iri  = IRI.create(java.io.File(tbox_onto).getAbsoluteFile().toURI())
    manager.saveOntology(onto, tbox_iri)
    print(f"  TBox-enriched ontology saved to {tbox_onto}")

    # Save Turtle serialisation
    tbox_ttl = tbox_onto.replace(".owl", ".ttl")
    TurtleDocumentFormat = jpype.JClass("org.semanticweb.owlapi.formats.TurtleDocumentFormat")
    ttl_iri = IRI.create(java.io.File(tbox_ttl).getAbsoluteFile().toURI())
    manager.saveOntology(onto, TurtleDocumentFormat(), ttl_iri)
    print(f"  Turtle serialisation saved to {tbox_ttl}")

    # ------------------------------------------------------------------ #
    # Phase 3b: ABox annotation chains via mowl insert_annotations        #
    # ------------------------------------------------------------------ #
    

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function that orchestrates all three phases.
    """
    print("\n" + "="*70)
    print("BIO-ONTOLOGY CONSTRUCTION PIPELINE")
    print("="*70)
    
    # Define input file paths
    GENE_TO_GO_FILE  = "triplets/gene_to_go.tsv"
    CHEBI_TO_GO_FILE = "triplets/chebi_to_go.tsv"
    CPP_CSV_FILE     = "/Users/hadmin1/Desktop/POSEIDON_CPPSite/Cell-penetrating-peptides/data/Natural_CPP3_download_annotated_preprocessed_Ontology_Normalization.csv"

    genes_ontology    = "Ontology/sio_genes.owl"
    inhibitors_ontology = "Ontology/sio_genes_inhibitors.owl"

    # PHASE 1: Load sio.owl, populate gene individuals and GO mechanism instances
    if os.path.exists(GENE_TO_GO_FILE):
        extend_gene_regulation(GENE_TO_GO_FILE, output_file=genes_ontology)

    # PHASE 2: Chain inhibitor individuals onto the gene ontology
    if os.path.exists(CHEBI_TO_GO_FILE) and os.path.exists(genes_ontology):
        extend_inhibitor_regulation(
            CHEBI_TO_GO_FILE,
            input_file=genes_ontology,
            output_file=inhibitors_ontology
        )

    # PHASE 3: TBox enrichment + ABox annotation chain
    if os.path.exists(CPP_CSV_FILE) and os.path.exists(inhibitors_ontology):
        print("\nPreparing annotation TSV files from CSV ...")
        cpp_mech_file, cpp_cargo_file, cpp_location_file, cpp_cell_file = \
            prepare_annotation_files(CPP_CSV_FILE)

        extend_ontology_with_annotations(
            ontology=inhibitors_ontology,
            cpp_csv_file=CPP_CSV_FILE,
            cpp_mech_file=cpp_mech_file,
            cpp_cargo_file=cpp_cargo_file,
            cpp_location_file=cpp_location_file,
            cpp_cell_file=cpp_cell_file,
            output_file="Ontology/extended_cpp_ontology.owl"
        )


if __name__ == "__main__":
    main()
