"""Evidence, plausibility, and applicability policy for the canonical pKa pipeline.

The policy deliberately separates four questions which were previously
collapsed into one ``site_assignment_confidence`` value:

* did the structure matcher find the intended atoms?
* did the experimental source identify those atoms/site?
* is the acid/base transition itself identified coherently?
* are the measurement conditions sufficiently documented?

Manual atom selection can answer only the first question.  Measurements that
remain molecule-level are retained as weak labels, but cannot silently become
local-site Stage 1 truth.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from rdkit import Chem  # type: ignore


EVIDENCE_POLICY_SCHEMA_VERSION = "1.2.0"
DEFAULT_REFERENCE_PRIORS = "data/curation/stage1_reference_pka_priors.csv"
AQUEOUS_ACTIVE_PKA_RANGE = (0.0, 14.0)

# Detection and ionization are deliberately different contracts. These labels
# describe chemistry but do not create an aqueous Brønsted pair by themselves.
STRUCTURAL_ONLY_LABELS = frozenset({
    "azo", "azide", "ether", "ester", "lactone", "nitrile", "nitro",
    "nitrite", "nitroso", "phosphate_ester", "sulfone", "sulfoxide",
    "thioester", "furan", "thiophene", "sulfamoyl_nitrogen",
    "phosphoramidate_nitrogen",
})
PERMANENT_CHARGE_LABELS = frozenset({"quaternary_ammonium"})
EXTREME_RANGE_DISABLED_LABELS = frozenset({
    "ketone", "aldehyde", "ester", "lactone", "ether",
    "amide_primary", "amide_secondary", "amide_tertiary",
})
UNCLASSIFIED_NITROGEN_LABELS = frozenset({
    "unclassified_aromatic_ring_N", "unclassified_aliphatic_ring_N",
    "unclassified_cationic_ring_N", "unclassified_cationic_nonring_N",
    "unclassified_nonring_N",
})


def ionization_policy_role(label: object) -> str:
    """Return the aqueous model role without confusing formal charge with pKa."""
    value = str(label or "")
    if value in UNCLASSIFIED_NITROGEN_LABELS:
        return "unclassified_requires_quarantine"
    if value in PERMANENT_CHARGE_LABELS:
        return "permanent_charge_descriptor"
    if value in EXTREME_RANGE_DISABLED_LABELS:
        return "extreme_range_disabled"
    if value in STRUCTURAL_ONLY_LABELS:
        return "structural_descriptor_only"
    if value in UNRESOLVED_IONIZABLE_LABELS:
        return "potentially_ionizable_unresolved"
    return "aqueous_pair_candidate"

# These detector labels can plausibly own an experimental acid/base endpoint,
# but do not yet have a complete conjugate-pair enumerator.  Their presence
# invalidates the inference "one enumerable site == one ionizable site".
UNRESOLVED_IONIZABLE_LABELS = frozenset({
    "boric_acid", "peroxide", "percaboxylic_acid", "sulfinic_acid",
    "sulfenic_acid", "amidenium", "other_aniline", "other_anilinium",
    "tetrazolium", "cyanamide", "enamine", "amide_primary",
    "amide_secondary", "amide_tertiary", "2amide_anhydride",
    "amide_anhydride", "carbamide", "hydroxyamide", "isoamide", "lactam",
    "pyrazolidine",
    "n_substituted_tetrazole",
    "124triazine", "7-azaindole", "n-indole",
    "pyridazinone", "n-indazole", "nn-indazole", "indole", "Nindole",
    "isoNindole", "Sindole", "n-hydroxy", "hydrazone", "imide", "thioide",
    "thiophenol", "thioamide", "thioamideamide_anhydride",
    "thioamideamide", "thioamidethio",
    "isoxazolidine", "sulfonamide", "sulfamide", "sulfamate",
    "hydroxamic_acid", "thioimidate", "aromatic_thiosulfamide",
    "123_oxadiazole", "124_oxadiazole", "125_oxadiazole", "134_oxadiazole",
    "123_thiadiazole", "124_thiadiazole", "125_thiadiazole", "134_thiadiazole",
    *UNCLASSIFIED_NITROGEN_LABELS,
})


@dataclass(frozen=True)
class ReferencePrior:
    label: str
    family: str
    pka: float
    uncertainty: float
    transition_definition: str
    provenance: str


def _candidate_paths(path: str) -> Iterable[str]:
    yield path
    if not os.path.isabs(path):
        yield os.path.join(os.path.dirname(os.path.dirname(__file__)), path)


@lru_cache(maxsize=8)
def load_reference_priors(path: str = DEFAULT_REFERENCE_PRIORS) -> Dict[str, ReferencePrior]:
    resolved = next((candidate for candidate in _candidate_paths(path) if os.path.exists(candidate)), None)
    if resolved is None:
        return {}
    result: Dict[str, ReferencePrior] = {}
    with open(resolved, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("enabled", "true")).strip().lower() not in {"1", "true", "yes"}:
                continue
            prior = ReferencePrior(
                label=str(row["site_label"]),
                family=str(row["site_family"]),
                pka=float(row["reference_pka"]),
                uncertainty=max(0.25, float(row["reference_uncertainty"])),
                transition_definition=str(row["transition_definition"]),
                provenance=str(row["provenance"]),
            )
            result[prior.label] = prior
    return result


def reference_prior(
    label: object,
    family: object = "",
    path: str = DEFAULT_REFERENCE_PRIORS,
) -> Optional[ReferencePrior]:
    priors = load_reference_priors(path)
    exact = priors.get(str(label))
    if exact is not None:
        return exact
    family_matches = [prior for prior in priors.values() if prior.family == str(family)]
    if not family_matches:
        return None
    # A family fallback is allowed only when its enabled entries agree closely;
    # otherwise different transition chemistry would be silently conflated.
    values = [prior.pka for prior in family_matches]
    if max(values) - min(values) > 2.0:
        return None
    return ReferencePrior(
        label=str(label),
        family=str(family),
        pka=float(sum(values) / len(values)),
        uncertainty=float(max(prior.uncertainty for prior in family_matches)),
        transition_definition="family_consensus:" + family_matches[0].transition_definition,
        provenance="family consensus from enabled reference priors",
    )


def unresolved_ionizable_contexts(
    resolved_sites: Sequence[Mapping[str, object]],
    pair_sites: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    pair_keys = {
        (str(site.get("label", "")), tuple(sorted(int(value) for value in site.get("atom_set", set()))))
        for site in pair_sites
    }
    contexts = []
    for site in resolved_sites:
        label = str(site.get("label", ""))
        key = (label, tuple(sorted(int(value) for value in site.get("atom_set", set()))))
        if label not in UNRESOLVED_IONIZABLE_LABELS or key in pair_keys:
            continue
        contexts.append({
            "label": label,
            "site_type": str(site.get("type", site.get("site_type", "unknown"))),
            "ionization_role": ionization_policy_role(label),
            "atom_indices": list(key[1]),
            "reason": "detected_potentially_ionizable_context_without_complete_pair_enumerator",
        })
    return contexts


def plausibility_against_reference(
    pka: float,
    label: object,
    family: object,
    path: str = DEFAULT_REFERENCE_PRIORS,
) -> Dict[str, object]:
    prior = reference_prior(label, family, path)
    if prior is None or not math.isfinite(float(pka)):
        return {
            "status": "not_assessed_no_reference",
            "reference_pka": None,
            "reference_uncertainty": None,
            "z_distance": None,
            "transition_definition": "",
            "reference_provenance": "",
        }
    z_distance = abs(float(pka) - prior.pka) / prior.uncertainty
    status = (
        "conflict"
        if z_distance > 3.5
        else "review"
        if z_distance > 2.5
        else "consistent"
    )
    return {
        "status": status,
        "reference_pka": prior.pka,
        "reference_uncertainty": prior.uncertainty,
        "z_distance": float(z_distance),
        "transition_definition": prior.transition_definition,
        "reference_provenance": prior.provenance,
    }


def _formal_charge(smiles: str) -> Optional[int]:
    mol = Chem.MolFromSmiles(str(smiles)) if str(smiles).strip() else None
    if mol is None:
        return None
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def assess_measurement_evidence(
    *,
    pka: float,
    label: object,
    family: object,
    structural_mapping_confidence: str,
    source_site_evidence_confidence: str,
    unresolved_contexts: Sequence[Mapping[str, object]],
    source_temperature: str = "",
    source_original_smiles: str = "",
    input_smiles: str = "",
    reviewer_verified_site_attribution: bool = False,
    reviewer_verified_transition_identity: bool = False,
    reviewer_verified_measurement_conditions: bool = False,
    reference_path: str = DEFAULT_REFERENCE_PRIORS,
) -> Dict[str, object]:
    plausibility = plausibility_against_reference(
        pka, label, family, path=reference_path
    )
    source_conf = (
        "high"
        if reviewer_verified_site_attribution
        else str(source_site_evidence_confidence or "none").lower()
    )
    mapping_conf = str(structural_mapping_confidence or "none").lower()
    original_charge = _formal_charge(source_original_smiles)
    input_charge = _formal_charge(input_smiles)
    charge_state_changed = (
        original_charge is not None
        and input_charge is not None
        and original_charge != input_charge
    )

    if reviewer_verified_transition_identity:
        transition_confidence = "high"
    elif plausibility["status"] == "conflict" or charge_state_changed:
        transition_confidence = "low"
    elif plausibility["status"] == "consistent":
        transition_confidence = "high" if source_conf == "high" else "medium"
    else:
        transition_confidence = "medium" if source_conf in {"high", "medium"} else "low"

    conditions_confidence = (
        "high"
        if reviewer_verified_measurement_conditions
        else "medium"
        if str(source_temperature).strip()
        else "low"
    )
    if source_conf == "high" and transition_confidence != "low":
        tier = "gold"
        reason = "source_identifies_site_and_transition_is_not_in_conflict"
    elif (
        mapping_conf == "high"
        and not unresolved_contexts
        and transition_confidence in {"high", "medium"}
    ):
        tier = "silver"
        reason = "unique_complete_structural_site_with_reference_consistent_transition"
    else:
        tier = "ambiguous"
        reasons = []
        if unresolved_contexts:
            reasons.append("competing_unresolved_ionizable_context")
        if source_conf not in {"high", "medium"}:
            reasons.append("source_does_not_identify_site")
        if transition_confidence == "low":
            reasons.append("transition_identity_low_confidence")
        reason = "+".join(reasons) or "insufficient_site_transition_evidence"

    return {
        "evidence_policy_schema_version": EVIDENCE_POLICY_SCHEMA_VERSION,
        "evidence_tier": tier,
        "evidence_tier_reason": reason,
        "structural_mapping_confidence": mapping_conf,
        "experimental_site_attribution_confidence": source_conf,
        "transition_identity_confidence": transition_confidence,
        "measurement_conditions_confidence": conditions_confidence,
        "reference_plausibility_status": plausibility["status"],
        "reference_pka": plausibility["reference_pka"],
        "reference_uncertainty": plausibility["reference_uncertainty"],
        "reference_z_distance": plausibility["z_distance"],
        "reference_transition_definition": plausibility["transition_definition"],
        "reference_provenance": plausibility["reference_provenance"],
        "source_structure_charge_changed": charge_state_changed,
        "unresolved_ionizable_context_count": len(unresolved_contexts),
        "unresolved_ionizable_context_labels": sorted({
            str(context.get("label", "")) for context in unresolved_contexts
        }),
        "stage1_local_supervision_eligible": tier in {"gold", "silver"},
        "stage2_exact_site_supervision_eligible": tier in {"gold", "silver"},
        "stage2_weak_molecule_supervision_eligible": tier == "ambiguous",
    }
