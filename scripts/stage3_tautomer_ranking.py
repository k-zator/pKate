"""Enumerate and rank tautomers inside populated Stage 3 configurations.

Stage 2/3 thermodynamics first determine the population of each protonation
configuration.  This module does not alter those populations.  It subsequently
enumerates tautomer drawings from every equivalent reference-node seed,
deduplicates them within the configuration, and ranks the retained candidates.

RDKit's tautomer score is a deterministic rule score, not an energy or a
calibrated probability.  The normalized weights are conditional on the
retained candidate set and are ranking aids only.  A truncated candidate set
therefore remains explicitly flagged even though its retained weights sum to
one.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem, rdBase  # type: ignore
from rdkit.Chem.MolStandardize import rdMolStandardize  # type: ignore


TAUTOMER_RANKING_SCHEMA_VERSION = "1.1.0"
TAUTOMER_RANKING_METHOD = (
    "rdkit_configuration_level_enumeration_rule_score_softmax_rank_only"
)
DEFAULT_TAUTOMER_SCORE_TEMPERATURE = 5.0
DEFAULT_MAX_TAUTOMERS_PER_CONFIGURATION = 16


def _configuration_signature(node: Dict) -> Tuple[Tuple[str, object], ...]:
    """Return the site-level state shared by tautomer-equivalent nodes.

    Atom/bond drawings and proton-location alternatives are deliberately not
    part of this key.  Consequently, nodes with the same coupled-group levels
    contribute candidates to one configuration-level tautomer enumeration.
    """
    values = node.get("group_levels") or node.get("site_forms") or {}
    return tuple(sorted((str(key), value) for key, value in values.items()))


def _configuration_id(signature: Tuple[Tuple[str, object], ...]) -> str:
    """Create a short deterministic identifier from a configuration signature."""
    encoded = json.dumps(signature, separators=(",", ":"), sort_keys=False)
    return "configuration_" + hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def _score_tautomer(smiles: str, enumerator: object) -> float:
    """Return RDKit's rule score, failing loudly for an invalid retained SMILES."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Could not parse enumerated tautomer: {smiles}")
    return float(enumerator.ScoreTautomer(mol))


def _mapped_smiles(mol: Chem.Mol) -> str:
    """Serialize a candidate deterministically while preserving atom maps."""
    return str(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))


def _configuration_seed_smiles(node: Dict) -> str | None:
    """Choose the chemically interpretable reference drawing for enumeration."""
    reference = node.get("reference_atom_mapped_smiles")
    if reference:
        return str(reference)
    # Compatibility fallback for small programmatic fixtures and older snapshots.
    stored = node.get("tautomers") or []
    return str(stored[0]) if stored else None


def _enumerate_configuration_tautomers(
    configuration_nodes: List[Dict],
    max_tautomers_per_configuration: int,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    """Enumerate and deduplicate one fixed-charge protonation configuration.

    Every equivalent node contributes a reference seed.  Candidate SMILES map
    to the deterministic first seed that generated them; this provenance is
    later used to recover a compatible source node, not to assign a separate
    thermodynamic population.  The retained limit is applied once after the
    seed unions are deduplicated and ranked.  The caller performs that final
    score-based limit because RDKit enumeration order is not a chemical rank.

    The diagnostic dictionary separates four failure modes: unavailable seed
    structures, rejected generated candidates, RDKit search truncation, and
    inconsistent seed charges.  The historical node-level truncation flag is
    reported independently and does not make a successfully re-enumerated
    configuration incomplete.
    """
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(int(max_tautomers_per_configuration))
    enumerator.SetMaxTransforms(max(100, int(max_tautomers_per_configuration) * 20))

    tautomer_to_node: Dict[str, str] = {}
    seed_count = 0
    failed_seed_count = 0
    rejected_candidate_count = 0
    rdkit_truncated_seed_count = 0
    seed_charges = set()
    # This is provenance from the network builder's old four-per-node search.
    # It must not determine completion of the fresh configuration-level search.
    source_node_truncated = any(
        bool(node.get("tautomer_enumeration_truncated", False))
        for node in configuration_nodes
    )
    for node in sorted(configuration_nodes, key=lambda item: str(item.get("node_id", ""))):
        node_id = str(node.get("node_id", ""))
        seed_smiles = _configuration_seed_smiles(node)
        if not seed_smiles:
            failed_seed_count += 1
            continue
        seed = Chem.MolFromSmiles(seed_smiles)
        if seed is None:
            failed_seed_count += 1
            continue
        seed_count += 1
        target_charge = int(Chem.GetFormalCharge(seed))
        seed_charges.add(target_charge)
        # Always retain the reference itself, even if RDKit later aborts or
        # cannot discover any additional tautomer from this seed.
        canonical_seed = _mapped_smiles(seed)
        tautomer_to_node.setdefault(canonical_seed, node_id)
        with rdBase.BlockLogs():
            try:
                result = enumerator.Enumerate(seed)
                if str(result.status) != "Completed":
                    rdkit_truncated_seed_count += 1
                candidates = result
            except Exception:
                failed_seed_count += 1
                candidates = (seed,)
            for candidate in candidates:
                # Tautomer enumeration must not leak into another protomer;
                # protonation configurations have already been populated.
                if int(Chem.GetFormalCharge(candidate)) != target_charge:
                    continue
                try:
                    sanitized = Chem.Mol(candidate)
                    Chem.SanitizeMol(sanitized)
                    candidate_smiles = _mapped_smiles(sanitized)
                except Exception:
                    rejected_candidate_count += 1
                    continue
                tautomer_to_node.setdefault(candidate_smiles, node_id)

    return tautomer_to_node, {
        "seed_node_count": int(seed_count),
        "failed_seed_count": int(failed_seed_count),
        "rejected_candidate_count": int(rejected_candidate_count),
        "rdkit_truncated_seed_count": int(rdkit_truncated_seed_count),
        "inconsistent_seed_charges": bool(len(seed_charges) > 1),
        "source_node_enumeration_truncated": bool(source_node_truncated),
    }


def select_top_tautomer_in_configuration(
    records: List[Dict], configuration_id: str
) -> Dict | None:
    """Select rank 1 without allowing tautomer score to change configuration.

    The caller must first choose the protonation configuration from the Stage 3
    free-energy population.  This function then resolves only the drawing
    within that already selected physical configuration.
    """
    candidates = [
        record for record in records
        if str(record["configuration_id"]) == str(configuration_id)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda record: (
        int(record["tautomer_rank_within_configuration"]),
        str(record["atom_mapped_smiles"]),
    ))


def rank_tautomers_within_configurations(
    nodes: List[Dict],
    molecule_id: str,
    score_temperature: float = DEFAULT_TAUTOMER_SCORE_TEMPERATURE,
    max_tautomers_per_configuration: int = DEFAULT_MAX_TAUTOMERS_PER_CONFIGURATION,
) -> Tuple[List[Dict], Dict, Dict | None]:
    """Enumerate, rank, and annotate every protonation configuration.

    Returns a flat auditable candidate table, a molecule-level summary, and the
    globally largest heuristic candidate for diagnostics.  The third return
    value is *not* permitted to override the thermodynamically dominant
    configuration; ``apply_stage3`` reselects rank 1 inside that configuration.

    Candidate weights are a softmax of RDKit rule-score differences and sum to
    one inside each retained configuration candidate set.  Multiplying by the
    fixed configuration population makes the molecule-wide heuristic weights
    sum to one without feeding tautomer scores back into protonation physics.
    """
    if not math.isfinite(float(score_temperature)) or float(score_temperature) <= 0.0:
        raise ValueError("Tautomer score temperature must be finite and positive")
    if int(max_tautomers_per_configuration) < 1:
        raise ValueError("max_tautomers_per_configuration must be at least one")
    if not nodes:
        return [], {
            "status": "unavailable_no_populated_nodes",
            "ranking_method": TAUTOMER_RANKING_METHOD,
            "ranked_tautomer_count": 0,
            "complete_microstate_heuristic_population_sum": 0.0,
        }, None

    # Group before enumeration so equivalent node seeds share one candidate
    # set and one explicit, configuration-wide retention limit.
    grouped: Dict[Tuple[Tuple[str, object], ...], List[Dict]] = defaultdict(list)
    for node in nodes:
        grouped[_configuration_signature(node)].append(node)

    enumerator = rdMolStandardize.TautomerEnumerator()
    records: List[Dict] = []
    truncated_configuration_count = 0
    complete_configuration_count = 0
    source_truncated_configuration_count = 0
    tied_top_configuration_count = 0
    for signature, configuration_nodes in sorted(grouped.items()):
        configuration_id = _configuration_id(signature)
        configuration_population = float(
            configuration_nodes[0].get("stage3_configuration_population_at_ph", 0.0)
        )
        tautomer_to_node, enumeration = _enumerate_configuration_tautomers(
            configuration_nodes,
            max_tautomers_per_configuration=int(max_tautomers_per_configuration),
        )
        if not tautomer_to_node:
            continue

        scored_all = [
            (smiles, node_id, _score_tautomer(smiles, enumerator))
            for smiles, node_id in tautomer_to_node.items()
        ]
        # Enumerate first, then impose the single configuration cap by score.
        # Lexical SMILES and node id make tied results reproducible.
        scored_all.sort(key=lambda item: (-item[2], item[0], item[1]))
        unique_candidates_before_limit = int(len(scored_all))
        scored = scored_all[:int(max_tautomers_per_configuration)]
        best_score = float(scored[0][2])
        scaled = np.asarray([
            math.exp((float(score) - best_score) / float(score_temperature))
            for _, _, score in scored
        ], dtype=float)
        # These weights are conditional on the retained candidates.  They must
        # not be interpreted as solution-phase tautomer populations.
        conditional_weights = scaled / float(np.sum(scaled))
        top_tied = sum(abs(float(score) - best_score) < 1e-12 for _, _, score in scored) > 1
        # Completion is conservative: any failed seed, rejected candidate,
        # search-limit event, charge inconsistency, or final cap is surfaced.
        truncated = bool(
            int(enumeration["failed_seed_count"]) > 0
            or int(enumeration["rejected_candidate_count"]) > 0
            or int(enumeration["rdkit_truncated_seed_count"]) > 0
            or bool(enumeration["inconsistent_seed_charges"])
            or unique_candidates_before_limit > int(max_tautomers_per_configuration)
        )
        enumeration_complete = not truncated
        tied_top_configuration_count += int(top_tied)
        truncated_configuration_count += int(truncated)
        complete_configuration_count += int(enumeration_complete)
        source_truncated_configuration_count += int(
            bool(enumeration["source_node_enumeration_truncated"])
        )
        positive = conditional_weights[conditional_weights > 0.0]
        effective_count = float(math.exp(float(-np.sum(positive * np.log(positive)))))

        node_rankings: Dict[str, List[Dict]] = defaultdict(list)
        configuration_records: List[Dict] = []
        for rank, ((smiles, node_id, score), conditional_weight) in enumerate(
            zip(scored, conditional_weights), start=1
        ):
            record = {
                "molecule_id": str(molecule_id),
                "configuration_id": configuration_id,
                "configuration_signature_json": json.dumps(signature, separators=(",", ":")),
                "origin_node_id": node_id,
                "atom_mapped_smiles": smiles,
                "tautomer_rank_within_configuration": int(rank),
                "rdkit_tautomer_score": float(score),
                "score_below_configuration_best": float(best_score - float(score)),
                "tautomer_heuristic_conditional_weight": float(conditional_weight),
                "configuration_population_at_ph": configuration_population,
                "complete_microstate_heuristic_population_at_ph": float(
                    configuration_population * float(conditional_weight)
                ),
                "configuration_tautomer_count": int(len(scored)),
                "configuration_tautomer_effective_count": effective_count,
                "configuration_top_score_tied": bool(top_tied),
                "tautomer_enumeration_truncated": bool(truncated),
                "configuration_enumeration_complete": bool(enumeration_complete),
                "configuration_enumeration_seed_node_count": int(
                    enumeration["seed_node_count"]
                ),
                "configuration_enumeration_failed_seed_count": int(
                    enumeration["failed_seed_count"]
                ),
                "configuration_enumeration_rejected_candidate_count": int(
                    enumeration["rejected_candidate_count"]
                ),
                "configuration_rdkit_truncated_seed_count": int(
                    enumeration["rdkit_truncated_seed_count"]
                ),
                "configuration_inconsistent_seed_charges": bool(
                    enumeration["inconsistent_seed_charges"]
                ),
                "configuration_unique_candidate_count_before_limit": int(
                    unique_candidates_before_limit
                ),
                "max_tautomers_per_configuration": int(
                    max_tautomers_per_configuration
                ),
                "source_node_enumeration_truncated": bool(
                    enumeration["source_node_enumeration_truncated"]
                ),
                "ranking_method": TAUTOMER_RANKING_METHOD,
                "ranking_schema_version": TAUTOMER_RANKING_SCHEMA_VERSION,
                "score_temperature_rule_units": float(score_temperature),
                "ranking_semantics": (
                    "heuristic_rank_and_weight_conditional_on_retained_configuration_"
                    "candidates_not_physical_free_energy_or_calibrated_probability"
                ),
            }
            records.append(record)
            configuration_records.append(record)
            node_rankings[node_id].append({
                key: record[key]
                for key in (
                    "atom_mapped_smiles",
                    "tautomer_rank_within_configuration",
                    "rdkit_tautomer_score",
                    "score_below_configuration_best",
                    "tautomer_heuristic_conditional_weight",
                    "complete_microstate_heuristic_population_at_ph",
                )
            })

        # Annotate every equivalent node with the same configuration-level top
        # candidate.  The per-seed field is retained only for provenance.
        for node in configuration_nodes:
            node_id = str(node.get("node_id", ""))
            own_rankings = sorted(
                node_rankings.get(node_id, []),
                key=lambda item: int(item["tautomer_rank_within_configuration"]),
            )
            node["stage3_tautomer_configuration_id"] = configuration_id
            node["stage3_tautomer_ranking_method"] = TAUTOMER_RANKING_METHOD
            node["stage3_tautomer_ranking_schema_version"] = TAUTOMER_RANKING_SCHEMA_VERSION
            node["stage3_configuration_tautomer_count"] = int(len(scored))
            node["stage3_configuration_tautomer_effective_count"] = effective_count
            node["stage3_configuration_top_tautomer_tied"] = bool(top_tied)
            node["stage3_tautomer_ranking_limited_by_truncation"] = bool(truncated)
            node["stage3_configuration_tautomer_enumeration_complete"] = bool(
                enumeration_complete
            )
            node["stage3_max_tautomers_per_configuration"] = int(
                max_tautomers_per_configuration
            )
            configuration_top = configuration_records[0] if configuration_records else None
            node["stage3_top_ranked_tautomer"] = ({
                key: configuration_top[key]
                for key in (
                    "atom_mapped_smiles",
                    "tautomer_rank_within_configuration",
                    "rdkit_tautomer_score",
                    "score_below_configuration_best",
                    "tautomer_heuristic_conditional_weight",
                    "complete_microstate_heuristic_population_at_ph",
                )
            } if configuration_top else None)
            node["stage3_top_ranked_tautomer_from_this_seed"] = (
                own_rankings[0] if own_rankings else None
            )
            node["stage3_node_ranked_tautomer_population_at_ph"] = float(sum(
                float(item["complete_microstate_heuristic_population_at_ph"])
                for item in own_rankings
            ))

    records.sort(key=lambda record: (
        -float(record["complete_microstate_heuristic_population_at_ph"]),
        str(record["configuration_id"]),
        int(record["tautomer_rank_within_configuration"]),
    ))
    # This global maximum is diagnostic only.  The final molecular structure is
    # selected later from rank 1 of the thermodynamically dominant configuration.
    dominant = records[0] if records else None
    population_sum = float(sum(
        float(record["complete_microstate_heuristic_population_at_ph"])
        for record in records
    ))
    summary = {
        "status": (
            "available_configuration_level_tautomer_ranking"
            if records else "unavailable_no_tautomers"
        ),
        "ranking_method": TAUTOMER_RANKING_METHOD,
        "ranking_schema_version": TAUTOMER_RANKING_SCHEMA_VERSION,
        "score_temperature_rule_units": float(score_temperature),
        "configuration_count": int(len(grouped)),
        "ranked_tautomer_count": int(len(records)),
        "truncated_configuration_count": int(truncated_configuration_count),
        "complete_configuration_count": int(complete_configuration_count),
        "source_truncated_configuration_count": int(
            source_truncated_configuration_count
        ),
        "tied_top_configuration_count": int(tied_top_configuration_count),
        "max_tautomers_per_configuration": int(max_tautomers_per_configuration),
        "enumeration_scope": "deduplicated_protonation_configuration",
        "complete_microstate_heuristic_population_sum": population_sum,
        "global_max_complete_microstate_heuristic_population": (
            float(dominant["complete_microstate_heuristic_population_at_ph"])
            if dominant else 0.0
        ),
        "semantics": (
            "protonation_configuration_population_times_an_RDKit_rule_score_weight_"
            "conditional_on_retained_candidates; ranking_only_not_a_physical_"
            "tautomer_population"
        ),
    }
    return records, summary, dominant
