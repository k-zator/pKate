"""Thermodynamically consistent Stage 2 one-body and pair-coupling closure.

The experimental inputs are macroscopic pKa values.  They do not uniquely
identify a separate interaction energy for every pair of sites.  This module
therefore constructs the minimum-complexity, regularized pairwise free-energy
model that reproduces the Stage 2 macroscopic ladder while remaining close to
the Stage 1 plus Stage 2 site-wise proposal.  The inferred pair terms are
explicitly marked prior-dependent; they are not presented as direct
experimental measurements.

For protonated-side occupancies x_i in {0, 1}, the pH-independent log10 weight
of a microstate is

    sum_i b_i x_i + sum_{i<j} J_ij x_i x_j

and its pH-dependent log10 weight subtracts n_H * pH.  A positive J stabilizes
joint protonation in log-equilibrium units.  The corresponding interaction
free energy is -R T ln(10) J.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize  # type: ignore


STAGE2_FREE_ENERGY_SCHEMA_VERSION = "1.0.0"
STAGE2_FREE_ENERGY_METHOD = (
    "minimum_norm_pairwise_closure_of_stage2_predicted_macro_ladder"
)
DEFAULT_TEMPERATURE_K = 298.15
R_KJ_MOL_K = 8.31446261815324e-3


PairKey = Tuple[str, str]


def canonical_pair_key(site_a: str, site_b: str) -> PairKey:
    """Return an order-independent key for an interaction between distinct sites."""
    a = str(site_a)
    b = str(site_b)
    if a == b:
        raise ValueError("A pair coupling requires two different sites")
    return (a, b) if a < b else (b, a)


def pair_coupling_map(pair_terms: Iterable[Mapping]) -> Dict[PairKey, float]:
    """Convert serialized pair terms to the canonical lookup used by thermodynamics."""
    result: Dict[PairKey, float] = {}
    for term in pair_terms:
        key = canonical_pair_key(str(term["site_i"]), str(term["site_j"]))
        result[key] = float(term["coupling_log10_units"])
    return result


def _configuration_signature(node: Mapping) -> Tuple:
    """Identify a protonation configuration independently of its tautomer drawing."""
    levels = node.get("group_levels") or {}
    if levels:
        return ("levels", tuple(sorted((str(k), int(v)) for k, v in levels.items())))
    forms = node.get("site_forms") or {}
    return ("forms", tuple(sorted((str(k), str(v)) for k, v in forms.items())))


def site_acid_side_occupancy(node: Mapping, site: Mapping) -> int:
    """Return whether a transition is on its protonated (acid-form) side."""
    site_id = str(site["site_id"])
    form = (node.get("site_forms") or {}).get(site_id)
    if form == "acid_form":
        return 1
    if form == "base_form":
        return 0
    if form == "outside_transition" or form is None:
        group_id = str(site.get("coupling_group_id", ""))
        levels = node.get("group_levels") or {}
        if group_id and group_id in levels and site.get("acid_level") is not None:
            return int(int(levels[group_id]) >= int(site["acid_level"]))
    raise ValueError(
        f"Node {node.get('node_id')} has no interpretable form for site {site_id}"
    )


def _configuration_records(
    nodes: Sequence[Mapping],
    sites: Sequence[Mapping],
) -> List[Dict]:
    """Collapse tautomeric drawings into unique, validated occupancy configurations."""
    by_signature: Dict[Tuple, Dict] = {}
    site_ids = [str(site["site_id"]) for site in sites]
    for node in nodes:
        signature = _configuration_signature(node)
        occupancies = tuple(site_acid_side_occupancy(node, site) for site in sites)
        proton_count = int(node.get("protonated_site_count", sum(occupancies)))
        if proton_count != sum(occupancies):
            raise ValueError(
                f"Node {node.get('node_id')} proton count does not match transition occupancies"
            )
        existing = by_signature.get(signature)
        if existing is not None and existing["occupancies"] != occupancies:
            raise ValueError("Equivalent microstate drawings disagree on site occupancies")
        by_signature.setdefault(signature, {
            "signature": signature,
            "node": node,
            "occupancies": occupancies,
            "proton_count": proton_count,
            "site_ids": site_ids,
        })
    return list(by_signature.values())


def _log10_sum(values: Sequence[float]) -> float:
    """Compute ``log10(sum(10**values))`` without numerical overflow."""
    if not values:
        return -math.inf
    high = max(values)
    return float(high + math.log10(sum(10.0 ** (value - high) for value in values)))


def configuration_log10_coefficient(
    occupancies: Sequence[int],
    site_ids: Sequence[str],
    one_body_pkas: Mapping[str, float],
    pair_couplings: Mapping[PairKey, float],
) -> float:
    """Evaluate the pH-independent log10 binding coefficient of a configuration.

    Occupancy 1 denotes the protonated (acid-form) side. The coefficient is the
    sum of occupied one-body pKas plus couplings for jointly occupied site pairs.
    """
    coefficient = sum(
        int(occupied) * float(one_body_pkas[str(site_id)])
        for site_id, occupied in zip(site_ids, occupancies)
    )
    for left in range(len(site_ids)):
        if not occupancies[left]:
            continue
        for right in range(left + 1, len(site_ids)):
            if occupancies[right]:
                coefficient += float(pair_couplings.get(
                    canonical_pair_key(site_ids[left], site_ids[right]), 0.0
                ))
    return float(coefficient)


def macro_pka_values_from_free_energy(
    nodes: Sequence[Mapping],
    sites: Sequence[Mapping],
    one_body_pkas: Mapping[str, float],
    pair_couplings: Mapping[PairKey, float] | None = None,
) -> List[float]:
    """Derive the ordered macroscopic pKa ladder from configuration partition sums."""
    pair_map = pair_couplings or {}
    records = _configuration_records(nodes, sites)
    if not records:
        return []
    by_count: Dict[int, List[float]] = {}
    site_ids = [str(site["site_id"]) for site in sites]
    for record in records:
        coefficient = configuration_log10_coefficient(
            record["occupancies"], site_ids, one_body_pkas, pair_map
        )
        by_count.setdefault(int(record["proton_count"]), []).append(coefficient)
    partition_coefficients = {
        count: _log10_sum(values) for count, values in by_count.items()
    }
    counts = sorted(partition_coefficients)
    if any(upper != lower + 1 for lower, upper in zip(counts, counts[1:])):
        raise ValueError("Microstate graph has a non-contiguous proton-count ladder")
    return [
        float(partition_coefficients[upper] - partition_coefficients[lower])
        for lower, upper in zip(counts, counts[1:])
    ]


def contextual_edge_pkas(
    nodes: Sequence[Mapping],
    edges: Sequence[Mapping],
    sites: Sequence[Mapping],
    one_body_pkas: Mapping[str, float],
    pair_couplings: Mapping[PairKey, float] | None = None,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """Return context-specific edge pKas and the maximum cycle-closure error."""
    pair_map = pair_couplings or {}
    node_lookup = {str(node["node_id"]): node for node in nodes}
    site_lookup = {str(site["site_id"]): site for site in sites}
    site_ids = [str(site["site_id"]) for site in sites]
    occupancy_lookup = {
        node_id: tuple(
            site_acid_side_occupancy(node, site_lookup[site_id])
            for site_id in site_ids
        )
        for node_id, node in node_lookup.items()
    }
    result: Dict[str, Dict[str, float]] = {}
    max_cycle_error = 0.0
    for edge in edges:
        site_id = str(edge["site_id"])
        site_index = site_ids.index(site_id)
        acid_id = str(edge["acid_node_id"])
        base_id = str(edge["base_node_id"])
        acid_occ = occupancy_lookup[acid_id]
        base_occ = occupancy_lookup[base_id]
        if acid_occ[site_index] != 1 or base_occ[site_index] != 0:
            raise ValueError(f"Edge {edge.get('edge_id')} has reversed site occupancy")
        if any(
            acid_occ[idx] != base_occ[idx]
            for idx in range(len(site_ids)) if idx != site_index
        ):
            raise ValueError(f"Edge {edge.get('edge_id')} changes more than one site")
        interaction_shift = 0.0
        for other_index, other_id in enumerate(site_ids):
            if other_id == site_id or not base_occ[other_index]:
                continue
            interaction_shift += float(pair_map.get(
                canonical_pair_key(site_id, other_id), 0.0
            ))
        contextual_pka = float(one_body_pkas[site_id]) + interaction_shift
        acid_coefficient = configuration_log10_coefficient(
            acid_occ, site_ids, one_body_pkas, pair_map
        )
        base_coefficient = configuration_log10_coefficient(
            base_occ, site_ids, one_body_pkas, pair_map
        )
        cycle_error = abs((acid_coefficient - base_coefficient) - contextual_pka)
        max_cycle_error = max(max_cycle_error, float(cycle_error))
        result[str(edge["edge_id"])] = {
            "contextual_edge_pka": contextual_pka,
            "one_body_pka": float(one_body_pkas[site_id]),
            "pair_interaction_shift": float(interaction_shift),
            "cycle_closure_error": float(cycle_error),
        }
    return result, float(max_cycle_error)


def infer_regularized_pairwise_free_energy(
    stage1_local_pkas: Sequence[float],
    stage2_macro_pkas: Sequence[float],
    adjustable: Sequence[bool],
    proposed_deltas: Sequence[float],
    nodes: Sequence[Mapping],
    sites: Sequence[Mapping],
    observed_macro_anchor_count: int = 0,
    one_body_regularization: float = 0.05,
    pair_regularization: float = 0.25,
    max_abs_pair_coupling: float = 4.0,
) -> Tuple[Dict[str, float], List[Dict], Dict]:
    """Infer a conservative pairwise closure of a predicted macro-pKa ladder.

    One-body values start at Stage 1 plus the supported Stage 2 proposal.  Pair
    terms and small one-body refinements are optimized jointly.  Ridge terms
    choose one solution from an underdetermined family; diagnostics therefore
    always disclose that pair values are prior-dependent.
    """
    stage1 = np.asarray(stage1_local_pkas, dtype=float)
    target = np.asarray(stage2_macro_pkas, dtype=float)
    adjustable_mask = np.asarray(adjustable, dtype=bool)
    deltas = np.asarray(proposed_deltas, dtype=float)
    site_ids = [str(site["site_id"]) for site in sites]
    n_sites = len(site_ids)
    if not (
        len(stage1) == len(target) == len(adjustable_mask) == len(deltas) == n_sites
    ):
        raise ValueError("Stage 2 free-energy inputs must have one value per transition")
    if not np.isfinite(stage1).all() or not np.isfinite(target).all() or not np.isfinite(deltas).all():
        raise ValueError("Stage 2 free-energy closure received non-finite values")
    prior = stage1 + np.where(adjustable_mask, deltas, 0.0)
    adjustable_indices = list(np.where(adjustable_mask)[0])
    candidate_pairs = [
        (left, right)
        for left in range(n_sites)
        for right in range(left + 1, n_sites)
        if adjustable_mask[left] or adjustable_mask[right]
    ]

    def unpack(values: np.ndarray) -> Tuple[np.ndarray, Dict[PairKey, float]]:
        """Expand optimizer variables into one-body values and canonical pair terms."""
        one_body = prior.copy()
        cursor = 0
        for idx in adjustable_indices:
            one_body[idx] = float(values[cursor])
            cursor += 1
        pairs: Dict[PairKey, float] = {}
        for left, right in candidate_pairs:
            pairs[canonical_pair_key(site_ids[left], site_ids[right])] = float(values[cursor])
            cursor += 1
        return one_body, pairs

    initial = np.asarray(
        [prior[idx] for idx in adjustable_indices] + [0.0] * len(candidate_pairs),
        dtype=float,
    )

    def predicted_macro(values: np.ndarray) -> np.ndarray:
        """Map one optimizer vector to its implied macroscopic pKa ladder."""
        one_body, pairs = unpack(values)
        mapping = {site_id: float(value) for site_id, value in zip(site_ids, one_body)}
        return np.asarray(
            macro_pka_values_from_free_energy(nodes, sites, mapping, pairs), dtype=float
        )

    def objective(values: np.ndarray) -> float:
        """Penalize macro mismatch plus movement from one-body and zero-pair priors."""
        predicted = predicted_macro(values)
        macro_loss = float(np.mean((predicted - target) ** 2))
        if adjustable_indices:
            one_body_loss = float(np.mean([
                (values[pos] - prior[idx]) ** 2
                for pos, idx in enumerate(adjustable_indices)
            ]))
        else:
            one_body_loss = 0.0
        pair_values = values[len(adjustable_indices):]
        pair_loss = float(np.mean(pair_values ** 2)) if len(pair_values) else 0.0
        return (
            macro_loss
            + float(one_body_regularization) * one_body_loss
            + float(pair_regularization) * pair_loss
        )

    if len(initial):
        bounds = [(-15.0, 30.0)] * len(adjustable_indices) + [
            (-float(max_abs_pair_coupling), float(max_abs_pair_coupling))
        ] * len(candidate_pairs)
        fit = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-12},
        )
        solution = fit.x if fit.success else initial
        optimizer_success = bool(fit.success)
        optimizer_iterations = int(getattr(fit, "nit", 0))
    else:
        solution = initial
        optimizer_success = True
        optimizer_iterations = 0

    # Store chemically irrelevant sub-nanopKa optimizer jitter at a stable
    # precision so identical inputs produce identical artifacts.
    solution = np.round(solution, 10)
    one_body_array, pair_map = unpack(solution)
    one_body_map = {
        site_id: float(value) for site_id, value in zip(site_ids, one_body_array)
    }
    reconstructed = np.asarray(
        macro_pka_values_from_free_energy(nodes, sites, one_body_map, pair_map), dtype=float
    )
    errors = np.abs(reconstructed - target)

    # Local numerical rank describes the chosen parameterization, not the
    # amount of experimental information.  It is reported separately so a
    # full-rank fit to a model-predicted ladder is not mistaken for measured
    # pair-energy identifiability.
    if len(solution):
        epsilon = 1e-5
        jacobian = np.zeros((len(target), len(solution)), dtype=float)
        for column in range(len(solution)):
            shifted = solution.copy()
            shifted[column] += epsilon
            jacobian[:, column] = (predicted_macro(shifted) - reconstructed) / epsilon
        jacobian_rank = int(np.linalg.matrix_rank(jacobian, tol=1e-6))
    else:
        jacobian_rank = 0
    parameter_count = int(len(solution))
    nullity = max(0, parameter_count - jacobian_rank)
    pair_terms = []
    conversion = -R_KJ_MOL_K * DEFAULT_TEMPERATURE_K * math.log(10.0)
    for (left, right), coupling in sorted(pair_map.items()):
        pair_terms.append({
            "site_i": left,
            "site_j": right,
            "coupling_log10_units": float(coupling),
            "interaction_delta_g_kj_mol_at_298k": float(conversion * coupling),
            "source": STAGE2_FREE_ENERGY_METHOD,
            "experimentally_identified": False,
        })
    identifiability = (
        "no_pair_parameter"
        if not candidate_pairs
        else "regularized_prior_dependent_all_macro_steps_anchored"
        if int(observed_macro_anchor_count) >= len(target)
        else "regularized_prior_dependent_predicted_macro_ladder"
    )
    diagnostics = {
        "status": "optimized" if optimizer_success else "optimizer_fallback_to_prior",
        "method": STAGE2_FREE_ENERGY_METHOD,
        "optimizer_success": optimizer_success,
        "optimizer_iterations": optimizer_iterations,
        "macro_reconstruction_mae": float(np.mean(errors)) if len(errors) else 0.0,
        "macro_reconstruction_max_abs_error": float(np.max(errors)) if len(errors) else 0.0,
        "one_body_regularization": float(one_body_regularization),
        "pair_regularization": float(pair_regularization),
        "pair_coupling_bound_log10_units": float(max_abs_pair_coupling),
        "adjustable_one_body_parameter_count": int(len(adjustable_indices)),
        "pair_parameter_count": int(len(candidate_pairs)),
        "active_pair_parameter_count": int(sum(
            abs(value) > 1e-4 for value in pair_map.values()
        )),
        "free_parameter_count": parameter_count,
        "local_macro_jacobian_rank": jacobian_rank,
        "local_parameter_nullity": nullity,
        "observed_macro_anchor_count": int(observed_macro_anchor_count),
        "macro_ladder_step_count": int(len(target)),
        "pair_coupling_identifiability": identifiability,
        "pair_couplings_experimentally_identified": False,
    }
    return one_body_map, pair_terms, diagnostics


def annotate_pairwise_microstate_populations(
    nodes: List[Dict],
    sites: Sequence[Mapping],
    one_body_pkas: Mapping[str, float],
    pair_couplings: Mapping[PairKey, float] | None,
    ph: float,
) -> Dict:
    """Annotate nodes with normalized pairwise-model populations at the requested pH.

    Thermodynamic weight is assigned once per protonation configuration, then
    divided equally among duplicate tautomer drawings; Stage 3 can subsequently
    replace that equal division with its explicit tautomer ranking.
    """
    records = _configuration_records(nodes, sites)
    pair_map = pair_couplings or {}
    site_ids = [str(site["site_id"]) for site in sites]
    signature_multiplicity = Counter(_configuration_signature(node) for node in nodes)
    log_weights: Dict[Tuple, float] = {}
    for record in records:
        log_weights[record["signature"]] = (
            configuration_log10_coefficient(
                record["occupancies"], site_ids, one_body_pkas, pair_map
            )
            - int(record["proton_count"]) * float(ph)
        )
    finite = [value for value in log_weights.values() if math.isfinite(value)]
    if not finite:
        return {
            "status": "no_complete_microstate_configuration",
            "population_sum": 0.0,
            "dominant_population": 0.0,
            "configuration_entropy": float("nan"),
            "effective_configuration_count": float("nan"),
        }
    high = max(finite)
    scaled = {
        signature: 10.0 ** (value - high)
        for signature, value in log_weights.items()
    }
    total = float(sum(scaled.values()))
    configuration_populations = {
        signature: value / total for signature, value in scaled.items()
    }
    node_populations = []
    for node in nodes:
        signature = _configuration_signature(node)
        configuration_population = float(configuration_populations[signature])
        node_population = configuration_population / max(1, signature_multiplicity[signature])
        node["stage3_configuration_population_at_ph"] = configuration_population
        node["stage3_node_population_at_ph"] = float(node_population)
        node["stage3_log10_binding_coefficient"] = float(
            log_weights[signature] + int(node.get("protonated_site_count", 0)) * float(ph)
        )
        node_populations.append(float(node_population))
    positive = np.asarray(
        [value for value in configuration_populations.values() if value > 0.0], dtype=float
    )
    entropy = float(-np.sum(positive * np.log(positive))) if len(positive) else float("nan")
    ordered = sorted(node_populations, reverse=True)
    return {
        "status": "complete_pairwise_coupled_population",
        "population_model": "enumerated_pairwise_free_energy",
        "population_sum": float(sum(node_populations)),
        "dominant_population": float(ordered[0]),
        "second_population": float(ordered[1]) if len(ordered) > 1 else 0.0,
        "dominant_population_gap": float(
            ordered[0] - (ordered[1] if len(ordered) > 1 else 0.0)
        ),
        "configuration_entropy": entropy,
        "effective_configuration_count": (
            float(math.exp(entropy)) if math.isfinite(entropy) else float("nan")
        ),
        "pair_coupling_count": int(len(pair_map)),
    }


def site_marginals_from_pairwise_model(
    nodes: Sequence[Mapping],
    sites: Sequence[Mapping],
    one_body_pkas: Mapping[str, float],
    pair_couplings: Mapping[PairKey, float] | None,
    ph: float,
) -> Dict[str, float]:
    """Marginalize coupled microstate populations into protonation per site."""
    copied_nodes = [dict(node) for node in nodes]
    annotate_pairwise_microstate_populations(
        copied_nodes, sites, one_body_pkas, pair_couplings, ph
    )
    records = {
        _configuration_signature(node): tuple(
            site_acid_side_occupancy(node, site) for site in sites
        )
        for node in copied_nodes
    }
    result = {str(site["site_id"]): 0.0 for site in sites}
    for node in copied_nodes:
        occupancy = records[_configuration_signature(node)]
        probability = float(node.get("stage3_node_population_at_ph", 0.0))
        for idx, site in enumerate(sites):
            if occupancy[idx]:
                result[str(site["site_id"])] += probability
    return result
