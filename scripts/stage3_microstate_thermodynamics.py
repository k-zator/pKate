"""Thermodynamic helpers for canonical Stage 3 microstate inference."""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize  # type: ignore

from build_molecule_microstate_network_dataset import independent_site_macro_pkas
from stage2_network_context import project_macro_pkas_nonincreasing_with_fixed


STAGE3_SCHEMA_VERSION = "1.6.0"
STAGE3_LOCAL_PKA_METHOD = (
    "direct_stage2_pairwise_free_energy_model"
)


def _log10_add(a: float, b: float) -> float:
    """Add two base-10 log weights without overflowing their linear values."""
    if math.isinf(a) and a < 0:
        return b
    if math.isinf(b) and b < 0:
        return a
    high = max(a, b)
    return high + math.log10(10.0 ** (a - high) + 10.0 ** (b - high))


def _coupled_configuration_coefficients(
    nodes: Sequence[Dict],
    site_pka_map: Dict[str, float],
) -> Tuple[Dict[Tuple[Tuple[str, int], ...], float], Dict[Tuple[Tuple[str, int], ...], int]]:
    """Propagate local pKas through a connected coordinate graph.

    Returns a pH-independent log10 coefficient and proton count for every
    unique configuration; an empty result signals an incomplete/disconnected graph.
    """
    configs: Dict[Tuple[Tuple[str, int], ...], Dict] = {}
    for node in nodes:
        levels = node.get("group_levels")
        if not levels:
            continue
        key = tuple(sorted((str(group), int(level)) for group, level in levels.items()))
        configs.setdefault(key, node)
    if not configs:
        return {}, {}
    counts = {
        config: int(node.get("protonated_site_count", 0))
        for config, node in configs.items()
    }
    adjacency: Dict[Tuple[Tuple[str, int], ...], List[Tuple[Tuple[Tuple[str, int], ...], float]]] = {}
    config_items = list(configs.items())
    for site_id, pka in site_pka_map.items():
        acid = [
            config for config, node in config_items
            if node.get("site_forms", {}).get(site_id) == "acid_form"
        ]
        base = [
            config for config, node in config_items
            if node.get("site_forms", {}).get(site_id) == "base_form"
        ]
        for acid_config in acid:
            acid_levels = dict(acid_config)
            for base_config in base:
                base_levels = dict(base_config)
                changed = [
                    group for group in acid_levels
                    if acid_levels[group] != base_levels.get(group)
                ]
                if len(changed) != 1 or counts[acid_config] != counts[base_config] + 1:
                    continue
                group = changed[0]
                if acid_levels[group] != base_levels[group] + 1:
                    continue
                adjacency.setdefault(base_config, []).append((acid_config, float(pka)))
                adjacency.setdefault(acid_config, []).append((base_config, -float(pka)))

    minimum = min(counts.values())
    roots = sorted(config for config, count in counts.items() if count == minimum)
    coefficients = {root: 0.0 for root in roots}
    queue = list(roots)
    while queue:
        config = queue.pop(0)
        for neighbor, delta in adjacency.get(config, []):
            if neighbor not in coefficients:
                coefficients[neighbor] = coefficients[config] + delta
                queue.append(neighbor)
    if set(coefficients) != set(configs):
        return {}, {}
    return coefficients, counts


def macro_pka_values(
    local_pkas: Sequence[float],
    nodes: Sequence[Dict] | None = None,
    site_ids: Sequence[str] | None = None,
) -> List[float]:
    """Convert local transition pKas into a macroscopic binding-polynomial ladder.

    Enumerated coupled coordinates are used when available; otherwise the
    independent-site analytical construction provides the compatibility fallback.
    """
    if nodes and site_ids and any(node.get("group_levels") for node in nodes):
        site_map = {
            str(site_id): float(value)
            for site_id, value in zip(site_ids, local_pkas)
        }
        coefficients, counts = _coupled_configuration_coefficients(nodes, site_map)
        if coefficients:
            by_count: Dict[int, float] = {}
            for config, coefficient in coefficients.items():
                count = counts[config]
                by_count[count] = _log10_add(by_count.get(count, -math.inf), coefficient)
            ordered_counts = sorted(by_count)
            return [
                float(by_count[upper] - by_count[lower])
                for lower, upper in zip(ordered_counts, ordered_counts[1:])
                if upper == lower + 1
            ]
    steps, _ = independent_site_macro_pkas(
        [(f"site_{idx}", float(value)) for idx, value in enumerate(local_pkas)]
    )
    return [float(step["predicted_macro_pka"]) for step in steps]


def infer_effective_local_pkas(
    stage1_local_pkas: Sequence[float],
    stage2_macro_pkas: Sequence[float],
    adjustable: Sequence[bool],
    proposed_deltas: Sequence[float] | None = None,
    regularization: float = 1e-3,
    nodes: Sequence[Dict] | None = None,
    site_ids: Sequence[str] | None = None,
) -> Tuple[List[float], Dict[str, float | int | str | bool]]:
    """Infer ordered effective local pKas without moving fallback sites.

    This follows the enumerated coordinate topology (including serial
    amphoteric edges), but is not a learned context-dependent edge model. It
    finds local values whose binding-polynomial macro ladder most closely
    matches Stage 2 while remaining close to Stage 1.
    """
    stage1 = np.asarray(stage1_local_pkas, dtype=float)
    target = np.asarray(stage2_macro_pkas, dtype=float)
    adjustable_mask = np.asarray(adjustable, dtype=bool)
    if len(stage1) != len(target) or len(stage1) != len(adjustable_mask):
        raise ValueError("local pKas, macro pKas and adjustable mask must have equal length")
    if not np.isfinite(stage1).all() or not np.isfinite(target).all():
        raise ValueError("Stage 3 inversion received non-finite pKas")
    if len(stage1) == 0:
        return [], {
            "status": "empty_network",
            "optimizer_success": True,
            "adjustable_site_count": 0,
            "macro_reconstruction_mae": 0.0,
            "macro_reconstruction_max_abs_error": 0.0,
        }

    deltas = (
        np.zeros(len(stage1), dtype=float)
        if proposed_deltas is None
        else np.asarray(proposed_deltas, dtype=float)
    )
    if len(deltas) != len(stage1):
        raise ValueError("proposed_deltas must match the site count")
    initial_full = stage1 + np.where(adjustable_mask, deltas, 0.0)
    initial_full = np.asarray(
        project_macro_pkas_nonincreasing_with_fixed(
            initial_full.tolist(), adjustable_mask.tolist()
        ),
        dtype=float,
    )
    adjustable_indices = np.where(adjustable_mask)[0]

    if not len(adjustable_indices):
        reconstructed = np.asarray(macro_pka_values(stage1, nodes, site_ids), dtype=float)
        errors = np.abs(reconstructed - target)
        return stage1.tolist(), {
            "status": "stage1_fallback_no_adjustable_sites",
            "optimizer_success": True,
            "adjustable_site_count": 0,
            "macro_reconstruction_mae": float(np.mean(errors)),
            "macro_reconstruction_max_abs_error": float(np.max(errors)),
        }

    def compose(values: np.ndarray) -> np.ndarray:
        """Insert adjustable optimizer values while holding fallback sites fixed."""
        full = stage1.copy()
        full[adjustable_indices] = values
        return full

    def objective(values: np.ndarray) -> float:
        """Balance macro-ladder reconstruction against displacement from Stage 1."""
        full = compose(values)
        reconstructed = np.asarray(macro_pka_values(full, nodes, site_ids), dtype=float)
        macro_loss = float(np.mean((reconstructed - target) ** 2))
        prior_loss = float(np.mean((full[adjustable_indices] - stage1[adjustable_indices]) ** 2))
        return macro_loss + (float(regularization) * prior_loss)

    constraints = ({
        "type": "ineq",
        "fun": lambda values: compose(values)[:-1] - compose(values)[1:],
    },)
    result = minimize(
        objective,
        initial_full[adjustable_indices],
        method="SLSQP",
        bounds=[(-15.0, 30.0)] * len(adjustable_indices),
        constraints=constraints,
        options={"maxiter": 250, "ftol": 1e-10},
    )
    effective = compose(result.x if result.success else initial_full[adjustable_indices])
    # Numerical guard: the optimization constraints should already enforce
    # this, but the fixed projection makes the deployment invariant explicit.
    effective = np.asarray(
        project_macro_pkas_nonincreasing_with_fixed(
            effective.tolist(), adjustable_mask.tolist()
        ),
        dtype=float,
    )
    effective[~adjustable_mask] = stage1[~adjustable_mask]
    reconstructed = np.asarray(macro_pka_values(effective, nodes, site_ids), dtype=float)
    errors = np.abs(reconstructed - target)
    return effective.tolist(), {
        "status": "optimized" if result.success else "optimizer_fallback_initial_shift",
        "optimizer_success": bool(result.success),
        "optimizer_iterations": int(getattr(result, "nit", 0)),
        "adjustable_site_count": int(len(adjustable_indices)),
        "objective": float(objective(effective[adjustable_indices])),
        "macro_reconstruction_mae": float(np.mean(errors)),
        "macro_reconstruction_max_abs_error": float(np.max(errors)),
    }


def annotate_microstate_populations(
    nodes: List[Dict],
    site_pka_map: Dict[str, float],
    ph: float,
) -> Dict[str, float | int | str]:
    """Annotate nodes and return population diagnostics and site marginals."""
    coupled = any(node.get("group_levels") for node in nodes)
    signatures = [
        tuple(sorted(node.get("group_levels", {}).items()))
        if coupled
        else tuple(sorted(node.get("site_forms", {}).items()))
        for node in nodes
    ]
    multiplicity = Counter(signatures)
    if coupled:
        coefficients, counts = _coupled_configuration_coefficients(nodes, site_pka_map)
        log_weights = [
            coefficients.get(signature, -math.inf)
            - counts.get(signature, 0) * float(ph)
            for signature in signatures
        ]
    else:
        log_weights = []
        complete = set(site_pka_map)
        for node in nodes:
            forms = node.get("site_forms", {})
            if set(forms) != complete:
                log_weights.append(-math.inf)
                continue
            log_weights.append(
                sum(
                    float(site_pka_map[site_id]) - float(ph)
                    for site_id, form in forms.items()
                    if form == "acid_form"
                )
            )
    finite = [value for value in log_weights if math.isfinite(value)]
    if not finite:
        return {
            "status": "no_complete_microstate_configuration",
            "population_sum": 0.0,
            "dominant_population": 0.0,
            "configuration_entropy": float("nan"),
            "effective_configuration_count": float("nan"),
        }
    high = max(finite)
    configuration_scaled = {
        signature: 10.0 ** (log_weight - high)
        for signature, log_weight in zip(signatures, log_weights)
        if math.isfinite(log_weight)
    }
    total = float(sum(configuration_scaled.values()))
    configuration_populations = {
        signature: value / total
        for signature, value in configuration_scaled.items()
    }
    node_populations = []
    for node, signature in zip(nodes, signatures):
        configuration_population = configuration_populations.get(signature, 0.0)
        node_population = configuration_population / max(1, multiplicity[signature])
        node["stage3_configuration_population_at_ph"] = float(configuration_population)
        node["stage3_node_population_at_ph"] = float(node_population)
        node_populations.append(float(node_population))

    positive = np.asarray(
        [value for value in configuration_populations.values() if value > 0.0],
        dtype=float,
    )
    entropy = float(-np.sum(positive * np.log(positive))) if len(positive) else float("nan")
    sorted_populations = sorted(node_populations, reverse=True)
    return {
        "status": "complete_independent_site_population",
        "population_model": (
            "enumerated_coupled_coordinates" if coupled else "independent_binary_sites"
        ),
        "population_sum": float(sum(node_populations)),
        "dominant_population": float(sorted_populations[0]),
        "second_population": float(sorted_populations[1]) if len(sorted_populations) > 1 else 0.0,
        "dominant_population_gap": float(
            sorted_populations[0] - (sorted_populations[1] if len(sorted_populations) > 1 else 0.0)
        ),
        "configuration_entropy": entropy,
        "effective_configuration_count": (
            float(math.exp(entropy)) if math.isfinite(entropy) else float("nan")
        ),
    }


def site_protonated_marginals(nodes: Sequence[Dict], site_ids: Sequence[str]) -> Dict[str, float]:
    """Sum populated node probabilities on the protonated side of each transition."""
    marginals = {str(site_id): 0.0 for site_id in site_ids}
    transition_levels: Dict[str, Tuple[str, int]] = {}
    if any(node.get("group_levels") for node in nodes):
        for site_id in marginals:
            acid_nodes = [
                node for node in nodes
                if node.get("site_forms", {}).get(site_id) == "acid_form"
            ]
            base_nodes = [
                node for node in nodes
                if node.get("site_forms", {}).get(site_id) == "base_form"
            ]
            for acid_node in acid_nodes:
                acid_levels = acid_node.get("group_levels", {})
                for base_node in base_nodes:
                    base_levels = base_node.get("group_levels", {})
                    changed = [
                        group for group in acid_levels
                        if int(acid_levels[group]) == int(base_levels.get(group, 10**9)) + 1
                        and all(
                            other == group
                            or int(acid_levels[other]) == int(base_levels.get(other, 10**9))
                            for other in acid_levels
                        )
                    ]
                    if len(changed) == 1:
                        transition_levels[site_id] = (
                            str(changed[0]), int(acid_levels[changed[0]])
                        )
                        break
                if site_id in transition_levels:
                    break
    for node in nodes:
        population = float(node.get("stage3_node_population_at_ph", 0.0))
        for site_id in marginals:
            form = node.get("site_forms", {}).get(site_id)
            is_acid_side = form == "acid_form"
            if form == "outside_transition" and site_id in transition_levels:
                group, acid_level = transition_levels[site_id]
                is_acid_side = int(node.get("group_levels", {}).get(group, -10**9)) >= acid_level
            if is_acid_side:
                marginals[site_id] += population
    return marginals
