#!/usr/bin/env python3
"""Render clearly named Stage 3 validation and microstate diagnostic galleries."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd  # type: ignore
from rdkit import Chem  # type: ignore
from rdkit.Chem import Draw  # type: ignore


DEFAULT_STAGE1_OOF = (
    "data/processed/ml_models_experimental_only/stage1_intrinsic/"
    "scaffold_oof_predictions.csv"
)
DEFAULT_STAGE2_EVAL = (
    "data/processed/ml_models_experimental_only/stage2_network_context/"
    "scaffold_eval_predictions.csv"
)
DEFAULT_STAGE3_SITES = (
    "data/processed/ml_models_experimental_only/stage3_microstates/"
    "stage3_site_predictions.csv"
)
DEFAULT_OUT_DIR = "data/processed/stage3_failure_visualization_canonical"

SECONDARY_GUESS_DEFAULT_ERROR_THRESHOLD = 3.0
SECONDARY_GUESS_DEFAULT_MIN_IMPROVEMENT = 1.0
SECONDARY_GUESS_DEFAULT_MAX_ABS_ERROR = 2.0


def _finite_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if pd.notna(parsed) else None


def _secondary_transition_guess(
    row: pd.Series,
    *,
    error_threshold: float = SECONDARY_GUESS_DEFAULT_ERROR_THRESHOLD,
    min_improvement: float = SECONDARY_GUESS_DEFAULT_MIN_IMPROVEMENT,
    max_secondary_abs_error: float = SECONDARY_GUESS_DEFAULT_MAX_ABS_ERROR,
) -> Dict[str, object]:
    """Return the best alternative edge as an audit suggestion, never a label.

    The candidate ranking uses the blind Stage 1 network macro-pKa values already
    serialized in the Stage 2 scaffold holdout.  The experimental value is used
    only by this post-hoc audit to rank candidates; it is not a deployment input
    and the primary curated assignment remains unchanged until reviewed.
    """
    result: Dict[str, object] = {
        "secondary_guess_available": False,
        "secondary_guess_site_id": "",
        "secondary_guess_site_label": "",
        "secondary_guess_site_family": "",
        "secondary_guess_coupling_group": "",
        "secondary_guess_assignment_mode": "unavailable",
        "secondary_guess_existing_anchor_pka": float("nan"),
        "secondary_guess_requires_joint_swap": False,
        "secondary_guess_stage1_macro_pka": float("nan"),
        "secondary_guess_abs_error": float("nan"),
        "primary_assignment_stage1_abs_error": float("nan"),
        "secondary_guess_improvement_over_primary_stage1": float("nan"),
        "secondary_guess_joint_swap_current_total_abs_error": float("nan"),
        "secondary_guess_joint_swap_proposed_total_abs_error": float("nan"),
        "secondary_guess_joint_swap_improvement": float("nan"),
        "secondary_guess_joint_swap_reciprocal_abs_error": float("nan"),
        "secondary_guess_review_recommended": False,
        "secondary_guess_confidence": "unavailable",
        "secondary_guess_reason": "not_a_multisite_stage2_validation_row",
        "secondary_guess_site_atom_maps_json": "[]",
        "secondary_guess_site_center_maps_json": "[]",
        "secondary_guess_is_diagnostic_not_training_label": True,
    }
    if str(row.get("validation_scope", "")) != "stage2_scaffold_validation_multisite":
        return result
    observed = _finite_float(row.get("experimental_anchor_pka"))
    validation_error = _finite_float(row.get("abs_error"))
    if observed is None or validation_error is None:
        result["secondary_guess_reason"] = "missing_experimental_or_validation_value"
        return result
    try:
        sites = json.loads(str(row.get("network_sites_json", "[]")))
    except (TypeError, ValueError, json.JSONDecodeError):
        sites = []
    if not isinstance(sites, list) or len(sites) < 2:
        result["secondary_guess_reason"] = "no_alternative_enumerated_transition"
        return result

    primary_site_id = str(row.get("site_id", ""))
    primary_stage1 = _finite_float(row.get("stage1_network_macro_pka"))
    if primary_stage1 is None:
        primary_site = next(
            (site for site in sites if str(site.get("site_id", "")) == primary_site_id),
            None,
        )
        if primary_site is not None:
            primary_stage1 = _finite_float(primary_site.get("rank_associated_macro_pka"))
    if primary_stage1 is None:
        result["secondary_guess_reason"] = "primary_stage1_network_pka_unavailable"
        return result

    candidates = []
    for site in sites:
        if str(site.get("site_id", "")) == primary_site_id:
            continue
        candidate_pka = _finite_float(site.get("rank_associated_macro_pka"))
        if candidate_pka is None:
            continue
        candidates.append((
            abs(candidate_pka - observed),
            str(site.get("site_id", "")),
            site,
            candidate_pka,
            _finite_float(site.get("experimental_anchor_pka")),
        ))
    if not candidates:
        result["secondary_guess_reason"] = "alternative_stage1_network_pka_unavailable"
        return result
    secondary_error, _, secondary, secondary_pka, secondary_anchor = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    primary_error = abs(primary_stage1 - observed)
    improvement = primary_error - secondary_error
    requires_joint_swap = secondary_anchor is not None
    reciprocal_error = float("nan")
    joint_current_error = float("nan")
    joint_proposed_error = float("nan")
    joint_improvement = float("nan")
    if requires_joint_swap:
        reciprocal_error = abs(primary_stage1 - float(secondary_anchor))
        joint_current_error = primary_error + abs(secondary_pka - float(secondary_anchor))
        joint_proposed_error = secondary_error + reciprocal_error
        joint_improvement = joint_current_error - joint_proposed_error
        review_recommended = bool(
            validation_error > float(error_threshold)
            and secondary_error <= float(max_secondary_abs_error)
            and reciprocal_error <= float(max_secondary_abs_error)
            and joint_improvement >= float(min_improvement)
        )
    else:
        review_recommended = bool(
            validation_error > float(error_threshold)
            and secondary_error <= float(max_secondary_abs_error)
            and improvement >= float(min_improvement)
        )
    confidence_improvement = joint_improvement if requires_joint_swap else improvement
    confidence_error = max(secondary_error, reciprocal_error) if requires_joint_swap else secondary_error
    if review_recommended and confidence_error <= 1.0 and confidence_improvement >= 2.0:
        confidence = "strong"
    elif review_recommended:
        confidence = "moderate"
    else:
        confidence = "low"
    if review_recommended and requires_joint_swap:
        reason = "large_heldout_error_and_reciprocal_anchor_swap_fits_better"
    elif review_recommended:
        reason = "large_heldout_error_and_alternative_stage1_edge_fits_better"
    elif validation_error <= float(error_threshold):
        reason = "primary_heldout_error_below_review_threshold"
    elif requires_joint_swap and joint_improvement < float(min_improvement):
        reason = "reciprocal_anchor_swap_does_not_improve_enough"
    elif requires_joint_swap:
        reason = "reciprocal_anchor_swap_remains_too_far_from_experimental_pkas"
    elif improvement < float(min_improvement):
        reason = "alternative_does_not_improve_enough"
    else:
        reason = "alternative_remains_too_far_from_experimental_pka"
    result.update({
        "secondary_guess_available": True,
        "secondary_guess_site_id": str(secondary.get("site_id", "")),
        "secondary_guess_site_label": str(secondary.get("label", "")),
        "secondary_guess_site_family": str(secondary.get("family", "")),
        "secondary_guess_coupling_group": str(secondary.get("coupling_group", "")),
        "secondary_guess_assignment_mode": (
            "joint_swap_with_existing_anchor" if requires_joint_swap
            else "unoccupied_alternative_edge"
        ),
        "secondary_guess_existing_anchor_pka": (
            float(secondary_anchor) if secondary_anchor is not None else float("nan")
        ),
        "secondary_guess_requires_joint_swap": requires_joint_swap,
        "secondary_guess_stage1_macro_pka": float(secondary_pka),
        "secondary_guess_abs_error": float(secondary_error),
        "primary_assignment_stage1_abs_error": float(primary_error),
        "secondary_guess_improvement_over_primary_stage1": float(improvement),
        "secondary_guess_joint_swap_current_total_abs_error": float(joint_current_error),
        "secondary_guess_joint_swap_proposed_total_abs_error": float(joint_proposed_error),
        "secondary_guess_joint_swap_improvement": float(joint_improvement),
        "secondary_guess_joint_swap_reciprocal_abs_error": float(reciprocal_error),
        "secondary_guess_review_recommended": review_recommended,
        "secondary_guess_confidence": confidence,
        "secondary_guess_reason": reason,
        "secondary_guess_site_atom_maps_json": json.dumps(
            secondary.get("atom_maps", []), separators=(",", ":")
        ),
        "secondary_guess_site_center_maps_json": json.dumps(
            secondary.get("center_maps", []), separators=(",", ":")
        ),
    })
    return result


def build_pka_failure_table(
    stage1_oof: pd.DataFrame,
    stage2_eval: pd.DataFrame,
    stage3_sites: pd.DataFrame,
    secondary_guess_error_threshold: float = SECONDARY_GUESS_DEFAULT_ERROR_THRESHOLD,
    secondary_guess_min_improvement: float = SECONDARY_GUESS_DEFAULT_MIN_IMPROVEMENT,
    secondary_guess_max_abs_error: float = SECONDARY_GUESS_DEFAULT_MAX_ABS_ERROR,
) -> pd.DataFrame:
    stage1_label_column = "site_label" if "site_label" in stage1_oof.columns else "candidate_label"
    stage1_columns = [
        "molecule_id", "site_id", stage1_label_column, "site_family",
        "experimental_anchor_pka", "pred_intrinsic_pka",
    ] + [
        column for column in (
            "validation_regime", "exact_label_training_rows_in_fold",
            "exact_label_training_scaffolds_in_fold",
        ) if column in stage1_oof.columns
    ]
    stage1 = stage1_oof[stage1_columns].copy().rename(
        columns={stage1_label_column: "site_label"}
    )
    stage1 = stage1.rename(columns={"pred_intrinsic_pka": "validation_predicted_pka"})
    stage1["validation_scope"] = "stage1_scaffold_oof_single_site"

    stage2_prediction_column = (
        "stage2_supported_pairwise_free_energy_macro_pka"
        if "stage2_supported_pairwise_free_energy_macro_pka" in stage2_eval.columns
        else "stage2_supported_projected_macro_pka"
    )
    stage2_columns = [
        "molecule_id", "site_id", "site_label", "site_family",
        "experimental_anchor_pka", stage2_prediction_column,
    ] + [
        column for column in (
            "stage1_network_macro_pka", "network_sites_json",
            "experimental_values_json", "experimental_measurement_count",
            "experimental_evidence_tier",
        )
        if column in stage2_eval.columns
    ]
    stage2 = stage2_eval[stage2_columns].copy().rename(columns={
        stage2_prediction_column: "validation_predicted_pka"
    })
    stage2["validation_scope"] = "stage2_scaffold_validation_multisite"
    combined = pd.concat([stage1, stage2], ignore_index=True)
    combined["signed_error"] = (
        combined["validation_predicted_pka"] - combined["experimental_anchor_pka"]
    )
    combined["abs_error"] = combined["signed_error"].abs()

    context_columns = [
        "molecule_id", "site_id", "input_smiles", "atom_mapped_smiles",
        "input_member_form",
        "site_atom_maps_json", "site_center_maps_json", "population_ph",
        "stage1_intrinsic_pka", "stage2_predicted_macro_pka",
        "stage3_reconstructed_macro_pka",
        "stage3_one_body_pka", "stage3_effective_local_pka",
        "stage3_one_body_pka_interval_low", "stage3_one_body_pka_interval_high",
        "stage3_contextual_edge_pka_at_dominant_background",
        "stage3_contextual_edge_pka_min", "stage3_contextual_edge_pka_max",
        "stage2_pair_coupling_identifiability", "stage3_protonated_probability_at_ph",
        "stage3_uncoupled_protonated_probability_at_ph",
        "stage3_site_call_changed_by_pair_coupling",
        "stage3_predicted_site_form_at_ph", "stage3_site_state_decision_confidence",
        "stage3_uncertainty_aware_site_state_confidence",
        "stage3_empirical_pka_side_call_confidence",
        "stage3_empirical_pka_side_centered_evidence",
        "stage3_empirical_macro_pka_interval_90_low",
        "stage3_empirical_macro_pka_interval_90_high",
        "stage3_empirical_calibration_source", "stage3_empirical_calibration_scope",
        "stage3_empirical_calibration_label_rows",
        "stage3_empirical_calibration_family_rows",
        "stage3_overall_site_state_confidence",
        "stage3_overall_site_state_confidence_tier",
        "stage3_effective_local_pka_ci_low", "stage3_effective_local_pka_ci_high",
        "stage3_protonated_probability_ci_low", "stage3_protonated_probability_ci_high",
        "stage1_applicability_domain", "stage1_exact_label_training_rows",
        "population_status", "dominant_atom_mapped_smiles",
        "dominant_configuration_population",
    ]
    context = stage3_sites[[
        column for column in context_columns if column in stage3_sites.columns
    ]].drop_duplicates(
        subset=["molecule_id", "site_id"]
    )
    result = combined.merge(context, on=["molecule_id", "site_id"], how="left")
    result["deployed_assigned_macro_pka"] = result.get("stage2_predicted_macro_pka")
    if "stage3_one_body_pka" in result.columns:
        result["deployed_assigned_one_body_pka"] = result["stage3_one_body_pka"]
    else:
        result["deployed_assigned_one_body_pka"] = result.get(
            "stage3_effective_local_pka"
        )
    # Compatibility alias for review queues generated before Stage 3 schema 1.4.
    result["deployed_assigned_local_pka"] = result["deployed_assigned_one_body_pka"]
    result["deployed_macro_signed_difference_from_experimental"] = (
        result["deployed_assigned_macro_pka"] - result["experimental_anchor_pka"]
    )
    result["deployed_macro_abs_difference_from_experimental"] = result[
        "deployed_macro_signed_difference_from_experimental"
    ].abs()
    result["deployed_one_body_signed_difference_from_experimental"] = (
        result["deployed_assigned_one_body_pka"] - result["experimental_anchor_pka"]
    )
    result["deployed_local_signed_difference_from_experimental"] = result[
        "deployed_one_body_signed_difference_from_experimental"
    ]
    result["deployed_pka_interpretation"] = (
        "macro is comparable to the experimental site-associated scalar; "
        "Stage3 one-body pKa plus pair terms gives each context-specific edge pKa"
    )
    result["input_drawing_is_experimental_state_observation"] = False
    implied = result.apply(
        lambda row: _experimental_pka_implied_form(
            row.get("experimental_anchor_pka"), row.get("population_ph")
        ),
        axis=1,
        result_type="expand",
    )
    implied.columns = [
        "experimental_pka_implied_protonated_probability_at_population_ph",
        "experimental_pka_implied_dominant_site_form",
    ]
    result = pd.concat([result, implied], axis=1)
    result["stage3_form_vs_experimental_pka_check"] = result.apply(
        _form_agreement_check, axis=1
    )
    result["experimental_pka_form_check_caveat"] = result["validation_scope"].map(
        lambda scope: (
            "direct single-site Henderson-Hasselbalch check"
            if scope == "stage1_scaffold_oof_single_site"
            else "approximate only: a site-associated macroscopic pKa does not uniquely "
            "determine a local form in a coupled multisite network"
        )
    )
    secondary = result.apply(
        lambda row: _secondary_transition_guess(
            row,
            error_threshold=secondary_guess_error_threshold,
            min_improvement=secondary_guess_min_improvement,
            max_secondary_abs_error=secondary_guess_max_abs_error,
        ),
        axis=1,
        result_type="expand",
    )
    result = pd.concat([result, secondary], axis=1)
    return result


def _experimental_pka_implied_form(
    experimental_pka: object,
    ph: object,
) -> Tuple[float, str]:
    """Return HH protonated fraction and dominant pair member implied by a pKa."""
    if experimental_pka is None or ph is None or pd.isna(experimental_pka) or pd.isna(ph):
        return float("nan"), "unavailable"
    delta = float(ph) - float(experimental_pka)
    if delta > 300.0:
        probability = 0.0
    elif delta < -300.0:
        probability = 1.0
    else:
        probability = 1.0 / (1.0 + (10.0 ** delta))
    if probability > 0.5:
        form = "acid_form"
    elif probability < 0.5:
        form = "base_form"
    else:
        form = "equal_acid_and_base"
    return float(probability), form


def _form_agreement_check(row: pd.Series) -> str:
    predicted = str(row.get("stage3_predicted_site_form_at_ph", "unavailable"))
    implied = str(row.get("experimental_pka_implied_dominant_site_form", "unavailable"))
    if predicted not in {"acid_form", "base_form"} or implied not in {
        "acid_form", "base_form",
    }:
        return "unavailable_or_boundary"
    return "agrees" if predicted == implied else "disagrees"


def _parse_maps(raw: object) -> List[int]:
    try:
        values = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return [int(value) for value in values]


def _mapped_molecule_and_highlights(
    mapped_smiles: object,
    atom_maps: object,
    center_maps: object,
) -> Tuple[Optional[Chem.Mol], List[int], Dict[int, Tuple[float, float, float]]]:
    mol = Chem.MolFromSmiles(str(mapped_smiles))
    if mol is None:
        return None, [], {}
    site_maps = set(_parse_maps(atom_maps))
    centers = set(_parse_maps(center_maps))
    colors: Dict[int, Tuple[float, float, float]] = {}
    for atom in mol.GetAtoms():
        map_number = int(atom.GetAtomMapNum())
        if map_number in site_maps:
            colors[atom.GetIdx()] = (0.95, 0.75, 0.25)
        if map_number in centers:
            colors[atom.GetIdx()] = (0.90, 0.20, 0.20)
    return mol, sorted(colors), colors


def _fmt(value: object, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _legend(row: pd.Series, category: str) -> str:
    identity = f"{row.get('site_family', 'NA')} / {row.get('site_label', 'NA')} | {row.get('molecule_id', '')}"
    if category == "heldout_pka_errors":
        secondary = ""
        if bool(row.get("secondary_guess_review_recommended", False)):
            mode = (
                "JOINT SWAP"
                if bool(row.get("secondary_guess_requires_joint_swap", False))
                else "SECONDARY"
            )
            secondary = (
                f"\n{mode} REVIEW: {row.get('secondary_guess_site_label', 'NA')} "
                f"pKa={_fmt(row.get('secondary_guess_stage1_macro_pka'))} "
                f"(blind Stage1 error={_fmt(row.get('secondary_guess_abs_error'))})"
            )
        return (
            f"{row.get('validation_scope', '')} / {row.get('validation_regime', 'NA')}\n{identity}\n"
            f"exp={_fmt(row.get('experimental_anchor_pka'))} pred={_fmt(row.get('validation_predicted_pka'))} "
            f"error={_fmt(row.get('signed_error'))}\n"
            f"deployed macro pKa={_fmt(row.get('stage2_predicted_macro_pka'))}; "
            f"Stage3 one-body={_fmt(row.get('stage3_one_body_pka', row.get('stage3_effective_local_pka')))}; "
            f"label support in fold={row.get('exact_label_training_rows_in_fold', 'NA')}"
            f"{secondary}"
        )
    if category == "secondary_transition_assignment_guesses":
        return (
            f"SECONDARY TRANSITION AUDIT ({row.get('secondary_guess_confidence', 'NA')})\n"
            f"{row.get('molecule_id', '')}\n"
            f"primary={row.get('site_label', 'NA')} | held-out error={_fmt(row.get('abs_error'))}\n"
            f"secondary={row.get('secondary_guess_site_label', 'NA')} | "
            f"blind Stage1 pKa={_fmt(row.get('secondary_guess_stage1_macro_pka'))} | "
            f"error={_fmt(row.get('secondary_guess_abs_error'))}\n"
            f"experimental pKa={_fmt(row.get('experimental_anchor_pka'))}; "
            f"mode={row.get('secondary_guess_assignment_mode', 'NA')}; "
            f"improvement={_fmt(row.get('secondary_guess_joint_swap_improvement') if bool(row.get('secondary_guess_requires_joint_swap', False)) else row.get('secondary_guess_improvement_over_primary_stage1'))}"
        )
    if category == "low_confidence_state_calls_at_ph_7_4":
        return (
            f"low-confidence site-state call @ pH {_fmt(row.get('population_ph'), 1)}\n{identity}\n"
            f"one-body pKa={_fmt(row.get('stage3_one_body_pka', row.get('stage3_effective_local_pka')))} "
            f"CI=[{_fmt(row.get('stage3_effective_local_pka_ci_low'))},{_fmt(row.get('stage3_effective_local_pka_ci_high'))}]\n"
            f"P(prot)={_fmt(row.get('stage3_protonated_probability_at_ph'), 3)} "
            f"CI=[{_fmt(row.get('stage3_protonated_probability_ci_low'), 3)},{_fmt(row.get('stage3_protonated_probability_ci_high'), 3)}]\n"
            f"form={row.get('stage3_predicted_site_form_at_ph', 'NA')} "
            f"empirical pKa-side conf="
            f"{_fmt(row.get('stage3_empirical_pka_side_call_confidence'), 3)} | "
            f"overall conf={_fmt(row.get('stage3_overall_site_state_confidence'), 3)}"
        )
    if category == "empirical_pka_side_conflicts":
        return (
            f"SCALAR-pKa / COUPLED-STATE CONFLICT @ pH "
            f"{_fmt(row.get('population_ph'), 1)}\n{identity}\n"
            f"macro pKa={_fmt(row.get('stage2_predicted_macro_pka'))} | "
            f"dominant-background edge pKa="
            f"{_fmt(row.get('stage3_contextual_edge_pka_at_dominant_background'))}\n"
            f"P(prot)={_fmt(row.get('stage3_protonated_probability_at_ph'), 3)} | "
            f"form={row.get('stage3_predicted_site_form_at_ph', 'NA')}\n"
            f"empirical pKa-side support="
            f"{_fmt(row.get('stage3_empirical_pka_side_call_confidence'), 3)}"
        )
    if category == "stage2_free_energy_reconstruction_gaps":
        return (
            f"Stage2 free-energy reconstruction gap\n{identity}\n"
            f"Stage2 macro={_fmt(row.get('stage2_predicted_macro_pka'))} "
            f"Stage3 macro={_fmt(row.get('stage3_reconstructed_macro_pka'))}\n"
            f"abs gap={_fmt(row.get('stage3_macro_reconstruction_abs_error'), 3)} "
            f"one-body={_fmt(row.get('stage3_one_body_pka', row.get('stage3_effective_local_pka')))}"
        )
    if category == "pair_coupling_changed_site_calls":
        return (
            f"pair coupling changes site call @ pH {_fmt(row.get('population_ph'), 1)}\n{identity}\n"
            f"P(prot), coupled={_fmt(row.get('stage3_protonated_probability_at_ph'), 3)} "
            f"vs J=0={_fmt(row.get('stage3_uncoupled_protonated_probability_at_ph'), 3)}\n"
            f"edge pKa range=[{_fmt(row.get('stage3_contextual_edge_pka_min'))},"
            f"{_fmt(row.get('stage3_contextual_edge_pka_max'))}]\n"
            f"identifiability={row.get('stage2_pair_coupling_identifiability', 'NA')}"
        )
    return (
        f"incomplete microstate network\n{identity}\n"
        f"network={row.get('network_confidence', 'NA')}\n"
        f"{row.get('population_status', 'NA')}"
    )


def render_gallery(
    rows: pd.DataFrame,
    category: str,
    output_png: Path,
    output_csv: Path,
    max_examples: int,
) -> int:
    sample = rows.head(int(max_examples)).copy()
    # The PNG is intentionally capped, but the companion review table must
    # retain every case rather than silently truncating to the rendered sample.
    rows.to_csv(output_csv, index=False)
    mols = []
    legends = []
    highlights = []
    colors = []
    for _, row in sample.iterrows():
        mapped_smiles = row.get("atom_mapped_smiles")
        if category in {
            "low_confidence_state_calls_at_ph_7_4",
            "empirical_pka_side_conflicts",
        } and pd.notna(row.get("dominant_atom_mapped_smiles")):
            mapped_smiles = row.get("dominant_atom_mapped_smiles")
        atom_maps = row.get("site_atom_maps_json")
        center_maps = row.get("site_center_maps_json")
        if category == "secondary_transition_assignment_guesses":
            atom_maps = row.get("secondary_guess_site_atom_maps_json")
            center_maps = row.get("secondary_guess_site_center_maps_json")
        mol, atom_indices, atom_colors = _mapped_molecule_and_highlights(
            mapped_smiles,
            atom_maps,
            center_maps,
        )
        if mol is None:
            continue
        mols.append(mol)
        legends.append(_legend(row, category))
        highlights.append(atom_indices)
        colors.append(atom_colors)
    if mols:
        image = Draw.MolsToGridImage(
            mols,
            molsPerRow=3,
            subImgSize=(600, 390),
            legends=legends,
            highlightAtomLists=highlights,
            highlightAtomColors=colors,
            useSVG=False,
        )
        image.save(str(output_png))
    else:
        output_png.unlink(missing_ok=True)
    return len(mols)


def _comparison_legend(row: pd.Series, panel: str) -> str:
    identity = (
        f"{row.get('site_family', 'NA')} / {row.get('site_label', 'NA')} | "
        f"{row.get('molecule_id', '')}"
    )
    ph = _fmt(row.get("population_ph"), 1)
    if panel == "input":
        return (
            f"INPUT DRAWING (not an observed equilibrium state)\n{identity}\n"
            f"drawn site member={row.get('input_member_form', 'NA')}\n"
            f"experimental pKa={_fmt(row.get('experimental_anchor_pka'))} implies "
            f"{row.get('experimental_pka_implied_dominant_site_form', 'NA')} @ pH {ph}\n"
            f"implied P(protonated)="
            f"{_fmt(row.get('experimental_pka_implied_protonated_probability_at_population_ph'), 3)}"
        )
    return (
        f"STAGE 3 DOMINANT MICROSTATE @ pH {ph}\n{identity}\n"
        f"predicted site member={row.get('stage3_predicted_site_form_at_ph', 'NA')} | "
        f"P(protonated)={_fmt(row.get('stage3_protonated_probability_at_ph'), 3)}\n"
        f"deployed macro/one-body pKa={_fmt(row.get('deployed_assigned_macro_pka'))}/"
        f"{_fmt(row.get('deployed_assigned_one_body_pka'))}\n"
        f"vs experimental-pKa implication: "
        f"{row.get('stage3_form_vs_experimental_pka_check', 'NA')}"
    )


def render_deployment_comparison_gallery(
    rows: pd.DataFrame,
    output_png: Path,
    output_csv: Path,
    max_examples: int,
) -> int:
    """Render each input drawing beside the deployed dominant microstate."""
    rows.to_csv(output_csv, index=False)
    mols = []
    legends = []
    highlights = []
    colors = []
    rendered_cases = 0
    for _, row in rows.head(int(max_examples)).iterrows():
        pair = []
        for panel, smiles_column in (
            ("input", "atom_mapped_smiles"),
            ("stage3", "dominant_atom_mapped_smiles"),
        ):
            mol, atom_indices, atom_colors = _mapped_molecule_and_highlights(
                row.get(smiles_column),
                row.get("site_atom_maps_json"),
                row.get("site_center_maps_json"),
            )
            if mol is None:
                pair = []
                break
            pair.append((mol, _comparison_legend(row, panel), atom_indices, atom_colors))
        if not pair:
            continue
        rendered_cases += 1
        for mol, legend, atom_indices, atom_colors in pair:
            mols.append(mol)
            legends.append(legend)
            highlights.append(atom_indices)
            colors.append(atom_colors)
    if mols:
        image = Draw.MolsToGridImage(
            mols,
            molsPerRow=2,
            subImgSize=(760, 430),
            legends=legends,
            highlightAtomLists=highlights,
            highlightAtomColors=colors,
            useSVG=False,
        )
        image.save(str(output_png))
    else:
        output_png.unlink(missing_ok=True)
    return rendered_cases


def _deployment_comparison_table_html(rows: pd.DataFrame) -> str:
    columns = [
        "rank", "site", "experimental macro pKa", "deployed macro pKa",
        "Stage3 one-body pKa coefficient", "dominant-background edge pKa",
        "all-background edge pKa range", "one-body sensitivity interval", "P(protonated)",
        "empirical pKa-side call confidence", "empirical macro-pKa 90% interval",
        "input drawing form", "pKa-implied P(protonated)", "pKa-implied form",
        "Stage3 form", "form check",
        "scaffold-held-out pKa", "held-out error", "secondary transition guess",
        "secondary blind Stage1 pKa", "secondary error", "secondary mode",
        "secondary existing anchor", "review recommended", "SMILES",
    ]
    body = []
    for rank, (_, row) in enumerate(rows.iterrows(), start=1):
        values = [
            str(rank),
            f"{row.get('site_family', 'NA')} / {row.get('site_label', 'NA')}",
            _fmt(row.get("experimental_anchor_pka")),
            _fmt(row.get("deployed_assigned_macro_pka")),
            _fmt(row.get("deployed_assigned_one_body_pka")),
            _fmt(row.get("stage3_contextual_edge_pka_at_dominant_background")),
            (
                f"[{_fmt(row.get('stage3_contextual_edge_pka_min'))}, "
                f"{_fmt(row.get('stage3_contextual_edge_pka_max'))}]"
            ),
            (
                f"[{_fmt(row.get('stage3_one_body_pka_interval_low', row.get('stage3_effective_local_pka_ci_low')))}, "
                f"{_fmt(row.get('stage3_one_body_pka_interval_high', row.get('stage3_effective_local_pka_ci_high')))}]"
            ),
            _fmt(row.get("stage3_protonated_probability_at_ph"), 3),
            _fmt(row.get("stage3_empirical_pka_side_call_confidence"), 3),
            (
                f"[{_fmt(row.get('stage3_empirical_macro_pka_interval_90_low'))}, "
                f"{_fmt(row.get('stage3_empirical_macro_pka_interval_90_high'))}]"
            ),
            str(row.get("input_member_form", "NA")),
            _fmt(
                row.get(
                    "experimental_pka_implied_protonated_probability_at_population_ph"
                ),
                3,
            ),
            str(row.get("experimental_pka_implied_dominant_site_form", "NA")),
            str(row.get("stage3_predicted_site_form_at_ph", "NA")),
            str(row.get("stage3_form_vs_experimental_pka_check", "NA")),
            _fmt(row.get("validation_predicted_pka")),
            _fmt(row.get("signed_error")),
            str(row.get("secondary_guess_site_label", "")),
            _fmt(row.get("secondary_guess_stage1_macro_pka")),
            _fmt(row.get("secondary_guess_abs_error")),
            str(row.get("secondary_guess_assignment_mode", "")),
            _fmt(row.get("secondary_guess_existing_anchor_pka")),
            str(bool(row.get("secondary_guess_review_recommended", False))),
            str(row.get("input_smiles", "")),
        ]
        body.append(
            "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in values) + "</tr>"
        )
    header = "".join(f"<th>{html.escape(value)}</th>" for value in columns)
    return (
        "<p><b>Interpretation:</b> the left molecule in each image row is merely "
        "the source drawing; it is not an experimentally measured protonation "
        "state. The right molecule is Stage 3's dominant predicted microstate at "
        "the requested pH. The pKa-implied form is a Henderson-Hasselbalch check; "
        "for multisite molecules it is approximate because a site-associated "
        "macroscopic pKa does not uniquely identify a local microstate. The one-body "
        "coefficient is a model parameter, while the dominant-background edge pKa is "
        "the corresponding microscopic transition in the displayed background. The "
        "empirical confidence is calibrated from scaffold-held-out scalar-pKa residuals; "
        "it is not direct experimental microstate-state validation.</p>"
        f"<div class='table-wrap'><table><thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-oof", default=DEFAULT_STAGE1_OOF)
    parser.add_argument("--stage2-eval", default=DEFAULT_STAGE2_EVAL)
    parser.add_argument("--stage3-sites", default=DEFAULT_STAGE3_SITES)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-examples", type=int, default=24)
    parser.add_argument("--comparison-examples", type=int, default=12)
    parser.add_argument("--pka-error-threshold", type=float, default=3.0)
    parser.add_argument(
        "--secondary-guess-min-improvement", type=float,
        default=SECONDARY_GUESS_DEFAULT_MIN_IMPROVEMENT,
        help="Minimum blind-Stage1 absolute-error improvement required for review.",
    )
    parser.add_argument(
        "--secondary-guess-max-abs-error", type=float,
        default=SECONDARY_GUESS_DEFAULT_MAX_ABS_ERROR,
        help="Maximum blind-Stage1 error allowed for a reviewable secondary edge.",
    )
    parser.add_argument(
        "--low-confidence-threshold", "--ambiguity-confidence-threshold",
        dest="low_confidence_threshold", type=float, default=0.2,
    )
    parser.add_argument(
        "--macro-reconstruction-error-threshold", "--inverse-error-threshold",
        dest="macro_reconstruction_error_threshold", type=float, default=0.3,
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1 = pd.read_csv(args.stage1_oof, low_memory=False)
    stage2 = pd.read_csv(args.stage2_eval, low_memory=False)
    sites = pd.read_csv(args.stage3_sites, low_memory=False)
    pka = build_pka_failure_table(
        stage1,
        stage2,
        sites,
        secondary_guess_error_threshold=float(args.pka_error_threshold),
        secondary_guess_min_improvement=float(args.secondary_guess_min_improvement),
        secondary_guess_max_abs_error=float(args.secondary_guess_max_abs_error),
    )
    heldout_pka_errors = pka[pka["abs_error"] > float(args.pka_error_threshold)].sort_values(
        "abs_error", ascending=False
    )
    deployment_view = heldout_pka_errors.assign(
        _form_check_priority=heldout_pka_errors["stage3_form_vs_experimental_pka_check"].map({
            "disagrees": 0,
            "unavailable_or_boundary": 1,
            "agrees": 2,
        }).fillna(3)
    ).sort_values(["_form_check_priority", "abs_error"], ascending=[True, False]).drop(
        columns="_form_check_priority"
    )
    confidence_column = (
        "stage3_overall_site_state_confidence"
        if "stage3_overall_site_state_confidence" in sites.columns
        else "stage3_site_state_decision_confidence"
    )
    low_confidence = sites[
        sites[confidence_column].notna()
        & (
            sites[confidence_column]
            < float(args.low_confidence_threshold)
        )
    ].sort_values(confidence_column).drop_duplicates("molecule_id")
    empirical_pka_side_conflicts = sites[
        sites.get(
            "stage3_empirical_pka_side_call_confidence",
            pd.Series(float("nan"), index=sites.index),
        ).notna()
        & (
            sites.get(
                "stage3_empirical_pka_side_call_confidence",
                pd.Series(float("nan"), index=sites.index),
            ) < 0.5
        )
    ].sort_values("stage3_empirical_pka_side_call_confidence")
    reconstruction_failures = sites[
        sites["stage3_macro_reconstruction_abs_error"]
        > float(args.macro_reconstruction_error_threshold)
    ].sort_values("stage3_macro_reconstruction_abs_error", ascending=False).drop_duplicates("molecule_id")
    incomplete_networks = sites[
        ~sites["population_status"].astype(str).str.startswith("complete_")
    ].drop_duplicates("molecule_id")
    pair_changed = sites[
        sites.get(
            "stage3_site_call_changed_by_pair_coupling",
            pd.Series(False, index=sites.index),
        ).fillna(False).astype(bool)
    ].sort_values("stage3_overall_site_state_confidence").drop_duplicates("molecule_id")
    secondary_guesses = heldout_pka_errors[
        heldout_pka_errors["secondary_guess_review_recommended"].fillna(False).astype(bool)
    ].sort_values(
        "secondary_guess_improvement_over_primary_stage1",
        ascending=False,
    )

    categories = {
        "heldout_pka_errors": heldout_pka_errors,
        "secondary_transition_assignment_guesses": secondary_guesses,
        "low_confidence_state_calls_at_ph_7_4": low_confidence,
        "empirical_pka_side_conflicts": empirical_pka_side_conflicts,
        "stage2_free_energy_reconstruction_gaps": reconstruction_failures,
        "pair_coupling_changed_site_calls": pair_changed,
        "incomplete_microstate_networks": incomplete_networks,
    }
    rendered = {}
    for category, frame in categories.items():
        rendered[category] = render_gallery(
            frame,
            category,
            out_dir / f"stage3_{category}.png",
            out_dir / f"stage3_{category}.csv",
            args.max_examples,
        )
    comparison_category = "deployment_view_of_heldout_errors"
    rendered[comparison_category] = render_deployment_comparison_gallery(
        deployment_view,
        out_dir / f"stage3_{comparison_category}.png",
        out_dir / f"stage3_{comparison_category}.csv",
        args.comparison_examples,
    )

    summary = {
        "important_interpretation": (
            "held-out pKa errors are internal scaffold-validation errors; low-confidence "
            "state calls are uncertainty flags, not experimentally proven state failures"
        ),
        "pka_error_threshold": float(args.pka_error_threshold),
        "heldout_pka_error_rows": int(len(heldout_pka_errors)),
        "secondary_transition_guess_method": (
            "post-hoc audit only: among alternative enumerated edges, choose the one "
            "whose blind Stage1 network macro-pKa is closest to experiment; an edge "
            "already occupied by another exact anchor is evaluated as a reciprocal "
            "joint swap, and the primary curated label is never replaced without review"
        ),
        "secondary_transition_guess_min_improvement": float(
            args.secondary_guess_min_improvement
        ),
        "secondary_transition_guess_max_abs_error": float(
            args.secondary_guess_max_abs_error
        ),
        "secondary_transition_review_rows": int(len(secondary_guesses)),
        "secondary_transition_joint_swap_review_rows": int(
            secondary_guesses["secondary_guess_requires_joint_swap"].fillna(False).astype(bool).sum()
        ),
        "deployment_comparison_interpretation": (
            "same scaffold-validation failure cases: left is the representative input "
            "drawing (not an observed protonation state), right is the Stage3 dominant "
            "microstate at the requested pH; deployment differences are fitted-model "
            "diagnostics, not held-out errors"
        ),
        "deployment_rows_with_deployed_pka": int(
            heldout_pka_errors["deployed_assigned_macro_pka"].notna().sum()
        ),
        "deployment_rows_agreeing_with_experimental_pka_implied_form": int((
            heldout_pka_errors["stage3_form_vs_experimental_pka_check"] == "agrees"
        ).sum()),
        "deployment_rows_disagreeing_with_experimental_pka_implied_form": int((
            heldout_pka_errors["stage3_form_vs_experimental_pka_check"] == "disagrees"
        ).sum()),
        "low_confidence_state_call_threshold": float(args.low_confidence_threshold),
        "low_confidence_site_rows": int((
            sites[confidence_column].notna()
            & (sites[confidence_column] < float(args.low_confidence_threshold))
        ).sum()),
        "low_confidence_molecules": int(len(low_confidence)),
        "empirical_pka_side_conflict_rows": int(len(empirical_pka_side_conflicts)),
        "empirical_pka_side_conflict_molecules": int(
            empirical_pka_side_conflicts["molecule_id"].nunique()
        ),
        "empirical_pka_side_conflict_interpretation": (
            "the coupled Stage 3 site call is opposed by the majority of the "
            "scaffold-held-out scalar-pKa residual distribution; review the macro-step "
            "to site mapping and inferred pair closure, not only the raw pKa regressor"
        ),
        "stage2_free_energy_reconstruction_error_threshold": float(
            args.macro_reconstruction_error_threshold
        ),
        "stage2_free_energy_reconstruction_gap_molecules": int(
            len(reconstruction_failures)
        ),
        "pair_coupling_changed_site_call_rows": int(
            sites.get(
                "stage3_site_call_changed_by_pair_coupling",
                pd.Series(False, index=sites.index),
            ).fillna(False).astype(bool).sum()
        ),
        "pair_coupling_changed_site_call_molecules": int(len(pair_changed)),
        "incomplete_microstate_network_molecules": int(len(incomplete_networks)),
        "rendered_examples": rendered,
    }
    (out_dir / "stage3_failure_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Canonical Stage 3 review</title>",
        "<style>body{font-family:sans-serif;margin:2rem;max-width:1900px}img{max-width:100%;border:1px solid #bbb}code{background:#eee;padding:.1rem .3rem}.table-wrap{overflow:auto;max-height:52rem;border:1px solid #bbb}table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{padding:.42rem .55rem;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}th{position:sticky;top:0;background:#eee;z-index:1}th:nth-child(2),td:nth-child(2),th:last-child,td:last-child{text-align:left}tr:nth-child(even){background:#fafafa}</style>",
        "</head><body><h1>Canonical Stage 3 review galleries</h1>",
        f"<p>{summary['important_interpretation']}</p>",
    ]
    display_order = [
        "heldout_pka_errors",
        "secondary_transition_assignment_guesses",
        comparison_category,
        "low_confidence_state_calls_at_ph_7_4",
        "empirical_pka_side_conflicts",
        "stage2_free_energy_reconstruction_gaps",
        "pair_coupling_changed_site_calls",
        "incomplete_microstate_networks",
    ]
    for category in display_order:
        gallery_content = (
            f"<img src='stage3_{category}.png' alt='{category}'>"
            if int(rendered.get(category, 0)) > 0
            else "<p><em>No cases at the current thresholds.</em></p>"
        )
        html.extend([
            f"<h2>{category.replace('_', ' ').title()}</h2>",
            f"<p><a href='stage3_{category}.csv'>CSV details</a></p>",
            (
                _deployment_comparison_table_html(deployment_view)
                if category == comparison_category else ""
            ),
            gallery_content,
        ])
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Gallery index: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
