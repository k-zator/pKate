import argparse
import json
import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from functional_group_pka_analysis import (
    ACIDIC_FAMILIES,
    BASIC_FAMILIES,
    CONJUGATE_FAMILY_MAP,
    PAIR_TYPE_FAMILY_FORMS,
    training_group_label,
)
from ml_dataset_builder import DatasetBuildConfig, build_candidate_dataset
from ml_features import build_feature_matrix, molecule_descriptors, morgan_bits
from ionization_context_features import build_carboxyl_form_feature_frame, build_ionization_context_frame
from thermo_pair_form_model import apply_thermo_pair_form_model
from train_intrinsic_single_group_model import build_intrinsic_features
from training_data_resolver import (
    DEFAULT_CURATED_ASSIGNMENTS_PATH,
    DEFAULT_CURATED_SUMMARY_PATH,
    DEFAULT_RAW_DIR,
    ensure_curated_training_data,
)


DEFAULT_SCORING_ASSIGNMENTS_PATH = "data/processed/functional_group_assignments_pruned_eval_unpooled_families.csv"
DEFAULT_SCORING_SUMMARY_PATH = "data/processed/functional_group_pka_summary_pruned_eval_unpooled_families.csv"
DEFAULT_BUNDLE_ROOT = "data/processed/ml_models_pruned_eval_unpooled_families_predicted_pka"
DEFAULT_STAGE3_OUTPUT_DIR = os.path.join(DEFAULT_BUNDLE_ROOT, "stage3_protonation")
DEFAULT_STAGE1_MODEL_PATH = os.path.join(DEFAULT_BUNDLE_ROOT, "stage1_intrinsic", "stage1_intrinsic_model.pkl")
DEFAULT_STAGE2_MODEL_PATH = os.path.join(DEFAULT_BUNDLE_ROOT, "stage2_delta", "stage2_delta_model.pkl")
DEFAULT_SITE_MODEL_PATH = os.path.join(DEFAULT_BUNDLE_ROOT, "site_state_baseline", "site_state_scaffold_model.pkl")
DEFAULT_CARBOXYL_FORM_MODEL_PATH = os.path.join(DEFAULT_BUNDLE_ROOT, "carboxyl_form_head", "carboxyl_form_head.pkl")
DEFAULT_PAIR_FORM_HEADS_PATH = os.path.join(DEFAULT_BUNDLE_ROOT, "pair_form_heads", "pair_form_heads.pkl")
DEFAULT_THERMO_PAIR_FORM_MODEL_PATH = os.path.join(
    DEFAULT_BUNDLE_ROOT,
    "thermo_pair_form_model_weak_prior",
    "thermo_pair_form_model.pkl",
)


def _load_bundle(path: str) -> Dict:
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    if isinstance(bundle, dict) and "heads" in bundle:
        return bundle
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle
    return {"model": bundle}


def _normalized_family(label: str) -> str:
    return CONJUGATE_FAMILY_MAP.get(label, label)


def _label_form(label: str) -> Tuple[Optional[str], Optional[str]]:
    family = _normalized_family(label)
    forms = PAIR_TYPE_FAMILY_FORMS.get(family)
    if forms is None:
        return family, None
    if label in forms.get("acid_form", set()):
        return family, "acid_form"
    if label in forms.get("base_form", set()):
        return family, "base_form"
    return family, "unknown_form"


def _build_stage1_like_features(df: pd.DataFrame, feature_columns, fp_bits: int, fp_radius: int) -> pd.DataFrame:
    X = build_intrinsic_features(
        df.reset_index(drop=True),
        nbits=fp_bits,
        radius=fp_radius,
        label_col="candidate_label",
    )
    X = X.reindex(columns=feature_columns, fill_value=0.0)
    return X


def _resolved_group_multihot(series: pd.Series) -> pd.DataFrame:
    labels = sorted({token for value in series for token in str(value).split("|") if token})
    if not labels:
        return pd.DataFrame(index=series.index)
    data = []
    for value in series:
        tokens = set(token for token in str(value).split("|") if token)
        data.append({f"ctx_has_{label}": float(label in tokens) for label in labels})
    return pd.DataFrame(data)


def _build_stage2_features(df: pd.DataFrame, feature_columns, fp_bits: int, fp_radius: int) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    smiles_unique = df["smiles"].dropna().unique()
    smiles_to_desc = {smi: molecule_descriptors(smi) for smi in smiles_unique}
    smiles_to_fp = {smi: morgan_bits(smi, nbits=fp_bits, radius=fp_radius) for smi in smiles_unique}

    desc_df = pd.DataFrame(df["smiles"].map(smiles_to_desc).tolist())
    fp_matrix = np.vstack([smiles_to_fp.get(smi, np.zeros(fp_bits, dtype=np.float32)) for smi in df["smiles"]])
    fp_df = pd.DataFrame(fp_matrix, columns=[f"fp_{i}" for i in range(fp_bits)])

    training_labels = df["candidate_label"].astype(str).map(training_group_label)
    group_df = pd.get_dummies(training_labels, prefix="group")
    type_df = pd.get_dummies(df["pka_type_canonical"].fillna("unknown"), prefix="pka_type")
    mode_df = pd.get_dummies(df.get("group_mode", "unknown").fillna("unknown"), prefix="mode")
    form_df = pd.get_dummies(df.get("candidate_own_form", df.get("pair_member_form", "unknown")).fillna("unknown"), prefix="pair_form")
    resolved_df = _resolved_group_multihot(df["resolved_groups"]) 
    ion_ctx_df = build_ionization_context_frame(
        df,
        resolved_col="resolved_groups",
        candidate_col="candidate_label",
    )

    numeric_df = pd.DataFrame(
        {
            "resolved_group_count": pd.to_numeric(df["resolved_group_count"], errors="coerce").fillna(0.0),
            "intrinsic_pred_pka": pd.to_numeric(df["intrinsic_pred_pka"], errors="coerce").fillna(0.0),
            "molecule_formal_charge": pd.to_numeric(df.get("molecule_formal_charge", 0.0), errors="coerce").fillna(0.0),
            "neutral_input_risk": pd.to_numeric(df.get("neutral_input_risk", 0.0), errors="coerce").fillna(0.0),
        }
    )

    X = pd.concat(
        [numeric_df, ion_ctx_df, desc_df, group_df, type_df, mode_df, form_df, resolved_df, fp_df],
        axis=1,
    )
    X = X.reindex(columns=feature_columns, fill_value=0.0)
    return X


def _infer_site_probability(df: pd.DataFrame, site_model_bundle: Optional[Dict]) -> pd.Series:
    if site_model_bundle is None:
        return pd.Series(np.ones(len(df), dtype=float), index=df.index)

    model = site_model_bundle.get("model")
    feature_columns = site_model_bundle.get("feature_columns")
    if model is None or feature_columns is None:
        return pd.Series(np.ones(len(df), dtype=float), index=df.index)

    include_observed_pka = bool(site_model_bundle.get("include_observed_pka", False))
    X_site, _ = build_feature_matrix(
        df,
        nbits=int(site_model_bundle.get("fp_bits", 512)),
        radius=int(site_model_bundle.get("fp_radius", 2)),
        include_observed_pka=include_observed_pka,
    )
    X_site = X_site.reindex(columns=feature_columns, fill_value=0.0)
    proba = model.predict_proba(X_site)[:, 1]
    return pd.Series(proba, index=df.index)


def _infer_carboxyl_form_probability(df: pd.DataFrame, carboxyl_bundle: Optional[Dict]) -> pd.Series:
    if carboxyl_bundle is None:
        return pd.Series(np.nan, index=df.index)

    model = carboxyl_bundle.get("model")
    feature_columns = carboxyl_bundle.get("feature_columns")
    if model is None or feature_columns is None:
        return pd.Series(np.nan, index=df.index)

    X = build_carboxyl_form_feature_frame(
        df,
        candidate_col=str(carboxyl_bundle.get("candidate_col", "candidate_label")),
        resolved_col=str(carboxyl_bundle.get("resolved_col", "resolved_groups")),
        smiles_col=str(carboxyl_bundle.get("smiles_col", "smiles")),
        pka_type_col=str(carboxyl_bundle.get("pka_type_col", "pka_type_canonical")),
    )
    X = X.reindex(columns=feature_columns, fill_value=0.0)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = np.asarray(model.predict(X), dtype=float)
    return pd.Series(proba, index=df.index)


def _apply_carboxyl_form_head(
    df: pd.DataFrame,
    carboxyl_bundle: Optional[Dict],
    confidence_margin: float = 0.10,
) -> pd.DataFrame:
    if carboxyl_bundle is None:
        return df

    work = df.copy()
    acid_prob = _infer_carboxyl_form_probability(work, carboxyl_bundle)
    work["carboxyl_form_prob_acid"] = acid_prob

    mask = (
        (work["candidate_family"] == "carboxyl")
        & work["candidate_member_form"].isin(["acid_form", "base_form"])
        & work["carboxyl_form_prob_acid"].notna()
        & ((work["carboxyl_form_prob_acid"] - 0.5).abs() >= float(confidence_margin))
    )

    if "pred_member_form_rule_source" not in work.columns:
        work["pred_member_form_rule_source"] = "baseline_henderson"

    work.loc[mask, "pred_member_form"] = np.where(
        work.loc[mask, "carboxyl_form_prob_acid"] >= 0.5,
        "acid_form",
        "base_form",
    )
    work.loc[mask, "pred_member_form_rule_source"] = "carboxyl_form_head"
    return work


def _apply_pair_form_heads(
    df: pd.DataFrame,
    heads_bundle: Optional[Dict],
    confidence_margin: float = 0.10,
) -> pd.DataFrame:
    if heads_bundle is None:
        return df

    heads = heads_bundle.get("heads")
    if not isinstance(heads, dict) or not heads:
        return df

    work = df.copy()
    if "pred_member_form_rule_source" not in work.columns:
        work["pred_member_form_rule_source"] = "baseline_henderson"

    for head_name, head in heads.items():
        model = head.get("model")
        feature_columns = head.get("feature_columns")
        if model is None or feature_columns is None:
            continue

        acid_labels = set(head.get("acid_labels", []))
        base_labels = set(head.get("base_labels", []))
        all_labels = acid_labels | base_labels
        if not all_labels:
            continue

        head_threshold = float(head.get("decision_threshold", 0.5))
        head_margin = float(head.get("decision_margin", confidence_margin))
        is_carboxyl_head = head_name == "carboxyl"

        mask = work["candidate_label"].isin(all_labels)
        if not mask.any():
            continue

        sub = work.loc[mask].copy()
        X = build_carboxyl_form_feature_frame(
            sub,
            candidate_col=str(head.get("candidate_col", "candidate_label")),
            resolved_col=str(head.get("resolved_col", "resolved_groups")),
            smiles_col=str(head.get("smiles_col", "smiles")),
            pka_type_col=str(head.get("pka_type_col", "pka_type_canonical")),
            include_pka_type=bool(head.get("include_pka_type", False)),
        )
        X = X.reindex(columns=feature_columns, fill_value=0.0)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        else:
            proba = np.asarray(model.predict(X), dtype=float)

        prob_col = f"pair_head_prob_acid__{head_name}"
        work.loc[mask, prob_col] = proba

        if is_carboxyl_head:
            pred_pka = pd.to_numeric(sub.get("pred_effective_pka", np.nan), errors="coerce").fillna(np.nan).to_numpy()
            ph = pd.to_numeric(sub.get("ph_target", np.nan), errors="coerce").fillna(np.nan).to_numpy()
            if np.isnan(ph).all():
                ph = np.full(len(sub), 7.4, dtype=float)

            strong_acid_threshold = max(0.80, head_threshold + 0.20)
            strong_base_threshold = min(0.25, head_threshold - 0.10)
            acid_context_gate = pred_pka >= (ph + 0.75)

            acid_mask = (proba >= strong_acid_threshold) & acid_context_gate
            base_mask = proba <= strong_base_threshold

            if np.any(acid_mask):
                acid_idx = sub.index.to_numpy()[acid_mask]
                work.loc[acid_idx, "pred_member_form"] = "acid_form"
                work.loc[acid_idx, "pred_member_form_rule_source"] = "pair_form_head__carboxyl_conservative"

            if np.any(base_mask):
                base_idx = sub.index.to_numpy()[base_mask]
                work.loc[base_idx, "pred_member_form"] = "base_form"
                work.loc[base_idx, "pred_member_form_rule_source"] = "pair_form_head__carboxyl_conservative"
        elif head_name in {"pyridine_like", "pyrimidine_like"}:
            pred_pka = pd.to_numeric(sub.get("pred_effective_pka", np.nan), errors="coerce").fillna(np.nan).to_numpy()
            ph = pd.to_numeric(sub.get("ph_target", np.nan), errors="coerce").fillna(np.nan).to_numpy()
            if np.isnan(ph).all():
                ph = np.full(len(sub), 7.4, dtype=float)

            strong_acid_threshold = max(0.75, head_threshold + 0.15)
            strong_base_threshold = min(0.35, head_threshold - 0.05)
            acid_context_gate = pred_pka >= (ph + 0.75)

            acid_mask = (proba >= strong_acid_threshold) & acid_context_gate
            base_mask = proba <= strong_base_threshold

            if np.any(acid_mask):
                acid_idx = sub.index.to_numpy()[acid_mask]
                work.loc[acid_idx, "pred_member_form"] = "acid_form"
                work.loc[acid_idx, "pred_member_form_rule_source"] = f"pair_form_head__{head_name}_conservative"

            if np.any(base_mask):
                base_idx = sub.index.to_numpy()[base_mask]
                work.loc[base_idx, "pred_member_form"] = "base_form"
                work.loc[base_idx, "pred_member_form_rule_source"] = f"pair_form_head__{head_name}_conservative"
        else:
            # Generic pair-form heads: apply Henderson–Hasselbalch
            # consistency gate so the head cannot override clear-cut
            # thermodynamic predictions.
            pred_pka = pd.to_numeric(sub.get("pred_effective_pka", np.nan), errors="coerce").fillna(np.nan).to_numpy()
            ph = pd.to_numeric(sub.get("ph_target", np.nan), errors="coerce").fillna(np.nan).to_numpy()
            if np.isnan(ph).all():
                ph = np.full(len(sub), 7.4, dtype=float)

            # For the HH gate, acid_form is allowed only when pred_pka
            # is above pH (protonated dominates).  We use a 0.75-unit
            # margin so the head can only flip the prediction when the
            # thermodynamic signal is weak.
            acid_context_gate = pred_pka >= (ph + 0.75)
            base_context_gate = pred_pka <= (ph - 0.75)

            confident = np.abs(proba - head_threshold) >= float(head_margin)
            ml_acid = confident & (proba >= head_threshold)
            ml_base = confident & (proba < head_threshold)

            # Apply HH gate: ML acid prediction only allowed when pKa
            # supports protonation; otherwise fall through to baseline.
            acid_mask = ml_acid & acid_context_gate
            base_mask = ml_base & base_context_gate

            # Also accept the ML prediction when the gate is inconclusive
            # (pred_pka is within the margin around pH).
            inconclusive_gate = ~acid_context_gate & ~base_context_gate
            acid_mask |= ml_acid & inconclusive_gate
            base_mask |= ml_base & inconclusive_gate

            if np.any(acid_mask):
                acid_idx = sub.index.to_numpy()[acid_mask]
                work.loc[acid_idx, "pred_member_form"] = "acid_form"
                work.loc[acid_idx, "pred_member_form_rule_source"] = f"pair_form_head__{head_name}_hh_gated"

            if np.any(base_mask):
                base_idx = sub.index.to_numpy()[base_mask]
                work.loc[base_idx, "pred_member_form"] = "base_form"
                work.loc[base_idx, "pred_member_form_rule_source"] = f"pair_form_head__{head_name}_hh_gated"

    return work


def _apply_pair_form_layer(
    df: pd.DataFrame,
    pair_form_mode: str,
    thermo_bundle: Optional[Dict] = None,
    carboxyl_bundle: Optional[Dict] = None,
    pair_heads_bundle: Optional[Dict] = None,
) -> pd.DataFrame:
    mode = str(pair_form_mode or "thermo").strip().lower()
    if mode == "thermo":
        return apply_thermo_pair_form_model(df, thermo_bundle)
    if mode == "legacy":
        work = _apply_carboxyl_form_head(df, carboxyl_bundle=carboxyl_bundle)
        return _apply_pair_form_heads(work, heads_bundle=pair_heads_bundle)
    if mode == "none":
        return df
    raise ValueError(f"Unsupported pair-form mode: {pair_form_mode}")


def _normalized_site_ranking_mode(site_ranking_mode: str) -> str:
    mode = str(site_ranking_mode or "joint").strip().lower().replace("-", "_")
    if mode not in {"joint", "site_only", "site_then_state"}:
        raise ValueError(f"Unsupported site ranking mode: {site_ranking_mode}")
    return mode


def _apply_selection_score(df: pd.DataFrame, site_ranking_mode: str) -> pd.DataFrame:
    mode = _normalized_site_ranking_mode(site_ranking_mode)
    work = df.copy()
    site_prob = pd.to_numeric(work.get("site_prob", 0.0), errors="coerce").fillna(0.0)
    member_presence_prob = pd.to_numeric(work.get("member_presence_prob", 0.0), errors="coerce").fillna(0.0)

    work["joint_score"] = site_prob * member_presence_prob
    if mode == "site_only":
        work["selection_primary_score"] = site_prob
        work["selection_secondary_score"] = 0.0
    elif mode == "site_then_state":
        rounded_site_prob = site_prob.round(3)
        work["selection_primary_score"] = rounded_site_prob
        work["selection_secondary_score"] = member_presence_prob
    else:
        work["selection_primary_score"] = work["joint_score"]
        work["selection_secondary_score"] = 0.0

    work["selection_score"] = (
        pd.to_numeric(work["selection_primary_score"], errors="coerce").fillna(0.0)
        + (1e-6 * pd.to_numeric(work["selection_secondary_score"], errors="coerce").fillna(0.0))
    )
    work["combined_score"] = work["selection_score"]
    return work


def _protonated_fraction(pka_value: np.ndarray, pH: float) -> np.ndarray:
    return 1.0 / (1.0 + np.power(10.0, pH - pka_value))


def _add_selection_confidence(df: pd.DataFrame, pH: float) -> pd.DataFrame:
    work = df.copy()
    work["rank_in_molecule"] = (
        work.groupby("molecule_key")["combined_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    grouped = work.groupby("molecule_key")["combined_score"]
    top_score = grouped.transform("max")
    second_score_map = (
        work.sort_values(["molecule_key", "combined_score"], ascending=[True, False])
        .groupby("molecule_key")["combined_score"]
        .nth(1)
        .to_dict()
    )
    second_score = work["molecule_key"].map(second_score_map).fillna(0.0)

    work["top_score_in_molecule"] = top_score
    work["second_score_in_molecule"] = second_score
    work["score_gap_to_second"] = (top_score - second_score).clip(lower=0.0)
    work["score_gap_ratio"] = work["score_gap_to_second"] / (top_score.abs() + 1e-9)

    protonation_margin = (work["pred_protonated_fraction"] - 0.5).abs() * 2.0
    base_conf = (0.5 * work["score_gap_ratio"].clip(0.0, 1.0)) + (0.5 * protonation_margin.clip(0.0, 1.0))
    work["selection_confidence"] = (base_conf / work["rank_in_molecule"]).clip(0.0, 1.0)

    ph_distance = (work["pred_effective_pka"] - float(pH)).abs()
    distance_term = ph_distance / (ph_distance + 1.0)
    work["pka_eff_confidence"] = (work["selection_confidence"] * distance_term).clip(0.0, 1.0)

    half_width = 0.35 + (1.96 * (1.0 - work["pka_eff_confidence"]))
    work["pred_effective_pka_ci_low"] = work["pred_effective_pka"] - half_width
    work["pred_effective_pka_ci_high"] = work["pred_effective_pka"] + half_width
    return work


def _apply_uncertainty_ph_fallback(
    df: pd.DataFrame,
    pH: float,
    uncertain_threshold: float,
) -> pd.DataFrame:
    work = df.copy()
    work["pred_member_form_baseline"] = work["pred_member_form"]
    if "pred_member_form_rule_source" not in work.columns:
        work["pred_member_form_rule_source"] = "baseline_henderson"
    else:
        work["pred_member_form_rule_source"] = work["pred_member_form_rule_source"].fillna("baseline_henderson")

    work["candidate_family_type"] = np.where(
        work["candidate_family"].isin(list(ACIDIC_FAMILIES)),
        "acidic",
        np.where(work["candidate_family"].isin(list(BASIC_FAMILIES)), "basic", "unknown"),
    )

    is_pair_candidate = work["candidate_member_form"].isin(["acid_form", "base_form"])
    uncertain = work["selection_confidence"] < float(uncertain_threshold)
    fallback_mask = is_pair_candidate & uncertain

    acidic_fallback = fallback_mask & (work["candidate_family_type"] == "acidic")
    basic_fallback = fallback_mask & (work["candidate_family_type"] == "basic")

    carboxyl_fallback = acidic_fallback & (work["candidate_family"] == "carboxyl")
    generic_acidic_fallback = acidic_fallback & ~carboxyl_fallback

    if carboxyl_fallback.any():
        pos_n_within_3 = pd.to_numeric(work.get("ion_ctx_local_pos_n_within_3", 0.0), errors="coerce").fillna(0.0)
        amide_within_4 = pd.to_numeric(work.get("ion_ctx_local_amide_within_4", 0.0), errors="coerce").fillna(0.0)
        min_dist_pos_n = pd.to_numeric(work.get("ion_ctx_local_min_dist_pos_n", 99.0), errors="coerce").fillna(99.0)
        no_basic_context = (pd.to_numeric(work.get("ion_ctx_basic_groups", 0.0), errors="coerce").fillna(0.0) <= 0.0).astype(float)

        adjusted_ph = (
            float(pH)
            - 0.35
            - (0.20 * pos_n_within_3)
            - (0.02 * amide_within_4)
            - (0.01 * (5.0 - min_dist_pos_n.clip(upper=5.0)).clip(lower=0.0))
            + (0.15 * no_basic_context)
        )
        work["carboxyl_adjusted_ph_threshold"] = adjusted_ph

        work.loc[carboxyl_fallback, "pred_member_form"] = np.where(
            work.loc[carboxyl_fallback, "pred_effective_pka"] < adjusted_ph.loc[carboxyl_fallback],
            "base_form",
            "acid_form",
        )
        work.loc[carboxyl_fallback, "pred_member_form_rule_source"] = "uncertainty_ph_fallback_carboxyl_local"

    work.loc[generic_acidic_fallback, "pred_member_form"] = np.where(
        work.loc[generic_acidic_fallback, "pred_effective_pka"] < float(pH),
        "base_form",
        "acid_form",
    )

    # For basic families (amines, pyridines, …) the predicted pKa is
    # pKaH — the deprotonation equilibrium of the conjugate acid.
    # Henderson–Hasselbalch: pKaH > pH → acid form dominates (protonated);
    #                        pKaH < pH → base form dominates (deprotonated).
    work.loc[basic_fallback, "pred_member_form"] = np.where(
        work.loc[basic_fallback, "pred_effective_pka"] > float(pH),
        "acid_form",
        "base_form",
    )

    work.loc[(generic_acidic_fallback | basic_fallback), "pred_member_form_rule_source"] = "uncertainty_ph_fallback"
    return work


def _molecule_level_summary(df: pd.DataFrame) -> Dict[str, float]:
    grouped = df.sort_values("combined_score", ascending=False).groupby("molecule_key", as_index=False).first()
    if grouped.empty:
        return {
            "molecules": 0,
            "site_top1_acc": 0.0,
            "pair_form_acc": 0.0,
        }

    site_acc = float((grouped["candidate_label"] == grouped["true_label"]).mean())

    pair_df = grouped[grouped["true_pair_member_form"].isin(["acid_form", "base_form"])].copy()
    pair_acc = float((pair_df["pred_member_form"] == pair_df["true_pair_member_form"]).mean()) if not pair_df.empty else float("nan")
    mean_conf = float(grouped["selection_confidence"].mean()) if "selection_confidence" in grouped.columns else float("nan")
    return {
        "molecules": int(grouped["molecule_key"].nunique()),
        "site_top1_acc": site_acc,
        "pair_form_acc": pair_acc,
        "mean_top1_selection_confidence": mean_conf,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: infer protonation state/site at a target pH.")
    parser.add_argument("--assignments", default=DEFAULT_SCORING_ASSIGNMENTS_PATH)
    parser.add_argument("--summary", default=DEFAULT_SCORING_SUMMARY_PATH)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--stage1-model", default=DEFAULT_STAGE1_MODEL_PATH)
    parser.add_argument("--stage2-model", default=DEFAULT_STAGE2_MODEL_PATH)
    parser.add_argument("--site-model", default=DEFAULT_SITE_MODEL_PATH)
    parser.add_argument("--pair-form-mode", choices=["thermo", "legacy", "none"], default="thermo")
    parser.add_argument("--site-ranking-mode", choices=["joint", "site_only", "site_then_state"], default="site_only")
    parser.add_argument("--thermo-pair-form-model", default=DEFAULT_THERMO_PAIR_FORM_MODEL_PATH)
    parser.add_argument("--carboxyl-form-model", default=DEFAULT_CARBOXYL_FORM_MODEL_PATH)
    parser.add_argument("--pair-form-heads", default=DEFAULT_PAIR_FORM_HEADS_PATH)
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--out-csv", default=os.path.join(DEFAULT_STAGE3_OUTPUT_DIR, "stage3_candidate_scores.csv"))
    parser.add_argument("--out-json", default=os.path.join(DEFAULT_STAGE3_OUTPUT_DIR, "stage3_metrics.json"))
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument(
        "--uncertain-threshold",
        type=float,
        default=0.55,
        help="If selection_confidence is below this threshold, apply pKa-vs-pH fallback rules.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Preserve compatibility with programmatic callers that construct the
    # pre-thermodynamic-layer argparse Namespace used by older releases.
    pair_form_mode = str(getattr(args, "pair_form_mode", "legacy"))
    site_ranking_mode = str(getattr(args, "site_ranking_mode", "site_only"))
    thermo_pair_form_model = getattr(
        args, "thermo_pair_form_model", DEFAULT_THERMO_PAIR_FORM_MODEL_PATH
    )
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    assignments_path, summary_path = ensure_curated_training_data(
        assignments_path=args.assignments,
        summary_path=args.summary,
        raw_dir=getattr(args, "raw_dir", DEFAULT_RAW_DIR),
    )
    candidate_df = build_candidate_dataset(
        DatasetBuildConfig(assignments_path=assignments_path, summary_path=summary_path)
    )
    if args.max_rows > 0:
        candidate_df = candidate_df.head(args.max_rows).copy()
    if candidate_df.empty:
        raise RuntimeError("Candidate dataset is empty for Stage 3 inference.")
    candidate_df["ph_target"] = float(args.ph)

    stage1 = _load_bundle(args.stage1_model)
    stage2 = _load_bundle(args.stage2_model)
    site_bundle = _load_bundle(args.site_model) if args.site_model else None
    thermo_bundle = None
    carboxyl_bundle = None
    pair_heads_bundle = None
    if pair_form_mode == "thermo":
        thermo_bundle = _load_bundle(thermo_pair_form_model) if thermo_pair_form_model else None
    elif pair_form_mode == "legacy":
        carboxyl_bundle = _load_bundle(args.carboxyl_form_model) if args.carboxyl_form_model else None
        pair_heads_bundle = _load_bundle(args.pair_form_heads) if args.pair_form_heads else None

    X_stage1 = _build_stage1_like_features(
        candidate_df,
        feature_columns=stage1.get("feature_columns"),
        fp_bits=int(stage1.get("fp_bits", 512)),
        fp_radius=int(stage1.get("fp_radius", 2)),
    )
    candidate_df["intrinsic_pred_pka"] = stage1["model"].predict(X_stage1)

    # Stage 2 delta is only meaningful for multi-group molecules.
    # For single-group molecules the delta is forced to 0.
    is_multi = pd.to_numeric(candidate_df["resolved_group_count"], errors="coerce").fillna(1) > 1
    candidate_df["pred_delta_pka"] = 0.0
    if is_multi.any():
        X_stage2 = _build_stage2_features(
            candidate_df.loc[is_multi],
            feature_columns=stage2.get("feature_columns"),
            fp_bits=int(stage2.get("fp_bits", 512)),
            fp_radius=int(stage2.get("fp_radius", 2)),
        )
        candidate_df.loc[is_multi, "pred_delta_pka"] = stage2["model"].predict(X_stage2)

    candidate_df["pred_effective_pka"] = candidate_df["intrinsic_pred_pka"] + candidate_df["pred_delta_pka"]

    family_form = candidate_df["candidate_label"].map(_label_form)
    candidate_df["candidate_family"] = family_form.map(lambda x: x[0])
    candidate_df["candidate_member_form"] = family_form.map(lambda x: x[1] if x[1] is not None else "non_pair")

    frac_prot = _protonated_fraction(candidate_df["pred_effective_pka"].to_numpy(dtype=float), pH=float(args.ph))
    candidate_df["pred_protonated_fraction"] = frac_prot
    candidate_df["pred_is_deprotonated"] = candidate_df["pred_effective_pka"] < float(args.ph)
    candidate_df["pred_is_protonated"] = ~candidate_df["pred_is_deprotonated"]

    candidate_df["pred_member_form"] = np.where(
        candidate_df["candidate_member_form"] == "acid_form",
        np.where(candidate_df["pred_is_protonated"], "acid_form", "base_form"),
        np.where(
            candidate_df["candidate_member_form"] == "base_form",
            np.where(candidate_df["pred_is_protonated"], "acid_form", "base_form"),
            candidate_df["candidate_member_form"],
        ),
    )

    candidate_df["member_presence_prob"] = np.where(
        candidate_df["candidate_member_form"] == "acid_form",
        candidate_df["pred_protonated_fraction"],
        np.where(
            candidate_df["candidate_member_form"] == "base_form",
            1.0 - candidate_df["pred_protonated_fraction"],
            0.5,
        ),
    )

    candidate_df["site_prob"] = _infer_site_probability(candidate_df, site_bundle)
    candidate_df = _apply_selection_score(candidate_df, site_ranking_mode=site_ranking_mode)
    candidate_df = _add_selection_confidence(candidate_df, pH=float(args.ph))
    candidate_df = pd.concat(
        [
            candidate_df,
            build_ionization_context_frame(
                candidate_df,
                resolved_col="resolved_groups",
                candidate_col="candidate_label",
                smiles_col="smiles",
            ),
        ],
        axis=1,
    )
    candidate_df = _apply_uncertainty_ph_fallback(
        candidate_df,
        pH=float(args.ph),
        uncertain_threshold=float(args.uncertain_threshold),
    )
    candidate_df = _apply_pair_form_layer(
        candidate_df,
        pair_form_mode=pair_form_mode,
        thermo_bundle=thermo_bundle,
        carboxyl_bundle=carboxyl_bundle,
        pair_heads_bundle=pair_heads_bundle,
    )

    inferred_true_form = candidate_df["true_label"].map(
        lambda label: (_label_form(str(label))[1] if pd.notna(label) else None)
    )
    candidate_df["true_pair_member_form"] = inferred_true_form

    # Fallback for labels without explicit pair forms (or unexpected blanks):
    # recover from true-site rows keyed by (molecule, true_label) without cross-label bleed.
    missing_mask = ~candidate_df["true_pair_member_form"].isin(["acid_form", "base_form"])
    if missing_mask.any():
        true_form_df = candidate_df[candidate_df["is_true_site"] == 1][
            ["molecule_key", "true_label", "pair_member_form"]
        ].copy()
        if not true_form_df.empty:
            true_form_df["_pair_rank"] = true_form_df["pair_member_form"].map(
                {"acid_form": 2, "base_form": 1}
            ).fillna(0)
            true_form_df = (
                true_form_df.sort_values(["molecule_key", "true_label", "_pair_rank"], ascending=[True, True, False])
                .drop_duplicates(subset=["molecule_key", "true_label"], keep="first")
            )
            true_form_map = true_form_df.set_index(["molecule_key", "true_label"])["pair_member_form"].to_dict()
            candidate_df.loc[missing_mask, "true_pair_member_form"] = [
                true_form_map.get((mk, tl))
                for mk, tl in zip(
                    candidate_df.loc[missing_mask, "molecule_key"],
                    candidate_df.loc[missing_mask, "true_label"],
                )
            ]

    candidate_df.to_csv(args.out_csv, index=False)

    metrics = _molecule_level_summary(candidate_df)
    metrics.update(
        {
            "rows": int(len(candidate_df)),
            "ph": float(args.ph),
            "uncertain_threshold": float(args.uncertain_threshold),
            "pair_form_mode": pair_form_mode,
            "site_ranking_mode": site_ranking_mode,
            "used_site_model": bool(site_bundle is not None and "feature_columns" in site_bundle),
            "used_thermo_pair_form_model": bool(thermo_bundle is not None and "feature_columns" in thermo_bundle),
            "thermo_feature_mode": thermo_bundle.get("feature_mode") if thermo_bundle is not None else None,
            "used_carboxyl_form_model": bool(carboxyl_bundle is not None and "feature_columns" in carboxyl_bundle),
            "used_pair_form_heads": bool(pair_heads_bundle is not None and pair_heads_bundle.get("heads")),
        }
    )
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Saved Stage 3 candidate scores: {args.out_csv}")
    print(f"Saved Stage 3 metrics: {args.out_json}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
