from typing import Dict, Optional

import numpy as np
import pandas as pd


def _normalized_feature_mode(feature_mode: str) -> str:
    mode = str(feature_mode or "full").strip().lower().replace("-", "_")
    if mode not in {"full", "weak_prior"}:
        raise ValueError(f"Unsupported thermo feature mode: {feature_mode}")
    return mode


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def build_thermo_pair_form_feature_frame(df: pd.DataFrame, feature_mode: str = "full") -> pd.DataFrame:
    feature_mode = _normalized_feature_mode(feature_mode)
    ph_target = _numeric_series(df, "ph_target", default=7.4)
    pred_effective_pka = _numeric_series(df, "pred_effective_pka")
    intrinsic_pred_pka = _numeric_series(df, "intrinsic_pred_pka")
    pred_delta_pka = _numeric_series(df, "pred_delta_pka")
    candidate_prior_mean = _numeric_series(df, "candidate_prior_mean_pka")
    family_prior_mean = _numeric_series(df, "family_prior_mean_pka")
    textbook_reference = _numeric_series(df, "textbook_reference_pka")
    pka_margin = pred_effective_pka - ph_target

    numeric_blocks = {
        "ph_target": ph_target,
        "pred_effective_pka": pred_effective_pka,
        "intrinsic_pred_pka": intrinsic_pred_pka,
        "pred_delta_pka": pred_delta_pka,
        "pred_protonated_fraction": _numeric_series(df, "pred_protonated_fraction", default=0.5),
        "resolved_group_count": _numeric_series(df, "resolved_group_count"),
        "molecule_formal_charge": _numeric_series(df, "molecule_formal_charge"),
        "neutral_input_risk": _numeric_series(df, "neutral_input_risk"),
        "pred_pka_minus_ph": pka_margin,
        "abs_pred_pka_margin": pka_margin.abs(),
        "intrinsic_pka_minus_ph": intrinsic_pred_pka - ph_target,
        "abs_pred_delta_pka": pred_delta_pka.abs(),
    }
    if feature_mode == "full":
        numeric_blocks.update(
            {
                "candidate_prior_mean_pka": candidate_prior_mean,
                "candidate_prior_median_pka": _numeric_series(df, "candidate_prior_median_pka"),
                "candidate_prior_iqr_pka": _numeric_series(df, "candidate_prior_iqr_pka"),
                "candidate_prior_count": _numeric_series(df, "candidate_prior_count"),
                "family_prior_mean_pka": family_prior_mean,
                "family_prior_median_pka": _numeric_series(df, "family_prior_median_pka"),
                "family_prior_count": _numeric_series(df, "family_prior_count"),
                "textbook_reference_pka": textbook_reference,
                "prior_mean_minus_ph": candidate_prior_mean - ph_target,
                "family_prior_mean_minus_ph": family_prior_mean - ph_target,
                "textbook_reference_minus_ph": textbook_reference - ph_target,
            }
        )
    numeric = pd.DataFrame(numeric_blocks, index=df.index)

    ion_cols = sorted(col for col in df.columns if col.startswith("ion_ctx_"))
    if ion_cols:
        ion_ctx = pd.DataFrame(
            {
                col: pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                for col in ion_cols
            },
            index=df.index,
        )
    else:
        ion_ctx = pd.DataFrame(index=df.index)

    family_series = (
        df.get("candidate_family", pd.Series("unknown", index=df.index))
        .fillna("unknown")
        .astype(str)
    )
    family_df = pd.get_dummies(family_series, prefix="family")

    if "pka_type_canonical" in df.columns:
        type_df = pd.get_dummies(df["pka_type_canonical"].fillna("unknown"), prefix="pka_type")
    else:
        type_df = pd.DataFrame(index=df.index)

    family_margin_df = family_df.mul(pka_margin, axis=0)
    family_margin_df.columns = [f"{col}__x_pred_pka_minus_ph" for col in family_df.columns]

    return pd.concat([numeric, ion_ctx, family_df, type_df, family_margin_df], axis=1)


def infer_thermo_pair_form_probability(
    df: pd.DataFrame,
    bundle: Optional[Dict],
) -> pd.Series:
    if bundle is None:
        return pd.Series(np.nan, index=df.index, dtype=float)

    model = bundle.get("model")
    feature_columns = bundle.get("feature_columns")
    if model is None or feature_columns is None:
        return pd.Series(np.nan, index=df.index, dtype=float)

    X = build_thermo_pair_form_feature_frame(df, feature_mode=str(bundle.get("feature_mode", "full")))
    X = X.reindex(columns=feature_columns, fill_value=0.0)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        proba = np.asarray(model.predict(X), dtype=float)
    return pd.Series(proba, index=df.index, dtype=float)


def apply_thermo_pair_form_model(
    df: pd.DataFrame,
    bundle: Optional[Dict],
) -> pd.DataFrame:
    if bundle is None:
        return df

    work = df.copy()
    acid_prob = infer_thermo_pair_form_probability(work, bundle)
    work["thermo_pair_form_prob_acid"] = acid_prob

    threshold = float(bundle.get("decision_threshold", 0.5))
    margin = float(bundle.get("decision_margin", 0.0))

    if "pred_member_form_rule_source" not in work.columns:
        work["pred_member_form_rule_source"] = "baseline_henderson"

    candidate_member_form = work.get("candidate_member_form", pd.Series("", index=work.index)).fillna("")
    mask = (
        candidate_member_form.isin(["acid_form", "base_form"])
        & work["thermo_pair_form_prob_acid"].notna()
        & ((work["thermo_pair_form_prob_acid"] - threshold).abs() >= margin)
    )
    if not mask.any():
        return work

    work.loc[mask, "pred_member_form"] = np.where(
        work.loc[mask, "thermo_pair_form_prob_acid"] >= threshold,
        "acid_form",
        "base_form",
    )
    work.loc[mask, "pred_member_form_rule_source"] = "thermo_pair_form_model"
    return work