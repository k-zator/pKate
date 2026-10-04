import argparse
import glob
import json
import os
import shutil
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from rdkit import Chem  # type: ignore

from functional_group_pka_analysis import (
    BASIC_FAMILIES,
    PAIR_TYPE_FAMILY_FORMS,
    assignment_status,
    build_functional_group_table,
    build_summary_table,
    canonical_family_label,
    classify_pair_type,
    conjugate_check_flag,
    distribution_sanity_check,
    expand_pair_site_transitions,
    normalize_conjugate_family,
    refine_group_label,
    reinterpret_pka_type_for_two_faced,
    training_group_label,
    training_group_reason,
)
from ml_dataset_builder import DatasetBuildConfig, build_candidate_dataset
from substructure_match import assign_single_group_label, find_sites_with_metadata, resolve_overlapping_sites
from training_data_resolver import ensure_curated_training_data


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_EXTERNAL_ROOT = os.path.join(REPO_ROOT, "data", "raw", "external_datasets")
DEFAULT_OUTPUT_ROOT = os.path.join(REPO_ROOT, "data", "processed", "external_datasets")

DEFAULT_CZODROWSKI_DATASET_DIR = os.path.join(
    DEFAULT_EXTERNAL_ROOT,
    "Machine-learning-meets-pKa",
    "datasets",
)
DEFAULT_SAMPL7_ROOT = os.path.join(DEFAULT_EXTERNAL_ROOT, "SAMPL7")
DEFAULT_SAMPL8_ROOT = os.path.join(DEFAULT_EXTERNAL_ROOT, "SAMPL8")

EMPTY_SUMMARY_COLUMNS = ["final_group_label", "count", "mean", "median", "std", "min", "max", "q1", "q3", "iqr"]
EMPTY_CANDIDATE_COLUMNS = [
    "molecule_key",
    "source_file",
    "record_index",
    "smiles",
    "pka_value",
    "pka_type_canonical",
    "resolved_group_count",
    "resolved_groups",
    "resolved_group_instances",
    "assignment_status",
    "candidate_label",
    "candidate_training_group_label",
    "candidate_instance_id",
    "true_label",
    "is_true_site",
    "final_group_family",
    "pair_member_form",
    "candidate_own_form",
    "candidate_family",
    "group_mode",
    "distribution_ok",
    "molecule_formal_charge",
    "neutral_input_risk",
    "candidate_global_count",
    "candidate_uniqueness",
    "candidate_prior_mean_pka",
    "candidate_prior_median_pka",
    "candidate_prior_iqr_pka",
    "candidate_prior_count",
    "family_prior_mean_pka",
    "family_prior_median_pka",
    "family_prior_count",
    "textbook_reference_pka",
]
EMPTY_BENCHMARK_ASSIGNMENT_COLUMNS = [
    "source_file",
    "record_index",
    "smiles",
    "pka_value",
    "pka_source_method",
    "pka_type_raw",
    "pka_type_canonical",
    "pka_type_effective",
    "two_faced_type_flip",
    "atom_index_raw",
    "atom_index_mode",
    "atom_index_matched",
    "atom_site_match",
    "atom_candidate_count",
    "atom_candidate_labels",
    "site_identifier_raw",
    "all_candidate_groups",
    "resolved_groups",
    "resolved_group_instances",
    "resolved_group_instances_indexed",
    "resolved_group_label_counts",
    "resolved_group_types",
    "resolved_group_count",
    "rejected_group_count",
    "assignment_status",
    "selected_group_label",
    "selected_group_family",
    "final_group_label",
    "final_group_refined",
    "final_group_summary_label",
    "group_mode",
    "pair_family",
    "pair_member_form",
    "pair_assignment_confidence",
    "pair_candidate_count",
    "pair_candidate_labels",
    "pair_selection_reason",
    "selected_site_priority",
    "selected_site_specificity",
    "molecule_formal_charge",
    "neutral_input_risk",
    "distribution_ok",
    "distribution_reason",
    "final_group_family",
    "final_group_canonical",
    "conjugate_check_flag",
    "single_group_strict",
    "training_group_label",
    "training_group_reason",
    "benchmark_dataset",
    "benchmark_compound_id",
    "benchmark_observation_index",
    "benchmark_reference_microstate_id",
    "benchmark_truth_status",
    "benchmark_pka_sem",
]

CZODROWSKI_LOCAL_DUPLICATES = {
    "novartis_cleaned_mono_unique_notraindata.sdf": "novartis_testdata.sdf",
    "avlilumove_cleaned_mono_unique_notraindata.sdf": "AvLiLuMoVe_testdata.sdf",
}


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _formal_charge(mol: Optional[Chem.Mol]) -> int:
    if mol is None:
        return 0
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def _copy_curated_summary(out_path: str) -> str:
    _, curated_summary_path = ensure_curated_training_data()
    _ensure_dir(os.path.dirname(out_path))
    shutil.copyfile(curated_summary_path, out_path)
    return out_path


def _strict_summary_frame(assignments_df: pd.DataFrame) -> pd.DataFrame:
    if assignments_df.empty:
        return pd.DataFrame(columns=EMPTY_SUMMARY_COLUMNS)

    strict_df = assignments_df[
        assignments_df["single_group_strict"]
        & assignments_df["final_group_summary_label"].notna()
        & assignments_df["distribution_ok"]
    ].copy()
    if strict_df.empty:
        return pd.DataFrame(columns=EMPTY_SUMMARY_COLUMNS)

    strict_df["final_group_label"] = strict_df["training_group_label"]
    return build_summary_table(strict_df)


def _write_candidate_tables(assignments_path: str, summary_self_path: str, out_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    assignments_df = pd.read_csv(assignments_path)
    if assignments_df.empty:
        candidates_self = pd.DataFrame(columns=EMPTY_CANDIDATE_COLUMNS)
        candidates_self.to_csv(os.path.join(out_dir, "candidates_self_priors.csv"), index=False)

        curated_summary_copy_path = _copy_curated_summary(os.path.join(out_dir, "summary_curated_priors.csv"))
        candidates_curated = pd.DataFrame(columns=EMPTY_CANDIDATE_COLUMNS)
        candidates_curated.to_csv(os.path.join(out_dir, "candidates_curated_priors.csv"), index=False)
        return candidates_self, candidates_curated, curated_summary_copy_path

    candidates_self = build_candidate_dataset(
        DatasetBuildConfig(assignments_path=assignments_path, summary_path=summary_self_path)
    )
    if candidates_self.empty and not len(candidates_self.columns):
        candidates_self = pd.DataFrame(columns=EMPTY_CANDIDATE_COLUMNS)
    candidates_self.to_csv(os.path.join(out_dir, "candidates_self_priors.csv"), index=False)

    curated_summary_copy_path = _copy_curated_summary(os.path.join(out_dir, "summary_curated_priors.csv"))
    candidates_curated = build_candidate_dataset(
        DatasetBuildConfig(assignments_path=assignments_path, summary_path=curated_summary_copy_path)
    )
    if candidates_curated.empty and not len(candidates_curated.columns):
        candidates_curated = pd.DataFrame(columns=EMPTY_CANDIDATE_COLUMNS)
    candidates_curated.to_csv(os.path.join(out_dir, "candidates_curated_priors.csv"), index=False)
    return candidates_self, candidates_curated, curated_summary_copy_path


def _top_label_counts(series: pd.Series, n: int = 12) -> str:
    if series.empty:
        return "{}"
    counts = series.value_counts().head(n).to_dict()
    return json.dumps(counts, sort_keys=True)


def _safe_rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(pd.to_numeric(series, errors="coerce").fillna(0.0).mean())


def _czodrowski_role(file_name: str) -> str:
    lower_name = file_name.lower()
    if lower_name in CZODROWSKI_LOCAL_DUPLICATES:
        return "duplicate_local_curated"
    if "notraindata" in lower_name:
        return "external_test"
    if "combined_training" in lower_name:
        return "training_bundle"
    if "chembl25" in lower_name or "datawarrior" in lower_name or "avlilumove.sdf" in lower_name:
        return "source_training"
    return "other"


def _czodrowski_usage_metadata(file_name: str) -> Dict[str, object]:
    lower_name = file_name.lower()
    duplicate_local_source = CZODROWSKI_LOCAL_DUPLICATES.get(lower_name, "")
    dataset_role = _czodrowski_role(file_name)

    if duplicate_local_source:
        return {
            "dataset_role": dataset_role,
            "recommended_use": "do_not_use",
            "recommended_use_reason": "duplicate_of_local_curated_sdf",
            "duplicate_local_source": duplicate_local_source,
            "include_in_combined_recommended": False,
        }

    if dataset_role == "source_training":
        return {
            "dataset_role": dataset_role,
            "recommended_use": "use_with_caution",
            "recommended_use_reason": "macro_pka_without_ground_truth_site_assignment",
            "duplicate_local_source": "",
            "include_in_combined_recommended": True,
        }

    if dataset_role == "external_test":
        return {
            "dataset_role": dataset_role,
            "recommended_use": "external_eval_candidate",
            "recommended_use_reason": "held_out_source_without_curated_local_duplicate",
            "duplicate_local_source": "",
            "include_in_combined_recommended": True,
        }

    if dataset_role == "training_bundle":
        return {
            "dataset_role": dataset_role,
            "recommended_use": "audit_only",
            "recommended_use_reason": "prepooled_bundle_not_independent_raw_source",
            "duplicate_local_source": "",
            "include_in_combined_recommended": False,
        }

    return {
        "dataset_role": dataset_role,
        "recommended_use": "review_first",
        "recommended_use_reason": "unclassified_source",
        "duplicate_local_source": "",
        "include_in_combined_recommended": False,
    }


def _dataset_manifest_row(
    dataset_name: str,
    dataset_role: str,
    assignments_df: pd.DataFrame,
    candidates_self: pd.DataFrame,
    candidates_curated: pd.DataFrame,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "dataset_name": dataset_name,
        "dataset_role": dataset_role,
        "assignment_rows": int(len(assignments_df)),
        "molecules": int(assignments_df["smiles"].nunique()) if not assignments_df.empty else 0,
        "candidate_rows_self_priors": int(len(candidates_self)),
        "candidate_rows_curated_priors": int(len(candidates_curated)),
        "single_group_rows": int((assignments_df["assignment_status"] == "single_group").sum()) if not assignments_df.empty else 0,
        "multi_group_rows": int((assignments_df["assignment_status"] == "multi_group").sum()) if not assignments_df.empty else 0,
        "distribution_ok_rate": _safe_rate(assignments_df.get("distribution_ok", pd.Series(dtype=float))),
        "atom_site_match_rate": _safe_rate(assignments_df.get("atom_site_match", pd.Series(dtype=float))),
        "pair_type_rate": _safe_rate(assignments_df.get("group_mode", pd.Series(dtype=object)).eq("pair_type")),
        "top_final_group_labels_json": _top_label_counts(assignments_df.get("final_group_label", pd.Series(dtype=object))),
        "pka_source_counts_json": _top_label_counts(assignments_df.get("pka_source_method", pd.Series(dtype=object))),
    }
    if extra:
        row.update(extra)
    return row


def prepare_czodrowski_datasets(
    dataset_dir: str = DEFAULT_CZODROWSKI_DATASET_DIR,
    out_root: str = os.path.join(DEFAULT_OUTPUT_ROOT, "czodrowski"),
    include_files: Optional[Sequence[str]] = None,
    overlap_threshold: float = 0.75,
) -> pd.DataFrame:
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Czodrowski dataset directory not found: {dataset_dir}")

    _ensure_dir(out_root)
    if include_files:
        selected_files = list(include_files)
    else:
        selected_files = sorted(os.path.basename(path) for path in glob.glob(os.path.join(dataset_dir, "*.sdf")))

    manifest_rows: List[Dict[str, object]] = []
    combined_all_frames: List[pd.DataFrame] = []
    combined_recommended_frames: List[pd.DataFrame] = []

    for file_name in selected_files:
        dataset_path = os.path.join(dataset_dir, file_name)
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Missing Czodrowski SDF: {dataset_path}")

        dataset_name = os.path.splitext(file_name)[0]
        out_dir = os.path.join(out_root, dataset_name)
        _ensure_dir(out_dir)
        usage_metadata = _czodrowski_usage_metadata(file_name)

        assignments_df = build_functional_group_table(
            raw_dir=dataset_dir,
            overlap_threshold=float(overlap_threshold),
            include_files=[file_name],
            allow_epik=False,
        )
        assignments_path = os.path.join(out_dir, "assignments.csv")
        assignments_df.to_csv(assignments_path, index=False)

        summary_self_df = _strict_summary_frame(assignments_df)
        summary_self_path = os.path.join(out_dir, "summary_self_priors.csv")
        summary_self_df.to_csv(summary_self_path, index=False)

        candidates_self, candidates_curated, _ = _write_candidate_tables(assignments_path, summary_self_path, out_dir)
        assignments_df.get("final_group_label", pd.Series(dtype=object)).value_counts().rename_axis("final_group_label").reset_index(name="count").to_csv(
            os.path.join(out_dir, "final_group_counts.csv"),
            index=False,
        )

        combined_all_frames.append(assignments_df)
        if bool(usage_metadata["include_in_combined_recommended"]):
            combined_recommended_frames.append(assignments_df)
        manifest_rows.append(
            _dataset_manifest_row(
                dataset_name=dataset_name,
                dataset_role=str(usage_metadata["dataset_role"]),
                assignments_df=assignments_df,
                candidates_self=candidates_self,
                candidates_curated=candidates_curated,
                extra=usage_metadata,
            )
        )

    if combined_all_frames:
        combined_name = "combined_all"
        combined_dir = os.path.join(out_root, combined_name)
        _ensure_dir(combined_dir)

        combined_assignments_df = pd.concat(combined_all_frames, ignore_index=True)
        combined_assignments_path = os.path.join(combined_dir, "assignments.csv")
        combined_assignments_df.to_csv(combined_assignments_path, index=False)

        combined_summary_df = _strict_summary_frame(combined_assignments_df)
        combined_summary_path = os.path.join(combined_dir, "summary_self_priors.csv")
        combined_summary_df.to_csv(combined_summary_path, index=False)

        combined_candidates_self, combined_candidates_curated, _ = _write_candidate_tables(
            combined_assignments_path,
            combined_summary_path,
            combined_dir,
        )
        manifest_rows.append(
            _dataset_manifest_row(
                dataset_name=combined_name,
                dataset_role="combined",
                assignments_df=combined_assignments_df,
                candidates_self=combined_candidates_self,
                candidates_curated=combined_candidates_curated,
                extra={
                    "recommended_use": "inventory_only",
                    "recommended_use_reason": "contains_all_external_rows_including_do_not_use_duplicates",
                    "duplicate_local_source": "",
                    "include_in_combined_recommended": False,
                },
            )
        )

    if combined_recommended_frames:
        combined_name = "combined_recommended"
        combined_dir = os.path.join(out_root, combined_name)
        _ensure_dir(combined_dir)

        combined_assignments_df = pd.concat(combined_recommended_frames, ignore_index=True)
        combined_assignments_path = os.path.join(combined_dir, "assignments.csv")
        combined_assignments_df.to_csv(combined_assignments_path, index=False)

        combined_summary_df = _strict_summary_frame(combined_assignments_df)
        combined_summary_path = os.path.join(combined_dir, "summary_self_priors.csv")
        combined_summary_df.to_csv(combined_summary_path, index=False)

        combined_candidates_self, combined_candidates_curated, _ = _write_candidate_tables(
            combined_assignments_path,
            combined_summary_path,
            combined_dir,
        )
        manifest_rows.append(
            _dataset_manifest_row(
                dataset_name=combined_name,
                dataset_role="combined",
                assignments_df=combined_assignments_df,
                candidates_self=combined_candidates_self,
                candidates_curated=combined_candidates_curated,
                extra={
                    "recommended_use": "use_this_combined_view",
                    "recommended_use_reason": "excludes_do_not_use_local_duplicates_and_audit_only_bundles",
                    "duplicate_local_source": "",
                    "include_in_combined_recommended": True,
                },
            )
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["assignment_rows", "dataset_name"], ascending=[False, True])
    manifest_df.to_csv(os.path.join(out_root, "manifest.csv"), index=False)
    return manifest_df


_SMILES_ANALYSIS_CACHE: Dict[Tuple[str, float], Dict[str, object]] = {}


def _analyze_smiles(smiles: str, overlap_threshold: float) -> Dict[str, object]:
    smiles_text = str(smiles).strip()
    cache_key = (smiles_text, float(overlap_threshold))
    cached = _SMILES_ANALYSIS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mol = Chem.MolFromSmiles(smiles_text)
    if mol is None:
        payload = {
            "smiles": smiles_text,
            "mol": None,
            "formal_charge": 0,
            "candidates": [],
            "candidate_labels": [],
            "resolved_sites": [],
            "resolved_labels": [],
            "resolved_label_instances": [],
            "resolved_instances_indexed": [],
            "resolved_label_counts": Counter(),
            "resolved_types": [],
            "status": "no_group",
        }
        _SMILES_ANALYSIS_CACHE[cache_key] = payload
        return payload

    candidates = find_sites_with_metadata(mol)
    resolved_sites, _ = resolve_overlapping_sites(candidates, overlap_threshold=float(overlap_threshold))
    resolved_label_instances = [site["label"] for site in resolved_sites]
    resolved_label_counts = Counter(resolved_label_instances)
    payload = {
        "smiles": Chem.MolToSmiles(mol, canonical=True),
        "mol": mol,
        "formal_charge": _formal_charge(mol),
        "candidates": candidates,
        "candidate_labels": sorted({site["label"] for site in candidates}),
        "resolved_sites": resolved_sites,
        "resolved_labels": sorted({site["label"] for site in resolved_sites}),
        "resolved_label_instances": resolved_label_instances,
        "resolved_instances_indexed": [f"{site['label']}#{idx}" for idx, site in enumerate(resolved_sites)],
        "resolved_label_counts": resolved_label_counts,
        "resolved_types": sorted({site["type"] for site in resolved_sites}),
        "status": assignment_status(len(resolved_sites)),
    }
    _SMILES_ANALYSIS_CACHE[cache_key] = payload
    return payload


def _pair_family_form_map(resolved_sites: Sequence[Dict[str, object]]) -> Dict[str, Counter]:
    family_forms: Dict[str, Counter] = {}
    for site in resolved_sites:
        for transition in expand_pair_site_transitions(site):
            family = str(transition["pair_family"])
            pair_form = str(transition["pair_member_form"])
            family_forms.setdefault(family, Counter())[pair_form] += 1
    return family_forms


def _expanded_pair_sites(resolved_sites: Sequence[Dict[str, object]]) -> List[Dict]:
    return [
        transition
        for site in resolved_sites
        for transition in expand_pair_site_transitions(site)
    ]


def _read_sampl7_experimental(sampl7_root: str, include_compounds: Optional[Sequence[str]] = None) -> pd.DataFrame:
    exp_path = os.path.join(
        sampl7_root,
        "physical_property",
        "pKa",
        "analysis",
        "macrostate_analysis",
        "pKa_experimental_values.csv",
    )
    df = pd.read_csv(exp_path).rename(
        columns={
            "Molecule ID": "compound_id",
            "pKa mean": "pka_value",
            "pKa SEM": "pka_sem",
            "Isomeric SMILES": "reference_smiles",
        }
    )
    if include_compounds:
        df = df[df["compound_id"].isin(include_compounds)].copy()
    df["dataset_name"] = "SAMPL7"
    df["observation_index"] = df.groupby("compound_id").cumcount().astype(int)
    return df[["dataset_name", "compound_id", "observation_index", "pka_value", "pka_sem", "reference_smiles"]]


def _read_sampl8_experimental(sampl8_root: str, include_compounds: Optional[Sequence[str]] = None) -> pd.DataFrame:
    exp_path = os.path.join(sampl8_root, "physical_properties", "pKa", "experimental_pKas.csv")
    df = pd.read_csv(exp_path)
    df.columns = [column.strip() for column in df.columns]
    df = df.rename(
        columns={
            "compound ID": "compound_id",
            "pKa": "pka_value",
            "uncertainty (every pKa is followed by an uncertainty; multiple pairs indicate multiple pKa values; N/A if not measured)": "pka_sem",
        }
    )
    if include_compounds:
        df = df[df["compound_id"].isin(include_compounds)].copy()
    df["dataset_name"] = "SAMPL8"
    df["observation_index"] = df.groupby("compound_id").cumcount().astype(int)
    df["reference_smiles"] = ""
    return df[["dataset_name", "compound_id", "observation_index", "pka_value", "pka_sem", "reference_smiles"]]


def _load_microstates(
    microstate_dir: str,
    dataset_name: str,
    include_compounds: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for csv_path in sorted(glob.glob(os.path.join(microstate_dir, "*_microstates.csv"))):
        compound_id = os.path.basename(csv_path).replace("_microstates.csv", "")
        if include_compounds and compound_id not in include_compounds:
            continue

        df = pd.read_csv(csv_path)
        df.columns = [column.strip() for column in df.columns]
        for row in df.to_dict(orient="records"):
            microstate_id = str(row.get("microstate ID", "")).strip()
            smiles = str(row.get("canonical isomeric SMILES", "")).strip()
            mol = Chem.MolFromSmiles(smiles)
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "compound_id": compound_id,
                    "microstate_id": microstate_id,
                    "smiles": Chem.MolToSmiles(mol, canonical=True) if mol is not None else smiles,
                    "formal_charge": _formal_charge(mol),
                    "is_reference": microstate_id.endswith("micro000"),
                    "microstate_csv": os.path.basename(csv_path),
                }
            )

    return pd.DataFrame(rows)


def _reference_transition_records(microstates_df: pd.DataFrame, overlap_threshold: float) -> List[Dict[str, object]]:
    if microstates_df.empty:
        return []

    reference_rows = microstates_df[microstates_df["is_reference"]]
    if reference_rows.empty:
        return []

    reference_row = reference_rows.iloc[0]
    reference_analysis = _analyze_smiles(str(reference_row["smiles"]), overlap_threshold=float(overlap_threshold))
    reference_family_forms = _pair_family_form_map(reference_analysis["resolved_sites"])

    transition_rows: List[Dict[str, object]] = []
    for other_row in microstates_df.itertuples(index=False):
        if str(other_row.microstate_id) == str(reference_row["microstate_id"]):
            continue

        charge_delta = int(other_row.formal_charge) - int(reference_row["formal_charge"])
        if abs(charge_delta) != 1:
            continue

        other_analysis = _analyze_smiles(str(other_row.smiles), overlap_threshold=float(overlap_threshold))
        other_family_forms = _pair_family_form_map(other_analysis["resolved_sites"])
        # Compare transitions represented on both endpoints. Amphoteric neutral
        # structures expose both adjacent edges, whereas a terminal charged
        # form exposes only the edge that reaches it.
        changed_families = sorted(
            family
            for family in (set(reference_family_forms) & set(other_family_forms))
            if reference_family_forms.get(family, Counter()) != other_family_forms.get(family, Counter())
        )

        base_row: Dict[str, object] = {
            "dataset_name": str(reference_row["dataset_name"]),
            "compound_id": str(reference_row["compound_id"]),
            "reference_microstate_id": str(reference_row["microstate_id"]),
            "other_microstate_id": str(other_row.microstate_id),
            "reference_formal_charge": int(reference_row["formal_charge"]),
            "other_formal_charge": int(other_row.formal_charge),
            "charge_delta": charge_delta,
            "changed_family_count": len(changed_families),
            "changed_families": "|".join(changed_families),
            "reference_smiles": str(reference_row["smiles"]),
            "other_smiles": str(other_row.smiles),
            "truth_status": "ambiguous",
            "true_label_on_reference": "",
            "true_family_on_reference": "",
            "true_pair_member_form": "",
            "true_atom_index": None,
            "pka_type_canonical": "",
        }

        if len(changed_families) != 1:
            base_row["truth_status"] = "ambiguous_changed_family_count"
            transition_rows.append(base_row)
            continue

        changed_family = changed_families[0]
        reference_family_sites = [
            site for site in _expanded_pair_sites(reference_analysis["resolved_sites"])
            if str(site.get("pair_family")) == changed_family
        ]
        other_family_sites = [
            site for site in _expanded_pair_sites(other_analysis["resolved_sites"])
            if str(site.get("pair_family")) == changed_family
        ]
        if len(reference_family_sites) != 1:
            base_row["truth_status"] = "ambiguous_reference_site_count"
            transition_rows.append(base_row)
            continue
        if not other_family_sites:
            base_row["truth_status"] = "ambiguous_missing_other_family_site"
            transition_rows.append(base_row)
            continue

        reference_site = reference_family_sites[0]
        true_label = str(reference_site["label"])
        pair_member_form = str(reference_site.get("pair_member_form", ""))
        if pair_member_form not in {"acid_form", "base_form"}:
            base_row["truth_status"] = "ambiguous_pair_form"
            transition_rows.append(base_row)
            continue

        expected_delta = -1 if pair_member_form == "acid_form" else 1
        if charge_delta != expected_delta:
            base_row["truth_status"] = "ambiguous_charge_direction"
            transition_rows.append(base_row)
            continue

        atom_set = sorted(reference_site.get("atom_set", set()))
        base_row.update(
            {
                "truth_status": "unambiguous",
                "true_label_on_reference": true_label,
                "true_family_on_reference": changed_family,
                "true_pair_member_form": pair_member_form,
                "true_atom_index": int(atom_set[0]) if atom_set else None,
                "pka_type_canonical": "acidic" if pair_member_form == "acid_form" else "basic",
            }
        )
        transition_rows.append(base_row)

    return transition_rows


def _benchmark_assignment_row(
    observation_row: pd.Series,
    transition_row: pd.Series,
    overlap_threshold: float,
) -> Dict[str, object]:
    reference_analysis = _analyze_smiles(str(transition_row["reference_smiles"]), overlap_threshold=float(overlap_threshold))
    resolved_sites = reference_analysis["resolved_sites"]
    measurement_group = str(transition_row["true_label_on_reference"])
    measurement_family = str(transition_row["true_family_on_reference"])
    expanded_sites = _expanded_pair_sites(resolved_sites)
    selected_site = next(
        site
        for site in expanded_sites
        if str(site.get("label")) == measurement_group
        and str(site.get("pair_family")) == measurement_family
    )
    status = assignment_status(len(resolved_sites))
    final_group = assign_single_group_label(resolved_sites)
    measurement_canonical = canonical_family_label(measurement_group, family=measurement_family)
    measurement_refined = refine_group_label(reference_analysis["mol"], selected_site, measurement_group)
    group_mode, pair_member_form = classify_pair_type(measurement_group, measurement_family)
    formal_charge = int(reference_analysis["formal_charge"])
    neutral_input_risk = bool(
        group_mode == "pair_type"
        and measurement_family in BASIC_FAMILIES
        and pair_member_form == "base_form"
        and formal_charge == 0
    )
    summary_label = measurement_group if group_mode == "pair_type" else (measurement_canonical or measurement_refined)
    pka_type_canonical = str(transition_row["pka_type_canonical"])
    distribution_ok, distribution_reason = distribution_sanity_check(
        group_mode=group_mode,
        pair_member_form=pair_member_form,
        summary_label=summary_label,
        pka_value=float(observation_row["pka_value"]),
    )
    pka_type_effective, two_faced_type_flip = reinterpret_pka_type_for_two_faced(
        pka_type=pka_type_canonical,
        group_family=measurement_family,
    )
    conjugate_flag = conjugate_check_flag(
        group_family=measurement_family,
        pka_type=pka_type_canonical,
        pka_value=float(observation_row["pka_value"]),
    )
    pair_sites = expanded_sites
    pair_candidate_labels = "|".join(sorted({str(site.get("label")) for site in pair_sites if site.get("label")}))

    return {
        "source_file": f"{observation_row['dataset_name']}_{observation_row['compound_id']}.benchmark",
        "record_index": int(observation_row["observation_index"]),
        "smiles": str(reference_analysis["smiles"]),
        "pka_value": float(observation_row["pka_value"]),
        "pka_source_method": "experimental",
        "pka_type_raw": pka_type_canonical,
        "pka_type_canonical": pka_type_canonical,
        "pka_type_effective": pka_type_effective,
        "two_faced_type_flip": bool(two_faced_type_flip),
        "atom_index_raw": transition_row["true_atom_index"],
        "atom_index_mode": "reference_microstate_inferred",
        "atom_index_matched": transition_row["true_atom_index"],
        "atom_site_match": True,
        "atom_candidate_count": 1,
        "atom_candidate_labels": measurement_group,
        "site_identifier_raw": str(transition_row["other_microstate_id"]),
        "all_candidate_groups": "|".join(reference_analysis["candidate_labels"]),
        "resolved_groups": "|".join(reference_analysis["resolved_labels"]),
        "resolved_group_instances": "|".join(reference_analysis["resolved_label_instances"]),
        "resolved_group_instances_indexed": "|".join(reference_analysis["resolved_instances_indexed"]),
        "resolved_group_label_counts": "|".join(
            f"{label}:{count}" for label, count in sorted(reference_analysis["resolved_label_counts"].items())
        ),
        "resolved_group_types": "|".join(reference_analysis["resolved_types"]),
        "resolved_group_count": int(len(resolved_sites)),
        "rejected_group_count": 0,
        "assignment_status": status,
        "selected_group_label": measurement_group,
        "selected_group_family": measurement_family,
        "final_group_label": measurement_group,
        "final_group_refined": measurement_refined,
        "final_group_summary_label": summary_label,
        "group_mode": group_mode,
        "pair_family": measurement_family if measurement_family in PAIR_TYPE_FAMILY_FORMS else None,
        "pair_member_form": pair_member_form,
        "pair_assignment_confidence": "high",
        "pair_candidate_count": int(len(pair_sites)),
        "pair_candidate_labels": pair_candidate_labels,
        "pair_selection_reason": "reference_microstate_transition_inference",
        "selected_site_priority": selected_site.get("priority"),
        "selected_site_specificity": selected_site.get("specificity"),
        "molecule_formal_charge": formal_charge,
        "neutral_input_risk": neutral_input_risk,
        "distribution_ok": distribution_ok,
        "distribution_reason": distribution_reason,
        "final_group_family": measurement_family,
        "final_group_canonical": measurement_canonical,
        "conjugate_check_flag": conjugate_flag,
        "single_group_strict": status == "single_group",
        "training_group_label": training_group_label(measurement_group),
        "training_group_reason": training_group_reason(measurement_group),
        "benchmark_dataset": str(observation_row["dataset_name"]),
        "benchmark_compound_id": str(observation_row["compound_id"]),
        "benchmark_observation_index": int(observation_row["observation_index"]),
        "benchmark_reference_microstate_id": str(transition_row["reference_microstate_id"]),
        "benchmark_truth_status": str(transition_row["truth_status"]),
        "benchmark_pka_sem": float(observation_row["pka_sem"]),
    }


def _write_benchmark_outputs(
    out_dir: str,
    observations_df: pd.DataFrame,
    microstates_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
    assignments_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _ensure_dir(out_dir)
    observations_df.to_csv(os.path.join(out_dir, "macro_observations.csv"), index=False)
    microstates_df.to_csv(os.path.join(out_dir, "microstates.csv"), index=False)
    transitions_df.to_csv(os.path.join(out_dir, "transition_catalog.csv"), index=False)
    assignments_path = os.path.join(out_dir, "unambiguous_assignments.csv")
    if assignments_df.empty and not len(assignments_df.columns):
        assignments_df = pd.DataFrame(columns=EMPTY_BENCHMARK_ASSIGNMENT_COLUMNS)
    assignments_df.to_csv(assignments_path, index=False)

    summary_self_df = _strict_summary_frame(assignments_df)
    summary_self_path = os.path.join(out_dir, "summary_self_priors.csv")
    summary_self_df.to_csv(summary_self_path, index=False)
    candidates_self, candidates_curated, _ = _write_candidate_tables(assignments_path, summary_self_path, out_dir)
    return candidates_self, candidates_curated


def prepare_sampl_benchmarks(
    sampl7_root: str = DEFAULT_SAMPL7_ROOT,
    sampl8_root: str = DEFAULT_SAMPL8_ROOT,
    out_root: str = os.path.join(DEFAULT_OUTPUT_ROOT, "sampl"),
    sampl7_compounds: Optional[Sequence[str]] = None,
    sampl8_compounds: Optional[Sequence[str]] = None,
    overlap_threshold: float = 0.75,
) -> pd.DataFrame:
    _ensure_dir(out_root)

    dataset_configs = [
        {
            "dataset_name": "SAMPL7",
            "root": sampl7_root,
            "experimental_loader": _read_sampl7_experimental,
            "microstate_dir": os.path.join(sampl7_root, "physical_property", "pKa", "microstates"),
            "compound_filter": sampl7_compounds,
        },
        {
            "dataset_name": "SAMPL8",
            "root": sampl8_root,
            "experimental_loader": _read_sampl8_experimental,
            "microstate_dir": os.path.join(sampl8_root, "physical_properties", "microstates"),
            "compound_filter": sampl8_compounds,
        },
    ]

    manifest_rows: List[Dict[str, object]] = []
    combined_observations: List[pd.DataFrame] = []
    combined_microstates: List[pd.DataFrame] = []
    combined_transitions: List[pd.DataFrame] = []
    combined_assignments: List[pd.DataFrame] = []

    for config in dataset_configs:
        dataset_name = str(config["dataset_name"])
        dataset_root = str(config["root"])
        if not os.path.exists(dataset_root):
            raise FileNotFoundError(f"{dataset_name} root not found: {dataset_root}")

        compound_filter = config["compound_filter"]
        if compound_filter is not None and len(compound_filter) == 0:
            dataset_observations_df = pd.DataFrame(
                columns=[
                    "dataset_name",
                    "compound_id",
                    "observation_index",
                    "pka_value",
                    "pka_sem",
                    "reference_smiles",
                    "microstate_count",
                    "experimental_pka_count",
                    "unambiguous_transition_count",
                    "unique_unambiguous_label_count",
                    "unique_unambiguous_labels",
                    "usable_for_unambiguous_site_scoring",
                    "assigned_true_label",
                    "assigned_pka_type_canonical",
                ]
            )
            microstates_df = pd.DataFrame(
                columns=["dataset_name", "compound_id", "microstate_id", "smiles", "formal_charge", "is_reference", "microstate_csv"]
            )
            transitions_df = pd.DataFrame()
            assignments_df = pd.DataFrame()
            dataset_out_dir = os.path.join(out_root, dataset_name)
            candidates_self, candidates_curated = _write_benchmark_outputs(
                out_dir=dataset_out_dir,
                observations_df=dataset_observations_df,
                microstates_df=microstates_df,
                transitions_df=transitions_df,
                assignments_df=assignments_df,
            )
            manifest_rows.append(
                {
                    "dataset_name": dataset_name,
                    "macro_observations": 0,
                    "molecules": 0,
                    "microstates": 0,
                    "transition_rows": 0,
                    "unambiguous_assignments": 0,
                    "unambiguous_candidates_self_priors": int(len(candidates_self)),
                    "unambiguous_candidates_curated_priors": int(len(candidates_curated)),
                    "usable_macro_fraction": 0.0,
                }
            )
            combined_observations.append(dataset_observations_df)
            combined_microstates.append(microstates_df)
            combined_transitions.append(transitions_df)
            combined_assignments.append(assignments_df)
            continue

        observations_df = config["experimental_loader"](dataset_root, include_compounds=config["compound_filter"])
        microstates_df = _load_microstates(
            microstate_dir=str(config["microstate_dir"]),
            dataset_name=dataset_name,
            include_compounds=config["compound_filter"],
        )

        transition_rows: List[Dict[str, object]] = []
        assignments_rows: List[Dict[str, object]] = []
        transition_by_compound: Dict[str, List[Dict[str, object]]] = {}
        for compound_id, compound_microstates in microstates_df.groupby("compound_id", sort=True):
            compound_transitions = _reference_transition_records(compound_microstates.copy(), overlap_threshold=float(overlap_threshold))
            transition_by_compound[str(compound_id)] = compound_transitions
            transition_rows.extend(compound_transitions)

        transitions_df = pd.DataFrame(transition_rows)
        observation_rows: List[Dict[str, object]] = []
        for observation in observations_df.itertuples(index=False):
            compound_id = str(observation.compound_id)
            compound_microstates = microstates_df[microstates_df["compound_id"] == compound_id].copy()
            compound_transitions = transition_by_compound.get(compound_id, [])
            unambiguous = [row for row in compound_transitions if row.get("truth_status") == "unambiguous"]
            unique_labels = sorted({str(row["true_label_on_reference"]) for row in unambiguous if row.get("true_label_on_reference")})

            observation_row = {
                "dataset_name": dataset_name,
                "compound_id": compound_id,
                "observation_index": int(observation.observation_index),
                "pka_value": float(observation.pka_value),
                "pka_sem": float(observation.pka_sem),
                "reference_smiles": str(observation.reference_smiles or ""),
                "microstate_count": int(len(compound_microstates)),
                "experimental_pka_count": int(len(observations_df[observations_df["compound_id"] == compound_id])),
                "unambiguous_transition_count": int(len(unambiguous)),
                "unique_unambiguous_label_count": int(len(unique_labels)),
                "unique_unambiguous_labels": "|".join(unique_labels),
                "usable_for_unambiguous_site_scoring": False,
            }

            if observation_row["experimental_pka_count"] == 1 and len(unique_labels) == 1:
                selected_transition = next(row for row in unambiguous if str(row["true_label_on_reference"]) == unique_labels[0])
                assignment_row = _benchmark_assignment_row(
                    pd.Series(observation_row),
                    pd.Series(selected_transition),
                    overlap_threshold=float(overlap_threshold),
                )
                assignments_rows.append(assignment_row)
                observation_row["usable_for_unambiguous_site_scoring"] = True
                observation_row["assigned_true_label"] = unique_labels[0]
                observation_row["assigned_pka_type_canonical"] = str(selected_transition["pka_type_canonical"])
            else:
                observation_row["assigned_true_label"] = ""
                observation_row["assigned_pka_type_canonical"] = ""

            observation_rows.append(observation_row)

        dataset_observations_df = pd.DataFrame(observation_rows)
        assignments_df = pd.DataFrame(assignments_rows)

        dataset_out_dir = os.path.join(out_root, dataset_name)
        candidates_self, candidates_curated = _write_benchmark_outputs(
            out_dir=dataset_out_dir,
            observations_df=dataset_observations_df,
            microstates_df=microstates_df,
            transitions_df=transitions_df,
            assignments_df=assignments_df,
        )

        combined_observations.append(dataset_observations_df)
        combined_microstates.append(microstates_df)
        combined_transitions.append(transitions_df)
        combined_assignments.append(assignments_df)

        manifest_rows.append(
            {
                "dataset_name": dataset_name,
                "macro_observations": int(len(dataset_observations_df)),
                "molecules": int(dataset_observations_df["compound_id"].nunique()) if not dataset_observations_df.empty else 0,
                "microstates": int(len(microstates_df)),
                "transition_rows": int(len(transitions_df)),
                "unambiguous_assignments": int(len(assignments_df)),
                "unambiguous_candidates_self_priors": int(len(candidates_self)),
                "unambiguous_candidates_curated_priors": int(len(candidates_curated)),
                "usable_macro_fraction": _safe_rate(dataset_observations_df.get("usable_for_unambiguous_site_scoring", pd.Series(dtype=float))),
            }
        )

    if combined_observations:
        combined_dir = os.path.join(out_root, "combined")
        combined_observations_df = pd.concat(combined_observations, ignore_index=True)
        combined_microstates_df = pd.concat(combined_microstates, ignore_index=True) if combined_microstates else pd.DataFrame()
        combined_transitions_df = pd.concat(combined_transitions, ignore_index=True) if combined_transitions else pd.DataFrame()
        combined_assignments_df = pd.concat(combined_assignments, ignore_index=True) if combined_assignments else pd.DataFrame()
        combined_candidates_self, combined_candidates_curated = _write_benchmark_outputs(
            out_dir=combined_dir,
            observations_df=combined_observations_df,
            microstates_df=combined_microstates_df,
            transitions_df=combined_transitions_df,
            assignments_df=combined_assignments_df,
        )
        manifest_rows.append(
            {
                "dataset_name": "combined",
                "macro_observations": int(len(combined_observations_df)),
                "molecules": int(combined_observations_df["compound_id"].nunique()) if not combined_observations_df.empty else 0,
                "microstates": int(len(combined_microstates_df)),
                "transition_rows": int(len(combined_transitions_df)),
                "unambiguous_assignments": int(len(combined_assignments_df)),
                "unambiguous_candidates_self_priors": int(len(combined_candidates_self)),
                "unambiguous_candidates_curated_priors": int(len(combined_candidates_curated)),
                "usable_macro_fraction": _safe_rate(combined_observations_df.get("usable_for_unambiguous_site_scoring", pd.Series(dtype=float))),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows).sort_values("dataset_name")
    manifest_df.to_csv(os.path.join(out_root, "manifest.csv"), index=False)
    return manifest_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare external pKa datasets in pipeline-compatible formats.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cz_parser = subparsers.add_parser("czodrowski", help="Normalize Czodrowski SDF datasets into assignments/summary/candidate tables.")
    cz_parser.add_argument("--dataset-dir", default=DEFAULT_CZODROWSKI_DATASET_DIR)
    cz_parser.add_argument("--out-root", default=os.path.join(DEFAULT_OUTPUT_ROOT, "czodrowski"))
    cz_parser.add_argument("--include-files", nargs="*", default=[])
    cz_parser.add_argument("--overlap-threshold", type=float, default=0.75)

    sampl_parser = subparsers.add_parser("sampl", help="Prepare SAMPL7/8 benchmark tables and unambiguous candidate subsets.")
    sampl_parser.add_argument("--sampl7-root", default=DEFAULT_SAMPL7_ROOT)
    sampl_parser.add_argument("--sampl8-root", default=DEFAULT_SAMPL8_ROOT)
    sampl_parser.add_argument("--out-root", default=os.path.join(DEFAULT_OUTPUT_ROOT, "sampl"))
    sampl_parser.add_argument("--sampl7-compounds", nargs="*", default=[])
    sampl_parser.add_argument("--sampl8-compounds", nargs="*", default=[])
    sampl_parser.add_argument("--overlap-threshold", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "czodrowski":
        manifest_df = prepare_czodrowski_datasets(
            dataset_dir=args.dataset_dir,
            out_root=args.out_root,
            include_files=args.include_files or None,
            overlap_threshold=float(args.overlap_threshold),
        )
    else:
        manifest_df = prepare_sampl_benchmarks(
            sampl7_root=args.sampl7_root,
            sampl8_root=args.sampl8_root,
            out_root=args.out_root,
            sampl7_compounds=args.sampl7_compounds or None,
            sampl8_compounds=args.sampl8_compounds or None,
            overlap_threshold=float(args.overlap_threshold),
        )

    print(f"Saved manifest: {os.path.join(args.out_root, 'manifest.csv')}")
    print(manifest_df.to_string(index=False))


if __name__ == "__main__":
    main()
