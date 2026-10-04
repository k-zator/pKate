import os
from typing import List, Tuple, Optional


DEFAULT_RAW_DIR = "data/raw"
DEFAULT_CURATED_ASSIGNMENTS_PATH = "data/processed/functional_group_assignments_curated.csv"
DEFAULT_CURATED_SUMMARY_PATH = "data/processed/functional_group_pka_summary_curated.csv"
DEFAULT_SITE_OVERRIDES_PATH = "data/processed/site_resolution_overrides.csv"
DEFAULT_OVERLAP_THRESHOLD = 0.75
TRUSTED_TRAINING_SDF_FILES = (
    "literature_compilation.sdf",
    "novartis_testdata.sdf",
    "experimental_training_datasets.sdf",
    "AvLiLuMoVe_testdata.sdf",
    "FIXED_chembl26.sdf",
)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _trusted_raw_paths(raw_dir: str) -> List[str]:
    paths: List[str] = []
    missing: List[str] = []
    for file_name in TRUSTED_TRAINING_SDF_FILES:
        path = os.path.join(raw_dir, file_name)
        if os.path.exists(path):
            paths.append(path)
        else:
            missing.append(file_name)

    if missing:
        raise FileNotFoundError(
            f"Missing expected trusted training SDF files in {raw_dir}: {', '.join(sorted(missing))}"
        )
    return paths


def _latest_mtime(paths: List[str]) -> float:
    return max(os.path.getmtime(path) for path in paths)


def _dependency_paths(site_overrides_path: Optional[str] = None) -> List[str]:
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    dependency_names = [
        "training_data_resolver.py",
        "functional_group_pka_analysis.py",
        "SMARTS_library.py",
        "substructure_match.py",
    ]
    paths: List[str] = []
    for name in dependency_names:
        path = os.path.join(scripts_dir, name)
        if os.path.exists(path):
            paths.append(path)
    if site_overrides_path and os.path.exists(site_overrides_path):
        paths.append(site_overrides_path)
    return paths


def ensure_curated_training_data(
    assignments_path: str = DEFAULT_CURATED_ASSIGNMENTS_PATH,
    summary_path: str = DEFAULT_CURATED_SUMMARY_PATH,
    raw_dir: str = DEFAULT_RAW_DIR,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    site_overrides_path: str = DEFAULT_SITE_OVERRIDES_PATH,
) -> Tuple[str, str]:
    raw_paths = _trusted_raw_paths(raw_dir)
    dependency_paths = _dependency_paths(site_overrides_path=site_overrides_path)

    assignments_exists = os.path.exists(assignments_path)
    summary_exists = os.path.exists(summary_path)
    if assignments_exists and summary_exists:
        outputs_mtime = min(os.path.getmtime(assignments_path), os.path.getmtime(summary_path))
        newest_input_mtime = _latest_mtime(raw_paths + dependency_paths)
        if outputs_mtime >= newest_input_mtime:
            return assignments_path, summary_path

    from functional_group_pka_analysis import build_functional_group_table, build_summary_table, training_group_label

    _ensure_parent_dir(assignments_path)
    _ensure_parent_dir(summary_path)

    df = build_functional_group_table(
        raw_dir=raw_dir,
        overlap_threshold=overlap_threshold,
        include_files=list(TRUSTED_TRAINING_SDF_FILES),
        allow_epik=False,
        overrides_path=site_overrides_path,
    )

    strict_df = df[
        df["single_group_strict"]
        & df["final_group_summary_label"].notna()
        & df["distribution_ok"]
    ].copy()
    if "training_group_label" not in strict_df.columns:
        strict_df["training_group_label"] = strict_df["final_group_summary_label"].map(training_group_label)
    strict_df["final_group_label"] = strict_df["training_group_label"]
    summary = build_summary_table(strict_df)

    df.to_csv(assignments_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Refreshed curated training data from {len(raw_paths)} trusted raw SDF files.")
    print(f"  assignments: {assignments_path}")
    print(f"  summary:     {summary_path}")
    return assignments_path, summary_path