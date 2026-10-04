from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from rdkit import Chem  # type: ignore

from functional_group_pka_analysis import ACIDIC_FAMILIES, BASIC_FAMILIES, CONJUGATE_FAMILY_MAP
from substructure_match import find_sites_with_metadata, maximal_site_rank_key, resolve_overlapping_sites


AMMONIUM_LIKE_LABELS = {
    "primary_ammonium",
    "secondary_ammonium",
    "tertiary_ammonium",
    "quaternary_ammonium",
    "aryl_ammonium",
    "pyridinium",
    "quinolinium",
    "isoquinolinium",
    "pyrimidinium",
    "pyrazinium",
    "pyrrolium",
    "pyrrolium_n",
    "triazinium",
    "imidazolium",
    "iminium",
}

AMINE_LIKE_LABELS = {
    "primary_amine",
    "secondary_amine",
    "tertiary_amine",
    "aniline",
    "pyridine",
    "quinoline",
    "isoquinoline",
    "pyrimidine",
    "pyrazine",
    "pyrrole",
    "triazine",
    "imidazole",
    "imine",
}


def _split_groups(value: object) -> List[str]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [token for token in text.split("|") if token]


def _family_for_label(label: str) -> str:
    return CONJUGATE_FAMILY_MAP.get(label, label)


def _best_site_for_label(resolved_sites: List[Dict], label: str) -> Optional[Dict]:
    if not label:
        return None
    matches = [site for site in resolved_sites if str(site.get("label")) == label]
    if not matches:
        return None
    return sorted(matches, key=maximal_site_rank_key, reverse=True)[0]


def _min_graph_distance(
    distance_matrix,
    source_atoms: Set[int],
    target_atoms: Set[int],
) -> float:
    if not source_atoms or not target_atoms:
        return float("inf")
    return min(float(distance_matrix[a, b]) for a in source_atoms for b in target_atoms)


def _count_target_within(
    distance_matrix,
    source_atoms: Set[int],
    target_atoms: Set[int],
    radius: int,
) -> int:
    if not source_atoms or not target_atoms:
        return 0
    return sum(1 for atom in target_atoms if min(float(distance_matrix[s, atom]) for s in source_atoms) <= float(radius))


def _local_charge_sum_within(
    distance_matrix,
    mol: Chem.Mol,
    source_atoms: Set[int],
    radius: int,
) -> float:
    if not source_atoms:
        return 0.0
    total = 0.0
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        if min(float(distance_matrix[s, atom_idx]) for s in source_atoms) <= float(radius):
            total += float(atom.GetFormalCharge())
    return total


def build_ionization_context_frame(
    df: pd.DataFrame,
    resolved_col: str = "resolved_groups",
    candidate_col: str = "candidate_label",
    smiles_col: str = "smiles",
    overlap_threshold: float = 0.75,
) -> pd.DataFrame:
    if resolved_col not in df.columns:
        return pd.DataFrame(index=df.index)

    candidate_series = df[candidate_col] if candidate_col in df.columns else pd.Series([""] * len(df), index=df.index)
    smiles_series = df[smiles_col] if smiles_col in df.columns else pd.Series([""] * len(df), index=df.index)

    per_smiles_cache: Dict[str, Dict[str, object]] = {}

    def _smiles_payload(smiles: str) -> Dict[str, object]:
        if smiles in per_smiles_cache:
            return per_smiles_cache[smiles]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            payload = {
                "mol": None,
                "resolved": [],
                "distance": None,
                "positive_n": set(),
                "basic_atoms": set(),
                "amide_atoms": set(),
            }
            per_smiles_cache[smiles] = payload
            return payload

        candidates = find_sites_with_metadata(mol)
        resolved, _ = resolve_overlapping_sites(candidates, overlap_threshold=overlap_threshold)
        distance = Chem.GetDistanceMatrix(mol)

        positive_n = {
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() > 0
        }
        basic_atoms = set()
        amide_atoms = set()

        for site in resolved:
            atom_set = set(site.get("atom_set", set()))
            label = str(site.get("label", ""))
            family = str(site.get("family", ""))
            site_type = str(site.get("type", ""))
            if site_type == "basic" or family in BASIC_FAMILIES:
                basic_atoms.update(atom_set)
            if label.startswith("amide") or label == "lactam":
                amide_atoms.update(atom_set)

        payload = {
            "mol": mol,
            "resolved": resolved,
            "distance": distance,
            "positive_n": positive_n,
            "basic_atoms": basic_atoms,
            "amide_atoms": amide_atoms,
        }
        per_smiles_cache[smiles] = payload
        return payload

    rows = []
    for resolved_value, candidate_label, smiles in zip(df[resolved_col], candidate_series, smiles_series):
        tokens = _split_groups(resolved_value)
        families = [_family_for_label(token) for token in tokens]
        candidate_label = "" if pd.isna(candidate_label) else str(candidate_label)
        candidate_family = _family_for_label(candidate_label) if candidate_label else ""

        total = len(tokens)
        acidic = sum(1 for fam in families if fam in ACIDIC_FAMILIES)
        basic = sum(1 for fam in families if fam in BASIC_FAMILIES)
        carboxyl = sum(1 for fam in families if fam == "carboxyl")
        amide_like = sum(1 for token in tokens if token.startswith("amide") or token == "lactam")
        amine_like = sum(1 for token in tokens if token in AMINE_LIKE_LABELS)
        ammonium_like = sum(1 for token in tokens if token in AMMONIUM_LIKE_LABELS)
        same_family = sum(1 for fam in families if fam == candidate_family and candidate_family)

        local_min_dist_pos_n = float("inf")
        local_pos_n_within_3 = 0
        local_pos_n_within_5 = 0
        local_min_dist_basic = float("inf")
        local_basic_within_4 = 0
        local_min_dist_amide = float("inf")
        local_amide_within_4 = 0
        local_site_atom_count = 0.0
        local_site_charge_sum = 0.0
        local_charge_sum_r2 = 0.0

        smiles_text = "" if pd.isna(smiles) else str(smiles).strip()
        if smiles_text:
            payload = _smiles_payload(smiles_text)
            mol = payload["mol"]
            resolved_sites = payload["resolved"]
            distance = payload["distance"]
            if mol is not None and distance is not None and candidate_label:
                best_site = _best_site_for_label(resolved_sites, candidate_label)
                if best_site is not None:
                    source_atoms = set(best_site.get("atom_set", set()))
                    positive_n = set(payload["positive_n"])
                    basic_atoms = set(payload["basic_atoms"])
                    amide_atoms = set(payload["amide_atoms"])

                    local_site_atom_count = float(len(source_atoms))
                    local_site_charge_sum = float(
                        sum(mol.GetAtomWithIdx(atom_idx).GetFormalCharge() for atom_idx in source_atoms)
                    )
                    local_charge_sum_r2 = _local_charge_sum_within(
                        distance,
                        mol,
                        source_atoms,
                        radius=2,
                    )

                    local_min_dist_pos_n = _min_graph_distance(distance, source_atoms, positive_n)
                    local_pos_n_within_3 = _count_target_within(distance, source_atoms, positive_n, radius=3)
                    local_pos_n_within_5 = _count_target_within(distance, source_atoms, positive_n, radius=5)

                    local_min_dist_basic = _min_graph_distance(distance, source_atoms, basic_atoms)
                    local_basic_within_4 = _count_target_within(distance, source_atoms, basic_atoms, radius=4)

                    local_min_dist_amide = _min_graph_distance(distance, source_atoms, amide_atoms)
                    local_amide_within_4 = _count_target_within(distance, source_atoms, amide_atoms, radius=4)

        if local_min_dist_pos_n == float("inf"):
            local_min_dist_pos_n = 99.0
        if local_min_dist_basic == float("inf"):
            local_min_dist_basic = 99.0
        if local_min_dist_amide == float("inf"):
            local_min_dist_amide = 99.0

        rows.append(
            {
                "ion_ctx_total_groups": float(total),
                "ion_ctx_acidic_groups": float(acidic),
                "ion_ctx_basic_groups": float(basic),
                "ion_ctx_carboxyl_groups": float(carboxyl),
                "ion_ctx_amide_like_groups": float(amide_like),
                "ion_ctx_amine_like_groups": float(amine_like),
                "ion_ctx_ammonium_like_groups": float(ammonium_like),
                "ion_ctx_same_family_groups": float(same_family),
                "ion_ctx_other_family_groups": float(max(0, total - same_family)),
                "ion_ctx_candidate_is_acidic_family": float(candidate_family in ACIDIC_FAMILIES),
                "ion_ctx_candidate_is_basic_family": float(candidate_family in BASIC_FAMILIES),
                "ion_ctx_candidate_is_carboxyl": float(candidate_family == "carboxyl"),
                "ion_ctx_has_carboxyl_and_basic": float((carboxyl > 0) and (basic > 0)),
                "ion_ctx_charge_proxy": float(ammonium_like - carboxyl),
                "ion_ctx_local_site_atom_count": float(local_site_atom_count),
                "ion_ctx_local_site_charge_sum": float(local_site_charge_sum),
                "ion_ctx_local_charge_sum_r2": float(local_charge_sum_r2),
                "ion_ctx_local_min_dist_pos_n": float(local_min_dist_pos_n),
                "ion_ctx_local_pos_n_within_3": float(local_pos_n_within_3),
                "ion_ctx_local_pos_n_within_5": float(local_pos_n_within_5),
                "ion_ctx_local_min_dist_basic": float(local_min_dist_basic),
                "ion_ctx_local_basic_within_4": float(local_basic_within_4),
                "ion_ctx_local_min_dist_amide": float(local_min_dist_amide),
                "ion_ctx_local_amide_within_4": float(local_amide_within_4),
            }
        )

    return pd.DataFrame(rows, index=df.index)


def build_carboxyl_form_feature_frame(
    df: pd.DataFrame,
    candidate_col: str = "candidate_label",
    resolved_col: str = "resolved_groups",
    smiles_col: str = "smiles",
    pka_type_col: str = "pka_type_canonical",
    include_pka_type: bool = False,
) -> pd.DataFrame:
    ion_ctx = build_ionization_context_frame(
        df,
        resolved_col=resolved_col,
        candidate_col=candidate_col,
        smiles_col=smiles_col,
    )

    numeric = pd.DataFrame(
        {
            "resolved_group_count": pd.to_numeric(df.get("resolved_group_count", 0.0), errors="coerce").fillna(0.0),
            "molecule_formal_charge": pd.to_numeric(df.get("molecule_formal_charge", 0.0), errors="coerce").fillna(0.0),
            "neutral_input_risk": pd.to_numeric(df.get("neutral_input_risk", 0.0), errors="coerce").fillna(0.0),
        },
        index=df.index,
    )

    if include_pka_type and pka_type_col in df.columns:
        pka_type_dummies = pd.get_dummies(df[pka_type_col].fillna("unknown"), prefix="pka_type")
    else:
        pka_type_dummies = pd.DataFrame(index=df.index)

    return pd.concat([numeric, ion_ctx, pka_type_dummies], axis=1)
