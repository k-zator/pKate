import os
from typing import Dict, List

import pandas as pd # type: ignore
from rdkit import Chem # type: ignore

from substructure_match import assign_single_group_label, find_sites_with_metadata, resolve_overlapping_sites


PAIR_TYPE_FAMILIES = {
    "carboxyl",
    "sulfonyl_oxyacid",
    "carbamic",
    "sulfamic",
    "amine",
    "imine_iminium",
}


TOY_CASES = [
    {"case": "acetic_acid", "state": "acid", "smiles": "CC(=O)O"},
    {"case": "acetic_acid", "state": "conj_base", "smiles": "CC(=O)[O-]"},
    {"case": "benzoic_acid", "state": "acid", "smiles": "O=C(O)c1ccccc1"},
    {"case": "benzoic_acid", "state": "conj_base", "smiles": "O=C([O-])c1ccccc1"},
    {"case": "methylamine", "state": "base", "smiles": "CN"},
    {"case": "methylamine", "state": "conj_acid", "smiles": "C[NH3+]"},
    {"case": "dimethylamine", "state": "base", "smiles": "CNC"},
    {"case": "dimethylamine", "state": "conj_acid", "smiles": "C[NH2+]C"},
    {"case": "trimethylamine", "state": "base", "smiles": "CN(C)C"},
    {"case": "trimethylamine", "state": "conj_acid", "smiles": "C[NH+](C)C"},
    {"case": "aniline", "state": "base", "smiles": "Nc1ccccc1"},
    {"case": "aniline", "state": "conj_acid", "smiles": "[NH3+]c1ccccc1"},
    {"case": "imidazole", "state": "base", "smiles": "c1ncc[nH]1"},
    {"case": "imidazole", "state": "conj_acid", "smiles": "c1[nH+]cc[nH]1"},
]


def classify_smiles(smiles: str) -> Dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "final_group_label": None,
            "resolved_groups": "",
            "resolved_group_count": 0,
            "all_candidate_groups": "",
        }

    candidates = find_sites_with_metadata(mol)
    resolved, _ = resolve_overlapping_sites(candidates, overlap_threshold=0.5)
    final_label = assign_single_group_label(resolved)
    final_site = (
        sorted(
            resolved,
            key=lambda x: (x["priority"], x["specificity"], len(x["atom_set"])),
            reverse=True,
        )[0]
        if resolved
        else None
    )
    final_family = final_site.get("family") if final_site else None
    formal_charge = int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    pair_type = final_family in PAIR_TYPE_FAMILIES if final_family else False
    neutral_input_risk = bool(pair_type and final_family == "amine" and formal_charge == 0)

    candidate_labels = sorted({entry["label"] for entry in candidates})
    resolved_labels = sorted({entry["label"] for entry in resolved})

    return {
        "final_group_label": final_label,
        "final_group_family": final_family,
        "group_mode": "pair_type" if pair_type else "individual",
        "molecule_formal_charge": formal_charge,
        "neutral_input_risk": neutral_input_risk,
        "resolved_groups": "|".join(resolved_labels),
        "resolved_group_count": len(resolved),
        "all_candidate_groups": "|".join(candidate_labels),
    }


def run_determinism_passes(smiles: str, n_passes: int = 3) -> bool:
    signatures: List[str] = []
    for _ in range(n_passes):
        out = classify_smiles(smiles)
        signatures.append(
            f"{out['final_group_label']}::{out['resolved_groups']}::{out['resolved_group_count']}"
        )
    return len(set(signatures)) == 1


def main() -> None:
    rows = []
    for case in TOY_CASES:
        classification = classify_smiles(case["smiles"])
        deterministic = run_determinism_passes(case["smiles"], n_passes=3)
        rows.append({**case, **classification, "deterministic_over_3_runs": deterministic})

    df = pd.DataFrame(rows)
    out_dir = "data/processed"
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "toy_protonation_assignment_check.csv")
    df.to_csv(out_csv, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
