import argparse
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd  # type: ignore
from rdkit import Chem  # type: ignore
from rdkit import RDLogger  # type: ignore


EPKA_RE = re.compile(r"^r_epik_pKa_(\d+)$")
FORMAL_CHARGE_RE = re.compile(r"\[[^\]]*[+-][0-9]*[^\]]*\]")


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def has_formal_charge_token(smiles: Optional[str]) -> bool:
    if smiles is None:
        return False
    return bool(FORMAL_CHARGE_RE.search(smiles))


def get_mol_formal_charge(mol: Chem.Mol) -> int:
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def _apply_site_delta(
    mol: Chem.Mol,
    atom_idx: int,
    delta_charge: int,
    adjust_explicit_h: bool,
) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(atom_idx)
    atom.SetFormalCharge(int(atom.GetFormalCharge()) + int(delta_charge))

    if adjust_explicit_h:
        explicit_h = int(atom.GetNumExplicitHs())
        if delta_charge > 0:
            atom.SetNumExplicitHs(explicit_h + 1)
        elif delta_charge < 0 and explicit_h > 0:
            atom.SetNumExplicitHs(explicit_h - 1)

    atom.UpdatePropertyCache(strict=False)
    updated = rw.GetMol()
    return updated, None


def _mol_to_smiles_quick(mol: Chem.Mol) -> Optional[str]:
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def generate_site_state_smiles(
    mol: Chem.Mol,
    atom_idx: int,
    delta_charge: int,
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    last_error: Optional[str] = None
    for adjust_h in (True, False):
        updated, error = _apply_site_delta(mol, atom_idx, delta_charge, adjust_h)
        if updated is not None:
            smiles = _mol_to_smiles_quick(updated)
            if smiles is not None:
                charge = get_mol_formal_charge(updated)
                return smiles, charge, None
            try:
                Chem.SanitizeMol(updated)
                smiles = Chem.MolToSmiles(updated, canonical=True)
                charge = get_mol_formal_charge(updated)
                return smiles, charge, None
            except Exception as exc:
                last_error = str(exc)
        if error is not None:
            last_error = error
    return None, None, last_error


def collect_site_indices(props: Dict[str, Any]) -> List[int]:
    found = []
    for key in props.keys():
        match = EPKA_RE.match(str(key))
        if match:
            found.append(int(match.group(1)))
    return sorted(found)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Epik site-level charged canonical SMILES from raw Epik SDF annotations."
    )
    parser.add_argument(
        "--input",
        default="data/raw/04_mols_chembl_with_pka_v0_filtered.sdf",
        help="Input SDF containing r_epik_pKa_*, i_epik_pKa_atom_*, s_epik_pKa_identifier_*",
    )
    parser.add_argument(
        "--output",
        default="data/processed/chembl_epik_site_charged_smiles.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="Reference pH used to assign dominant protonation state from Epik pKa",
    )
    parser.add_argument(
        "--max-mols",
        type=int,
        default=0,
        help="Optional cap for number of input molecules to process (0 = all)",
    )
    parser.add_argument(
        "--emit-both-states",
        action="store_true",
        help="Also generate both protonated and deprotonated states (slower). Default outputs assigned state only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    suppl = Chem.SDMolSupplier(args.input, removeHs=False)
    rows: List[Dict[str, Any]] = []

    processed_mols = 0
    skipped_mols = 0
    site_rows = 0

    for mol_index, mol in enumerate(suppl, start=1):
        if args.max_mols > 0 and processed_mols >= args.max_mols:
            break
        if mol is None:
            skipped_mols += 1
            continue

        props = mol.GetPropsAsDict()
        site_indices = collect_site_indices(props)
        if not site_indices:
            continue

        processed_mols += 1
        source_smiles = Chem.MolToSmiles(mol, canonical=True)
        source_charge_calc = get_mol_formal_charge(mol)
        source_charge_epik = safe_float(props.get("i_epik_Tot_Q"))
        chembl_id = props.get("chembl_id")

        for site_idx in site_indices:
            pka_value = safe_float(props.get(f"r_epik_pKa_{site_idx}"))
            atom_1based = safe_float(props.get(f"i_epik_pKa_atom_{site_idx}"))
            identifier = props.get(f"s_epik_pKa_identifier_{site_idx}")

            if pka_value is None or atom_1based is None:
                continue

            atom_idx = int(atom_1based) - 1
            if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
                continue

            atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()

            if pka_value > args.ph:
                assigned_state_at_ph = "protonated"
                assigned_delta = +1
            else:
                assigned_state_at_ph = "deprotonated"
                assigned_delta = -1

            assigned_smiles_at_ph, assigned_formal_charge_at_ph, assigned_error = generate_site_state_smiles(
                mol=mol,
                atom_idx=atom_idx,
                delta_charge=assigned_delta,
            )

            protonated_smiles = None
            protonated_charge = None
            protonated_error = None
            deprotonated_smiles = None
            deprotonated_charge = None
            deprotonated_error = None

            if args.emit_both_states:
                protonated_smiles, protonated_charge, protonated_error = generate_site_state_smiles(
                    mol=mol,
                    atom_idx=atom_idx,
                    delta_charge=+1,
                )
                deprotonated_smiles, deprotonated_charge, deprotonated_error = generate_site_state_smiles(
                    mol=mol,
                    atom_idx=atom_idx,
                    delta_charge=-1,
                )

            rows.append(
                {
                    "record_index": mol_index,
                    "chembl_id": chembl_id,
                    "source_canonical_smiles": source_smiles,
                    "source_formal_charge_calc": source_charge_calc,
                    "source_formal_charge_epik": source_charge_epik,
                    "epik_site_index": site_idx,
                    "epik_pka": pka_value,
                    "epik_atom_idx_1based": int(atom_1based),
                    "epik_atom_symbol": atom_symbol,
                    "epik_identifier": identifier,
                    "site_class_by_ph": "basic_if_pka_gt_ph" if pka_value > args.ph else "acidic_if_pka_le_ph",
                    "protonated_smiles": protonated_smiles,
                    "protonated_formal_charge": protonated_charge,
                    "protonated_generation_ok": protonated_smiles is not None,
                    "protonated_has_formal_charge_token": has_formal_charge_token(protonated_smiles),
                    "deprotonated_smiles": deprotonated_smiles,
                    "deprotonated_formal_charge": deprotonated_charge,
                    "deprotonated_generation_ok": deprotonated_smiles is not None,
                    "deprotonated_has_formal_charge_token": has_formal_charge_token(deprotonated_smiles),
                    "assigned_state_at_ph": assigned_state_at_ph,
                    "assigned_smiles_at_ph": assigned_smiles_at_ph,
                    "assigned_formal_charge_at_ph": assigned_formal_charge_at_ph,
                    "assigned_generation_ok": assigned_smiles_at_ph is not None,
                    "assigned_has_formal_charge_token": has_formal_charge_token(assigned_smiles_at_ph),
                    "assigned_generation_error": assigned_error,
                    "protonated_generation_error": protonated_error,
                    "deprotonated_generation_error": deprotonated_error,
                }
            )
            site_rows += 1

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print(f"Processed molecules with Epik sites: {processed_mols}")
    print(f"Skipped unreadable molecules: {skipped_mols}")
    print(f"Generated site rows: {site_rows}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
