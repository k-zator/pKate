from rdkit import Chem # type: ignore
from rdkit import RDLogger # type: ignore
from SMARTS_library import BASIC_SMARTS, ACIDIC_SMARTS
from pka_evidence_policy import ionization_policy_role


FUNCTIONAL_GROUP_DETECTOR_SCHEMA_VERSION = "2.5.0"

UNCLASSIFIED_NITROGEN_LABELS = {
    "unclassified_aromatic_ring_N",
    "unclassified_aliphatic_ring_N",
    "unclassified_cationic_ring_N",
    "unclassified_cationic_nonring_N",
    "unclassified_nonring_N",
}
EXPLICIT_CENTER_PRIORITY_FAMILIES = {
    "n_oxide", "amine_n_oxide", "tetrazole_tetrazolate",
}


"""
`LABEL_PRIORITY` provides deterministic tie-breaking when multiple SMARTS match
the same local region. Higher priority values are assigned to more chemically
specific or protonation-state-defining motifs, so overlap resolution preserves
the most informative label for downstream conjugate-pair assignment.
"""


LABEL_PRIORITY = {
    "protonated_alcohol": 392,
    "protonated_thiol": 391,
    "hydrazinium": 389,
    "hydroxylammonium": 388,
    "hydrazine": 232,
    "conjugated_hydrazine": 233,
    "hydroxylamine": 231,
    "phosphoric_acid": 385,
    "phosphate_anion": 384,
    "phosphonic_acid": 383,
    "phosphonate_anion": 382,
    "phosphinic_acid": 381,
    "phosphinate_anion": 380,
    "phosphoramidic_acid": 379,
    "phosphoramidate_anion": 378,
    "carboxylate": 360,
    "carboxylic_acid": 350,
    "sulfonic_acid": 340,
    "sulfonate": 335,
    "phosphate_ester": 320,
    "phenol": 300,
    "phenolate": 298,
    "nitro": 290,
    "thiol": 295,
    "sulfide": 293,
    "amide": 285,
    "amide_NH": 280,
    "amide_primary": 287,
    "amide_secondary": 286,
    "amide_tertiary": 285,
    "ester": 275,
    "lactone": 276,
    "lactam": 286,
    "carbamate": 272,
    "carbamic_ester": 271,
    "carbamic_acid": 270,
    "sulfamic_acid": 268,
    "sulfamate_anion": 267,
    "sulfamate": 266,
    "sulfamoyl_nitrogen": 190,
    "phosphoramidate_nitrogen": 190,
    "quaternary_ammonium": 265,
    "protonated_n_oxide": 264,
    "protonated_amine_n_oxide": 264,
    "n-oxide": 263,
    "amine_n_oxide": 263,
    "tetrazole": 207,
    "tetrazolate": 208,
    "pyridazine": 198,
    "pyridazinium": 199,
    "benzimidazole": 204,
    "benzimidazolium": 205,
    "oxazole": 194,
    "oxazolium": 195,
    "isoxazole": 194,
    "isoxazolium": 195,
    "thiazole": 194,
    "thiazolium": 195,
    "isothiazole": 194,
    "isothiazolium": 195,
    "123triazole": 202,
    "123triazolate": 203,
    "123triazolate_alt2": 203,
    "123triazolate_alt3": 203,
    "124triazole": 202,
    "124triazolium": 203,
    "124triazolium_alt2": 203,
    "124triazolium_alt3": 203,
    "124triazolate": 203,
    "124triazolate_alt2": 203,
    "124triazolate_alt3": 203,
    "indazole": 204,
    "indazolium": 205,
    "indazolium_alt": 205,
    "indazolate": 205,
    "indazolate_alt": 205,
    "tertiary_ammonium": 260,
    "aziridinium": 262,
    "secondary_ammonium": 255,
    "primary_ammonium": 250,
    "aryl_ammonium": 253,
    "tertiary_amine": 230,
    "aziridine": 240,
    "secondary_amine": 225,
    "primary_amine": 220,
    "aniline": 210,
    "imine": 208,
    "iminium": 209,
    "imidazole": 205,
    "imidazolium": 206,
    "imidazolate": 207,
    "imidazolate_alt": 207,
    "pyridine": 200,
    "pyridinium": 201,
    "quinoline": 204,
    "quinolinium": 205,
    "isoquinoline": 204,
    "isoquinolinium": 205,
    "pyrimidine": 198,
    "pyrimidinium": 199,
    "pyrazine": 198,
    "pyrazinium": 199,
    "pyrrole": 196,
    "pyrrolium": 197,
    "pyrrolium_n": 197,
    "pyrazole": 196,
    "pyrazolium": 197,
    "pyrazolium_alt": 197,
    "pyrazolate": 198,
    "pyrazolate_alt": 198,
    "triazine": 198,
    "triazinium": 199,
    "tertiary_alcohol": 175,
    "secondary_alcohol": 170,
    "primary_alcohol": 165,
    "alkoxide": 165,
}


LABEL_FAMILY = {
    "protonated_alcohol": "alcohol_oxonium",
    "protonated_thiol": "thiol_thiolium",
    "hydrazine": "hydrazine_like",
    "conjugated_hydrazine": "hydrazine_like",
    "hydrazinium": "hydrazine_like",
    "hydroxylamine": "hydroxylamine_like",
    "hydroxylammonium": "hydroxylamine_like",
    "phosphoric_acid": "phosphate_oxyacid",
    "phosphate_anion": "phosphate_oxyacid",
    "phosphonic_acid": "phosphonate_oxyacid",
    "phosphonate_anion": "phosphonate_oxyacid",
    "phosphinic_acid": "phosphinate_oxyacid",
    "phosphinate_anion": "phosphinate_oxyacid",
    "phosphoramidic_acid": "phosphoramidate_oxyacid",
    "phosphoramidate_anion": "phosphoramidate_oxyacid",
    "phosphate_ester": "phosphate_ester",
    "carboxylic_acid": "carboxyl",
    "carboxylate": "carboxyl",
    "sulfonic_acid": "sulfonyl_oxyacid",
    "sulfonate": "sulfonyl_oxyacid",
    "carbamic_acid": "carbamic",
    "carbamate": "carbamic",
    "sulfamic_acid": "sulfamic",
    "sulfamate_anion": "sulfamic",
    "sulfamate": "sulfamate_ester",
    "sulfamoyl_nitrogen": "sulfamoyl_nitrogen",
    "phosphoramidate_nitrogen": "phosphoramidate_nitrogen",
    "phenol": "phenol_phenolate",
    "phenolate": "phenol_phenolate",
    "thiol": "thiol_thiolate",
    "sulfide": "thiol_thiolate",  # historical label; SMARTS is thiolate S-
    "imine": "imine_iminium",
    "iminium": "imine_iminium",
    "amidine": "amidine_like",
    "aminoamidine": "amidine_like",
    "hydroxyamidine": "amidine_like",
    "n-hydroxyamidine": "amidine_like",
    "amidinium": "amidine_like",
    "amidenium": "protonated_amide",
    "guanidine": "guanidine_like",
    "guanidinium": "guanidine_like",
    "n-hydroxyguanidine": "guanidine_like",
    "hydroxyguanmidine": "guanidine_like",
    "hydroxyguanidinium": "guanidine_like",
    "ester": "ester",
    "lactone": "ester",
    "amide": "amide",
    "amide_NH": "amide",
    "amide_primary": "amide",
    "amide_secondary": "amide",
    "amide_tertiary": "amide",
    "lactam": "amide",
    "primary_amine": "amine",
    "aziridine": "aziridine",
    "secondary_amine": "amine",
    "tertiary_amine": "amine",
    "aniline": "amine",
    "primary_ammonium": "amine",
    "aziridinium": "aziridine",
    "secondary_ammonium": "amine",
    "tertiary_ammonium": "amine",
    "quaternary_ammonium": "permanent_ammonium",
    "aryl_ammonium": "amine",
    "pyridine": "amine",
    "pyridinium": "amine",
    "quinoline": "amine",
    "quinolinium": "amine",
    "isoquinoline": "amine",
    "isoquinolinium": "amine",
    "pyrimidine": "amine",
    "pyrimidinium": "amine",
    "pyrazine": "amine",
    "pyrazinium": "amine",
    "pyrrole": "amine",
    "pyrrolium": "amine",
    "pyrrolium_n": "amine",
    "pyrazole": "amine",
    "pyrazolium": "amine",
    "pyrazolium_alt": "amine",
    "pyrazolate": "pyrazole_acidity",
    "pyrazolate_alt": "pyrazole_acidity",
    "triazine": "amine",
    "triazinium": "amine",
    "imidazole": "amine",
    "imidazolium": "amine",
    "imidazolate": "imidazole_acidity",
    "imidazolate_alt": "imidazole_acidity",
    "primary_alcohol": "alcohol_alkoxide",
    "secondary_alcohol": "alcohol_alkoxide",
    "tertiary_alcohol": "alcohol_alkoxide",
    "alkoxide": "alcohol_alkoxide",
    "nitro": "nitro",
    "n-oxide": "n_oxide",
    "amine_n_oxide": "amine_n_oxide",
    "protonated_n_oxide": "n_oxide",
    "protonated_amine_n_oxide": "amine_n_oxide",
    "tetrazole": "tetrazole_tetrazolate",
    "tetrazolate": "tetrazole_tetrazolate",
    "n_substituted_tetrazole": "n_substituted_tetrazole",
    "pyridazine": "pyridazine",
    "pyridazinium": "pyridazine",
    "benzimidazole": "benzimidazole",
    "benzimidazolium": "benzimidazole",
    "oxazole": "oxazole",
    "oxazolium": "oxazole",
    "isoxazole": "isoxazole",
    "isoxazolium": "isoxazole",
    "thiazole": "thiazole",
    "thiazolium": "thiazole",
    "isothiazole": "isothiazole",
    "isothiazolium": "isothiazole",
    "123triazole": "123triazole",
    "123triazolate": "123triazole_acidity",
    "123triazolate_alt2": "123triazole_acidity",
    "123triazolate_alt3": "123triazole_acidity",
    "124triazole": "124triazole",
    "124triazolium": "124triazole_basicity",
    "124triazolium_alt2": "124triazole_basicity",
    "124triazolium_alt3": "124triazole_basicity",
    "124triazolate": "124triazole_acidity",
    "124triazolate_alt2": "124triazole_acidity",
    "124triazolate_alt3": "124triazole_acidity",
    "indazole": "indazole",
    "indazolium": "indazole_basicity",
    "indazolium_alt": "indazole_basicity",
    "indazolate": "indazole_acidity",
    "indazolate_alt": "indazole_acidity",
    "123_oxadiazole": "oxadiazole",
    "124_oxadiazole": "oxadiazole",
    "125_oxadiazole": "oxadiazole",
    "134_oxadiazole": "oxadiazole",
    "123_thiadiazole": "thiadiazole",
    "124_thiadiazole": "thiadiazole",
    "125_thiadiazole": "thiadiazole",
    "134_thiadiazole": "thiadiazole",
    **{label: "unclassified_nitrogen" for label in UNCLASSIFIED_NITROGEN_LABELS},
}


HIGH_PROFILE_FAMILIES = {
    "phosphate_oxyacid",
    "phosphonate_oxyacid",
    "phosphinate_oxyacid",
    "phosphoramidate_oxyacid",
    "carboxyl",
    "sulfonyl_oxyacid",
    "carbamic",
    "sulfamic",
    "amide",
    "ester",
}

_INVALID_SMARTS_WARNED = set()
_DEFAULT_COMPILED_SPECS = None


def get_default_pattern_specs():
    specs = []
    for label, smarts in ACIDIC_SMARTS.items():
        specs.append(
            {
                "label": label,
                "smarts": smarts,
                "site_type": "acidic",
                "priority": LABEL_PRIORITY.get(label, 100),
                "family": LABEL_FAMILY.get(label, label),
            }
        )
    for label, smarts in BASIC_SMARTS.items():
        specs.append(
            {
                "label": label,
                "smarts": smarts,
                "site_type": "basic",
                "priority": LABEL_PRIORITY.get(label, 50),
                "family": LABEL_FAMILY.get(label, label),
            }
        )
    return specs


def _mol_from_smarts_safely(smarts, label=""):
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    try:
        patt = Chem.MolFromSmarts(smarts)
    finally:
        RDLogger.EnableLog("rdApp.error")
        RDLogger.EnableLog("rdApp.warning")

    if patt is None and label and label not in _INVALID_SMARTS_WARNED:
        _INVALID_SMARTS_WARNED.add(label)
        print(f"Skipping invalid SMARTS for {label}: {smarts}")
    return patt


def _compile_pattern_specs(pattern_specs):
    compiled_specs = []
    for spec in pattern_specs:
        patt = _mol_from_smarts_safely(spec["smarts"], label=str(spec.get("label", "")))
        if patt is None:
            continue
        compiled_specs.append(
            {
                **spec,
                "pattern": patt,
                "smarts_atoms": patt.GetNumAtoms(),
                "center_query_indices": tuple(
                    atom.GetIdx()
                    for atom in patt.GetAtoms()
                    if atom.GetAtomMapNum() > 0
                ),
            }
        )
    return compiled_specs

def find_sites(mol, smarts_dict, site_type):
    sites = []
    for label, smarts in smarts_dict.items():
        patt = _mol_from_smarts_safely(smarts, label=str(label))
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        for match in matches:
            sites.append({
                "type": site_type,
                "label": label,
                "atoms": match
            })
    return sites


def _unclassified_nitrogen_label(atom):
    if atom.GetFormalCharge() > 0:
        return (
            "unclassified_cationic_ring_N"
            if atom.IsInRing()
            else "unclassified_cationic_nonring_N"
        )
    if atom.GetIsAromatic() and atom.IsInRing():
        return "unclassified_aromatic_ring_N"
    if atom.IsInRing():
        return "unclassified_aliphatic_ring_N"
    return "unclassified_nonring_N"


def _site_atom_set(mol, match, spec):
    atom_set = set(match)
    if spec.get("family") != "tetrazole_tetrazolate" or not match:
        return atom_set
    center = int(match[0])
    for ring in mol.GetRingInfo().AtomRings():
        if center not in ring or len(ring) != 5:
            continue
        atomic_numbers = sorted(mol.GetAtomWithIdx(idx).GetAtomicNum() for idx in ring)
        if atomic_numbers == [6, 7, 7, 7, 7]:
            return set(ring)
    return atom_set


def find_sites_with_metadata(mol, pattern_specs=None, include_unclassified_nitrogen=True):
    global _DEFAULT_COMPILED_SPECS
    if pattern_specs is None:
        if _DEFAULT_COMPILED_SPECS is None:
            _DEFAULT_COMPILED_SPECS = _compile_pattern_specs(get_default_pattern_specs())
        compiled_specs = _DEFAULT_COMPILED_SPECS
    else:
        compiled_specs = _compile_pattern_specs(pattern_specs)
    candidates = []
    for spec in compiled_specs:
        matches = mol.GetSubstructMatches(spec["pattern"])
        for match in matches:
            atom_set = _site_atom_set(mol, match, spec)
            center_atom_set = {
                match[index] for index in spec.get("center_query_indices", ())
            }
            hetero_atom_set = {
                atom_idx
                for atom_idx in match
                if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() not in {1, 6}
            }
            specificity = (len(atom_set) * 10000) + (spec["smarts_atoms"] * 100) + spec["priority"]
            candidates.append(
                {
                    "type": spec["site_type"],
                    "label": spec["label"],
                    "smarts": spec["smarts"],
                    "atoms": tuple(sorted(atom_set)),
                    "atom_set": atom_set,
                    "center_atom_set": center_atom_set,
                    "hetero_atom_set": hetero_atom_set,
                    "priority": spec["priority"],
                    "smarts_atoms": spec["smarts_atoms"],
                    "specificity": specificity,
                    "family": spec.get("family", spec["label"]),
                    "ionization_role": ionization_policy_role(spec["label"]),
                }
            )
    if include_unclassified_nitrogen:
        described_nitrogens = {
            atom_idx
            for site in candidates
            for atom_idx in site["atom_set"]
            if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 7
        }
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 7 or atom.GetIdx() in described_nitrogens:
                continue
            atom_idx = atom.GetIdx()
            label = _unclassified_nitrogen_label(atom)
            candidates.append({
                "type": "unresolved",
                "label": label,
                "smarts": "generated_unclassified_nitrogen_fallback",
                "atoms": (atom_idx,),
                "atom_set": {atom_idx},
                "center_atom_set": set(),
                "hetero_atom_set": {atom_idx},
                "priority": 1,
                "smarts_atoms": 1,
                "specificity": 10101,
                "family": "unclassified_nitrogen",
                "ionization_role": "unclassified_requires_quarantine",
                "is_fallback": True,
            })
    return candidates


def maximal_site_rank_key(site):
    return (
        int(
            site.get("family") in EXPLICIT_CENTER_PRIORITY_FAMILIES
            and bool(site.get("center_atom_set"))
        ),
        len(site.get("atom_set", set())),
        int(site.get("smarts_atoms", 0)),
        int(site.get("priority", 0)),
        int(site.get("specificity", 0)),
        str(site.get("label", "")),
    )


def resolve_overlapping_sites(candidates, overlap_threshold=0.75):
    sorted_candidates = sorted(
        candidates,
        key=maximal_site_rank_key,
        reverse=True,
    )

    accepted = []
    rejected = []
    for cand in sorted_candidates:
        overlap_conflict = False
        overlap_with = None

        for kept in accepted:
            overlap_atoms = cand["atom_set"] & kept["atom_set"]
            if not overlap_atoms:
                continue

            same_family = cand.get("family") == kept.get("family")
            carbon_boundary_only = (
                "hetero_atom_set" in cand
                and "hetero_atom_set" in kept
                and not (
                    overlap_atoms
                    & (set(cand.get("hetero_atom_set", set())) | set(kept.get("hetero_atom_set", set())))
                )
            )
            if carbon_boundary_only:
                continue
            cand_overlap_ratio = len(overlap_atoms) / max(1, len(cand["atom_set"]))
            kept_overlap_ratio = len(overlap_atoms) / max(1, len(kept["atom_set"]))

            is_subset_fragment = cand["atom_set"].issubset(kept["atom_set"]) and maximal_site_rank_key(kept) >= maximal_site_rank_key(cand)
            high_profile_fragment = (
                kept.get("family") in HIGH_PROFILE_FAMILIES
                and cand_overlap_ratio >= 0.5
                and maximal_site_rank_key(kept) > maximal_site_rank_key(cand)
            )

            active_overlap_threshold = 0.01 if same_family else overlap_threshold
            generic_overlap_conflict = (
                cand_overlap_ratio >= active_overlap_threshold
                or kept_overlap_ratio >= active_overlap_threshold
            )

            atom_subset_conflict = (
                cand["atom_set"].issubset(kept["atom_set"])
                or kept["atom_set"].issubset(cand["atom_set"])
            )

            if atom_subset_conflict:
                overlap_conflict = True
                overlap_with = kept
                break

            if is_subset_fragment or high_profile_fragment or generic_overlap_conflict:
                overlap_conflict = True
                overlap_with = kept
                break

        if overlap_conflict:
            rejected.append(
                {
                    **cand,
                    "reject_reason": "overlap",
                    "overlap_with": {
                        "label": overlap_with["label"],
                        "atoms": overlap_with["atoms"],
                    },
                }
            )
            continue

        accepted.append(cand)

    return accepted, rejected


def assign_single_group_label(resolved_sites):
    if not resolved_sites:
        return None

    best = sorted(
        resolved_sites,
        key=maximal_site_rank_key,
        reverse=True,
    )[0]
    return best["label"]

def extract_site_subgraph(mol, atom_idx, radius=2):
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
    atoms = set()
    for bidx in env:
        bond = mol.GetBondWithIdx(bidx)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())

    if not atoms:
        atoms.add(atom_idx)

    return Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms))

# mol = Chem.MolFromSmiles(smiles)
# basic_sites = find_sites(mol, BASIC_SMARTS, "basic")
# acidic_sites = find_sites(mol, ACIDIC_SMARTS, "acidic")
# all_sites = basic_sites + acidic_sites
