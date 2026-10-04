import argparse
import glob
import importlib
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib # type: ignore
import pandas as pd # type: ignore
from rdkit import Chem # type: ignore

matplotlib.use("Agg")
import matplotlib.pyplot as plt # type: ignore

from pka_evidence_policy import reference_prior
from substructure_match import (
    assign_single_group_label,
    find_sites_with_metadata,
    get_default_pattern_specs,
    resolve_overlapping_sites,
)


EPIK_PKA_PATTERN = re.compile(r"^r_epik_pKa_(\d+)$")

CONJUGATE_FAMILY_MAP = {
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
    "primary_alcohol": "alcohol_alkoxide",
    "secondary_alcohol": "alcohol_alkoxide",
    "tertiary_alcohol": "alcohol_alkoxide",
    "alkoxide": "alcohol_alkoxide",
    "carboxylic_acid": "carboxyl",
    "carboxylate": "carboxyl",
    "sulfonic_acid": "sulfonyl_oxyacid",
    "sulfonate": "sulfonyl_oxyacid",
    "carbamic_acid": "carbamic",
    "carbamate": "carbamic",
    "sulfamic_acid": "sulfamic",
    "sulfamate_anion": "sulfamic",
    "sulfamate": "sulfamate_ester",
    "phenol": "phenol_phenolate",
    "phenolate": "phenol_phenolate",
    "thiol": "thiol_thiolate",
    "sulfide": "thiol_thiolate",  # BASIC_SMARTS "sulfide" is really thiolate (S⁻)
    "iminium": "imine_iminium",
    "imine": "imine_iminium",
    "primary_amine": "amine",
    "secondary_amine": "amine",
    "aziridine": "aziridine",
    "tertiary_amine": "amine",
    "aniline": "amine",
    "primary_ammonium": "amine",
    "secondary_ammonium": "amine",
    "aziridinium": "aziridine",
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
    "triazine": "amine",
    "triazinium": "amine",
    "imidazole": "amine",
    "imidazolium": "amine",
    "pyrazole_basicity": "pyrazole_basicity",
    "pyrazole_acidity": "pyrazole_acidity",
    "pyrazolate": "pyrazole_acidity",
    "pyrazolate_alt": "pyrazole_acidity",
    "imidazole_basicity": "imidazole_basicity",
    "imidazole_acidity": "imidazole_acidity",
    "imidazolate": "imidazole_acidity",
    "imidazolate_alt": "imidazole_acidity",
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
    "tetrazole": "tetrazole_tetrazolate",
    "tetrazolate": "tetrazole_tetrazolate",
    "n-oxide": "n_oxide",
    "protonated_n_oxide": "n_oxide",
    "amine_n_oxide": "amine_n_oxide",
    "protonated_amine_n_oxide": "amine_n_oxide",
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
    # Transition-specific aliases are deliberately separate.  The neutral
    # 1,2,4-triazole and indazole structures each participate in two aqueous
    # equilibria, so assigning either detector label to one generic family
    # would conflate pKaH with N-H acidity.
    "123triazole_acidity": "123triazole_acidity",
    "123triazolate": "123triazole_acidity",
    "123triazolate_alt2": "123triazole_acidity",
    "123triazolate_alt3": "123triazole_acidity",
    "124triazole_basicity": "124triazole_basicity",
    "124triazolium": "124triazole_basicity",
    "124triazolium_alt2": "124triazole_basicity",
    "124triazolium_alt3": "124triazole_basicity",
    "124triazole_acidity": "124triazole_acidity",
    "124triazolate": "124triazole_acidity",
    "124triazolate_alt2": "124triazole_acidity",
    "124triazolate_alt3": "124triazole_acidity",
    "indazole_basicity": "indazole_basicity",
    "indazolium": "indazole_basicity",
    "indazolium_alt": "indazole_basicity",
    "indazole_acidity": "indazole_acidity",
    "indazolate": "indazole_acidity",
    "indazolate_alt": "indazole_acidity",
}

CONJUGATE_MAP = {
    "protonated_alcohol": "primary_alcohol",
    "protonated_thiol": "thiol",
    "hydrazine": "hydrazinium",
    "conjugated_hydrazine": "hydrazinium",
    "hydroxylamine": "hydroxylammonium",
    "phosphoric_acid": "phosphate_anion",
    "phosphonic_acid": "phosphonate_anion",
    "phosphinic_acid": "phosphinate_anion",
    "phosphoramidic_acid": "phosphoramidate_anion",
    "primary_alcohol": "alkoxide",
    "secondary_alcohol": "alkoxide",
    "tertiary_alcohol": "alkoxide",
    "carboxylic_acid": "carboxylate",
    "sulfonic_acid": "sulfonate",
    "iminium": "imine",
    "carbamic_acid": "carbamate",
    "sulfamic_acid": "sulfamate_anion",
    "phenol": "phenolate",
    "thiol": "sulfide",   # sulfide label = thiolate (S⁻)
    "primary_amine": "primary_ammonium",
    "secondary_amine": "secondary_ammonium",
    "aziridine": "aziridinium",
    "tertiary_amine": "tertiary_ammonium",
    "aniline": "aryl_ammonium",
    "pyridine": "pyridinium",
    "quinoline": "quinolinium",
    "isoquinoline": "isoquinolinium",
    "pyrimidine": "pyrimidinium",
    "pyrazine": "pyrazinium",
    "pyrrole": "pyrrolium",
    "pyrazole": "pyrazolium",
    "pyrazolium_alt": "pyrazole",
    "triazine": "triazinium",
    "imidazole": "imidazolium",
    "pyrazole_basicity": "pyrazolium",
    "pyrazole_acidity": "pyrazolate",
    "imidazole_basicity": "imidazolium",
    "imidazole_acidity": "imidazolate",
    "amidine": "amidinium",
    "aminoamidine": "amidinium",
    "hydroxyamidine": "amidinium",
    "n-hydroxyamidine": "amidinium",
    "guanidine": "guanidinium",
    "n-hydroxyguanidine": "hydroxyguanidinium",
    "hydroxyguanmidine": "hydroxyguanidinium",
    "tetrazole": "tetrazolate",
    "n-oxide": "protonated_n_oxide",
    "amine_n_oxide": "protonated_amine_n_oxide",
    "pyridazine": "pyridazinium",
    "benzimidazole": "benzimidazolium",
    "oxazole": "oxazolium",
    "isoxazole": "isoxazolium",
    "thiazole": "thiazolium",
    "isothiazole": "isothiazolium",
    "123triazole_acidity": "123triazolate",
    "124triazole_basicity": "124triazolium",
    "124triazole_acidity": "124triazolate",
    "indazole_basicity": "indazolium",
    "indazole_acidity": "indazolate",
}

CANONICAL_FAMILY_LABEL = {
    "alcohol_oxonium": "protonated_alcohol",
    "thiol_thiolium": "protonated_thiol",
    "hydrazine_like": "hydrazine",
    "hydroxylamine_like": "hydroxylamine",
    "phosphate_oxyacid": "phosphoric_acid",
    "phosphonate_oxyacid": "phosphonic_acid",
    "phosphinate_oxyacid": "phosphinic_acid",
    "phosphoramidate_oxyacid": "phosphoramidic_acid",
    "alcohol_alkoxide": "primary_alcohol",
    "carboxyl": "carboxylic_acid",
    "sulfonyl_oxyacid": "sulfonic_acid",
    "imine_iminium": "imine",
    "carbamic": "carbamic_acid",
    "sulfamic": "sulfamic_acid",
    "phenol_phenolate": "phenol",
    "thiol_thiolate": "thiol",
    "amine": "primary_amine",
    "aziridine": "aziridine",
    "amidine_like": "amidine",
    "guanidine_like": "guanidine",
    "tetrazole_tetrazolate": "tetrazole",
    "n_oxide": "n-oxide",
    "amine_n_oxide": "amine_n_oxide",
    "pyridazine": "pyridazine",
    "benzimidazole": "benzimidazole",
    "oxazole": "oxazole",
    "isoxazole": "isoxazole",
    "thiazole": "thiazole",
    "isothiazole": "isothiazole",
    "pyrazole_basicity": "pyrazole_basicity",
    "pyrazole_acidity": "pyrazole_acidity",
    "imidazole_basicity": "imidazole_basicity",
    "imidazole_acidity": "imidazole_acidity",
    "123triazole_acidity": "123triazole_acidity",
    "124triazole_basicity": "124triazole_basicity",
    "124triazole_acidity": "124triazole_acidity",
    "indazole_basicity": "indazole_basicity",
    "indazole_acidity": "indazole_acidity",
}

ACIDIC_FAMILIES = {
    "alcohol_oxonium",
    "thiol_thiolium",
    "carboxyl",
    "sulfonyl_oxyacid",
    "carbamic",
    "sulfamic",
    "phenol_phenolate",
    "thiol_thiolate",
    "phosphate_oxyacid",
    "phosphonate_oxyacid",
    "phosphinate_oxyacid",
    "phosphoramidate_oxyacid",
    "alcohol_alkoxide",
    "tetrazole_tetrazolate",
    "123triazole_acidity",
    "124triazole_acidity",
    "indazole_acidity",
    "pyrazole_acidity",
    "imidazole_acidity",
}
BASIC_FAMILIES = {
    "amine", "aziridine", "imine_iminium", "amidine_like", "guanidine_like",
    "hydrazine_like", "hydroxylamine_like",
    "n_oxide", "amine_n_oxide", "pyridazine", "benzimidazole",
    "oxazole", "isoxazole", "thiazole", "isothiazole",
    "124triazole_basicity", "indazole_basicity",
    "pyrazole_basicity", "imidazole_basicity",
}

# These families require transition-level handling.  A neutral amphoteric
# azole is the middle member of a serial cation <-> neutral <-> anion
# coordinate; the supported 1,2,3-triazole edge currently covers N-H acidity.
SERIAL_AZOLE_PAIR_FAMILIES = frozenset({
    "123triazole_acidity",
    "124triazole_basicity",
    "124triazole_acidity",
    "indazole_basicity",
    "indazole_acidity",
    "pyrazole_basicity",
    "pyrazole_acidity",
    "imidazole_basicity",
    "imidazole_acidity",
})

PAIR_TYPE_FAMILY_FORMS = {
    "alcohol_oxonium": {
        "acid_form": {"protonated_alcohol"},
        "base_form": set(),
    },
    "thiol_thiolium": {
        "acid_form": {"protonated_thiol"},
        "base_form": set(),
    },
    "hydrazine_like": {
        "acid_form": {"hydrazinium"},
        "base_form": {"hydrazine", "conjugated_hydrazine"},
    },
    "hydroxylamine_like": {
        "acid_form": {"hydroxylammonium"},
        "base_form": {"hydroxylamine"},
    },
    "phosphate_oxyacid": {
        "acid_form": {"phosphoric_acid"},
        "base_form": {"phosphate_anion"},
    },
    "phosphonate_oxyacid": {
        "acid_form": {"phosphonic_acid"},
        "base_form": {"phosphonate_anion"},
    },
    "phosphinate_oxyacid": {
        "acid_form": {"phosphinic_acid"},
        "base_form": {"phosphinate_anion"},
    },
    "phosphoramidate_oxyacid": {
        "acid_form": {"phosphoramidic_acid"},
        "base_form": {"phosphoramidate_anion"},
    },
    "alcohol_alkoxide": {
        "acid_form": {"primary_alcohol", "secondary_alcohol", "tertiary_alcohol"},
        "base_form": {"alkoxide"},
    },
    "carboxyl": {
        "acid_form": {"carboxylic_acid"},
        "base_form": {"carboxylate"},
    },
    "sulfonyl_oxyacid": {
        "acid_form": {"sulfonic_acid"},
        "base_form": {"sulfonate"},
    },
    "carbamic": {
        "acid_form": {"carbamic_acid"},
        "base_form": {"carbamate"},
    },
    "sulfamic": {
        "acid_form": {"sulfamic_acid"},
        "base_form": {"sulfamate_anion"},
    },
    "amine": {
        "acid_form": {
            "primary_ammonium",
            "secondary_ammonium",
            "tertiary_ammonium",
            "aryl_ammonium",
            "pyridinium",
            "quinolinium",
            "isoquinolinium",
            "pyrimidinium",
            "pyrazinium",
            "pyrrolium",
            "pyrazolium",
            "pyrazolium_alt",
            "triazinium",
            "imidazolium",
        },
        "base_form": {
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
            "pyrazole",
            "triazine",
            "imidazole",
        },
    },
    "aziridine": {
        "acid_form": {"aziridinium"},
        "base_form": {"aziridine"},
    },
    "imine_iminium": {
        "acid_form": {"iminium"},
        "base_form": {"imine"},
    },
    "amidine_like": {
        "acid_form": {"amidinium"},
        "base_form": {"amidine", "aminoamidine", "hydroxyamidine", "n-hydroxyamidine"},
    },
    "guanidine_like": {
        "acid_form": {"guanidinium", "hydroxyguanidinium"},
        "base_form": {"guanidine", "n-hydroxyguanidine", "hydroxyguanmidine"},
    },
    "phenol_phenolate": {
        "acid_form": {"phenol"},
        "base_form": {"phenolate"},
    },
    "thiol_thiolate": {
        "acid_form": {"thiol"},
        "base_form": {"sulfide"},  # sulfide label = thiolate (S⁻)
    },
    "tetrazole_tetrazolate": {
        "acid_form": {"tetrazole"},
        "base_form": {"tetrazolate"},
    },
    "n_oxide": {
        "acid_form": {"protonated_n_oxide"},
        "base_form": {"n-oxide"},
    },
    "amine_n_oxide": {
        "acid_form": {"protonated_amine_n_oxide"},
        "base_form": {"amine_n_oxide"},
    },
    "pyridazine": {
        "acid_form": {"pyridazinium"},
        "base_form": {"pyridazine"},
    },
    "benzimidazole": {
        "acid_form": {"benzimidazolium"},
        "base_form": {"benzimidazole"},
    },
    "oxazole": {
        "acid_form": {"oxazolium"},
        "base_form": {"oxazole"},
    },
    "isoxazole": {
        "acid_form": {"isoxazolium"},
        "base_form": {"isoxazole"},
    },
    "thiazole": {
        "acid_form": {"thiazolium"},
        "base_form": {"thiazole"},
    },
    "isothiazole": {
        "acid_form": {"isothiazolium"},
        "base_form": {"isothiazole"},
    },
    "pyrazole_basicity": {
        "acid_form": {"pyrazolium"},
        "base_form": {"pyrazole_basicity"},
    },
    "pyrazole_acidity": {
        "acid_form": {"pyrazole_acidity"},
        "base_form": {"pyrazolate", "pyrazolate_alt"},
    },
    "imidazole_basicity": {
        "acid_form": {"imidazolium"},
        "base_form": {"imidazole_basicity"},
    },
    "imidazole_acidity": {
        "acid_form": {"imidazole_acidity"},
        "base_form": {"imidazolate", "imidazolate_alt"},
    },
    "123triazole_acidity": {
        "acid_form": {"123triazole_acidity"},
        "base_form": {"123triazolate"},
    },
    "124triazole_basicity": {
        "acid_form": {"124triazolium"},
        "base_form": {"124triazole_basicity"},
    },
    "124triazole_acidity": {
        "acid_form": {"124triazole_acidity"},
        "base_form": {"124triazolate"},
    },
    "indazole_basicity": {
        "acid_form": {"indazolium"},
        "base_form": {"indazole_basicity"},
    },
    "indazole_acidity": {
        "acid_form": {"indazole_acidity"},
        "base_form": {"indazolate"},
    },
}

# Detector labels describe structures; transition aliases describe a single
# thermodynamic edge.  A neutral amphoteric structure expands to two aliases
# that share one coupled protonation coordinate.  Charged forms expose only
# the edge on which that form lies.
DETECTOR_TRANSITION_VARIANTS = {
    "pyrazole": (
        ("pyrazole_basicity", "pyrazole_basicity", "base_form", "pyrazole"),
        ("pyrazole_acidity", "pyrazole_acidity", "acid_form", "pyrazole"),
    ),
    "pyrazolium": (
        ("pyrazolium", "pyrazole_basicity", "acid_form", "pyrazole"),
    ),
    "pyrazolium_alt": (
        ("pyrazolium", "pyrazole_basicity", "acid_form", "pyrazole"),
    ),
    "pyrazolate": (
        ("pyrazolate", "pyrazole_acidity", "base_form", "pyrazole"),
    ),
    "pyrazolate_alt": (
        ("pyrazolate", "pyrazole_acidity", "base_form", "pyrazole"),
    ),
    "imidazole": (
        ("imidazole_basicity", "imidazole_basicity", "base_form", "imidazole"),
        ("imidazole_acidity", "imidazole_acidity", "acid_form", "imidazole"),
    ),
    "imidazolium": (
        ("imidazolium", "imidazole_basicity", "acid_form", "imidazole"),
    ),
    "imidazolate": (
        ("imidazolate", "imidazole_acidity", "base_form", "imidazole"),
    ),
    "imidazolate_alt": (
        ("imidazolate", "imidazole_acidity", "base_form", "imidazole"),
    ),
    "123triazole": (
        ("123triazole_acidity", "123triazole_acidity", "acid_form", "123triazole"),
    ),
    "123triazolate": (
        ("123triazolate", "123triazole_acidity", "base_form", "123triazole"),
    ),
    "123triazolate_alt2": (
        ("123triazolate", "123triazole_acidity", "base_form", "123triazole"),
    ),
    "123triazolate_alt3": (
        ("123triazolate", "123triazole_acidity", "base_form", "123triazole"),
    ),
    "124triazole": (
        ("124triazole_basicity", "124triazole_basicity", "base_form", "124triazole"),
        ("124triazole_acidity", "124triazole_acidity", "acid_form", "124triazole"),
    ),
    "124triazolium": (
        ("124triazolium", "124triazole_basicity", "acid_form", "124triazole"),
    ),
    "124triazolium_alt2": (
        ("124triazolium", "124triazole_basicity", "acid_form", "124triazole"),
    ),
    "124triazolium_alt3": (
        ("124triazolium", "124triazole_basicity", "acid_form", "124triazole"),
    ),
    "124triazolate": (
        ("124triazolate", "124triazole_acidity", "base_form", "124triazole"),
    ),
    "124triazolate_alt2": (
        ("124triazolate", "124triazole_acidity", "base_form", "124triazole"),
    ),
    "124triazolate_alt3": (
        ("124triazolate", "124triazole_acidity", "base_form", "124triazole"),
    ),
    "indazole": (
        ("indazole_basicity", "indazole_basicity", "base_form", "indazole"),
        ("indazole_acidity", "indazole_acidity", "acid_form", "indazole"),
    ),
    "indazolium": (
        ("indazolium", "indazole_basicity", "acid_form", "indazole"),
    ),
    "indazolium_alt": (
        ("indazolium", "indazole_basicity", "acid_form", "indazole"),
    ),
    "indazolate": (
        ("indazolate", "indazole_acidity", "base_form", "indazole"),
    ),
    "indazolate_alt": (
        ("indazolate", "indazole_acidity", "base_form", "indazole"),
    ),
}

INDIVIDUAL_PKA_WINDOWS = {
    "ketone": (14.0, 30.0),
    "aldehyde": (12.0, 26.0),
    "primary_alcohol": (12.0, 20.0),
    "secondary_alcohol": (12.0, 20.0),
    "tertiary_alcohol": (12.0, 20.0),
    "ester_aliphatic": (18.0, 30.0),
    "ester_aromatic": (18.0, 30.0),
    "amide_aliphatic": (12.0, 22.0),
    "amide_aromatic": (12.0, 22.0),
}

MIN_REASONABLE_PKA = -5.0
MAX_REASONABLE_PKA = 40.0

SITE_OVERRIDE_KEY_COLUMNS = [
    "source_file",
    "record_index",
    "smiles",
    "pka_value",
    "pka_source_method",
    "pka_type_canonical",
    "atom_index_raw",
]


TRAINING_LABEL_FALLBACKS = {
    "7-azaindole": "indole_like_fused_heteroaromatic",
    "124triazine": "azine_like_weak_base",
    "amide_anhydride": "amide_like",
    "amide_primary": "amide_like",
    "amide_secondary": "amide_like",
    "amide_tertiary": "amide_like",
    "furan": "neutral_heterocycle_like",
    "hydrazone": "imine_like_neutral",
    "hydroxamic_acid": "amide_like",
    "isoNindole": "indole_like_fused_heteroaromatic",
    "isoquinoline": "quinoline",
    "isothiazole": "azole_like_heteroaromatic",
    "isoxazolidine": "azole_like_heteroaromatic",
    "lactone": "ester_like",
    "nitroso": "imine_like_neutral",
    "nn-indazole": "indole_like_fused_heteroaromatic",
    "oxazole": "azole_like_heteroaromatic",
    "percaboxylic_acid": "peroxide_like",
    "peroxide": "peroxide_like",
    "primary_alcohol": "alcohol_like",
    "pyridazinone": "azine_like_weak_base",
    # Pyrrole carbon protonation and pyrazole N protonation have very
    # different reference pKas.  Sparse data must not erase that distinction;
    # Stage 1 now shrinks each label toward an explicit chemistry prior.
    "pyrrole": "pyrrole",
    "pyrazine": "azine_like_weak_base",
    "pyrazole": "weak_azole_like",
    "pyrazolidine": "weak_azole_like",
    "sulfinic_acid": "sulfur_oxyacid_like",
    "sulfonate": "sulfur_oxyacid_like",
    "sulfoxide": "sulfur_neutral_like",
    "Sindole": "indole_like_fused_heteroaromatic",
    "thiazole": "azole_like_heteroaromatic",
    "thioamideamide": "thioamide_like",
    "thioamideamide_anhydride": "thioamide_like",
    "thioamidethio": "thioamide_like",
    "thioester": "ester_like",
    "triazine": "azine_like_weak_base",
}


TRAINING_LABEL_FALLBACK_REASON = {
    "7-azaindole": "count < 5 and fused aza-indole chemistry is too sparse as a standalone leaf",
    "124triazine": "count < 5 and very sparse azine subtype; collapse to a broader weak azine bucket",
    "amide_anhydride": "amide subtype too sparse; use shared amide bucket",
    "amide_primary": "amide subtype too sparse; use shared amide bucket",
    "amide_secondary": "amide subtype too sparse; use shared amide bucket",
    "amide_tertiary": "amide subtype too sparse; use shared amide bucket",
    "furan": "neutral heterocycle leaf is too sparse for a standalone regressor",
    "hydrazone": "too sparse and closest to imine-like neutral behavior in this dataset",
    "hydroxamic_acid": "count < 5 and chemically closest to amide-like acidity here",
    "isoNindole": "very sparse fused indole-like heteroaromatic; collapse to shared fused bucket",
    "isoquinoline": "count < 5 and best-covered neighboring scaffold is quinoline",
    "isothiazole": "sparse five-membered heteroaromatic; use shared azole-like bucket",
    "isoxazolidine": "sparse five-membered heteroaromatic; use shared azole-like bucket",
    "lactone": "single-example cyclic ester; use ester-like fallback",
    "nitroso": "very sparse neutral N/O multiple-bond leaf; use imine-like neutral fallback",
    "nn-indazole": "sparse fused indazole-like heteroaromatic; collapse to shared fused bucket",
    "oxazole": "sparse five-membered heteroaromatic; use shared azole-like bucket",
    "percaboxylic_acid": "too sparse for its own regressor; use peroxide/peracid-like fallback",
    "peroxide": "too sparse for standalone regression; use shared peroxide-like bucket",
    "primary_alcohol": "count < 5 and neutral alcohol acidity is too sparse to estimate separately",
    "pyridazinone": "very sparse weak azine-like base; collapse to broader azine bucket",
    "pyrrole": "kept distinct because its carbon-protonation scale is incompatible with pyrazole",
    "pyrazine": "count < 20 and weak azine subtype is better pooled with neighboring azines",
    "pyrazole": "count < 20 and weak azole-like subtype is better pooled",
    "pyrazolidine": "very sparse weak azole-like subtype; collapse to broader weak azole bucket",
    "sulfinic_acid": "count < 20 and sulfur oxyacid subtype is too sparse standalone",
    "sulfonate": "single-example conjugate base; use sulfur oxyacid fallback",
    "sulfoxide": "single-example neutral sulfur oxide; use broader sulfur-neutral fallback",
    "Sindole": "very sparse fused sulfur-indole analog; collapse to shared fused bucket",
    "thiazole": "very sparse five-membered heteroaromatic; use shared azole-like bucket",
    "thioamideamide": "very sparse thioamide analog; use thioamide-like fallback",
    "thioamideamide_anhydride": "count < 20 and better pooled with other thioamide-like labels",
    "thioamidethio": "count < 20 and better pooled with other thioamide-like labels",
    "thioester": "very sparse sulfur ester analog; use ester-like fallback",
    "triazine": "very sparse weak azine subtype; collapse to broader azine bucket",
}


def training_group_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    label_text = str(label)
    return TRAINING_LABEL_FALLBACKS.get(label_text, label_text)


def training_group_reason(label: Optional[str]) -> str:
    if label is None:
        return ""
    label_text = str(label)
    return TRAINING_LABEL_FALLBACK_REASON.get(label_text, "kept as standalone training label")


def _label_type_lookup() -> Dict[str, str]:
    mapping = {}
    for spec in get_default_pattern_specs():
        mapping[spec["label"]] = spec["site_type"]
    return mapping


LABEL_TYPE_MAP = _label_type_lookup()


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iter_sdf_mols(sdf_path: str) -> Iterable[Tuple[int, Chem.Mol]]:
    with open(sdf_path, "rb") as handle:
        supplier = Chem.ForwardSDMolSupplier(handle, removeHs=False)
        for record_idx, mol in enumerate(supplier):
            if mol is None:
                continue
            yield record_idx, mol


def _extract_epik_measurements(mol: Chem.Mol, prop_names: List[str]) -> List[Dict]:
    records: List[Dict] = []
    for prop in prop_names:
        match = EPIK_PKA_PATTERN.match(prop)
        if not match:
            continue

        suffix = match.group(1)
        pka_value = _safe_float(mol.GetProp(prop))
        if pka_value is None:
            continue

        atom_key = f"i_epik_pKa_atom_{suffix}"
        ident_key = f"s_epik_pKa_identifier_{suffix}"

        atom_idx_raw = _safe_int(mol.GetProp(atom_key)) if mol.HasProp(atom_key) else None
        site_identifier = mol.GetProp(ident_key) if mol.HasProp(ident_key) else None

        pka_type_raw = None
        if site_identifier:
            lower_ident = site_identifier.lower()
            if "acid" in lower_ident:
                pka_type_raw = "acidic"
            elif "base" in lower_ident:
                pka_type_raw = "basic"

        records.append(
            {
                "pka_value": pka_value,
                "pka_source_method": "epik",
                "pka_type_raw": pka_type_raw,
                "atom_index_raw": atom_idx_raw,
                "site_identifier_raw": site_identifier,
            }
        )

    return records


def _extract_standard_measurements(mol: Chem.Mol) -> List[Dict]:
    records: List[Dict] = []

    if mol.HasProp("pKa"):
        value = _safe_float(mol.GetProp("pKa"))
        if value is not None:
            pka_type_raw = None
            for type_key in ("ionization_type", "type", "pKa_type"):
                if mol.HasProp(type_key):
                    pka_type_raw = mol.GetProp(type_key)
                    break

            records.append(
                {
                    "pka_value": value,
                    "pka_source_method": "experimental",
                    "pka_type_raw": pka_type_raw,
                    "atom_index_raw": None,
                    "site_identifier_raw": None,
                }
            )

    if mol.HasProp("marvin_pKa"):
        value = _safe_float(mol.GetProp("marvin_pKa"))
        if value is not None:
            atom_idx_raw = _safe_int(mol.GetProp("marvin_atom")) if mol.HasProp("marvin_atom") else None
            pka_type_raw = None
            for type_key in ("marvin_pKa_type", "ionization_type", "type"):
                if mol.HasProp(type_key):
                    pka_type_raw = mol.GetProp(type_key)
                    break

            records.append(
                {
                    "pka_value": value,
                    "pka_source_method": "marvin",
                    "pka_type_raw": pka_type_raw,
                    "atom_index_raw": atom_idx_raw,
                    "site_identifier_raw": None,
                }
            )

    return records


def extract_measurements(mol: Chem.Mol, allow_epik: bool = False) -> List[Dict]:
    prop_names = list(mol.GetPropNames())

    if allow_epik:
        epik_records = _extract_epik_measurements(mol, prop_names)
        if epik_records:
            return epik_records

    return _extract_standard_measurements(mol)


def canonicalize_pka_type(pka_type_raw: Optional[str]) -> Optional[str]:
    if not pka_type_raw:
        return None
    value = pka_type_raw.strip().lower()
    if "acid" in value:
        return "acidic"
    if "base" in value:
        return "basic"
    return value


def normalize_conjugate_family(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    return CONJUGATE_FAMILY_MAP.get(label, label)


def canonical_family_label(label: Optional[str], family: Optional[str] = None) -> Optional[str]:
    if label is None and family is None:
        return None
    family_value = family if family is not None else normalize_conjugate_family(label)
    if family_value is None:
        return label
    return CANONICAL_FAMILY_LABEL.get(family_value, label)


def assignment_status(resolved_group_count: int) -> str:
    if resolved_group_count == 0:
        return "no_group"
    if resolved_group_count == 1:
        return "single_group"
    return "multi_group"


def _select_best_site(resolved_sites: List[Dict], preferred_type: Optional[str] = None) -> Optional[Dict]:
    if not resolved_sites:
        return None
    candidate_sites = resolved_sites
    if preferred_type in {"acidic", "basic"}:
        typed_sites = [site for site in resolved_sites if site.get("type") == preferred_type]
        if typed_sites:
            candidate_sites = typed_sites
    best = sorted(
        candidate_sites,
        key=lambda x: (
            len(x["atom_set"]),
            x["specificity"],
            x["priority"],
        ),
        reverse=True,
    )[0]
    return best


def _site_rank_tuple(site: Dict) -> Tuple[int, int, int]:
    return (
        len(site.get("atom_set", set())),
        int(site.get("specificity", 0)),
        int(site.get("priority", 0)),
    )


def select_pair_first_site(
    resolved_sites: List[Dict],
    preferred_type: Optional[str],
    pka_value: Optional[float] = None,
) -> Tuple[Optional[Dict], Dict[str, object]]:
    if not resolved_sites:
        return None, {
            "pair_assignment_confidence": "none",
            "pair_candidate_count": 0,
            "pair_candidate_labels": "",
            "pair_selection_reason": "no_resolved_sites",
        }

    pair_candidates = [
        transition
        for site in resolved_sites
        for transition in expand_pair_site_transitions(site, preferred_type=preferred_type)
    ]

    if not pair_candidates:
        fallback = _select_best_site(resolved_sites, preferred_type=preferred_type)
        return fallback, {
            "pair_assignment_confidence": "none",
            "pair_candidate_count": 0,
            "pair_candidate_labels": "",
            "pair_selection_reason": "no_pair_family_match",
        }

    if preferred_type not in {"acidic", "basic"} and all(
        site.get("pair_family") in SERIAL_AZOLE_PAIR_FAMILIES
        for site in pair_candidates
    ):
        selected, reason = select_serial_azole_transition_by_reference(
            pair_candidates,
            pka_value,
        )
        metadata = {
            "pair_assignment_confidence": "medium" if selected is not None else "none",
            "pair_candidate_count": len(pair_candidates),
            "pair_candidate_labels": "|".join(sorted({
                site.get("label") for site in pair_candidates if site.get("label")
            })),
            "pair_selection_reason": reason,
        }
        return selected, metadata

    ranked = sorted(
        pair_candidates,
        key=_site_rank_tuple,
        reverse=True,
    )
    selected = ranked[0]

    confidence = "high"
    if len(ranked) > 1 and _site_rank_tuple(ranked[0]) == _site_rank_tuple(ranked[1]):
        confidence = "ambiguous"

    return selected, {
        "pair_assignment_confidence": confidence,
        "pair_candidate_count": len(pair_candidates),
        "pair_candidate_labels": "|".join(sorted({site.get("label") for site in pair_candidates if site.get("label")})),
        "pair_selection_reason": "pair_first",
    }


def _sites_matching_atom_index(
    resolved_sites: List[Dict],
    atom_index_raw: Optional[int],
) -> Tuple[List[Dict], Dict[str, object]]:
    if atom_index_raw is None:
        return [], {
            "atom_site_match": False,
            "atom_index_matched": None,
            "atom_index_mode": "none",
            "atom_candidate_count": 0,
            "atom_candidate_labels": "",
        }

    index_modes: List[Tuple[str, int]] = [("zero_based", atom_index_raw)]
    if atom_index_raw > 0:
        index_modes.append(("one_based_minus1", atom_index_raw - 1))

    for mode, atom_index in index_modes:
        matched_sites = [
            site for site in resolved_sites if atom_index in site.get("atom_set", set())
        ]
        if matched_sites:
            labels = sorted({site.get("label") for site in matched_sites if site.get("label")})
            return matched_sites, {
                "atom_site_match": True,
                "atom_index_matched": atom_index,
                "atom_index_mode": mode,
                "atom_candidate_count": len(matched_sites),
                "atom_candidate_labels": "|".join(labels),
            }

    return [], {
        "atom_site_match": False,
        "atom_index_matched": None,
        "atom_index_mode": "unmatched",
        "atom_candidate_count": 0,
        "atom_candidate_labels": "",
    }


def _site_has_aromatic_context(mol: Chem.Mol, atom_set: set) -> bool:
    for atom_idx in atom_set:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetIsAromatic():
            return True
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIsAromatic():
                return True
    return False


def _molecule_formal_charge(mol: Chem.Mol) -> int:
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def classify_pair_type(label: Optional[str], family: Optional[str]) -> Tuple[str, Optional[str]]:
    if label is None or family is None:
        return "individual", None

    family_forms = PAIR_TYPE_FAMILY_FORMS.get(family)
    if family_forms is None:
        return "individual", None

    if label in family_forms.get("acid_form", set()):
        return "pair_type", "acid_form"
    if label in family_forms.get("base_form", set()):
        return "pair_type", "base_form"
    return "pair_type", "unknown_form"


def expand_pair_site_transitions(
    site: Dict,
    preferred_type: Optional[str] = None,
) -> List[Dict]:
    """Expand one structural detector site into unambiguous pKa transitions.

    Most detector labels still yield one binary transition.  Amphoteric azoles
    yield an acidic and a basic transition that share ``coupling_group``; the
    network enumerator uses that marker to create three serial charge levels
    instead of two independent Boolean sites.
    """
    detector_label = str(site.get("label", ""))
    variants = DETECTOR_TRANSITION_VARIANTS.get(detector_label)
    if variants is None:
        family = str(site.get("family") or normalize_conjugate_family(detector_label) or "")
        mode, member_form = classify_pair_type(detector_label, family)
        if mode != "pair_type" or member_form not in {"acid_form", "base_form"}:
            return []
        return [{
            **site,
            "detector_label": detector_label,
            "pair_family": family,
            "pair_member_form": member_form,
            "coupling_group": family,
            "acid_level": 1,
            "base_level": 0,
        }]

    expanded = []
    for transition_label, family, member_form, coupling_group in variants:
        transition_type = "acidic" if family in ACIDIC_FAMILIES else "basic"
        if preferred_type in {"acidic", "basic"} and transition_type != preferred_type:
            continue
        expanded.append({
            **site,
            "label": transition_label,
            "detector_label": detector_label,
            "pair_family": family,
            "pair_member_form": member_form,
            "coupling_group": coupling_group,
            "acid_level": 1 if transition_type == "basic" else 0,
            "base_level": 0 if transition_type == "basic" else -1,
        })
    return expanded


def select_serial_azole_transition_by_reference(
    pair_candidates: List[Dict],
    pka_value: Optional[float],
    *,
    max_reference_z: float = 3.5,
    min_runner_up_separation: float = 2.0,
) -> Tuple[Optional[Dict], str]:
    """Conservatively resolve an untyped serial-azole endpoint.

    Every candidate must describe the same physical ring and have a
    transition-specific prior.  The measured pKa can identify an edge, but it
    is never treated as evidence for the atom-site assignment itself.
    """
    if not pair_candidates or pka_value is None:
        return None, "serial_azole_transition_requires_type_or_compatible_pka"
    try:
        observed = float(pka_value)
    except (TypeError, ValueError):
        return None, "serial_azole_transition_requires_type_or_compatible_pka"
    if pd.isna(observed):
        return None, "serial_azole_transition_requires_type_or_compatible_pka"

    physical_groups = {
        (
            str(site.get("coupling_group", site.get("pair_family", ""))),
            tuple(sorted(site.get("atom_set", set()))),
        )
        for site in pair_candidates
    }
    if (
        len(physical_groups) != 1
        or not all(
            site.get("pair_family") in SERIAL_AZOLE_PAIR_FAMILIES
            for site in pair_candidates
        )
    ):
        return None, "not_one_serial_azole_coordinate"

    scored = []
    for candidate in pair_candidates:
        prior = reference_prior(candidate.get("label"), candidate.get("pair_family"))
        if prior is None:
            return None, "serial_azole_transition_missing_reference_prior"
        delta = abs(observed - prior.pka)
        scored.append((delta, delta / prior.uncertainty, candidate))
    scored.sort(key=lambda item: (item[0], str(item[2].get("label", ""))))

    nearest = scored[0]
    if nearest[1] > max_reference_z:
        return None, "serial_azole_pka_conflicts_with_supported_transitions"
    if len(scored) == 1:
        return nearest[2], "sole_enabled_serial_azole_edge_reference_consistent"
    if scored[1][0] - nearest[0] < min_runner_up_separation:
        return None, "serial_azole_pka_does_not_separate_adjacent_transitions"
    return nearest[2], "serial_azole_transition_disambiguated_by_reference"


def distribution_sanity_check(
    group_mode: str,
    pair_member_form: Optional[str],
    summary_label: Optional[str],
    pka_value: float,
) -> Tuple[bool, str]:
    if summary_label is None:
        return False, "missing_group"

    if group_mode == "pair_type":
        if pair_member_form in {"acid_form", "base_form"}:
            return True, "pair_type"
        return False, "pair_unknown_form"

    window = INDIVIDUAL_PKA_WINDOWS.get(summary_label)
    if window is None:
        return True, "individual_no_window"

    low, high = window
    if low <= pka_value <= high:
        return True, "individual_in_window"
    return False, "individual_out_of_window"


def refine_group_label(mol: Chem.Mol, site: Optional[Dict], base_label: Optional[str]) -> Optional[str]:
    if base_label is None:
        return None
    if site is None:
        return base_label

    atom_set = site.get("atom_set", set())
    aromatic_context = _site_has_aromatic_context(mol, atom_set)

    if base_label in {"amide", "amide_NH", "amide_primary", "amide_secondary", "amide_tertiary"}:
        return "amide_aromatic" if aromatic_context else "amide_aliphatic"

    if base_label == "ester":
        return "ester_aromatic" if aromatic_context else "ester_aliphatic"

    if base_label in {"primary_amine", "secondary_amine", "tertiary_amine", "aniline"}:
        return "amine_aromatic" if aromatic_context else "amine_aliphatic"

    if base_label in {"primary_ammonium", "secondary_ammonium", "tertiary_ammonium", "quaternary_ammonium", "aryl_ammonium"}:
        return "ammonium_aromatic" if aromatic_context else "ammonium_aliphatic"

    return base_label


def conjugate_check_flag(
    group_family: Optional[str],
    pka_type: Optional[str],
    pka_value: float,
) -> str:
    if group_family is None:
        if pka_value <= 0 or pka_value >= 12:
            return "unknown_type_extreme"
        return "ok"

    if group_family in ACIDIC_FAMILIES:
        if pka_type == "basic":
            return "acidic_family_vs_basic_type"
        if pka_type == "acidic" and pka_value >= 13:
            return "acidic_too_high_hard"
        if pka_type == "acidic" and pka_value >= 11:
            return "acidic_high_soft"

    if group_family in BASIC_FAMILIES:
        if pka_type == "acidic":
            return "basic_family_vs_acidic_type"
        if pka_type == "basic" and pka_value <= 1:
            return "basic_too_low_hard"
        if pka_type == "basic" and pka_value <= 3:
            return "basic_low_soft"

    if pka_type == "basic" and pka_value <= 1:
        return "basic_too_low_hard"
    if pka_type == "basic" and pka_value <= 3:
        return "basic_low_soft"

    return "ok"


def reinterpret_pka_type_for_two_faced(
    pka_type: Optional[str],
    group_family: Optional[str],
) -> Tuple[Optional[str], bool]:
    if pka_type not in {"acidic", "basic"}:
        return pka_type, False

    if group_family in BASIC_FAMILIES and pka_type == "acidic":
        return "basic", True

    if group_family in ACIDIC_FAMILIES and pka_type == "basic":
        return "acidic", True

    return pka_type, False


def _resolve_sdf_files(raw_dir: str, include_files: Optional[List[str]] = None) -> List[str]:
    if include_files is None:
        return sorted(
            path
            for path in glob.glob(os.path.join(raw_dir, "*.sdf"))
            if not os.path.basename(path).startswith("INCORRECT_")
        )

    sdf_files: List[str] = []
    missing: List[str] = []
    for file_name in include_files:
        path = os.path.join(raw_dir, file_name)
        if os.path.basename(path).startswith("INCORRECT_"):
            continue
        if os.path.exists(path):
            sdf_files.append(path)
        else:
            missing.append(file_name)

    if missing:
        raise FileNotFoundError(
            f"Missing expected .sdf files in {raw_dir}: {', '.join(sorted(missing))}"
        )
    return sdf_files


def _collapse_exact_duplicate_measurements(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    key_cols = ["source_file", "smiles", "pka_value", "pka_source_method"]
    work = df.sort_values(["source_file", "record_index", "smiles", "pka_value"]).copy()
    grouped = work.groupby(key_cols, dropna=False)["record_index"].agg(list)
    count_map = grouped.map(len).to_dict()
    indices_map = grouped.map(lambda values: "|".join(str(int(value)) for value in values)).to_dict()

    keys = list(zip(work["source_file"], work["smiles"], work["pka_value"], work["pka_source_method"]))
    work["source_exact_duplicate_count"] = [int(count_map[key]) for key in keys]
    work["source_exact_duplicate_record_indices"] = [str(indices_map[key]) for key in keys]

    work = work.drop_duplicates(subset=key_cols, keep="first").copy()
    work.reset_index(drop=True, inplace=True)
    return work


def _normalize_override_key_value(column: str, value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return None
    if column in {"record_index", "atom_index_raw"}:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None
    if column == "pka_value":
        try:
            return round(float(value), 6)
        except (TypeError, ValueError):
            return None
    return str(value)


def _site_override_key(row: pd.Series) -> Tuple[object, ...]:
    return tuple(_normalize_override_key_value(column, row.get(column)) for column in SITE_OVERRIDE_KEY_COLUMNS)


def apply_site_resolution_overrides(
    df: pd.DataFrame,
    overrides_path: Optional[str] = None,
) -> pd.DataFrame:
    if not overrides_path or not os.path.exists(overrides_path):
        if "site_override_applied" not in df.columns:
            df = df.copy()
            df["site_override_applied"] = False
            df["site_override_label"] = ""
            df["site_override_note"] = ""
        return df

    overrides = pd.read_csv(overrides_path)
    required = set(SITE_OVERRIDE_KEY_COLUMNS + ["override_group_label"])
    missing = required - set(overrides.columns)
    if missing:
        raise ValueError(
            f"Site override file missing columns: {sorted(missing)}"
        )

    work = df.copy()
    if "auto_selected_group_label" not in work.columns:
        work["auto_selected_group_label"] = work["selected_group_label"]
    if "auto_final_group_label" not in work.columns:
        work["auto_final_group_label"] = work["final_group_label"]

    overrides = overrides.dropna(subset=["override_group_label"]).copy()
    overrides["override_group_label"] = overrides["override_group_label"].astype(str).str.strip()
    overrides = overrides[overrides["override_group_label"] != ""].copy()
    if overrides.empty:
        work["site_override_applied"] = False
        work["site_override_label"] = ""
        work["site_override_note"] = ""
        return work

    overrides["__override_key"] = overrides.apply(_site_override_key, axis=1)
    override_map = {
        key: row
        for key, row in overrides.drop_duplicates(subset=["__override_key"], keep="last").set_index("__override_key").iterrows()
    }

    work["site_override_applied"] = False
    work["site_override_label"] = ""
    work["site_override_note"] = ""

    for idx, row in work.iterrows():
        override = override_map.get(_site_override_key(row))
        if override is None:
            continue

        override_label = str(override["override_group_label"]).strip()
        available_labels = {
            token
            for token in str(row.get("all_candidate_groups", "")).split("|")
            if token and token.lower() not in {"nan", "none"}
        }
        if override_label not in available_labels:
            work.at[idx, "site_override_note"] = (
                f"rejected: override label {override_label!r} is not a current SMARTS candidate"
            )
            continue
        preferred_type = str(row.get("pka_type_canonical", "") or "").lower()
        transition_variants = expand_pair_site_transitions(
            {"label": override_label},
            preferred_type=preferred_type,
        )
        if len(transition_variants) == 1 and (
            preferred_type in {"acidic", "basic"}
            or override_label not in {
                "123triazole", "124triazole", "indazole", "pyrazole", "imidazole"
            }
        ):
            override_label = str(transition_variants[0]["label"])
        override_family = normalize_conjugate_family(override_label)
        override_canonical = canonical_family_label(
            override_label,
            family=override_family,
        )
        group_mode, pair_member_form = classify_pair_type(
            label=override_label,
            family=override_family,
        )
        if group_mode == "pair_type":
            summary_label = override_label
        elif override_family in (ACIDIC_FAMILIES | BASIC_FAMILIES):
            summary_label = override_canonical
        else:
            summary_label = override_label

        dist_ok, dist_reason = distribution_sanity_check(
            group_mode=group_mode,
            pair_member_form=pair_member_form,
            summary_label=summary_label,
            pka_value=float(row["pka_value"]),
        )

        work.at[idx, "selected_group_label"] = override_label
        work.at[idx, "selected_group_family"] = override_family
        work.at[idx, "final_group_label"] = override_label
        work.at[idx, "final_group_refined"] = override_label
        work.at[idx, "final_group_summary_label"] = summary_label
        work.at[idx, "group_mode"] = group_mode
        work.at[idx, "pair_family"] = override_family if override_family in PAIR_TYPE_FAMILY_FORMS else None
        work.at[idx, "pair_member_form"] = pair_member_form
        work.at[idx, "final_group_family"] = override_family
        work.at[idx, "final_group_canonical"] = override_canonical
        work.at[idx, "neutral_input_risk"] = bool(
            group_mode == "pair_type"
            and override_family in BASIC_FAMILIES
            and pair_member_form == "base_form"
            and int(row.get("molecule_formal_charge", 0) or 0) == 0
        )
        work.at[idx, "distribution_ok"] = dist_ok
        work.at[idx, "distribution_reason"] = dist_reason
        work.at[idx, "conjugate_check_flag"] = conjugate_check_flag(
            group_family=override_family,
            pka_type=row.get("pka_type_canonical"),
            pka_value=float(row["pka_value"]),
        )
        work.at[idx, "training_group_label"] = training_group_label(override_label)
        work.at[idx, "training_group_reason"] = training_group_reason(override_label)
        work.at[idx, "site_override_applied"] = True
        work.at[idx, "site_override_label"] = override_label
        work.at[idx, "site_override_note"] = str(override.get("note", "") or "")

    return work


def build_functional_group_table(
    raw_dir: str,
    overlap_threshold: float = 0.5,
    include_files: Optional[List[str]] = None,
    allow_epik: bool = False,
    overrides_path: Optional[str] = None,
) -> pd.DataFrame:
    sdf_files = _resolve_sdf_files(raw_dir, include_files=include_files)
    if not sdf_files:
        raise FileNotFoundError(f"No valid .sdf files found in {raw_dir} (all may be excluded by INCORRECT_ prefix)")

    all_rows = []
    for sdf_path in sdf_files:
        source_file = os.path.basename(sdf_path)
        for record_idx, mol in _iter_sdf_mols(sdf_path):
            smiles = Chem.MolToSmiles(mol, canonical=True)
            measurements = extract_measurements(mol, allow_epik=allow_epik)
            if not measurements:
                continue

            candidates = find_sites_with_metadata(mol)
            resolved_sites, rejected_sites = resolve_overlapping_sites(
                candidates,
                overlap_threshold=overlap_threshold,
            )
            final_group = assign_single_group_label(resolved_sites)

            resolved_labels = sorted({site["label"] for site in resolved_sites})
            resolved_label_instances = [site["label"] for site in resolved_sites]
            resolved_label_counts = Counter(resolved_label_instances)
            resolved_instances_with_index = [
                f"{site['label']}#{idx}"
                for idx, site in enumerate(resolved_sites)
            ]
            candidate_labels = sorted({site["label"] for site in candidates})
            resolved_types = sorted({site["type"] for site in resolved_sites})

            status = assignment_status(len(resolved_sites))

            for measurement in measurements:
                pka_type_canonical = canonicalize_pka_type(measurement["pka_type_raw"])
                pka_value = measurement["pka_value"]
                atom_matched_sites, atom_meta = _sites_matching_atom_index(
                    resolved_sites,
                    measurement["atom_index_raw"],
                )
                selection_pool = atom_matched_sites if atom_matched_sites else resolved_sites
                measurement_site, pair_meta = select_pair_first_site(
                    selection_pool,
                    preferred_type=pka_type_canonical,
                    pka_value=pka_value,
                )
                if atom_meta["atom_site_match"]:
                    pair_meta = {
                        **pair_meta,
                        "pair_selection_reason": f"atom_index_then_{pair_meta['pair_selection_reason']}",
                    }
                measurement_group = measurement_site.get("label") if measurement_site else final_group
                measurement_family = (
                    measurement_site.get("pair_family", measurement_site.get("family"))
                    if measurement_site
                    else normalize_conjugate_family(measurement_group)
                )
                pair_family = measurement_family if measurement_family in PAIR_TYPE_FAMILY_FORMS else None
                measurement_canonical = canonical_family_label(
                    measurement_group,
                    family=measurement_family,
                )
                measurement_refined = refine_group_label(
                    mol=mol,
                    site=measurement_site,
                    base_label=measurement_group,
                )
                group_mode, pair_member_form = classify_pair_type(
                    label=measurement_group,
                    family=measurement_family,
                )
                formal_charge = _molecule_formal_charge(mol)
                neutral_input_risk = bool(
                    group_mode == "pair_type"
                    and measurement_family in BASIC_FAMILIES
                    and pair_member_form == "base_form"
                    and formal_charge == 0
                )
                if group_mode == "pair_type":
                    summary_label = measurement_group
                elif measurement_family in (ACIDIC_FAMILIES | BASIC_FAMILIES):
                    summary_label = measurement_canonical
                else:
                    summary_label = measurement_refined
                dist_ok, dist_reason = distribution_sanity_check(
                    group_mode=group_mode,
                    pair_member_form=pair_member_form,
                    summary_label=summary_label,
                    pka_value=pka_value,
                )
                pka_type_effective, two_faced_type_flip = reinterpret_pka_type_for_two_faced(
                    pka_type=pka_type_canonical,
                    group_family=measurement_family,
                )
                conjugate_flag = conjugate_check_flag(
                    group_family=measurement_family,
                    pka_type=pka_type_canonical,
                    pka_value=pka_value,
                )
                row = {
                    "source_file": source_file,
                    "record_index": record_idx,
                    "smiles": smiles,
                    "pka_value": pka_value,
                    "pka_source_method": measurement["pka_source_method"],
                    "pka_type_raw": measurement["pka_type_raw"],
                    "pka_type_canonical": pka_type_canonical,
                    "pka_type_effective": pka_type_effective,
                    "two_faced_type_flip": two_faced_type_flip,
                    "atom_index_raw": measurement["atom_index_raw"],
                    "atom_index_mode": atom_meta["atom_index_mode"],
                    "atom_index_matched": atom_meta["atom_index_matched"],
                    "atom_site_match": atom_meta["atom_site_match"],
                    "atom_candidate_count": atom_meta["atom_candidate_count"],
                    "atom_candidate_labels": atom_meta["atom_candidate_labels"],
                    "site_identifier_raw": measurement["site_identifier_raw"],
                    "all_candidate_groups": "|".join(candidate_labels),
                    "resolved_groups": "|".join(resolved_labels),
                    "resolved_group_instances": "|".join(resolved_label_instances),
                    "resolved_group_instances_indexed": "|".join(resolved_instances_with_index),
                    "resolved_group_label_counts": "|".join(
                        [f"{label}:{count}" for label, count in sorted(resolved_label_counts.items())]
                    ),
                    "resolved_group_types": "|".join(resolved_types),
                    "resolved_group_count": len(resolved_sites),
                    "rejected_group_count": len(rejected_sites),
                    "assignment_status": status,
                    "selected_group_label": measurement_group,
                    "selected_group_family": measurement_family,
                    "final_group_label": measurement_group,
                    "final_group_refined": measurement_refined,
                    "final_group_summary_label": summary_label,
                    "group_mode": group_mode,
                    "pair_family": pair_family,
                    "pair_member_form": pair_member_form,
                    "pair_assignment_confidence": pair_meta["pair_assignment_confidence"],
                    "pair_candidate_count": pair_meta["pair_candidate_count"],
                    "pair_candidate_labels": pair_meta["pair_candidate_labels"],
                    "pair_selection_reason": pair_meta["pair_selection_reason"],
                    "selected_site_priority": measurement_site.get("priority") if measurement_site else None,
                    "selected_site_specificity": measurement_site.get("specificity") if measurement_site else None,
                    "molecule_formal_charge": formal_charge,
                    "neutral_input_risk": neutral_input_risk,
                    "distribution_ok": dist_ok,
                    "distribution_reason": dist_reason,
                    "final_group_family": measurement_family,
                    "final_group_canonical": measurement_canonical,
                    "conjugate_check_flag": conjugate_flag,
                }
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    df["pka_value"] = pd.to_numeric(df["pka_value"], errors="coerce")
    df = df.dropna(subset=["pka_value", "smiles"])

    df = df[(df["pka_value"] > MIN_REASONABLE_PKA) & (df["pka_value"] < MAX_REASONABLE_PKA)]
    df = _collapse_exact_duplicate_measurements(df)

    df = apply_site_resolution_overrides(df, overrides_path=overrides_path)

    df["single_group_strict"] = df["assignment_status"] == "single_group"
    df["training_group_label"] = df["final_group_label"].map(training_group_label)
    df["training_group_reason"] = df["final_group_label"].map(training_group_reason)
    return df


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("final_group_label")["pka_value"]
    summary = grouped.agg(["count", "mean", "median", "std", "min", "max"]).reset_index()

    q1 = grouped.quantile(0.25).rename("q1")
    q3 = grouped.quantile(0.75).rename("q3")

    summary = summary.merge(q1, left_on="final_group_label", right_index=True)
    summary = summary.merge(q3, left_on="final_group_label", right_index=True)
    summary["iqr"] = summary["q3"] - summary["q1"]

    summary = summary.sort_values("count", ascending=False)
    return summary


def build_assignment_strictness_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("assignment_status", dropna=False)
        .agg(
            n_records=("pka_value", "size"),
            mean_pka=("pka_value", "mean"),
            median_pka=("pka_value", "median"),
            mean_resolved_groups=("resolved_group_count", "mean"),
        )
        .reset_index()
    )
    total = max(1, int(summary["n_records"].sum()))
    summary["fraction"] = summary["n_records"] / total
    return summary.sort_values("n_records", ascending=False)


def build_multi_group_subset(df: pd.DataFrame) -> pd.DataFrame:
    multi_df = df[df["assignment_status"] == "multi_group"].copy()
    return multi_df


def build_multi_group_group_summary(multi_df: pd.DataFrame) -> pd.DataFrame:
    if multi_df.empty:
        return pd.DataFrame(columns=["group_label", "count", "fraction_in_multi"])

    group_col = "resolved_group_instances" if "resolved_group_instances" in multi_df.columns else "resolved_groups"

    exploded = (
        multi_df.assign(group_label=multi_df[group_col].str.split("|"))
        .explode("group_label")
        .dropna(subset=["group_label"])
    )
    summary = exploded.groupby("group_label").size().reset_index(name="count")
    total = max(1, int(summary["count"].sum()))
    summary["fraction_in_multi"] = summary["count"] / total
    return summary.sort_values("count", ascending=False)


def build_multi_group_combo_summary(multi_df: pd.DataFrame) -> pd.DataFrame:
    if multi_df.empty:
        return pd.DataFrame(columns=["group_combo", "count", "fraction_in_multi_rows"])

    combo_df = multi_df.copy()
    group_col = "resolved_group_instances" if "resolved_group_instances" in combo_df.columns else "resolved_groups"
    combo_df["group_combo"] = combo_df[group_col].apply(
        lambda x: "|".join(sorted([v for v in str(x).split("|") if v]))
    )
    summary = combo_df.groupby("group_combo").size().reset_index(name="count")
    total = max(1, int(summary["count"].sum()))
    summary["fraction_in_multi_rows"] = summary["count"] / total
    return summary.sort_values("count", ascending=False)


def build_conjugate_check_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(
            ["final_group_family", "pka_type_canonical", "conjugate_check_flag"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return summary


def build_two_faced_resolution_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(
            ["final_group_family", "pka_type_canonical", "pka_type_effective", "two_faced_type_flip"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return summary


def build_pair_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(
            [
                "group_mode",
                "pair_family",
                "final_group_family",
                "pair_member_form",
                "pair_assignment_confidence",
                "neutral_input_risk",
                "pka_type_canonical",
                "pka_type_effective",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return summary


def build_pair_training_labels(df: pd.DataFrame) -> pd.DataFrame:
    pair_df = df[
        (df["group_mode"] == "pair_type")
        & (df["pair_member_form"].isin(["acid_form", "base_form"]))
        & (df["pair_assignment_confidence"] == "high")
        & (df["distribution_ok"])
    ].copy()

    if pair_df.empty:
        return pair_df

    pair_df["observed_member_form"] = pair_df["pair_member_form"]
    pair_df["target_label"] = pair_df["pair_member_form"]

    keep_cols = [
        "source_file",
        "record_index",
        "smiles",
        "pka_value",
        "pka_source_method",
        "pka_type_canonical",
        "pka_type_effective",
        "pair_family",
        "observed_member_form",
        "target_label",
        "final_group_label",
        "final_group_family",
        "pair_candidate_count",
        "pair_candidate_labels",
        "pair_assignment_confidence",
        "neutral_input_risk",
        "molecule_formal_charge",
    ]

    return pair_df[keep_cols].sort_values(["pair_family", "pka_value"]).reset_index(drop=True)


def plot_group_distributions(df: pd.DataFrame, out_png: str, min_group_count: int = 25) -> None:
    plot_df = df.dropna(subset=["final_group_label"]).copy()
    counts = plot_df["final_group_label"].value_counts()
    keep_groups = counts[counts >= min_group_count].index
    plot_df = plot_df[plot_df["final_group_label"].isin(keep_groups)]

    if plot_df.empty:
        raise ValueError("No groups satisfy min_group_count for plotting.")

    # Determine per-group whether the plotted value is pKa (acids) or
    # pKaH (conjugate acids of bases).  For BASIC_FAMILIES the reported
    # "pKa" in the SDF is really pKaH — label it honestly.
    def _pka_label_for_group(group_label: str) -> str:
        family = CONJUGATE_FAMILY_MAP.get(group_label, group_label)
        if family in BASIC_FAMILIES:
            return "pKaH"
        return "pKa"

    plot_df["_pka_axis_label"] = plot_df["final_group_label"].map(_pka_label_for_group)

    try:
        sns = importlib.import_module("seaborn")

        n_groups = plot_df["final_group_label"].nunique()
        col_wrap = 4
        height = 2.8 if n_groups > 8 else 3.2

        grid = sns.FacetGrid(
            plot_df,
            col="final_group_label",
            col_wrap=col_wrap,
            sharex=True,
            sharey=False,
            height=height,
        )
        grid.map_dataframe(sns.histplot, x="pka_value", stat="density", bins=30, kde=True)
        # Relabel each subplot title to include pKa vs pKaH
        for ax, title_text in zip(grid.axes.flat, grid.col_names):
            family = CONJUGATE_FAMILY_MAP.get(str(title_text), str(title_text))
            tag = "pKaH" if family in BASIC_FAMILIES else "pKa"
            ax.set_title(f"{title_text} ({tag})")
        grid.set_axis_labels("pKa / pKaH", "Density")
        grid.fig.subplots_adjust(top=0.92)
        grid.fig.suptitle("pKa / pKaH Distributions by Functional Group (single-group subset)")
        grid.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(grid.fig)
    except ImportError:
        groups = sorted(plot_df["final_group_label"].unique())
        ncols = 4
        nrows = (len(groups) + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 2.8 * nrows), sharex=True)
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, group in enumerate(groups):
            ax = axes[idx]
            vals = plot_df.loc[plot_df["final_group_label"] == group, "pka_value"]
            tag = _pka_label_for_group(group)
            ax.hist(vals, bins=25, density=True, alpha=0.85)
            ax.set_title(f"{group} ({tag})")
            ax.set_xlabel(tag)
            ax.set_ylabel("Density")

        for idx in range(len(groups), len(axes)):
            axes[idx].axis("off")

        fig.suptitle("pKa / pKaH Distributions by Functional Group (single-group subset)")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_png, dpi=200)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse raw SDFs, assign functional groups, and plot pKa distributions."
    )
    parser.add_argument("--raw-dir", default="data/raw", help="Directory with input .sdf files")
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Directory for output tables and plots",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=0.75,
        help="Fractional atom overlap required to reject a lower-specificity match",
    )
    parser.add_argument(
        "--min-group-count",
        type=int,
        default=10,
        help="Minimum per-group sample count required for plotting",
    )
    parser.add_argument(
        "--allow-epik",
        action="store_true",
        help="Include Epik-derived pKa annotations when present. Default keeps only quoted pKa values.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    full_out = os.path.join(args.out_dir, "functional_group_assignments.csv")
    strict_out = os.path.join(args.out_dir, "functional_group_single_group_subset.csv")
    summary_out = os.path.join(args.out_dir, "functional_group_pka_summary.csv")
    plot_out = os.path.join(args.out_dir, "functional_group_pka_distributions.png")
    strictness_out = os.path.join(args.out_dir, "functional_group_strictness_checks.csv")
    multi_out = os.path.join(args.out_dir, "functional_group_multi_group_subset.csv")
    multi_groups_out = os.path.join(args.out_dir, "functional_group_multi_group_identified_groups.csv")
    multi_combo_out = os.path.join(args.out_dir, "functional_group_multi_group_combinations.csv")
    multi_pka_summary_out = os.path.join(args.out_dir, "functional_group_pka_summary_multi_group.csv")
    conjugate_out = os.path.join(args.out_dir, "functional_group_conjugate_check.csv")
    two_faced_out = os.path.join(args.out_dir, "functional_group_two_faced_resolution.csv")
    pair_type_out = os.path.join(args.out_dir, "functional_group_pair_type_summary.csv")
    pair_labels_out = os.path.join(args.out_dir, "functional_group_pair_training_labels.csv")

    df = build_functional_group_table(
        raw_dir=args.raw_dir,
        overlap_threshold=args.overlap_threshold,
        allow_epik=bool(args.allow_epik),
    )

    if df.empty:
        raise RuntimeError("No pKa-bearing records were parsed from raw SDF files.")

    strict_single_mask = df["single_group_strict"]
    strict_df = df[
        strict_single_mask
        & df["final_group_summary_label"].notna()
        & df["distribution_ok"]
    ].copy()
    strict_df["strict_inclusion_reason"] = "single_group"
    strict_df["final_group_label"] = strict_df["final_group_summary_label"]
    summary = build_summary_table(strict_df)

    strictness_summary = build_assignment_strictness_summary(df)
    multi_df = build_multi_group_subset(df)
    multi_group_summary = build_multi_group_group_summary(multi_df)
    multi_combo_summary = build_multi_group_combo_summary(multi_df)
    multi_df_for_summary = multi_df[multi_df["final_group_summary_label"].notna()].copy()
    multi_df_for_summary["final_group_label"] = multi_df_for_summary["final_group_summary_label"]
    multi_pka_summary = build_summary_table(multi_df_for_summary)
    conjugate_summary = build_conjugate_check_summary(df)
    two_faced_summary = build_two_faced_resolution_summary(df)
    pair_type_summary = build_pair_type_summary(df)
    pair_labels = build_pair_training_labels(df)

    df.to_csv(full_out, index=False)
    strict_df.to_csv(strict_out, index=False)
    summary.to_csv(summary_out, index=False)
    strictness_summary.to_csv(strictness_out, index=False)
    multi_df.to_csv(multi_out, index=False)
    multi_group_summary.to_csv(multi_groups_out, index=False)
    multi_combo_summary.to_csv(multi_combo_out, index=False)
    multi_pka_summary.to_csv(multi_pka_summary_out, index=False)
    conjugate_summary.to_csv(conjugate_out, index=False)
    two_faced_summary.to_csv(two_faced_out, index=False)
    pair_type_summary.to_csv(pair_type_out, index=False)
    pair_labels.to_csv(pair_labels_out, index=False)

    plot_group_distributions(strict_df, plot_out, min_group_count=args.min_group_count)

    print(f"Saved full assignments: {full_out}")
    print(f"Saved strict subset:   {strict_out}")
    print(f"Saved summary table:   {summary_out}")
    print(f"Saved strictness:      {strictness_out}")
    print(f"Saved multi subset:    {multi_out}")
    print(f"Saved multi groups:    {multi_groups_out}")
    print(f"Saved multi combos:    {multi_combo_out}")
    print(f"Saved multi pKa sum:   {multi_pka_summary_out}")
    print(f"Saved conjugate check: {conjugate_out}")
    print(f"Saved two-faced fix:   {two_faced_out}")
    print(f"Saved pair-type sum:   {pair_type_out}")
    print(f"Saved pair labels:     {pair_labels_out}")
    print(f"Saved plot:            {plot_out}")


if __name__ == "__main__":
    main()
