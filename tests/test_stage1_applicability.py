from stage1_network_inference import stage1_applicability
from build_molecule_microstate_network_dataset import stage1_local_prediction_for_site


def test_stage1_applicability_is_exact_label_specific():
    bundle = {
        "fp_bits": 64,
        "fp_radius": 2,
        "training_site_label_counts": {"pyrrole": 3, "primary_amine": 100},
        "training_smiles_by_label": {
            "pyrrole": ["c1cc[nH]c1", "Cc1cc[nH]c1", "Cn1cccc1"],
            "primary_amine": ["CN"],
        },
    }
    result = stage1_applicability("c1cc[nH]c1", "pyrrole", bundle)
    assert result["stage1_exact_label_training_rows"] == 3
    assert result["stage1_nearest_same_label_tanimoto"] == 1.0
    assert result["stage1_applicability_domain"] == "interpolation_limited_support"


def test_stage1_applicability_marks_absent_exact_label_zero_shot():
    bundle = {
        "fp_bits": 64,
        "fp_radius": 2,
        "training_site_label_counts": {"amine": 1000},
        "training_smiles_by_label": {"amine": ["CN"]},
    }
    result = stage1_applicability("c1nccs1", "thiazole", bundle)
    assert result["stage1_exact_label_training_rows"] == 0
    assert result["stage1_nearest_same_label_tanimoto"] is None
    assert result["stage1_applicability_domain"] == "zero_shot_exact_label_absent"


def test_zero_shot_transition_uses_enabled_reference_prior():
    value, provenance, confidence, uncertainty = stage1_local_prediction_for_site(
        {"label": "pyrazole_acidity", "family": "pyrazole_acidity"},
        raw_prediction=5.0,
        applicability_domain="zero_shot_exact_label_absent",
    )
    assert value == 14.21
    assert provenance == "stage1_reference_prior_zero_shot_exact_label_fallback"
    assert confidence == "low_reference_prior_zero_shot_exact_label"
    assert uncertainty == 2.5


def test_supported_transition_keeps_stage1_model_prediction():
    value, provenance, confidence, uncertainty = stage1_local_prediction_for_site(
        {"label": "pyrazole_basicity", "family": "pyrazole_basicity"},
        raw_prediction=2.42,
        applicability_domain="extrapolation_low_similarity_or_sparse_label",
    )
    assert value == 2.42
    assert provenance == "stage1_experimental_only_intrinsic_prediction"
    assert confidence == ""
    assert uncertainty is None
