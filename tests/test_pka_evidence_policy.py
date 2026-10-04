from pka_evidence_policy import (
    assess_measurement_evidence,
    ionization_policy_role,
    plausibility_against_reference,
    unresolved_ionizable_contexts,
)


def test_aqueous_policy_separates_descriptor_permanent_and_unknown_nitrogen():
    assert ionization_policy_role("nitro") == "structural_descriptor_only"
    assert ionization_policy_role("quaternary_ammonium") == "permanent_charge_descriptor"
    assert ionization_policy_role("ketone") == "extreme_range_disabled"
    assert ionization_policy_role("unclassified_aromatic_ring_N") == "unclassified_requires_quarantine"
    assert ionization_policy_role("tetrazole") == "aqueous_pair_candidate"


def test_unenumerated_thioamide_blocks_single_site_claim():
    contexts = unresolved_ionizable_contexts(
        [{"label": "thioamide", "site_type": "basic", "atom_set": {0, 1, 2}}],
        [],
    )
    assert [context["label"] for context in contexts] == ["thioamide"]


def test_structural_selection_alone_is_silver_not_source_verified_gold():
    evidence = assess_measurement_evidence(
        pka=15.9,
        label="primary_alcohol",
        family="alcohol_alkoxide",
        structural_mapping_confidence="high",
        source_site_evidence_confidence="none",
        unresolved_contexts=[],
        source_original_smiles="CCO",
        input_smiles="CCO",
    )
    assert evidence["evidence_tier"] == "silver"
    assert evidence["experimental_site_attribution_confidence"] == "none"
    assert evidence["stage1_local_supervision_eligible"] is True


def test_transformed_charged_alcohol_drawing_does_not_become_exact_truth():
    evidence = assess_measurement_evidence(
        pka=15.9,
        label="primary_alcohol",
        family="alcohol_alkoxide",
        structural_mapping_confidence="high",
        source_site_evidence_confidence="none",
        unresolved_contexts=[],
        source_original_smiles="OCC",
        input_smiles="CC[OH2+]",
    )
    assert evidence["source_structure_charge_changed"] is True
    assert evidence["transition_identity_confidence"] == "low"
    assert evidence["evidence_tier"] == "ambiguous"
    assert evidence["stage1_local_supervision_eligible"] is False


def test_pyrrole_seven_point_five_is_a_reference_conflict():
    result = plausibility_against_reference(7.5, "pyrrole", "amine")
    assert result["status"] == "conflict"
    assert result["z_distance"] > 3.5


def test_user_approved_azole_transitions_have_separate_reference_priors():
    assert ionization_policy_role("123triazole") == "aqueous_pair_candidate"
    assert ionization_policy_role("124triazole") == "aqueous_pair_candidate"
    assert ionization_policy_role("indazole") == "aqueous_pair_candidate"
    assert ionization_policy_role("pyrazole") == "aqueous_pair_candidate"
    assert ionization_policy_role("imidazole") == "aqueous_pair_candidate"
    assert plausibility_against_reference(
        2.3, "124triazole_basicity", "124triazole_basicity"
    )["reference_pka"] == 2.3
    assert plausibility_against_reference(
        9.8, "124triazole_acidity", "124triazole_acidity"
    )["reference_pka"] == 9.8
    assert plausibility_against_reference(
        1.0, "indazole_basicity", "indazole_basicity"
    )["reference_pka"] == 1.0
    assert plausibility_against_reference(
        13.9, "indazole_acidity", "indazole_acidity"
    )["reference_pka"] == 13.9
    assert plausibility_against_reference(
        2.49, "pyrazole_basicity", "pyrazole_basicity"
    )["reference_pka"] == 2.49
    assert plausibility_against_reference(
        14.21, "pyrazole_acidity", "pyrazole_acidity"
    )["reference_pka"] == 14.21
    assert plausibility_against_reference(
        7.0, "imidazole_basicity", "imidazole_basicity"
    )["reference_pka"] == 7.0
    assert plausibility_against_reference(
        14.2, "imidazole_acidity", "imidazole_acidity"
    )["reference_pka"] == 14.2


def test_unresolved_competitor_prevents_unique_site_silver_tier():
    evidence = assess_measurement_evidence(
        pka=7.5,
        label="pyrrole",
        family="amine",
        structural_mapping_confidence="high",
        source_site_evidence_confidence="none",
        unresolved_contexts=[{"label": "thioamide"}],
        input_smiles="CNC(=S)NCCCCc1cc[nH]c1",
    )
    assert evidence["evidence_tier"] == "ambiguous"
    assert "competing_unresolved_ionizable_context" in evidence["evidence_tier_reason"]
