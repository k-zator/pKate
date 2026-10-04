import pytest

from stage3_microstate_thermodynamics import (
    annotate_microstate_populations,
    infer_effective_local_pkas,
    macro_pka_values,
    site_protonated_marginals,
)


def test_inverse_reconstructs_an_independent_site_macro_ladder():
    local = [9.0, 4.0]
    target = macro_pka_values(local)
    inferred, diagnostics = infer_effective_local_pkas(
        local,
        target,
        [True, True],
    )
    assert inferred == pytest.approx(local, abs=1e-4)
    assert diagnostics["macro_reconstruction_mae"] < 1e-6


def test_inverse_keeps_fallback_local_pka_fixed():
    local = [9.0, 4.0]
    target = [10.0, 3.0]
    inferred, _ = infer_effective_local_pkas(
        local,
        target,
        [True, False],
        proposed_deltas=[1.0, 0.0],
    )
    assert inferred[1] == pytest.approx(4.0)
    assert inferred[0] >= inferred[1]


def test_population_at_pka_is_even_for_one_site():
    nodes = [
        {"node_id": "base", "site_forms": {"site_1": "base_form"}},
        {"node_id": "acid", "site_forms": {"site_1": "acid_form"}},
    ]
    summary = annotate_microstate_populations(nodes, {"site_1": 7.4}, ph=7.4)
    marginals = site_protonated_marginals(nodes, ["site_1"])
    assert summary["population_sum"] == pytest.approx(1.0)
    assert marginals["site_1"] == pytest.approx(0.5)


def test_duplicate_drawings_do_not_gain_configuration_weight():
    nodes = [
        {"node_id": "base", "site_forms": {"site_1": "base_form"}},
        {"node_id": "acid_a", "site_forms": {"site_1": "acid_form"}},
        {"node_id": "acid_b", "site_forms": {"site_1": "acid_form"}},
    ]
    summary = annotate_microstate_populations(nodes, {"site_1": 7.4}, ph=7.4)
    marginals = site_protonated_marginals(nodes, ["site_1"])
    assert marginals["site_1"] == pytest.approx(0.5)
    acid_nodes = [node for node in nodes if node["node_id"].startswith("acid")]
    assert sum(node["stage3_node_population_at_ph"] for node in acid_nodes) == pytest.approx(0.5)
    assert summary["effective_configuration_count"] == pytest.approx(2.0)


def test_serial_amphoteric_coordinate_uses_both_transition_pkas():
    nodes = [
        {
            "node_id": "anion", "protonated_site_count": 0,
            "group_levels": {"group_1": -1},
            "site_forms": {"basicity": "outside_transition", "acidity": "base_form"},
        },
        {
            "node_id": "neutral", "protonated_site_count": 1,
            "group_levels": {"group_1": 0},
            "site_forms": {"basicity": "base_form", "acidity": "acid_form"},
        },
        {
            "node_id": "cation", "protonated_site_count": 2,
            "group_levels": {"group_1": 1},
            "site_forms": {"basicity": "acid_form", "acidity": "outside_transition"},
        },
    ]
    local = [9.8, 2.3]
    site_ids = ["acidity", "basicity"]
    assert macro_pka_values(local, nodes, site_ids) == pytest.approx([9.8, 2.3])
    inferred, diagnostics = infer_effective_local_pkas(
        local, [9.8, 2.3], [True, True], nodes=nodes, site_ids=site_ids
    )
    assert inferred == pytest.approx(local, abs=1e-4)
    assert diagnostics["macro_reconstruction_mae"] < 1e-6
    summary = annotate_microstate_populations(
        nodes, {"basicity": 2.3, "acidity": 9.8}, ph=7.4
    )
    marginals = site_protonated_marginals(nodes, site_ids)
    assert summary["population_model"] == "enumerated_coupled_coordinates"
    assert nodes[1]["stage3_configuration_population_at_ph"] > 0.99
    assert marginals["acidity"] > 0.99
    assert marginals["basicity"] < 1e-5
