import math

import pytest

from stage2_free_energy_coupling import (
    R_KJ_MOL_K,
    annotate_pairwise_microstate_populations,
    canonical_pair_key,
    contextual_edge_pkas,
    infer_regularized_pairwise_free_energy,
    macro_pka_values_from_free_energy,
    pair_coupling_map,
    site_marginals_from_pairwise_model,
)


def _two_site_network():
    sites = [{"site_id": "a"}, {"site_id": "b"}]
    nodes = [
        {"node_id": "00", "protonated_site_count": 0,
         "site_forms": {"a": "base_form", "b": "base_form"}},
        {"node_id": "10", "protonated_site_count": 1,
         "site_forms": {"a": "acid_form", "b": "base_form"}},
        {"node_id": "01", "protonated_site_count": 1,
         "site_forms": {"a": "base_form", "b": "acid_form"}},
        {"node_id": "11", "protonated_site_count": 2,
         "site_forms": {"a": "acid_form", "b": "acid_form"}},
    ]
    edges = [
        {"edge_id": "a0", "site_id": "a", "acid_node_id": "10", "base_node_id": "00"},
        {"edge_id": "a1", "site_id": "a", "acid_node_id": "11", "base_node_id": "01"},
        {"edge_id": "b0", "site_id": "b", "acid_node_id": "01", "base_node_id": "00"},
        {"edge_id": "b1", "site_id": "b", "acid_node_id": "11", "base_node_id": "10"},
    ]
    return sites, nodes, edges


def test_pair_coupling_changes_only_edges_with_other_site_protonated():
    sites, nodes, edges = _two_site_network()
    one_body = {"a": 9.0, "b": 4.0}
    pairs = {canonical_pair_key("a", "b"): -1.0}
    values, cycle_error = contextual_edge_pkas(nodes, edges, sites, one_body, pairs)

    assert values["a0"]["contextual_edge_pka"] == pytest.approx(9.0)
    assert values["a1"]["contextual_edge_pka"] == pytest.approx(8.0)
    assert values["b0"]["contextual_edge_pka"] == pytest.approx(4.0)
    assert values["b1"]["contextual_edge_pka"] == pytest.approx(3.0)
    assert cycle_error < 1e-12
    assert macro_pka_values_from_free_energy(nodes, sites, one_body, pairs) == pytest.approx(
        [9.0000043429, 2.9999956571]
    )


def test_pair_term_reports_consistent_free_energy_sign():
    sites, nodes, _ = _two_site_network()
    _, terms, diagnostics = infer_regularized_pairwise_free_energy(
        stage1_local_pkas=[9.0, 4.0],
        stage2_macro_pkas=[9.0, 3.0],
        adjustable=[True, True],
        proposed_deltas=[0.0, 0.0],
        nodes=nodes,
        sites=sites,
        observed_macro_anchor_count=1,
    )
    pairs = pair_coupling_map(terms)
    coupling = pairs[canonical_pair_key("a", "b")]
    reported_delta_g = terms[0]["interaction_delta_g_kj_mol_at_298k"]
    assert reported_delta_g == pytest.approx(
        -R_KJ_MOL_K * 298.15 * math.log(10.0) * coupling
    )
    assert diagnostics["pair_couplings_experimentally_identified"] is False
    assert diagnostics["pair_coupling_identifiability"] == (
        "regularized_prior_dependent_predicted_macro_ladder"
    )


def test_pairwise_populations_and_marginals_are_normalized():
    sites, nodes, _ = _two_site_network()
    one_body = {"a": 7.4, "b": 7.4}
    pairs = {canonical_pair_key("a", "b"): -2.0}
    summary = annotate_pairwise_microstate_populations(
        nodes, sites, one_body, pairs, ph=7.4
    )
    marginals = site_marginals_from_pairwise_model(nodes, sites, one_body, pairs, ph=7.4)
    assert summary["status"] == "complete_pairwise_coupled_population"
    assert summary["population_sum"] == pytest.approx(1.0)
    assert marginals["a"] == pytest.approx(marginals["b"])
    assert marginals["a"] < 0.5


def test_serial_coordinate_occupancies_remain_nested():
    sites = [
        {"site_id": "acidity", "coupling_group_id": "g", "acid_level": 0},
        {"site_id": "basicity", "coupling_group_id": "g", "acid_level": 1},
    ]
    nodes = [
        {"node_id": "anion", "protonated_site_count": 0,
         "group_levels": {"g": -1},
         "site_forms": {"acidity": "base_form", "basicity": "outside_transition"}},
        {"node_id": "neutral", "protonated_site_count": 1,
         "group_levels": {"g": 0},
         "site_forms": {"acidity": "acid_form", "basicity": "base_form"}},
        {"node_id": "cation", "protonated_site_count": 2,
         "group_levels": {"g": 1},
         "site_forms": {"acidity": "outside_transition", "basicity": "acid_form"}},
    ]
    one_body = {"acidity": 9.8, "basicity": 2.3}
    assert macro_pka_values_from_free_energy(nodes, sites, one_body, {}) == pytest.approx(
        [9.8, 2.3]
    )

