"""Validity criteria — the tool that judges experiments must itself be correct.

Includes a regression for a real miscalibration: V1 originally required a FRACTION of methods to
beat the floor, which marked the C-MAPSS win INCONCLUSIVE because only the winning method cleared
the floor — i.e. it penalised exactly what a win looks like.
"""
import numpy as np
import pytest

from eval.validity import (FAIL, NA, PASS, Audit, PreregisteredRule, comparison_hygiene,
                           effect_vs_noise, manipulation_isolation, resolving_power)


# ---- V1 resolving power ------------------------------------------------------------------------
def test_v1_focal_claim_passes_when_only_the_winner_clears_the_floor():
    """REGRESSION: 'only my method beats the floor' is a WIN, not a benchmark defect."""
    scores = {"mine": 0.68, "b1": 0.60, "b2": 0.61, "b3": 0.59, "b4": 0.58}
    assert resolving_power(scores, 0.65, focal="mine").status == PASS


def test_v1_focal_claim_fails_when_the_focal_method_is_below_the_floor():
    assert resolving_power({"mine": 0.40, "b1": 0.80}, 0.65, focal="mine").status == FAIL


def test_v1_ranking_claim_fails_when_everything_is_below_the_floor():
    """A ranking over methods that all lost to a trivial baseline orders failures."""
    assert resolving_power({"a": 0.3, "b": 0.35, "c": 0.2}, 0.8).status == FAIL


def test_v1_ranking_claim_passes_when_the_population_clears_the_floor():
    assert resolving_power({"a": 0.9, "b": 0.85, "c": 0.7}, 0.5).status == PASS


def test_v1_lower_is_better_direction():
    assert resolving_power({"m": 340.0}, 470.0, higher_better=False, focal="m").status == PASS


def test_v1_missing_floor_is_NA_not_pass():
    """No floor recorded must NOT silently pass — it is unassessable."""
    assert resolving_power({"a": 1.0}, None).status == NA


# ---- V2 effect vs noise ------------------------------------------------------------------------
def test_v2_single_seed_fails():
    assert effect_vs_noise(0.5, None, n_seeds=1).status == FAIL


def test_v2_effect_below_noise_fails_and_above_passes():
    assert effect_vs_noise(0.05, 0.073, n_seeds=3).status == FAIL       # the real alignment numbers
    assert effect_vs_noise(0.30, 0.050, n_seeds=3).status == PASS


# ---- V3 manipulation isolation -----------------------------------------------------------------
def test_v3_detects_a_control_variable_that_moved():
    assert manipulation_isolation([0.10, 0.10, 0.55], tol=0.05).status == FAIL
    assert manipulation_isolation([0.103, 0.094, 0.113], tol=0.05).status == PASS


# ---- V4 comparison hygiene ---------------------------------------------------------------------
def test_v4_flags_cells_evaluated_on_different_splits():
    """The real graph-JEPA bug: baselines on val, the new cell on test."""
    c = comparison_hygiene({"tjepa_h1": "val", "mae": "val", "tjepa_graph": "test"})
    assert c.status == FAIL and "tjepa_graph" in c.detail


def test_v4_passes_when_protocols_match():
    assert comparison_hygiene({"a": "val", "b": "val"}).status == PASS


# ---- V5 pre-registration -----------------------------------------------------------------------
def test_v5_rule_evaluates_against_a_fixed_threshold():
    rule = PreregisteredRule("corr(alpha, advantage) exceeds 0.3", 0.3, ">")
    assert rule.evaluate(0.45).status == PASS
    assert rule.evaluate(0.307).status == PASS      # met the rule...
    assert rule.evaluate(0.1).status == FAIL


# ---- Audit aggregation -------------------------------------------------------------------------
def test_audit_any_fail_blocks_entitlement():
    a = Audit("x").add(resolving_power({"m": 1.0}, 0.5, focal="m")).add(effect_vs_noise(1.0, None, n_seeds=1))
    assert not a.entitled and "INCONCLUSIVE" in a.verdict


def test_audit_na_does_not_block_but_is_surfaced():
    a = Audit("x").add(resolving_power({"m": 1.0}, None))
    assert a.entitled and "unassessed" in a.verdict


def test_audit_all_pass_is_clean():
    a = Audit("x").add(resolving_power({"m": 1.0}, 0.5, focal="m")).add(effect_vs_noise(0.4, 0.05, n_seeds=3))
    assert a.entitled and a.verdict == "SUPPORTED"
