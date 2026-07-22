"""Validity criteria for mechanistic SSL benchmarks — executable, not advisory.

A mechanistic claim ("objective O helps because of mechanism M") is only supported if the benchmark
that produced it could have shown otherwise. Every criterion below was derived from a failure that
actually occurred in this project and that cost, or nearly cost, a false published result:

  V1 RESOLVING POWER   a trivial floor (raw features / random init) must NOT dominate the learned
                       methods. If it does, method-vs-method deltas are contrasts between things that
                       both failed, and any trend across conditions is noise.
                       [caught: predictability sweep §3; alignment bench §5 — 0/30 cells beat raw]
  V2 EFFECT vs NOISE   the reported effect must exceed run-to-run (seed) variance. A single-seed
                       effect has NO measured noise floor and cannot clear this by construction.
                       [caught: alignment bench, sd +-0.09 on a 0.05 effect]
  V3 MANIPULATION      when claiming "we varied X holding Y fixed", Y must be VERIFIED constant, not
    ISOLATION          assumed. Assert it in tests, measure it in the run.
                       [designed in: generate_aligned holds Omega fixed to 1e-9]
  V4 COMPARISON        compared cells must share eval split, probe budget, and protocol. Differences
    HYGIENE            in these confound with the variable of interest.
                       [caught: graph JEPA probed on test vs baselines on val]
  V5 PRE-REGISTERED    the decision rule must be fixed before seeing results. A threshold chosen
    DECISION RULE      after the fact converts noise into a finding.
                       [caught: alignment_bench printed "H2 SUPPORTED" on corr>0.3, p=0.245]

V1-V4 are computable and implemented here. V5 is a process property: `PreregisteredRule` records a
rule so a run can state what it committed to, and whether the outcome met it.

Nothing here decides whether a hypothesis is TRUE. These decide whether an experiment is ENTITLED to
an opinion — a separate and logically prior question.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# A criterion returns one of these. INCONCLUSIVE is a first-class outcome, distinct from a negative:
# "the benchmark could not tell" is not "the hypothesis is false".
PASS, FAIL, NA = "PASS", "FAIL", "N/A"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    value: float = float("nan")

    def __str__(self):
        v = "" if not np.isfinite(self.value) else f" [{self.value:.3f}]"
        return f"{self.name:26s} {self.status:5s}{v}  {self.detail}"


def resolving_power(method_scores, floor_score, higher_better=True, focal=None, min_fraction=0.25):
    """V1 — can the benchmark resolve the claim at hand?

    `method_scores`: dict {method: score} of the LEARNED methods under comparison.
    `floor_score`: the trivial baseline (raw features / random init) — what a method must beat to
    have shown that learning did anything.

    TWO claim shapes need different questions, and conflating them mis-scores real results:

      focal given  -> the claim is "METHOD X beats/does-something". The benchmark resolves it iff
                      X ITSELF clears the floor. Other methods failing to clear it is a finding
                      about them, not a defect of the benchmark. (Getting this wrong made an
                      earlier version of this file mark the C-MAPSS win INCONCLUSIVE because only
                      the winner beat the floor — which is precisely what winning looks like.)
      focal None   -> the claim is a RANKING over methods ("A is the best encoder"). That needs the
                      population to be above the floor: if every method is below it, the ordering is
                      an ordering of failures and carries no information.
    """
    if not method_scores or floor_score is None or not np.isfinite(floor_score):
        return Check("V1 resolving power", NA, "no floor recorded — CANNOT be assessed "
                                               "(a benchmark without a floor cannot show learning helped)")
    def beats(s):
        return (s > floor_score) if higher_better else (s < floor_score)
    frac = float(np.mean([beats(s) for s in method_scores.values()]))

    if focal is not None:
        if focal not in method_scores:
            return Check("V1 resolving power", NA, f"focal method {focal!r} absent from results")
        s = method_scores[focal]
        st = PASS if beats(s) else FAIL
        return Check("V1 resolving power", st,
                     f"focal {focal}={s:.3f} vs floor {floor_score:.3f} -> "
                     f"{'clears the floor' if st == PASS else 'BELOW the floor: learning not shown'}"
                     f"; {frac:.0%} of {len(method_scores)} methods clear it", frac)

    if frac < min_fraction:
        return Check("V1 resolving power", FAIL,
                     f"only {frac:.0%} of {len(method_scores)} methods beat the floor "
                     f"({floor_score:.3f}) -> a RANKING here orders failures; treat as INCONCLUSIVE",
                     frac)
    return Check("V1 resolving power", PASS,
                 f"{frac:.0%} of {len(method_scores)} methods beat the floor ({floor_score:.3f})", frac)


def effect_vs_noise(effect, seed_std, n_seeds=None, k=1.0):
    """V2 — is the effect bigger than the noise it was measured against?

    `effect`: the reported difference. `seed_std`: std of that difference across seeds.
    A single seed (n_seeds<2, or seed_std undefined) FAILS: with no measured noise floor the effect
    is unfalsifiable, not merely uncertain.
    """
    if n_seeds is not None and n_seeds < 2:
        return Check("V2 effect vs noise", FAIL,
                     f"single seed (n={n_seeds}) — no measured noise floor; effect {effect:+.3f} "
                     f"is directional only", float("nan"))
    if seed_std is None or not np.isfinite(seed_std):
        return Check("V2 effect vs noise", NA, "no seed variance recorded")
    if seed_std <= 1e-12:
        return Check("V2 effect vs noise", NA, "zero variance (identical runs?)")
    ratio = abs(effect) / seed_std
    st = PASS if ratio > k else FAIL
    return Check("V2 effect vs noise", st,
                 f"|effect| {abs(effect):.3f} vs seed sd {seed_std:.3f} -> ratio {ratio:.2f} "
                 f"({'>' if ratio > k else '<='}{k:g})", ratio)


def manipulation_isolation(control_values, tol=0.05, label="control"):
    """V3 — did the variable we claim to have held fixed actually stay fixed?

    `control_values`: the control quantity measured once per experimental condition.
    """
    v = np.asarray([x for x in np.ravel(control_values) if np.isfinite(x)], dtype=float)
    if v.size < 2:
        return Check("V3 manipulation isolation", NA, f"need >=2 {label} measurements")
    spread = float(v.max() - v.min())
    st = PASS if spread <= tol else FAIL
    return Check("V3 manipulation isolation", st,
                 f"{label} spread {spread:.4f} across conditions "
                 f"({'held fixed' if st == PASS else 'MOVED — confounded with the manipulation'})",
                 spread)


def comparison_hygiene(cell_protocols):
    """V4 — do the compared cells share an evaluation protocol?

    `cell_protocols`: dict {cell: hashable protocol descriptor}, e.g. the eval split or
    (split, probe_epochs). Any disagreement confounds protocol with the variable of interest.
    """
    if not cell_protocols:
        return Check("V4 comparison hygiene", NA, "no protocol metadata")
    uniq = set(cell_protocols.values())
    if len(uniq) == 1:
        return Check("V4 comparison hygiene", PASS,
                     f"all {len(cell_protocols)} cells share protocol {next(iter(uniq))!r}")
    groups = {}
    for c, p in cell_protocols.items():
        groups.setdefault(p, []).append(c)
    desc = "; ".join(f"{p!r}: {sorted(cs)}" for p, cs in sorted(groups.items(), key=lambda kv: str(kv[0])))
    return Check("V4 comparison hygiene", FAIL,
                 f"cells span {len(uniq)} protocols — comparison confounded ({desc})")


@dataclass
class PreregisteredRule:
    """V5 — a decision rule recorded BEFORE the run, so the verdict cannot be fitted to the data."""
    statement: str
    threshold: float
    direction: str = ">"          # ">" or "<"

    def evaluate(self, observed):
        met = observed > self.threshold if self.direction == ">" else observed < self.threshold
        return Check("V5 pre-registered rule", PASS if met else FAIL,
                     f"{self.statement} (rule: observed {self.direction} {self.threshold:g}; "
                     f"observed {observed:.3f})", float(observed))


@dataclass
class Audit:
    """A set of validity checks on one claim, with an overall entitlement verdict."""
    claim: str
    checks: list = field(default_factory=list)

    def add(self, check):
        self.checks.append(check)
        return self

    @property
    def entitled(self):
        """True iff no criterion FAILED. N/A does not block, but is reported."""
        return all(c.status != FAIL for c in self.checks)

    @property
    def verdict(self):
        if any(c.status == FAIL for c in self.checks):
            return "INCONCLUSIVE — benchmark not entitled to this claim"
        if any(c.status == NA for c in self.checks):
            return "SUPPORTED (with unassessed criteria)"
        return "SUPPORTED"

    def report(self):
        head = f"CLAIM: {self.claim}\n" + "-" * 100
        body = "\n".join(f"  {c}" for c in self.checks)
        return f"{head}\n{body}\n  => {self.verdict}\n"


CRITERIA = ["V1 resolving power", "V2 effect vs noise", "V3 manipulation isolation",
            "V4 comparison hygiene", "V5 pre-registered rule"]
