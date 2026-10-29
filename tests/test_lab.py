import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edgelab import simulate, stats  # noqa: E402
from edgelab.registry import Registry  # noqa: E402


def test_population_deterministic_and_counts():
    mix = {"skilled": 0.1, "loser": 0.9}
    a = simulate.population(mix, 100, 50, seed=7)
    b = simulate.population(mix, 100, 50, seed=7)
    assert len(a) == 100 and sum(t.kind == "skilled" for t in a) == 10
    assert all(np.array_equal(x.pnl, y.pnl) for x, y in zip(a, b))


def test_kinds_have_expected_means():
    rng = np.random.default_rng(1)
    big = 200_000
    assert simulate.make_trader("skilled", big, rng, edge=0.2).pnl.mean() == pytest.approx(0.2, abs=0.01)
    assert simulate.make_trader("breakeven", big, rng).pnl.mean() == pytest.approx(0.0, abs=0.01)
    d = simulate.make_trader("decaying", 1000, rng, edge=0.2, cost=0.05)
    assert d.true_mu[:500].min() == 0.2 and d.true_mu[500:].max() == -0.05


def test_martingale_doubles_after_loss_and_resets():
    t = simulate.make_trader("martingale", 500, np.random.default_rng(3), mart_cap=8)
    r = t.pnl / t.size
    for i in range(1, 500):
        exp = min(t.size[i - 1] * 2, 8) if r[i - 1] < 0 else 1.0
        assert t.size[i] == exp


def test_tilt_bursts_follow_losses():
    t = simulate.make_trader("tilt", 2000, np.random.default_rng(4), tilt_p=1.0, tilt_burst=3)
    first = np.nonzero((t.size[1:] == 1.5) & (t.size[:-1] == 1.0))[0]
    assert first.size and all(t.pnl[i] < 0 for i in first)


def test_challenge_pass():
    assert simulate.challenge_pass([5, 4]) is True
    assert simulate.challenge_pass([-11, 30]) is False
    assert simulate.challenge_pass([1, 1]) is False


def test_bh_matches_reference():
    p = [0.01, 0.04, 0.03, 0.2, 0.001]
    # reference step-up by hand at q=0.1, m=5: sorted .001 .01 .03 .04 .2 vs .02 .04 .06 .08 .1
    assert stats.benjamini_hochberg(p, 0.1).tolist() == [True, True, True, False, True]
    assert stats.benjamini_hochberg([], 0.1).size == 0


def test_sign_flip_and_t_agree_on_direction():
    rng = np.random.default_rng(5)
    good, bad = rng.normal(0.5, 1, 100), rng.normal(-0.5, 1, 100)
    assert stats.sign_flip_p(good) < 0.01 and stats.sign_flip_p(bad) > 0.9
    assert stats.t_p(good) < 0.01 and stats.t_p(bad) > 0.9


def test_credibility_shrinks_short_histories_more():
    rng = np.random.default_rng(6)
    tr = [rng.normal(0, 1, 20) for _ in range(50)] + [rng.normal(0, 1, 500) for _ in range(50)]
    tr += [rng.normal(0.3, 1, 20), rng.normal(0.3, 1, 500)]
    c = stats.credibility_mean(tr)
    assert c["z"][-2] < c["z"][-1]
    assert np.all(np.abs(c["mean"] - c["pop_mean"]) <= np.abs([t.mean() for t in tr]) + abs(c["pop_mean"]) + 1e-12)


def test_registry_chain_and_prereg_rules(tmp_path):
    reg = Registry(str(tmp_path / "r.jsonl"))
    reg.prereg("X-01", {"a": 1})
    with pytest.raises(PermissionError):
        reg.prereg("X-01", {"a": 2})
    with pytest.raises(KeyError):
        reg.result("Y-01", {})
    reg.result("X-01", {"ok": True})
    assert reg.verify()["ok"]
    lines = (tmp_path / "r.jsonl").read_text().splitlines()
    rec = json.loads(lines[1]); rec["payload"]["spec"]["a"] = 9
    lines[1] = json.dumps(rec)
    (tmp_path / "r.jsonl").write_text("\n".join(lines) + "\n")
    assert not reg.verify()["ok"]


def test_skill_scores_basic():
    from edgelab import skill
    assert skill.auc([3, 2, 1], [True, False, False]) == 1.0
    assert skill.auc([1, 1], [True, False]) == 0.5
    assert skill.precision_at_k([5, 4, 3, 2], [1, 0, 1, 0], 2) == 0.5
    assert skill.profit_factor([2, -1, 1]) == 3.0 and skill.win_rate([1, -1, 2, -2]) == 0.5
    rng = np.random.default_rng(8)
    tr = [rng.normal(0, 1, 100) for _ in range(200)] + [rng.normal(1.0, 1, 100)]
    p = skill.eb_prob(tr)["prob"]
    assert p[-1] > 0.9 and np.median(p[:-1]) < 0.9 and np.all((p >= 0) & (p <= 1))


def test_style_bench_mean_exact_and_style_independent_of_skill():
    rng = np.random.default_rng(11)
    for w in (0.3, 0.55, 0.8):
        for mu in (-0.05, 0.0, 0.15):
            assert simulate.style_outcomes(rng, 400_000, mu, w).mean() == pytest.approx(mu, abs=0.012)
    pop = simulate.population_v2({"skilled": 0.5, "loser": 0.5}, 400, 50, seed=3)
    ws = np.array([t.meta["win_rate"] for t in pop]); sk = np.array([t.kind == "skilled" for t in pop])
    assert abs(ws[sk].mean() - ws[~sk].mean()) < 0.05
    obs = np.array([np.mean(simulate.style_outcomes(rng, 5000, 0.0, w) > 0) for w in (0.3, 0.8)])
    assert obs[0] == pytest.approx(0.3, abs=0.03) and obs[1] == pytest.approx(0.8, abs=0.03)


def test_npmle_recovers_mixture_and_is_calibrated():
    from edgelab import skill
    rng = np.random.default_rng(12)
    mus = np.where(rng.random(3000) < 0.1, 0.5, 0.0)
    tr = [rng.normal(m, 1, 100) for m in mus]
    r = skill.npmle_prob(tr)
    p, y = r["prob"], mus > 0
    hi = p > 0.7
    assert hi.sum() > 50 and abs(p[hi].mean() - y[hi].mean()) < 0.1
    assert abs(r["prior"][r["grid"] > 0.25].sum() - 0.1) < 0.04


def test_npmle_threshold_excludes_breakeven_mass():
    from edgelab import skill
    rng = np.random.default_rng(13)
    mus = np.where(rng.random(3000) < 0.1, 0.5, 0.0)
    tr = [rng.normal(m, 1, 100) for m in mus]
    p0, p1 = skill.npmle_prob(tr)["prob"], skill.npmle_prob(tr, thr=0.25)["prob"]
    assert np.all(p1 <= p0 + 1e-12)
    assert p1[mus == 0].mean() < p0[mus == 0].mean() and p1[mus > 0].mean() > 0.8


def test_precision_at_k_ties_averages_the_tied_group():
    from edgelab import skill
    assert skill.precision_at_k([1, 1, 1, 1], [1, 0, 0, 0], 1) == 1.0
    assert skill.precision_at_k_ties([1, 1, 1, 1], [1, 0, 0, 0], 1) == 0.25
    assert skill.precision_at_k_ties([5, 4, 3, 2], [1, 0, 1, 0], 2) == 0.5
    assert skill.precision_at_k_ties([5, 3, 3, 1], [1, 1, 0, 0], 2) == 0.75
