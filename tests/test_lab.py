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


def test_npmle_variants_and_isotonic():
    from edgelab import skill
    rng = np.random.default_rng(14)
    mus = np.where(rng.random(800) < 0.1, 0.5, 0.0)
    tr = [rng.normal(m, 1, 80) for m in mus]
    base = skill.npmle_prob(tr)["prob"]
    assert np.allclose(base, skill.npmle_prob(tr, var="own", lik="normal")["prob"])
    for kw in ({"var": "pooled"}, {"lik": "t"}, {"lik": "boot", "boot_b": 100}):
        p = skill.npmle_prob(tr, **kw)["prob"]
        assert np.all((p >= 0) & (p <= 1 + 1e-9)) and skill.auc(p, mus > 0) > 0.95
    knots = skill.isotonic_fit([0.1, 0.2, 0.3, 0.4], [0, 1, 0, 1])
    assert list(knots[1]) == [0.0, 0.5, 1.0]
    assert list(skill.isotonic_apply(knots, [0.05, 0.25, 0.3, 0.9])) == [0.0, 0.5, 0.5, 1.0]


def _git_repo(tmp_path):
    import subprocess
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "edgelab").mkdir()
    (tmp_path / "edgelab" / "a.py").write_text("x = 1\n")
    (tmp_path / "study.py").write_text("print(1)\n")
    (tmp_path / "state").mkdir()
    run("init", "-q"); run("-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    return run


def test_code_lock_refuses_changed_or_uncommitted_code(tmp_path):
    from edgelab import study
    run = _git_repo(tmp_path)
    script = str(tmp_path / "study.py")
    reg = Registry(str(tmp_path / "state" / "r.jsonl"))
    pre = reg.prereg("X-01", {"a": 1}, lock=study.code_lock(script, str(tmp_path)))["payload"]
    assert study.check_lock(pre, script, str(tmp_path))["files_sha256"] == pre["lock"]["files_sha256"]
    (tmp_path / "state" / "out.log").write_text("running\n")          # outputs do not block a run
    study.check_lock(pre, script, str(tmp_path))
    (tmp_path / "edgelab" / "a.py").write_text("x = 2\n")
    with pytest.raises(SystemExit, match="uncommitted"):
        study.check_lock(pre, script, str(tmp_path))
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "change")
    with pytest.raises(SystemExit, match="differ"):
        study.check_lock(pre, script, str(tmp_path))
    with pytest.raises(SystemExit, match="no code lock"):
        study.check_lock(reg.prereg("X-02", {"a": 2})["payload"], script, str(tmp_path))


def test_mean_se_paired_diff_and_smoke_spec():
    from edgelab import study
    assert study.mean_se([1, 2, 3]) == {"mean": 2.0, "se": 0.5774, "n": 3}
    d = study.paired_diff([1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.0])
    assert d["mean"] == 0.625 and d["ci95"][0] < 0.625 < d["ci95"][1]
    spec = {"population": {"n_traders": 2000}, "seeds": list(range(5000, 5020))}
    sm = study.smoke_spec(spec)
    assert sm["population"]["n_traders"] == 300 and sm["seeds"] == [9990, 9991] and spec["seeds"][0] == 5000


def test_bench3_continuous_edges_variable_n_and_optional_correlation():
    a = simulate.population_v3(1000, (50, 200), seed=21)
    b = simulate.population_v3(1000, (50, 200), seed=21)
    assert all(np.array_equal(x.pnl, y.pnl) for x, y in zip(a, b))
    n = np.array([t.pnl.size for t in a])
    assert n.min() >= 50 and n.max() <= 200 and len(set(n)) > 50
    mu = np.array([t.meta["mu"] for t in a])
    assert len(np.unique(mu)) == len(a) and 0.05 < np.mean(mu > 0.05) < 0.16
    plain = [t for t in a if t.kind == "plain"]
    assert all(t.meta["edge"] == pytest.approx(t.meta["mu"]) for t in plain)
    rng = np.random.default_rng(22)
    big = simulate.make_bench3_trader(rng, 200_000, 0.1, "plain")
    assert big.pnl.mean() == pytest.approx(0.1, abs=0.015)
    c = simulate.population_v3(300, 100, seed=23, corr_sd=0.5)
    r = np.corrcoef(c[0].pnl[:50], c[1].pnl[:50])
    assert c[0].pnl.size == 100 and np.isfinite(r).all()


def test_pooled_ratio_ci_keeps_empty_seeds():
    from edgelab import study
    r = study.pooled_ratio_ci([3, 0, 8, 4], [4, 0, 10, 5])
    assert r["ratio"] == round(15 / 19, 4) and r["den"] == 19
    assert r["ci95"][0] <= r["ratio"] <= r["ci95"][1]
    assert study.pooled_ratio_ci([0, 0], [0, 0])["ratio"] is None


def test_moderation_prior_recovers_variance_prior():
    from edgelab import skill
    rng = np.random.default_rng(1)
    s2 = 2.0 * rng.chisquare(10, 5000) / 10 * rng.chisquare(20, 5000) / 20
    d0, s02 = skill.moderation_prior(s2, np.full(5000, 20))
    assert 8 < d0 < 12 and 1.8 < s02 < 2.2
    tr = [rng.normal(0.3 if i < 50 else 0.0, 1, 40) for i in range(500)]
    p = skill.npmle_prob(tr, var="moderated", lik="t")["prob"]
    assert np.all((p >= 0) & (p <= 1 + 1e-9)) and skill.auc(p, np.arange(500) < 50) > 0.8


def test_drift_bench_and_detectors():
    from edgelab import drift
    pop = simulate.population_drift(400, 400, seed=31)
    dec = [t for t in pop if t.kind == "decaying"]
    assert len(dec) == 200 and all(150 <= t.meta["tau"] <= 300 for t in dec)
    assert all(t.meta["mu_pre"] > 0.10 for t in pop)
    assert all(t.true_mu[t.meta["tau"]] == -0.05 and t.true_mu[t.meta["tau"] - 1] > 0.10 for t in dec)
    rng = np.random.default_rng(32)
    z = rng.normal(0, 1, (2000, 300))
    h = drift.calibrate(lambda zz, hh: drift.cusum_down(zz, 0.1, hh), z, 0.10, 0.1, 50)
    assert abs(np.mean(drift.cusum_down(z, 0.1, h) >= 0) - 0.10) < 0.02
    shifted = z.copy(); shifted[:, 100:] -= 0.5
    assert np.mean(drift.cusum_down(shifted, 0.1, h) >= 100) > 0.9
    L = drift.calibrate(lambda zz, ll: drift.ewma_down(zz, 0.05, ll), z, 0.10, 0.1, 10)
    assert np.mean(drift.ewma_down(shifted, 0.05, L) >= 100) > 0.9
    zz = drift.standardise([t.pnl for t in pop], 100)
    assert zz.shape == (400, 300)


def test_tilt_bench_and_size_after_loss_test():
    from edgelab import behaviour
    a = simulate.population_tilt(600, 100, seed=41)
    b = simulate.population_v3(600, 100, seed=41)
    assert all(x.meta["edge"] == y.meta["edge"] for x, y in zip(a, b))
    plain = [t for t in a if t.kind == "plain"]
    assert np.std(np.log(plain[0].size)) == pytest.approx(0.30, abs=0.08)
    rng = np.random.default_rng(42)
    pnl = rng.normal(0, 1, 200)
    size = np.ones(200); size[1:][pnl[:-1] < 0] = 2.0
    size *= np.exp(0.3 * rng.standard_normal(200))
    assert behaviour.size_after_loss_p(pnl, size) < 1e-6
    assert behaviour.size_ratio_after_loss(pnl, size) == pytest.approx(2.0, rel=0.25)
    null = [behaviour.size_after_loss_p(rng.normal(0, 1, 100), np.exp(0.3 * rng.standard_normal(100))) for _ in range(400)]
    assert 0.06 < np.mean(np.array(null) < 0.10) < 0.15
    assert behaviour.size_after_loss_p([1, 1, 1, 1], [1, 1, 1, 1]) == 1.0
