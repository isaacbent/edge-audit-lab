"""Survival of funded accounts (SURVIVAL-01): Kaplan-Meier, Cox proportional hazards, Harrell's concordance.

time = days survived (event day, or last day observed when censored); event = 1 breach, 0 censored.
  - km_survival(time, event, t): Kaplan-Meier S(t) (Kaplan & Meier 1958); censored accounts leave the risk set
    without counting as breaches.
  - cox_fit(X, time, event): Cox (1972) partial likelihood, Breslow ties, Newton-Raphson; returns coefficients.
  - concordance(risk, time, event): Harrell's C, share of usable pairs where the account that breached first had
    the higher risk score (ties in risk = 1/2).
"""
from __future__ import annotations

import numpy as np


def km_survival(time, event, t) -> float:
    time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=bool)
    s = 1.0
    for u in np.unique(time[event & (time <= t)]):
        at_risk = np.sum(time >= u)
        d = np.sum(event & (time == u))
        s *= 1.0 - d / at_risk
    return float(s)


def cox_fit(X, time, event, iters: int = 50, ridge: float = 1e-6) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    time, event = np.asarray(time, dtype=float), np.asarray(event, dtype=bool)
    order = np.argsort(-time, kind="mergesort")              # descending time: risk sets are prefixes
    X, time, event = X[order], time[order], event[order]
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        w = np.exp(np.clip(X @ beta, -50, 50))
        s0 = np.cumsum(w)
        s1 = np.cumsum(w[:, None] * X, axis=0)
        s2 = np.cumsum(w[:, None, None] * X[:, :, None] * X[:, None, :], axis=0)
        last = np.searchsorted(-time, -time, side="right") - 1  # Breslow: risk set = everyone with time >= t_i
        ev = np.nonzero(event)[0]
        a0, a1, a2 = s0[last[ev]], s1[last[ev]], s2[last[ev]]
        mean = a1 / a0[:, None]
        grad = (X[ev] - mean).sum(axis=0) - ridge * beta
        hess = -(a2 / a0[:, None, None] - mean[:, :, None] * mean[:, None, :]).sum(axis=0) - ridge * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        beta = beta - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def concordance(risk, time, event) -> float:
    risk, time = np.asarray(risk, dtype=float), np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    num = den = 0.0
    for i in np.nonzero(event)[0]:
        later = time > time[i]
        den += later.sum()
        num += (risk[i] > risk[later]).sum() + 0.5 * (risk[i] == risk[later]).sum()
    return float(num / den) if den else float("nan")
