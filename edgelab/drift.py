"""Detecting a funded trader's edge loss as trades arrive (DRIFT-01).

Each trader is standardised by their own warm-up record (first `warmup` trades: mean m0, sd s0), then monitored:
  - CUSUM (Page 1954), one-sided downward: S_t = max(0, S_{t-1} - z_t - k), alarm when S_t > h
  - EWMA chart (Roberts 1959): E_t = (1 - lam) E_{t-1} + lam z_t, E_0 = 0, alarm when
    E_t < -L * sqrt(lam / (2 - lam) * (1 - (1 - lam)^(2t)))
with z_t = (x_t - m0) / s0. Vectorised over traders; returns the index (in the full trade list) of the first
alarm, or -1. `calibrate` finds the threshold that gives a target alarm rate on in-control (no change) series.
"""
from __future__ import annotations

import numpy as np


def standardise(pnls, warmup: int):
    """Monitoring-period z-scores, one row per trader (all lists must have the same length)."""
    x = np.asarray(pnls, dtype=float)
    m0 = x[:, :warmup].mean(axis=1, keepdims=True)
    s0 = np.maximum(x[:, :warmup].std(axis=1, ddof=1, keepdims=True), 1e-12)
    return (x[:, warmup:] - m0) / s0


def cusum_down(z, k: float, h: float, warmup: int = 0) -> np.ndarray:
    s = np.zeros(z.shape[0])
    first = np.full(z.shape[0], -1)
    for t in range(z.shape[1]):
        s = np.maximum(0.0, s - z[:, t] - k)
        hit = (s > h) & (first < 0)
        first[hit] = t + warmup
    return first


def ewma_down(z, lam: float, L: float, warmup: int = 0) -> np.ndarray:
    e = np.zeros(z.shape[0])
    first = np.full(z.shape[0], -1)
    for t in range(z.shape[1]):
        e = (1 - lam) * e + lam * z[:, t]
        lim = L * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * (t + 1))))
        hit = (e < -lim) & (first < 0)
        first[hit] = t + warmup
    return first


def calibrate(alarm_fn, z, target: float, lo: float, hi: float, iters: int = 40) -> float:
    """Threshold (h for CUSUM, L for EWMA) whose alarm rate on in-control rows z is closest to target
    (alarm rate decreases with the threshold)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if np.mean(alarm_fn(z, mid) >= 0) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
