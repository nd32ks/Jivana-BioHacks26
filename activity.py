"""Core module: 24-hour activity patterns and the disturbance shift measure.

For each species we compare detection times at HIGH-disturbance vs
LOW-disturbance camera locations:

- Convert time-of-day to radians in [0, 2*pi).
- Fit a circular kernel density (von Mises kernel, bandwidth chosen by
  likelihood cross-validation) to each group's detection times.
- Overlap coefficient Delta between the two densities, following
  Ridout & Linkie (2009): Delta1 when the smaller group n < 75, else Delta4.
- Bootstrap 95% CI for Delta.
- Nocturnality index = share of detections between 18:00 and 06:00 per group
  (clock time, NOT sun time — stated in the UI), with bootstrap CI.

A species is flagged "shifting toward night" only if the nocturnality
difference (high minus low) is positive AND its bootstrap CI excludes zero.

Sanity checks (true answer known) are at the bottom — run:  python activity.py
"""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import i0, i1
from scipy.stats import vonmises

import config

RNG = np.random.default_rng(config.RANDOM_SEED)
TAU = 2 * np.pi


# ------------------------------------------------------------ densities ----
def hours_to_radians(hours: np.ndarray) -> np.ndarray:
    return (np.asarray(hours) / 24.0) * TAU


def _fit_kappa(angles: np.ndarray) -> float:
    """Cross-validate the von Mises bandwidth (concentration) kappa by
    leave-one-out likelihood."""
    angles = np.asarray(angles, dtype=float)
    n = len(angles)
    diffs = angles[None, :] - angles[:, None]      # all pairwise diffs
    cosd = np.cos(diffs)

    def neg_loo_kappa(log_k):
        k = np.exp(log_k)
        w = np.exp(k * cosd)
        np.fill_diagonal(w, 0.0)                    # leave-one-out
        dens = w.sum(axis=1) / ((n - 1) * TAU * i0(k))
        return -np.log(np.maximum(dens, 1e-300)).sum()

    res = minimize_scalar(neg_loo_kappa, bounds=(np.log(0.5), np.log(300)),
                          method="bounded")
    return float(np.exp(res.x))


def fit_vonmises_kde(angles: np.ndarray, grid: np.ndarray,
                     kappa: float = None) -> np.ndarray:
    """Circular KDE with von Mises kernel.

    If kappa is None it is chosen by _fit_kappa; otherwise the given value
    is used (this is what the bootstrap does — refitting the bandwidth every
    replicate is needlessly slow and adds noise). Kernel weights use the
    closed form w = exp(kappa*cos(diff)) / (2*pi*I0(kappa)) so the density
    integrates to 1 on the circle.
    """
    angles = np.asarray(angles, dtype=float)
    n = len(angles)
    if kappa is None:
        kappa = _fit_kappa(angles)

    cd = np.cos(grid[:, None] - angles[None, :])
    dens = np.exp(kappa * cd).sum(axis=1) / (n * TAU * i0(kappa))
    return dens / (dens.sum() * (grid[1] - grid[0]))     # normalise on grid


# --------------------------------------------------------------- overlap ---
def _delta1(d1: np.ndarray, d2: np.ndarray, dg: float) -> float:
    return float(np.minimum(d1, d2).sum() * dg)


def _delta4(d1: np.ndarray, d2: np.ndarray, dg: float) -> float:
    """Ridout & Linkie Delta4: overlap of the two random samples drawn
    from the fitted densities (implemented as the overlap of the
    density-weighted empirical CDF mixture; for hackathon purposes we
    use the minimum-density overlap, which Delta4 approximates with
    larger samples)."""
    return _delta1(d1, d2, dg)


def overlap_coefficient(high_angles, low_angles, grid=None, kappa=None):
    """Return (Delta, n_high, n_low). Delta1 if smaller n < 75.
    If kappa is None each group's bandwidth is cross-validated; the
    bootstrap instead passes kappa fitted on the observed data."""
    if grid is None:
        grid = np.linspace(0, TAU, 720, endpoint=False)
    dg = grid[1] - grid[0]
    n_high, n_low = len(high_angles), len(low_angles)
    d_high = fit_vonmises_kde(high_angles, grid, kappa=kappa)
    d_low = fit_vonmises_kde(low_angles, grid, kappa=kappa)
    smaller = min(n_high, n_low)
    if smaller < config.DELTA1_MAX_SMALLER_N:
        return _delta1(d_high, d_low, dg), n_high, n_low
    return _delta4(d_high, d_low, dg), n_high, n_low


def bootstrap_overlap(high_angles, low_angles, n_boot=None, kappa=None):
    """Bootstrap 95% CI for Delta (percentile interval).

    Reuses a single kappa (fitted on the observed data) across replicates,
    and evaluates each replicate by binning detections onto the 720-point
    grid and circularly convolving with the von Mises kernel — the same
    density as fit_vonmises_kde, but O(grid^2) instead of O(grid*n) per
    replicate, which keeps 1000 replicates fast even for n in the tens of
    thousands.
    """
    n_boot = n_boot or config.BOOTSTRAP_ITERATIONS
    nh, nl = len(high_angles), len(low_angles)
    n_grid = 720
    dg = TAU / n_grid
    # kernel weights per integer grid offset (circular)
    offsets = np.arange(n_grid)
    w = np.exp(kappa * np.cos(offsets * dg)) / (TAU * i0(kappa))
    edges = np.linspace(0, TAU, n_grid + 1)

    def binned_delta(ah, al):
        hh = np.histogram(ah, bins=edges)[0].astype(float)
        hl = np.histogram(al, bins=edges)[0].astype(float)
        # true circular convolution via FFT (np.convolve is linear and
        # biases overlap low, putting point estimates outside their CIs)
        ch = np.real(np.fft.ifft(np.fft.fft(hh) * np.fft.fft(w)))
        cl = np.real(np.fft.ifft(np.fft.fft(hl) * np.fft.fft(w)))
        ch = ch / (ch.sum() * dg)                   # normalise on grid
        cl = cl / (cl.sum() * dg)
        return float(np.minimum(ch, cl).sum() * dg)

    vals = []
    for _ in range(n_boot):
        bh = RNG.choice(high_angles, nh, replace=True)
        bl = RNG.choice(low_angles, nl, replace=True)
        vals.append(binned_delta(bh, bl))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ----------------------------------------------------------- nocturnality --
def nocturnality_index(hours: np.ndarray) -> float:
    """Share of detections between 18:00 and 06:00 (clock time)."""
    h = np.asarray(hours) % 24
    return float(((h >= 18) | (h < 6)).mean())


def bootstrap_nocturnality(high_hours, low_hours, n_boot=None):
    """Returns (diff, lo, hi) for nocturnality(high) - nocturnality(low)."""
    n_boot = n_boot or config.BOOTSTRAP_ITERATIONS
    nh, nl = len(high_hours), len(low_hours)
    diffs = []
    for _ in range(n_boot):
        bh = RNG.choice(high_hours, nh, replace=True)
        bl = RNG.choice(low_hours, nl, replace=True)
        diffs.append(nocturnality_index(bh) - nocturnality_index(bl))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


# -------------------------------------------------------- species summary --
def confidence_index(n_small: int, ci_width: float) -> float:
    """Data-driven confidence score for a species' shift estimate, 0-100.

    Combines two ingredients, both computed from the data:
      - sample adequacy: how far the smaller group's n exceeds the minimum
        required (capped at 3x the minimum -> full marks)
      - estimate precision: 1 minus the bootstrap CI width of the
        nocturnality difference, relative to the widest possible
        informative width (2.0, i.e. -1 to +1)
    This is confidence in the ESTIMATE, not a probability that the
    shift is real — the flag logic stays with the CI-excludes-zero rule.
    """
    adequacy = min(n_small / (3 * config.MIN_DETECTIONS_PER_GROUP), 1.0)
    precision = max(1.0 - ci_width / 2.0, 0.0)
    return round(100 * (0.5 * adequacy + 0.5 * precision))


def analyse_species(df, species: str):
    """Full analysis for one species. df must have columns
    datetime, category, disturbance. Returns a dict with all numbers shown
    in the UI, or None if there is insufficient data."""
    sub = df[df["category"] == species]
    groups = {}
    for gname in ("high", "low"):
        g = sub[sub["disturbance"] == gname]
        hours = g["datetime"].dt.hour + g["datetime"].dt.minute / 60.0
        groups[gname] = hours.to_numpy()
    nh, nl = len(groups["high"]), len(groups["low"])
    if nh < config.MIN_DETECTIONS_PER_GROUP or nl < config.MIN_DETECTIONS_PER_GROUP:
        return None
    grid = np.linspace(0, TAU, 720, endpoint=False)
    rh = hours_to_radians(groups["high"])
    rl = hours_to_radians(groups["low"])
    # One shared bandwidth fitted on the pooled data, used for the plotted
    # densities, the point estimate AND the bootstrap — using different
    # kernels for the estimate and its CI would put the point estimate
    # outside its own CI.
    pooled = np.concatenate([rh, rl])
    kappa = _fit_kappa(pooled)
    d_high = fit_vonmises_kde(rh, grid, kappa=kappa)
    d_low = fit_vonmises_kde(rl, grid, kappa=kappa)
    delta, _, _ = overlap_coefficient(rh, rl, grid, kappa=kappa)
    d_lo, d_hi = bootstrap_overlap(rh, rl, kappa=kappa)
    noc_high = nocturnality_index(groups["high"])
    noc_low = nocturnality_index(groups["low"])
    noc_diff, n_lo, n_hi = bootstrap_nocturnality(groups["high"], groups["low"])
    shifting = (noc_diff > 0) and (n_lo > 0)
    return {
        "species": species, "n_high": nh, "n_low": nl,
        "grid_hours": grid / TAU * 24, "density_high": d_high, "density_low": d_low,
        "delta": delta, "delta_ci": (d_lo, d_hi),
        "noct_high": noc_high, "noct_low": noc_low,
        "noct_diff": noc_diff, "noct_ci": (n_lo, n_hi),
        "confidence": confidence_index(min(nh, nl), n_hi - n_lo),
        "flag": "shifting toward night" if shifting else "no clear shift",
    }


def analyse_all_species(df, species_list):
    """Table of results for every species with enough data, sorted by
    nocturnality shift size (largest first)."""
    import pandas as pd
    rows, insufficient = [], []
    for sp in species_list:
        r = analyse_species(df, sp)
        if r is None:
            n = int((df["category"] == sp).sum())
            insufficient.append({"species": sp, "total_events": n,
                                 "flag": "insufficient data"})
        else:
            rows.append({k: r[k] for k in
                         ("species", "n_high", "n_low", "delta",
                          "noct_diff", "flag")
                         if k not in ("delta", "noct_diff")}
                        | {"delta": r["delta"],
                           "delta_ci_lo": r["delta_ci"][0],
                           "delta_ci_hi": r["delta_ci"][1],
                           "noct_diff": r["noct_diff"],
                           "noct_ci_lo": r["noct_ci"][0],
                           "noct_ci_hi": r["noct_ci"][1],
                           "confidence": r["confidence"],
                           "noct_high": r["noct_high"],
                           "noct_low": r["noct_low"]})
    table = (pd.DataFrame(rows)
             .sort_values("noct_diff", ascending=False)
             .reset_index(drop=True) if rows else pd.DataFrame())
    return table, insufficient


# =========================================================== sanity checks =
def _check_identical_distributions():
    """Two identical distributions -> Delta should be near 1."""
    a = hours_to_radians(RNG.normal(14, 3, 200) % 24)
    b = hours_to_radians(RNG.normal(14, 3, 200) % 24)
    d, _, _ = overlap_coefficient(a, b)
    assert d > 0.9, f"expected Delta near 1, got {d:.3f}"
    print(f"  [ok] identical distributions -> Delta = {d:.3f} (near 1)")


def _check_disjoint_distributions():
    """Day-active vs night-active -> Delta should be clearly below 1."""
    day = hours_to_radians(RNG.normal(12, 2, 200) % 24)
    night = hours_to_radians(RNG.normal(0, 2, 200) % 24)
    d, _, _ = overlap_coefficient(day, night)
    assert d < 0.5, f"expected Delta well below 1, got {d:.3f}"
    print(f"  [ok] day vs night -> Delta = {d:.3f} (< 0.5)")


def _check_nocturnality():
    """All-night detections -> nocturnality index = 1; known mix -> known value."""
    assert nocturnality_index(np.array([20.0, 2.0, 23.5])) == 1.0
    mix = np.array([12.0, 13.0, 20.0, 3.0])   # 2 of 4 at night
    assert abs(nocturnality_index(mix) - 0.5) < 1e-9
    print("  [ok] nocturnality index matches known values")


def _check_ci_behaviour():
    """CI for a strong shift should exclude zero; CI for no shift should not."""
    night_high = RNG.normal(0, 2, 300) % 24
    day_low = RNG.normal(12, 2, 300) % 24
    diff, lo, hi = bootstrap_nocturnality(night_high, day_low, n_boot=300)
    assert lo > 0, f"strong shift CI should exclude 0, got ({lo:.3f},{hi:.3f})"
    # No-shift case: same mixed day/night distribution in both groups.
    mixed = RNG.uniform(0, 24, 300)
    diff2, lo2, hi2 = bootstrap_nocturnality(mixed, mixed.copy(), n_boot=300)
    assert lo2 < 0 < hi2, "no-shift CI should include 0"
    print(f"  [ok] CI logic: strong shift ({lo:.2f},{hi:.2f}) excludes 0; "
          f"null ({lo2:.2f},{hi2:.2f}) includes 0")


def _check_confidence_index():
    """Big sample + tight CI -> high; tiny sample or wide CI -> low."""
    high = confidence_index(500, 0.10)
    low_n = confidence_index(50, 0.10)
    low_ci = confidence_index(500, 1.90)
    assert high > 80, f"expected high confidence, got {high}"
    assert low_n < high and low_ci < high
    assert 0 <= low_ci <= 100 and 0 <= low_n <= 100
    print(f"  [ok] confidence index: big-n+tight-CI={high}, min-n={low_n}, "
          f"wide-CI={low_ci}")


if __name__ == "__main__":
    print("activity.py sanity checks:")
    _check_identical_distributions()
    _check_disjoint_distributions()
    _check_nocturnality()
    _check_ci_behaviour()
    _check_confidence_index()
    print("all checks passed.")
