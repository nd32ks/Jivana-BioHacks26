"""Simple single-season occupancy model (maximum likelihood).

For each species we build a detection history per location using fixed
7-day occasions, then fit a single-season model with CONSTANT occupancy psi
and CONSTANT detection probability p by maximum likelihood. This is the
standard MacKenzie et al. (2002) model with no covariates.

UI label (required wording):
    "occupancy probability (share of sites used, accounting for imperfect
     detection)"

Occupancy is the probability a site is USED, corrected for imperfect
detection. It is NOT abundance and must never be presented as such.

Sanity check at the bottom simulates data with known psi and p and confirms
the estimates fall inside the 95% CI. Run:  python occupancy.py
"""

import numpy as np
from scipy.optimize import minimize

import config

RNG = np.random.default_rng(config.RANDOM_SEED)
OCCASION_DAYS = 7  # fixed occasion length


def build_detection_history(df, species: str, occasion_days: int = OCCASION_DAYS):
    """Binary detection history: rows = locations, cols = occasions."""
    sub = df[df["category"] == species]
    histories = {}
    # Iterate ALL locations (including ones where the species was never
    # detected): all-zero histories are what make occupancy (psi)
    # identifiable — an all-zero history means "unused site OR used but
    # never detected". Dropping them would bias psi toward 1.
    for loc, gloc in df.groupby("location"):
        g = sub[sub["location"] == loc]
        days = ((g["datetime"] - gloc["datetime"].min()).dt.days
                // occasion_days)
        hist = np.zeros(int(days.max()) + 1 if len(days) else
                        max((gloc["datetime"].max()
                             - gloc["datetime"].min()).days // occasion_days, 0) + 1,
                        dtype=int)
        for d in days.dropna():
            hist[int(d)] = 1
        histories[loc] = hist
    return histories


def _neg_log_lik(params, histories):
    """Likelihood for constant psi, p (MacKenzie et al. 2002, eq. for L)."""
    psi = 1 / (1 + np.exp(-params[0]))
    p = 1 / (1 + np.exp(-params[1]))
    nll = 0.0
    for h in histories.values():
        n, k = len(h), int(h.sum())
        if k > 0:
            # detected at least once: psi * p^k * (1-p)^(n-k)
            nll -= np.log(psi) + k * np.log(p) + (n - k) * np.log(1 - p)
        else:  # never happens here (we drop all-zero rows) but kept for completeness
            nll -= np.log((1 - psi) + psi * (1 - p) ** n)
    return nll


def fit_occupancy(histories, trace: list = None) -> dict | None:
    """MLE of (psi, p) with 95% CIs from the inverse Hessian (observed
    information). Returns None if the optimiser fails.

    If `trace` is a list, (negative log-likelihood, psi, p) is appended at
    every optimizer step — used by the Live page to animate convergence.
    """
    if len(histories) < 5:
        return None

    def _callback(xk):
        if trace is not None:
            trace.append((_neg_log_lik(xk, histories),
                          1 / (1 + np.exp(-xk[0])),
                          1 / (1 + np.exp(-xk[1]))))

    res = minimize(_neg_log_lik, x0=np.array([0.0, 0.0]),
                   args=(histories,), method="BFGS",
                   options={"gtol": 1e-4, "eps": 1e-6},
                   callback=_callback)
    if not res.success:
        return None
    psi = 1 / (1 + np.exp(-res.x[0]))
    p = 1 / (1 + np.exp(-res.x[1]))
    try:
        se = np.sqrt(np.diag(res.hess_inv))
        lo = 1 / (1 + np.exp(-(res.x - 1.96 * se)))
        hi = 1 / (1 + np.exp(-(res.x + 1.96 * se)))
        psi_ci = (float(np.clip(lo[0], 0, 1)), float(np.clip(hi[0], 0, 1)))
        p_ci = (float(np.clip(lo[1], 0, 1)), float(np.clip(hi[1], 0, 1)))
    except Exception:
        psi_ci = (np.nan, np.nan)
        p_ci = (np.nan, np.nan)
    return {"n_sites": len(histories), "psi": float(psi), "psi_ci": psi_ci,
            "p": float(p), "p_ci": p_ci}


def occupancy_by_group(df, species: str) -> dict:
    """Fit separately in the high- and low-disturbance location groups."""
    out = {}
    for gname in ("high", "low"):
        g = df[df["disturbance"] == gname]
        hist = build_detection_history(g, species)
        out[gname] = fit_occupancy(hist)
    return out


# =========================================================== sanity check ==
def _simulate(psi_true, p_true, n_sites=200, n_occasions=10, seed=1):
    """Simulate detection histories with known psi, p."""
    rng = np.random.default_rng(seed)
    occupied = rng.random(n_sites) < psi_true
    hists = {}
    for i, occ in enumerate(occupied):
        if occ:
            det = (rng.random(n_occasions) < p_true).astype(int)
            hists[i] = det
        else:
            hists[i] = np.zeros(n_occasions, dtype=int)
    return hists


def _check_recovery():
    psi_true, p_true = 0.7, 0.4
    hists = _simulate(psi_true, p_true)
    fit = fit_occupancy(hists)
    assert fit is not None, "optimiser failed on simulated data"
    lo, hi = fit["psi_ci"]
    assert lo <= psi_true <= hi, \
        f"true psi={psi_true} outside CI ({lo:.3f},{hi:.3f})"
    lo_p, hi_p = fit["p_ci"]
    assert lo_p <= p_true <= hi_p, \
        f"true p={p_true} outside CI ({lo_p:.3f},{hi_p:.3f})"
    print(f"  [ok] psi: true {psi_true}, estimated {fit['psi']:.3f} "
          f"CI ({lo:.3f},{hi:.3f})")
    print(f"  [ok] p:   true {p_true}, estimated {fit['p']:.3f} "
          f"CI ({lo_p:.3f},{hi_p:.3f})")
    print("all checks passed.")


if __name__ == "__main__":
    print("occupancy.py sanity check (known psi, p -> estimates recovered):")
    _check_recovery()
