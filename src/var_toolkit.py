"""Reduced-form VAR estimation, Cholesky IRFs, FEVD, bootstrap bands, and
counterfactual dynamics.

Written by hand rather than via `statsmodels.tsa.VAR` because the counterfactual
exercise needs individual structural coefficients zeroed and the system re-solved,
which no library exposes. `validate_toolkit()` checks the routines against
statsmodels and should be run once per session, before any estimation.

The model is

    A y_t = c + sum_{k=1}^{p} B_k y_{t-k} + u_t

estimated in reduced form, with A unit lower triangular (Cholesky).

All functions take and return plain numpy arrays; nothing here knows about the
notebook's specifications, variable names, or pandas objects.
"""

import numpy as np


def build_lagmat(Y, p, det=None):
    T = Y.shape[0]
    X = [np.ones((T - p, 1))]
    if det is not None:
        X.append(det[p:])
    X += [Y[p - lag: T - lag] for lag in range(1, p + 1)]
    return np.hstack(X), Y[p:]


def fit_var(Y, p, det=None):
    X, Yt = build_lagmat(Y, p, det)
    B, *_ = np.linalg.lstsq(X, Yt, rcond=None)
    U = Yt - X @ B
    Sigma = U.T @ U / (Yt.shape[0] - X.shape[1])
    n = Y.shape[1]
    ndet = 1 + (0 if det is None else det.shape[1])
    A = [B[ndet + n*(i - 1): ndet + n*i, :].T for i in range(1, p + 1)]
    D = X[:, :ndet] @ B[:ndet]
    return dict(B=B, A=A, U=U, Sigma=Sigma, D=D, p=p, n=n, T=Yt.shape[0])


def ma_reps(A, H):
    n, p = A[0].shape[0], len(A)
    Psi = [np.eye(n)]
    for h in range(1, H + 1):
        S = np.zeros((n, n))
        for i in range(1, min(h, p) + 1):
            S += A[i - 1] @ Psi[h - i]
        Psi.append(S)
    return Psi


def irf_chol(A, Sigma, H, shock=0, P=None):
    P = np.linalg.cholesky(Sigma) if P is None else P
    return np.array([Psi @ P for Psi in ma_reps(A, H)])[:, :, shock]


def fevd(A, Sigma, horizons):
    P = np.linalg.cholesky(Sigma)
    Th = np.array([Psi @ P for Psi in ma_reps(A, max(horizons))])
    out = {}
    for H in horizons:
        contrib = (Th[:H] ** 2).sum(axis=0)
        out[H] = contrib / contrib.sum(axis=1, keepdims=True)
    return out


def simulate_var(D, A, U, Y0):
    p, n, T = len(A), A[0].shape[0], U.shape[0]
    Y = np.zeros((T + p, n)); Y[:p] = Y0
    for t in range(p, T + p):
        y = D[t - p].copy()
        for i in range(1, p + 1):
            y = y + A[i - 1] @ Y[t - i]
        Y[t] = y + U[t - p]
    return Y


def irf_with_bands(Y, p, H, shock=0, det=None, nboot=2000, ci=90, seed=None,
                   transform=None):
    rng = np.random.default_rng(seed)
    m = fit_var(Y, p, det)
    point = transform(m, H, shock) if transform else irf_chol(m["A"], m["Sigma"], H, shock)
    U = m["U"] - m["U"].mean(axis=0); Tb = U.shape[0]
    draws, failed = [], 0
    for _ in range(nboot):
        Yb = simulate_var(m["D"], m["A"], U[rng.integers(0, Tb, Tb)], Y[:p])
        try:
            mb = fit_var(Yb, p, det)
            draws.append(transform(mb, H, shock) if transform
                         else irf_chol(mb["A"], mb["Sigma"], H, shock))
        except np.linalg.LinAlgError:
            failed += 1
    if failed:
        print(f"  ({failed} bootstrap draws discarded: non-PSD covariance)")
    draws = np.array(draws)
    a = (100 - ci)/2
    lo, hi = np.percentile(draws, [a, 100 - a], axis=0)
    return point, lo, hi, m


def to_structural(A, Sigma):
    P = np.linalg.cholesky(Sigma)
    D = np.diag(np.diag(P))
    L = P @ np.linalg.inv(D)
    return np.linalg.inv(L), D, [np.linalg.inv(L) @ Ai for Ai in A]


def counterfactual_dynamics(A, Sigma, keep, eq=None):
    """Zero every coefficient in structural equation `eq` (default: the last) except those
    on variables in `keep` and on `eq` itself, contemporaneously and at every lag."""
    n = Sigma.shape[0]
    eq = n - 1 if eq is None else eq
    A0, D, Bs = to_structural(A, Sigma)
    keep_set = set(keep) | {eq}
    A0c = A0.copy()
    for j in range(eq):
        if j not in keep_set:
            A0c[eq, j] = 0.0
    Bsc = [Bi.copy() for Bi in Bs]
    for Bi in Bsc:
        for j in range(n):
            if j not in keep_set:
                Bi[eq, j] = 0.0
    Linv = np.linalg.inv(A0c)
    return [Linv @ Bi for Bi in Bsc], Linv @ D


def make_cf_transform(keep, eq):
    """An `irf_with_bands` transform that applies `counterfactual_dynamics`."""
    def f(m, H, shock):
        Ac, Pc = counterfactual_dynamics(m["A"], m["Sigma"], keep=keep, eq=eq)
        return irf_chol(Ac, m["Sigma"], H, shock, P=Pc)
    return f


def validate_toolkit():
    """Check IRFs, FEVD, the counterfactual round-trip, and the impact normalization
    against statsmodels on simulated data. Raises AssertionError on failure."""
    from statsmodels.tsa.api import VAR as smVAR
    rng = np.random.default_rng(0)
    n, p, T = 5, 3, 220
    A_true = [0.5*np.eye(n) + 0.02*rng.standard_normal((n, n)), 0.2*np.eye(n), 0.05*np.eye(n)]
    S = np.eye(n) + 0.2*(np.ones((n, n)) - np.eye(n))
    Yv = simulate_var(np.zeros((T, n)), A_true,
                      rng.multivariate_normal(np.zeros(n), S, T), np.zeros((p, n)))
    m, ref = fit_var(Yv, p), smVAR(Yv).fit(p)
    d_irf  = np.abs(irf_chol(m["A"], m["Sigma"], 20, 0) - ref.irf(20).orth_irfs[:, :, 0]).max()
    d_fevd = np.abs(fevd(m["A"], m["Sigma"], [12])[12] - ref.fevd(12).decomp[:, -1, :]).max()
    Ak, Pk = counterfactual_dynamics(m["A"], m["Sigma"], keep=tuple(range(n)))
    d_cf   = np.abs(irf_chol(Ak, m["Sigma"], 20, 0, P=Pk)
                    - irf_chol(m["A"], m["Sigma"], 20, 0)).max()
    d_sd   = abs(irf_chol(m["A"], m["Sigma"], 5, 0)[0, 0] - np.sqrt(m["Sigma"][0, 0]))
    print(f"  IRF   vs statsmodels                 : {d_irf:.2e}")
    print(f"  FEVD  vs statsmodels                 : {d_fevd:.2e}")
    print(f"  counterfactual round-trip (keep all) : {d_cf:.2e}")
    print(f"  impact = 1 s.d. of shock             : {d_sd:.2e}")
    assert max(d_irf, d_fevd, d_cf, d_sd) < 1e-8, "toolkit validation FAILED"
    print("  all checks passed")
