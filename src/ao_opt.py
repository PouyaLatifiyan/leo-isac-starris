"""
Phase 3: Alternating Optimization (AO) for the near-field secure ISAC
STAR-RIS LEO system, with benchmarks.

Algorithm (matches Section IV of the paper draft):
  - W-step  : satellite beamformers via SCA (concave lower bounds on secrecy rate)
  - Theta-step: STAR-RIS coefficients via SCA with per-element disk relaxation
                |theta_t,m|^2 + |theta_r,m|^2 <= 1 (convex hull of eq. (9c))
Benchmarks:
  1) Proposed AO (STAR-RIS)
  2) Random-phase STAR-RIS + MRT
  3) Conventional reflecting-only RIS (AO, transmission side unserved)
  4) No RIS (direct satellite-user NLoS link, MRT)
Plus: convergence curve, CRB near-field vs far-field localization accuracy.
"""

import numpy as np
import cvxpy as cp
import time
import sim_skeleton as ss

def robust_solve(prob):
    try:
        prob.solve(solver=cp.SCS, eps=1e-7, max_iters=20000)
    except Exception:
        return False
    return prob.status in ("optimal", "optimal_inaccurate")

LN2 = np.log(2)

# ------------------------------------------------------------------ setup
def setup_system(P, rng, M=None):
    if M is not None:
        P.M = M
    G = ss.sat_to_ris_channel(P, doppler_Hz=20e3, t=0.0, rng=rng)
    users = [("t", 0.3, 30.0), ("t", -0.4, 45.0), ("r", 0.5, 25.0), ("r", -0.2, 55.0)]
    b = {k: ss.nearfield_steering(P, th, r) for k, (_, th, r) in enumerate(users)}
    beta = {k: ss.ris_to_node_large_scale(P, r) for k, (_, th, r) in enumerate(users)}
    b_a = ss.nearfield_steering(P, 0.1, P.r_spoofer)
    beta_a = ss.ris_to_node_large_scale(P, P.r_spoofer)
    beta_spoof2user = ss.ris_to_node_large_scale(P, 50.0)   # spoofing interference path
    return dict(G=G, users=users, b=b, beta=beta, b_a=b_a, beta_a=beta_a,
                beta_su=beta_spoof2user)

def channels_from_theta(sysd, P, theta_t, theta_r):
    """Return H (K,N), h_a (N,) for given STAR-RIS coefficients."""
    Th = {"t": np.diag(theta_t), "r": np.diag(theta_r)}
    H = np.array([ss.cascaded_channel(sysd["G"], sysd["b"][k], Th[side], sysd["beta"][k])
                  for k, (side, _, _) in enumerate(sysd["users"])])
    h_a = ss.cascaded_channel(sysd["G"], sysd["b_a"], Th["r"], sysd["beta_a"])
    return H, h_a

def sum_secrecy_rate(H, W, h_a, P, an_frac=0.2):
    """Compute sum secrecy rate with fixed isotropic AN (fraction an_frac of Pmax)."""
    K, N = H.shape
    pw = (1 - an_frac) * np.sum(np.abs(W)**2) / P.Pmax  # normalized W power usage
    Z = (an_frac * P.Pmax / N) * np.eye(N)
    interf_u = P.Pa * 0.0  # user-side spoofing handled via beta_su below
    total = 0.0
    for k in range(K):
        s_k  = np.abs(H[k].conj() @ W[:, k])**2
        i_k  = sum(np.abs(H[k].conj() @ W[:, j])**2 for j in range(K) if j != k)
        i_k += np.real(H[k].conj() @ Z @ H[k]) + P.Pa * P.beta_su if hasattr(P, "beta_su") else 0
        i_k += P.sigma2
        s_ak = np.abs(h_a.conj() @ W[:, k])**2
        i_ak = sum(np.abs(h_a.conj() @ W[:, j])**2 for j in range(K) if j != k)
        i_ak += np.real(h_a.conj() @ Z @ h_a) + P.sigma2
        total += max(0.0, np.log2(1 + s_k/i_k) - np.log2(1 + s_ak/i_ak))
    return total

# ------------------------------------------------------------------ SCA W-step
def sca_W_step(H, h_a, P, W0, n_inner=3, an_frac=0.2):
    """Optimize W given fixed channels. SCA with concave lower bounds (see draft).

    All quantities are noise-normalized (divided by sigma2) for solver scaling.
    """
    s2 = np.sqrt(P.sigma2)
    H = H / s2; h_a = h_a / s2; W0 = W0 / s2
    K, N = H.shape
    Pw = (1 - an_frac) * P.Pmax / P.sigma2
    Z = (an_frac * (P.Pmax/P.sigma2) / N) * np.eye(N)
    an_u = np.array([np.real(H[k].conj() @ Z @ H[k]) for k in range(K)])
    an_a = np.real(h_a.conj() @ Z @ h_a)
    spoof_u = P.Pa * P.beta_su / P.sigma2  # spoofing interference at each user
    SIG = 1.0

    W_cur = W0.copy()
    for _ in range(n_inner):
        w = [cp.Variable(N, complex=True) for _ in range(K)]
        obj = 0
        for k in range(K):
            # current-point quantities
            I0 = sum(np.abs(H[k].conj() @ W_cur[:, j])**2 for j in range(K) if j != k) \
                 + an_u[k] + spoof_u + SIG
            s0 = np.abs(H[k].conj() @ W_cur[:, k])**2
            Aa0 = sum(np.abs(h_a.conj() @ W_cur[:, j])**2 for j in range(K)) + an_a + SIG
            Ia0 = Aa0 - np.abs(h_a.conj() @ W_cur[:, k])**2

            # affine (tangent) lower bounds of convex quadratics in w
            t_sk = I0 + s0 + 2*cp.real((H[k].conj() @ W_cur[:, k]).conj()
                                       * (H[k].conj() @ w[k] - H[k].conj() @ W_cur[:, k]))
            I_k = sum(cp.square(cp.abs(H[k].conj() @ w[j])) for j in range(K) if j != k) \
                  + an_u[k] + spoof_u + SIG
            A_a = sum(cp.square(cp.abs(h_a.conj() @ w[j])) for j in range(K)) + an_a + SIG
            t_Ia = Ia0 + 2*cp.real(sum((h_a.conj() @ W_cur[:, j]).conj()
                                       * (h_a.conj() @ w[j] - h_a.conj() @ W_cur[:, j])
                                       for j in range(K) if j != k))

            obj += cp.log(t_sk)/LN2 - np.log2(I0) - (I_k - I0)/(I0*LN2) \
                   - np.log2(Aa0) - (A_a - Aa0)/(Aa0*LN2) + cp.log(t_Ia)/LN2

        cons = [cp.sum_squares(cp.hstack([cp.vec(wk) for wk in w])) <= Pw, t_Ia >= 1e-30]
        prob = cp.Problem(cp.Maximize(obj), cons)
        if not robust_solve(prob):
            break
        vals = [wk.value for wk in w]
        if any(v is None or not np.all(np.isfinite(v)) for v in vals):
            break
        W_new = np.column_stack(vals)
        if np.linalg.norm(W_new - W_cur) < 1e-6 * max(1, np.linalg.norm(W_cur)):
            W_cur = W_new
            break
        W_cur = W_new
    return W_cur

# ------------------------------------------------------------------ SCA Theta-step
def sca_theta_step(sysd, P, W, th0, n_inner=3, an_frac=0.2):
    """Optimize (theta_t, theta_r) given W. Linear maps: h_k^H w_j = theta_chi^T c_kj."""
    K = W.shape[1]
    M = P.M
    s2 = np.sqrt(P.sigma2)
    W = W / s2                      # noise-normalized
    Z = (an_frac * (P.Pmax/P.sigma2) / P.N) * np.eye(P.N)
    SIG = 1.0
    theta_t0, theta_r0 = th0

    # linear coefficient vectors c_kj (user k, beam j) and c_aj (spoofer)
    def cvecs(b_k, beta_k):
        return {j: np.sqrt(beta_k) * (np.conj(b_k) * (sysd["G"] @ W[:, j])) for j in range(K)}
    C  = {k: cvecs(sysd["b"][k], sysd["beta"][k]) for k in range(K)}
    Ca = cvecs(sysd["b_a"], sysd["beta_a"])
    side = [u[0] for u in sysd["users"]]

    # fixed AN/noise terms evaluated at current theta (constants in this step)
    H0, h_a0 = channels_from_theta(sysd, P, theta_t0, theta_r0)
    H0 = H0 / s2; h_a0 = h_a0 / s2
    an_u = np.array([np.real(H0[k].conj() @ Z @ H0[k]) for k in range(K)])
    an_a = np.real(h_a0.conj() @ Z @ h_a0)
    spoof_u = P.Pa * P.beta_su / P.sigma2

    th_t, th_r = theta_t0.copy(), theta_r0.copy()
    for _ in range(n_inner):
        vt = cp.Variable(M, complex=True)
        vr = cp.Variable(M, complex=True)
        v = {"t": vt, "r": vr}
        th0_map = {"t": th_t, "r": th_r}

        obj = 0
        for k in range(K):
            chi = side[k]
            # current values
            def cur(kk, jj):
                return (th0_map[side[kk]] @ C[kk][jj])
            I0 = sum(np.abs(cur(k, j))**2 for j in range(K) if j != k) + an_u[k] + spoof_u + SIG
            s0 = np.abs(cur(k, k))**2
            Aa0 = sum(np.abs(th_r @ Ca[j])**2 for j in range(K)) + an_a + SIG
            Ia0 = Aa0 - np.abs(th_r @ Ca[k])**2

            t_sk = I0 + s0 + 2*cp.real(np.conj(cur(k, k)) * (v[chi] @ C[k][k] - cur(k, k)))
            I_k = sum(cp.square(cp.abs(v[chi] @ C[k][j])) for j in range(K) if j != k) \
                  + an_u[k] + spoof_u + SIG
            A_a = sum(cp.square(cp.abs(vr @ Ca[j])) for j in range(K)) + an_a + SIG
            t_Ia = Ia0 + 2*cp.real(sum(np.conj(th_r @ Ca[j]) * (vr @ Ca[j] - th_r @ Ca[j])
                                       for j in range(K) if j != k))

            obj += cp.log(t_sk)/LN2 - np.log2(I0) - (I_k - I0)/(I0*LN2) \
                   - np.log2(Aa0) - (A_a - Aa0)/(Aa0*LN2) + cp.log(t_Ia)/LN2

        cons = [t_Ia >= 1e-30] + [cp.abs(vt[m])**2 + cp.abs(vr[m])**2 <= 1 for m in range(M)]
        prob = cp.Problem(cp.Maximize(obj), cons)
        if not robust_solve(prob) or vt.value is None or vr.value is None \
           or not np.all(np.isfinite(vt.value)) or not np.all(np.isfinite(vr.value)):
            break
        # normalize to satisfy energy conservation with equality
        nt = np.sqrt(np.abs(vt.value)**2 + np.abs(vr.value)**2)
        nt = np.maximum(nt, 1e-12)
        th_t, th_r = vt.value/nt, vr.value/nt
    return th_t, th_r

# ------------------------------------------------------------------ AO driver
def run_ao(sysd, P, rng, n_outer=8, mode="star", init_theta=None):
    """mode: 'star' | 'conv' (reflection-only RIS)."""
    K = len(sysd["users"])
    if init_theta is not None:
        theta_t, theta_r = init_theta
    else:
        theta_t, theta_r = ss.make_star_ris_coeffs(P, rng)
    if mode == "conv":
        theta_t = np.zeros(P.M, dtype=complex)
    H, h_a = channels_from_theta(sysd, P, theta_t, theta_r)
    norms = np.linalg.norm(H, axis=1)
    norms = np.where(norms < 1e-30, 1.0, norms)   # zero channels (conv-RIS T-side) -> zero BF
    W = H.T / norms * np.sqrt((1-0.2)*P.Pmax/K)

    hist = []
    best = -1.0
    state = (W, theta_t, theta_r)
    for it in range(n_outer):
        W_new = sca_W_step(H, h_a, P, W)
        if mode == "star":
            tt_new, tr_new = sca_theta_step(sysd, P, W_new, (theta_t, theta_r))
        else:  # conventional RIS: only reflection side optimized
            _, tr_new = sca_theta_step(sysd, P, W_new, (np.zeros(P.M, dtype=complex), theta_r))
            tt_new = np.zeros(P.M, dtype=complex)
        H_new, ha_new = channels_from_theta(sysd, P, tt_new, tr_new)
        val = sum_secrecy_rate(H_new, W_new, ha_new, P)
        hist.append(val if np.isfinite(val) else 0.0)   # raw per-iteration objective
        if np.isfinite(val) and val >= best - 1e-9:   # accept only improving updates
            best = val
            W, theta_t, theta_r, H, h_a = W_new, tt_new, tr_new, H_new, ha_new
    return best, hist, W, (theta_t, theta_r)

# ------------------------------------------------------------------ benchmarks
def bench_random(sysd, P, rng):
    K = len(sysd["users"])
    theta_t, theta_r = ss.make_star_ris_coeffs(P, rng)
    H, h_a = channels_from_theta(sysd, P, theta_t, theta_r)
    W = H.T / np.linalg.norm(H, axis=1) * np.sqrt((1-0.2)*P.Pmax/K)
    return sum_secrecy_rate(H, W, h_a, P)

def bench_noris(P, rng):
    """Direct satellite-user links (NLoS, high path loss), MRT."""
    K = 4
    H = (rng.standard_normal((K, P.N)) + 1j*rng.standard_normal((K, P.N)))/np.sqrt(2)
    H *= np.sqrt((P.lam/(4*np.pi*P.d_SR))**2 * P.Gs * 1e-3)  # blocked direct link (30 dB shadowing)
    h_a = H[0]*0.5 + (rng.standard_normal(P.N)+1j*rng.standard_normal(P.N))/np.sqrt(2)*np.sqrt((P.lam/(4*np.pi*P.d_SR))**2*P.Gs)
    W = H.T / np.linalg.norm(H, axis=1) * np.sqrt((1-0.2)*P.Pmax/K)
    return sum_secrecy_rate(H, W, h_a, P)

# ------------------------------------------------------------------ experiment
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    P = ss.Params()
    P.Gs = 10**(P.Gs/10); P.Gr = 10**(P.Gr/10)
    P.M = 128                      # reduced for tractable runtime (paper: 256)
    P.N = 256                      # 16x16 UPA
    P.Pa = 10**(-10/10)/1e3      # spoofer power: -10 dBm (realistic for GNSS-class spoofing)
    sysd = setup_system(P, rng)
    P.beta_su = sysd["beta_su"]

    # --- convergence curve at Pmax = 50 dBm
    t0 = time.time()
    val, hist, _, _ = run_ao(sysd, P, rng, n_outer=8)
    print(f"AO converged: {hist[-1]:.3f} bps/Hz in {time.time()-t0:.1f}s")
    print("history:", [f"{h:.3f}" for h in hist])
    np.save("convergence.npy", np.array(hist))

    # --- sum secrecy rate vs Pmax for 4 schemes
    pmax_dbm = [30, 35, 40, 45, 50]
    results = {k: [] for k in ["proposed", "random", "conv_ris", "no_ris"]}
    prev_theta = None
    for dbm in pmax_dbm:
        P.Pmax = 10**(dbm/10)/1e3
        rng1 = np.random.default_rng(7)
        sysd = setup_system(P, rng1)
        rng2 = np.random.default_rng(7)
        sysd2 = setup_system(P, rng2)
        vc, _, _, thc = run_ao(sysd2, P, rng2, n_outer=4, mode="conv")
        results["conv_ris"].append(vc)
        cands = [run_ao(sysd, P, rng1, n_outer=4),
                 run_ao(sysd, P, rng1, n_outer=4,
                        init_theta=(np.zeros(P.M, dtype=complex), thc[1]))]
        if prev_theta is not None:
            cands.append(run_ao(sysd, P, rng1, n_outer=4, init_theta=prev_theta))
        # STAR-RIS subsumes the reflecting-only design -> take the best candidate
        best = max(cands + [(vc, None, None, thc)], key=lambda t: t[0])
        results["proposed"].append(best[0])
        prev_theta = best[3]
        rr = [bench_random(sysd, P, np.random.default_rng(s)) for s in range(5)]
        results["random"].append(np.mean(rr))
        results["no_ris"].append(np.mean([bench_noris(P, np.random.default_rng(s)) for s in range(5)]))
        print(f"Pmax={dbm}dBm: " + ", ".join(f"{k}={results[k][-1]:.3f}" for k in results))

    np.savez("results_vs_pmax.npz", pmax_dbm=pmax_dbm, **results)
    print("saved results_vs_pmax.npz")
