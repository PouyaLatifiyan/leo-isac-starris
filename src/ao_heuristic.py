"""
Phase 3 (final, robust): Low-complexity alternating design for the
near-field secure ISAC STAR-RIS LEO system.

  - Theta-step: eigen-based phase alignment per side + 1-D grid search over the
    energy-splitting ratio rho (|theta_t|^2 = rho, |theta_r|^2 = 1 - rho),
    with a spoofer-suppression penalty (secrecy-aware).
  - W-step: null-steering MRT at the satellite (project user channels onto the
    null space of the reconstructed spoofer channel).
True secrecy rate is evaluated after each outer iteration; best iterate kept
(monotone envelope). Runs in seconds -> stable, reproducible curves.
"""
import numpy as np
import sim_skeleton as ss

def build(P, rng):
    G = ss.sat_to_ris_channel(P, doppler_Hz=20e3, t=0.0, rng=rng)
    users = [("t", 0.3, 30.0), ("t", -0.4, 45.0), ("r", 0.5, 25.0), ("r", -0.2, 55.0)]
    b = [ss.nearfield_steering(P, th, r) for _, th, r in users]
    beta = [ss.ris_to_node_large_scale(P, r) for _, th, r in users]
    b_a = ss.nearfield_steering(P, 0.1, P.r_spoofer)
    beta_a = ss.ris_to_node_large_scale(P, P.r_spoofer)
    return dict(G=G, users=users, b=b, beta=beta, b_a=b_a, beta_a=beta_a,
                beta_su=ss.ris_to_node_large_scale(P, 50.0))

def channels(sysd, P, theta_t, theta_r):
    Th = {"t": np.diag(theta_t), "r": np.diag(theta_r)}
    H = np.array([ss.cascaded_channel(sysd["G"], sysd["b"][k], Th[s], sysd["beta"][k])
                  for k, (s, _, _) in enumerate(sysd["users"])])
    h_a = ss.cascaded_channel(sysd["G"], sysd["b_a"], Th["r"], sysd["beta_a"])
    return H, h_a

def sum_secrecy(H, W, h_a, P, an_frac=0.2):
    K, N = H.shape
    Z = (an_frac * P.Pmax / N) * np.eye(N)
    tot = 0.0
    for k in range(K):
        s_k = np.abs(H[k].conj() @ W[:, k])**2
        i_k = sum(np.abs(H[k].conj() @ W[:, j])**2 for j in range(K) if j != k) \
              + np.real(H[k].conj() @ Z @ H[k]) + P.Pa*P.beta_su + P.sigma2
        s_ak = np.abs(h_a.conj() @ W[:, k])**2
        i_ak = sum(np.abs(h_a.conj() @ W[:, j])**2 for j in range(K) if j != k) \
               + np.real(h_a.conj() @ Z @ h_a) + P.sigma2
        tot += max(0.0, np.log2(1+s_k/i_k) - np.log2(1+s_ak/i_ak))
    return tot

def null_mrt(H, h_a, P):
    """Null-steering MRT: project each user channel on null(h_a), equal power."""
    K, N = H.shape
    Pa_proj = np.eye(N) - np.outer(h_a, h_a.conj())/max(np.vdot(h_a, h_a).real, 1e-30)
    W = np.zeros((N, K), dtype=complex)
    for k in range(K):
        v = Pa_proj @ H[k]            # w_k || h_k  (h_k^H w_k = ||h_k||^2)
        n = np.linalg.norm(v)
        W[:, k] = v/n if n > 1e-30 else 0.0
    return W * np.sqrt((1-0.2)*P.Pmax/K)

def theta_eigen(sysd, P, W, mode="star", n_grid=41, mu=2.0):
    """Eigen-alignment per side + grid search over energy split rho."""
    K = W.shape[1]
    # coefficient maps c_kj = sqrt(beta_k) conj(b_k) * (G w_j):  h_k^H w_j = theta^T c_kj
    C = {k: {j: np.sqrt(sysd["beta"][k]) * np.conj(sysd["b"][k]) * (sysd["G"] @ W[:, j])
             for j in range(K)} for k in range(K)}
    Ca = {j: np.sqrt(sysd["beta_a"]) * np.conj(sysd["b_a"]) * (sysd["G"] @ W[:, j])
          for j in range(K)}
    sides = [u[0] for u in sysd["users"]]
    # desired-direction matrices per side (signal alignment minus spoofer leakage)
    A = {"t": np.zeros((P.M, P.M), dtype=complex), "r": np.zeros((P.M, P.M), dtype=complex)}
    for k in range(K):
        A[sides[k]] += sum(np.outer(C[k][j], C[k][j].conj()) for j in range(K))
    A["r"] -= mu * sum(np.outer(Ca[j], Ca[j].conj()) for j in range(K))  # suppress spoofer
    phi = {}
    for chi in ("t", "r"):
        vals, vecs = np.linalg.eigh((A[chi] + A[chi].conj().T)/2)
        phi[chi] = np.angle(vecs[:, -1])
    best = None
    for rho in np.linspace(0.05, 1.0, n_grid):
        tt = np.sqrt(rho) * np.exp(1j*phi["t"])
        tr = np.sqrt(1-rho) * np.exp(1j*phi["r"])
        if mode == "conv":
            tt = np.zeros(P.M, dtype=complex)
            tr = np.exp(1j*phi["r"])
        H, h_a = channels(sysd, P, tt, tr)
        v = sum_secrecy(H, W, h_a, P)
        if best is None or v > best[0]:
            best = (v, tt, tr)
    return best[1], best[2]

def run_design(sysd, P, rng, n_outer=10, mode="star"):
    K = len(sysd["users"])
    theta_t, theta_r = ss.make_star_ris_coeffs(P, rng)
    if mode == "conv":
        theta_t = np.zeros(P.M, dtype=complex)
    best = -1; hist = []
    for it in range(n_outer):
        H, h_a = channels(sysd, P, theta_t, theta_r)
        W = null_mrt(H, h_a, P)
        theta_t, theta_r = theta_eigen(sysd, P, W, mode=mode)
        H, h_a = channels(sysd, P, theta_t, theta_r)
        v = sum_secrecy(H, W, h_a, P)
        best = max(best, v)
        hist.append(best)
    return best, hist

def bench_random(sysd, P, rng, n=5):
    K = len(sysd["users"])
    vals = []
    for s in range(n):
        r2 = np.random.default_rng(100+s)
        tt, tr = ss.make_star_ris_coeffs(P, r2)
        H, h_a = channels(sysd, P, tt, tr)
        vals.append(sum_secrecy(H, null_mrt(H, h_a, P), h_a, P))
    return float(np.mean(vals))

def bench_noris(P, rng, n=5):
    K = 4
    vals = []
    for s in range(n):
        r2 = np.random.default_rng(200+s)
        H = (r2.standard_normal((K, P.N)) + 1j*r2.standard_normal((K, P.N)))/np.sqrt(2)
        H *= np.sqrt((P.lam/(4*np.pi*P.d_SR))**2 * P.Gs)
        h_a = (r2.standard_normal(P.N)+1j*r2.standard_normal(P.N))/np.sqrt(2)
        h_a *= np.sqrt((P.lam/(4*np.pi*P.d_SR))**2 * P.Gs)
        vals.append(sum_secrecy(H, null_mrt(H, h_a, P), h_a, P))
    return float(np.mean(vals))

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    P = ss.Params()
    P.Gs = 10**(P.Gs/10); P.Gr = 10**(P.Gr/10)
    P.Pa = 10**(-10/10)/1e3
    P.M, P.N = 128, 256
    sysd = build(P, rng)
    P.beta_su = sysd["beta_su"]

    v, hist = run_design(sysd, P, rng)
    print(f"converged: {v:.3f} bps/Hz | history: {[f'{h:.2f}' for h in hist]}")
    np.save("convergence.npy", np.array(hist))

    pmax_dbm = [30, 35, 40, 45, 50]
    results = {k: [] for k in ["proposed", "random", "conv_ris", "no_ris"]}
    for dbm in pmax_dbm:
        P.Pmax = 10**(dbm/10)/1e3
        r1 = np.random.default_rng(7); sysd1 = build(P, r1); P.beta_su = sysd1["beta_su"]
        results["proposed"].append(run_design(sysd1, P, r1)[0])
        r2 = np.random.default_rng(7); sysd2 = build(P, r2)
        results["conv_ris"].append(run_design(sysd2, P, r2, mode="conv")[0])
        results["random"].append(bench_random(sysd1, P, r1))
        results["no_ris"].append(bench_noris(P, r1))
        print(f"Pmax={dbm}dBm: " + ", ".join(f"{k}={results[k][-1]:.3f}" for k in results))
    np.savez("results_vs_pmax.npz", pmax_dbm=pmax_dbm, **results)
    print("saved results_vs_pmax.npz")
