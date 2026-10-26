"""
Near-Field Secure ISAC with STAR-RIS for Spoofing-Resilient LEO Navigation
Simulation skeleton — Phase 2: channel generation & rate computation.

Scenario (per paper draft):
  LEO satellite (N-antenna UPA) -> STAR-RIS (M elements) -> ground users
  on transmission (T) and reflection (R) sides; one active spoofer.
  Sat->RIS: far-field LoS with orbital Doppler.
  RIS->nodes: near-field spherical-wave (Fresnel region).
"""

import numpy as np

# ---------------------------------------------------------------- parameters
C_LIGHT = 3e8

class Params:
    fc        = 2e9            # carrier frequency (Hz), S band
    lam       = C_LIGHT / fc   # wavelength (m)
    d         = lam / 2        # element spacing
    N         = 16 * 16        # satellite UPA antennas (16x16)
    M         = 256            # STAR-RIS elements (ULA equivalent)
    Kt, Kr    = 2, 2           # users on transmission / reflection side
    Pmax      = 10**(50/10) / 1e3   # 50 dBm -> W
    Pa        = 10**(-10/10) / 1e3  # spoofer power: -10 dBm (realistic GNSS-class)
    sigma2    = 10**(-110/10) / 1e3 # noise power (W)
    alt       = 600e3          # orbit altitude (m)
    elev_deg  = 60.0           # elevation angle (deg)
    d_SR      = alt / np.sin(np.deg2rad(elev_deg))  # sat-RIS distance
    Gs, Gr    = 45.0, 10.0     # antenna gains (dBi) -> linear below
    kappa     = 10.0           # Rician factor (linear) of the sat-RIS link
    r_user    = (20.0, 60.0)   # RIS-user near-field distance range (m)
    r_spoofer = 40.0           # RIS-spoofer distance (m)

P = Params()
P.Gs = 10**(P.Gs/10); P.Gr = 10**(P.Gr/10)

# ---------------------------------------------------------------- channels
def sat_to_ris_channel(P: Params, doppler_Hz: float, t: float, rng) -> np.ndarray:
    """Rician channel matrix G (M x N), eq. (1) of the draft:
    G = sqrt(beta) e^{j2pi fD t} [ sqrt(k/(1+k)) a_M a_N^H + sqrt(1/(1+k)) G~ ].
    """
    beta = (P.lam / (4*np.pi*P.d_SR))**2 * P.Gs * P.Gr
    n = np.arange(P.N)
    a_N = np.exp(-1j * 2*np.pi * P.d / P.lam * n * np.sin(np.deg2rad(P.elev_deg))) / np.sqrt(P.N)
    a_M = np.exp(-1j * 2*np.pi * P.d / P.lam * np.arange(P.M) * np.sin(np.deg2rad(P.elev_deg)))
    G_los  = np.outer(a_M, a_N.conj())
    G_nlos = (rng.standard_normal((P.M, P.N)) + 1j*rng.standard_normal((P.M, P.N))) / np.sqrt(2)
    k = P.kappa
    G = np.sqrt(beta) * np.exp(1j*2*np.pi*doppler_Hz*t) * (
        np.sqrt(k/(1+k)) * G_los + np.sqrt(1/(1+k)) * G_nlos)
    return G

def nearfield_steering(P: Params, theta: float, r: float) -> np.ndarray:
    """Near-field steering vector b(theta, r), eq. (2) of the draft."""
    m = np.arange(1, P.M+1)
    delta = (2*m - P.M - 1) / 2
    r_m = np.sqrt(r**2 + (delta*P.d)**2 - 2*r*delta*P.d*np.sin(theta))
    # element-level response (no 1/sqrt(M) normalization: aperture gain is physical)
    return np.exp(-1j * 2*np.pi/P.lam * (r_m - r))                       # (M,)

def cascaded_channel(G: np.ndarray, b: np.ndarray, Theta: np.ndarray, beta_k: float) -> np.ndarray:
    """h_k^H = sqrt(beta_k) b^H Theta G  ->  return h_k (N,).

    G: satellite-to-RIS channel matrix (M, N).
    b: near-field RIS->node steering vector (M,).
    Theta: STAR-RIS diag coefficient matrix (M, M).
    """
    return np.sqrt(beta_k) * np.conj(b.conj() @ Theta @ G)   # (N,)

def ris_to_node_large_scale(P: Params, r: float) -> float:
    return (P.lam / (4*np.pi*r))**2

def make_star_ris_coeffs(P: Params, rng, align_to=None, a_M=None):
    """STAR-RIS coefficients with beta_t^2 + beta_r^2 = 1.

    align_to: optional list of (side, b_vector) to align phases toward users
              (coherent benchmark); otherwise random phases.
    """
    bt = np.full(P.M, np.sqrt(0.5)); br = np.sqrt(1 - bt**2)
    if align_to is not None and a_M is not None:
        phit = np.angle(align_to[0] * np.conj(a_M))
        phir = np.angle(align_to[1] * np.conj(a_M))
    else:
        phit = rng.uniform(0, 2*np.pi, P.M)
        phir = rng.uniform(0, 2*np.pi, P.M)
    return bt*np.exp(1j*phit), br*np.exp(1j*phir)

# ---------------------------------------------------------------- rates
def sinr(h_k: np.ndarray, W: np.ndarray, Z: np.ndarray, k: int,
         interf_power: float, noise: float) -> float:
    sig  = np.abs(h_k.conj() @ W[:, k])**2
    interf = sum(np.abs(h_k.conj() @ W[:, j])**2 for j in range(W.shape[1]) if j != k)
    an = np.real(h_k.conj() @ Z @ h_k)
    return sig / (interf + an + interf_power + noise)

def secrecy_rate(gamma_k: float, gamma_ak: float) -> float:
    return max(0.0, np.log2(1+gamma_k) - np.log2(1+gamma_ak))

# ---------------------------------------------------------------- main demo
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    G = sat_to_ris_channel(P, doppler_Hz=20e3, t=0.0, rng=rng)

    # users: two on each side, near-field distances
    users = [("t", 0.3, 30.0), ("t", -0.4, 45.0), ("r", 0.5, 25.0), ("r", -0.2, 55.0)]
    b_users = [nearfield_steering(P, th, r) for _, th, r in users]

    # random-phase init (AO optimization comes in Phase 3)
    theta_t, theta_r = make_star_ris_coeffs(P, rng)

    # build cascaded user channels
    H = []
    for (side, th, r), b in zip(users, b_users):
        Th = np.diag(theta_t) if side == "t" else np.diag(theta_r)
        H.append(cascaded_channel(G, b, Th, ris_to_node_large_scale(P, r)))
    H = np.array(H)  # (K, N)

    # spoofer channel (reflection side)
    b_a = nearfield_steering(P, 0.1, P.r_spoofer)
    h_a = cascaded_channel(G, b_a, np.diag(theta_r), ris_to_node_large_scale(P, P.r_spoofer))

    # naive equal-power beamforming (placeholder for AO in Phase 3)
    K = len(users)
    W = H.T / np.linalg.norm(H, axis=1)                 # MRT direction: w_k = h_k/||h_k||
    W *= np.sqrt(P.Pmax / K)
    Z = np.zeros((P.N, P.N), dtype=complex)             # AN added in Phase 3

    total = 0.0
    for k in range(K):
        # spoofer->user interference with proper path loss (assume 50 m separation)
        beta_ak = ris_to_node_large_scale(P, 50.0)
        gk  = sinr(H[k],  W, Z, k, P.Pa*beta_ak, P.sigma2)
        gak = sinr(h_a,   W, Z, k, 0.0,          P.sigma2)
        rs  = secrecy_rate(gk, gak)
        total += rs
        print(f"User {k+1} ({users[k][0]}-side): SINR={gk:.3e}, secrecy rate={rs:.3f} bps/Hz")
    print(f"\nSum secrecy rate (random STAR-RIS, MRT BF): {total:.3f} bps/Hz")
    print("Phase 3 next: AO optimization of W, Z, theta_t, theta_r.")
