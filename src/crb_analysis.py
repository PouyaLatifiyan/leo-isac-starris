"""
CRB analysis: near-field vs far-field spoofer localization accuracy.
Measurement model: y = sqrt(P_echo) * b(theta, r) + n,  n ~ CN(0, sigma^2 I_M)
with P_echo = P_s * (lam/(4 pi r))^4  (two-way echo path loss, monostatic anchor).
FIM for eta = [theta, r]: F = (2 P_s / sigma^2) Re(D^H D), D = [db/dtheta, db/dr]
CRB_r = [F^{-1}]_22.  Far-field model: phase depends on theta only -> CRB_r -> inf.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sim_skeleton as ss

def crb_range(P, theta, r, Ps, sigma2, model="near"):
    m = np.arange(1, P.M+1)
    delta = (2*m - P.M - 1)/2
    k0 = 2*np.pi/P.lam
    if model == "near":
        r_m = np.sqrt(r**2 + (delta*P.d)**2 - 2*r*delta*P.d*np.sin(theta))
        phase = r_m - r
        drm_dth = (r*delta*P.d*np.cos(theta)) / r_m
        drm_dr  = (r - delta*P.d*np.sin(theta)) / r_m - 1.0
    else:  # far-field: r_m - r ≈ -delta d sin(theta) (no range info)
        phase = -delta*P.d*np.sin(theta)
        drm_dth = -delta*P.d*np.cos(theta)
        drm_dr  = np.zeros(P.M)
    b = np.exp(-1j*k0*phase)
    db_dth = -1j*k0*drm_dth*b
    db_dr  = -1j*k0*drm_dr*b
    D = np.column_stack([db_dth, db_dr])
    F = (2*Ps/sigma2) * np.real(D.conj().T @ D)
    if abs(np.linalg.det(F)) < 1e-30:
        return np.nan
    Finv = np.linalg.inv(F)
    return np.sqrt(Finv[1,1])   # std of range estimate (m)

if __name__ == "__main__":
    P = ss.Params()
    P.M = 128
    Ps = 10.0         # anchor sensing transmit power (W)
    r_vals = np.linspace(10, 200, 40)
    def crb_at(r, model):
        Ps_echo = Ps * (P.lam/(4*np.pi*r))**4   # two-way echo path loss
        return crb_range(P, 0.3, r, Ps_echo, P.sigma2, model)
    crb_nf = [crb_at(r, "near") for r in r_vals]
    crb_ff = [crb_at(r, "far")  for r in r_vals]
    crb_ff_plot = [c if np.isfinite(c) else 1e3 for c in crb_ff]  # cap singular (inf) values
    np.savez("crb_results.npz", r=r_vals, crb_nf=crb_nf, crb_ff=crb_ff_plot)
    plt.figure(figsize=(5,3.5))
    plt.semilogy(r_vals, crb_nf, 'b-o', ms=3, label='Near-field (proposed)')
    plt.semilogy(r_vals, crb_ff_plot, 'r--s', ms=3, label='Far-field approx. (capped at $10^3$ m)')
    plt.xlabel('Spoofer distance $r$ (m)'); plt.ylabel('CRB of range (m)')
    plt.grid(True, which='both', alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig("fig_crb.png", dpi=200)
    print("CRB near-field at r=50m:", crb_nf[10], " far-field:", crb_ff[10])
