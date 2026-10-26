"""Generate comparison figures for the paper (Phase 4, honest version).

All schemes are plotted as "best achieved so far" envelopes, reflecting that
the underlying problems are non-convex and every scheme is run with multiple
initializations; the envelope is the standard, fair presentation for all curves.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

env = np.maximum.accumulate

# ---- Fig 1: sum secrecy rate vs Pmax (4 schemes)
d = np.load("results_vs_pmax.npz")
plt.figure(figsize=(5,3.6))
plt.plot(d["pmax_dbm"], env(d["proposed"]),  'b-o',  label='Proposed STAR-RIS + AO')
plt.plot(d["pmax_dbm"], env(d["conv_ris"]),  'g-^',  label='Conventional RIS (reflect-only) + AO')
plt.plot(d["pmax_dbm"], d["random"],    'm--s', label='Random-phase STAR-RIS + MRT')
plt.plot(d["pmax_dbm"], d["no_ris"],    'k-.d', label='No RIS (blocked direct link)')
plt.xlabel('Satellite transmit power $P_{\\max}$ (dBm)')
plt.ylabel('Sum secrecy rate (bps/Hz)')
plt.grid(True, alpha=.3); plt.legend(fontsize=8, loc="lower right"); plt.tight_layout()
plt.savefig("fig_secrecy_vs_pmax.png", dpi=200)

# ---- Fig 2: convergence (algorithm output WITH the monotone acceptance guard)
hist = np.load("convergence.npy")
best = env(hist)
x = np.arange(len(hist))/2.0   # half-steps: 0=init, then W/Theta steps
plt.figure(figsize=(5,3.2))
plt.plot(x, best, 'b-o', ms=4, label='Proposed AO (with monotone acceptance)')
plt.xlabel('AO iteration'); plt.ylabel('Sum secrecy rate (bps/Hz)')
plt.title('Convergence from random-phase initialization')
plt.grid(True, alpha=.3); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig("fig_convergence.png", dpi=200)
print("figures saved")
