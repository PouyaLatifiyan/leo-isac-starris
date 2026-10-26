"""Robustness figure (Fig. 5): achieved sum secrecy rate when the TRUE spoofer
position deviates from the estimate used in the design.

Protocol (mismatch evaluation):
  1) python3 robust_perfect.py   # nominal AO design (W*, Theta*) using the
                                 # estimated spoofer position (Remark 1)
  2) this script: the TRUE spoofer position is perturbed by N(0, sigma_e^2)
     (in range, and the equivalent angular offset), the true spoofer channel
     is reconstructed from it, and the secrecy rate achieved by the fixed
     design (W*, Theta*) is evaluated over 60 Monte-Carlo realizations.

The figure marks the EKF tracking RMSE (~0.36 m, Fig. 6): the estimation
accuracy the ISAC anchor actually delivers.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sim_skeleton as ss
import ao_opt

def main():
    rng = np.random.default_rng(7)                      # same system geometry
    P = ss.Params(); P.Gs = 10**(P.Gs/10); P.Gr = 10**(P.Gr/10)
    P.M, P.N = 128, 256; P.Pa = 1e-4
    sysd = ao_opt.setup_system(P, rng); P.beta_su = sysd["beta_su"]

    d = np.load("perfect_design.npz")
    th_t, th_r, W, val0 = d["th_t"], d["th_r"], d["W"], float(d["val"])
    H, _ = ao_opt.channels_from_theta(sysd, P, th_t, th_r)
    print("nominal design value:", round(val0, 3), "bps/Hz", flush=True)

    th_true, r_true = 0.1, P.r_spoofer
    sig_errs = [0.0, 0.25, 0.36, 0.5, 1.0, 2.0, 4.0]
    n_mc = 60
    mean, std, vmin = [], [], []
    for sig in sig_errs:
        vals = []
        for s in range(n_mc):
            r2 = np.random.default_rng(9000 + s)
            dr = r2.normal(0, sig)
            dth = r2.normal(0, sig / max(r_true, 1.0))
            b_true = ss.nearfield_steering(P, th_true + dth, max(r_true + dr, 1.0))
            h_true = ss.cascaded_channel(sysd["G"], b_true, np.diag(th_r), sysd["beta_a"])
            vals.append(ao_opt.sum_secrecy_rate(H, W, h_true, P))
        vals = np.array(vals)
        mean.append(vals.mean()); std.append(vals.std()); vmin.append(vals.min())
        print(f"sigma={sig}m: mean={mean[-1]:.3f} std={std[-1]:.3f} min={vmin[-1]:.3f}",
              flush=True)
    np.savez("robustness.npz", sig=sig_errs, mean=mean, std=std, vmin=vmin,
             nominal=val0)

    mean, std, vmin = map(np.array, (mean, std, vmin))
    plt.figure(figsize=(5, 3.4))
    plt.fill_between(sig_errs, vmin, mean + std, color='b', alpha=0.15,
                     label='min / mean+std over 60 realizations')
    plt.errorbar(sig_errs, mean, yerr=std, fmt='b-o', capsize=3,
                 label='Mean achieved secrecy rate')
    plt.axhline(val0, color='k', ls='--', lw=1, label='Design value (nominal estimate)')
    plt.axvline(0.36, color='r', ls=':', lw=1.2, label='EKF RMSE $\\approx$ 0.36 m')
    plt.xlabel('Spoofer position estimation error $\\sigma_e$ (m)')
    plt.ylabel('Achieved sum secrecy rate (bps/Hz)')
    plt.grid(alpha=.3); plt.legend(fontsize=7.5, loc='center right')
    plt.tight_layout()
    plt.savefig("fig_robustness.png", dpi=200)
    print("fig_robustness.png saved")

if __name__ == "__main__":
    main()
