# Near-Field Secure ISAC with STAR-RIS for Spoofing-Resilient LEO Satellite Navigation (6G NTN)

Simulation code and paper for a satellite-communications course project:
a LEO satellite serves ground users on both sides of a ground-deployed
**STAR-RIS** in the presence of an active **spoofer**. A ground ISAC anchor
localizes the spoofer using **near-field spherical-wave** propagation
(FIM/CRB analysis), and the satellite beamforming + STAR-RIS
transmission/reflection coefficients are jointly optimized via
**alternating optimization (AO) with SCA** to maximize the sum secrecy rate.

## Repository structure

```
├── paper/paper.pdf        # IEEE-style paper (IEEEtran, compiled PDF)
├── paper/latex/           # Full LaTeX source of the paper (Overleaf-ready)
├── src/
│   ├── sim_skeleton.py    # System & channel model (Rician sat link, near-field RIS links)
│   ├── ao_opt.py          # AO algorithm (SCA W-step + Theta-step), benchmarks, Pmax sweep
│   ├── ao_heuristic.py    # Low-complexity closed-form alternating design (fast demo)
│   ├── crb_analysis.py    # Near-field vs far-field CRB of spoofer range estimation
│   ├── ekf_tracking.py    # EKF spoofer tracking demo (Fig. 6 of the paper)
│   ├── robust_perfect.py  # Nominal AO design used as anchor for the robustness test
│   ├── robustness.py      # Robustness: secrecy rate vs. spoofer-position error (Fig. 5)
│   ├── run_conv3.py       # Convergence experiment (half-step logging)
│   └── make_figs.py       # Regenerate the paper figures from saved results
├── figs/                  # Generated figures used in the paper (6 figures)
└── requirements.txt
```

## Quick start

```bash
git clone https://github.com/YOUR-USERNAME/leo-isac-starris.git
cd leo-isac-starris
pip install -r requirements.txt
cd src
python3 crb_analysis.py    # CRB figure (seconds)
python3 ao_heuristic.py    # fast low-complexity demo (seconds)
python3 ao_opt.py          # full AO + SCA experiments (slow, tens of minutes)
python3 ekf_tracking.py    # EKF tracking figure (seconds)
python3 robust_perfect.py  # nominal design for the robustness test (minutes)
python3 robustness.py      # robustness figure (seconds, needs robust_perfect.py first)
python3 make_figs.py       # regenerate figures
```

## Main results

- Proposed STAR-RIS + AO achieves up to **~17 bps/Hz** sum secrecy rate,
  outperforming a conventional reflection-only RIS, a random-phase STAR-RIS
  (near zero), and a blocked direct link.
- The AO algorithm converges within a few outer iterations.
- Near-field sensing yields a meter-level spoofer-range CRB (~0.84 m at 50 m,
  10 W anchor, two-way echo path loss) and an EKF tracking RMSE of ~0.36 m,
  while the far-field model makes the range fundamentally unobservable.
- Robustness: with the design based on the estimated spoofer position, a
  position mismatch of sigma_e = 0.36 m (the EKF RMSE) degrades the achieved
  secrecy rate by at most ~4.5% over 60 Monte-Carlo realizations.

## Notes / limitations

- Simulated demo uses M=128 STAR-RIS elements (paper target: 256) and a fixed
  isotropic AN covariance (20% power fraction); joint AN optimization is future work.
- The SCA solver benefits from warm starts (conventional-RIS solution and
  continuation over the power sweep), as described in the paper.

## Reference

If you use this code, please cite the course-project paper in `paper/paper.pdf`.
