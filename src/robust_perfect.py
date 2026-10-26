"""Step 1: run the perfect-CSI AO design once and save it (warm-start anchor)."""
import numpy as np
import sim_skeleton as ss
import ao_opt

rng = np.random.default_rng(7)
P = ss.Params(); P.Gs = 10**(P.Gs/10); P.Gr = 10**(P.Gr/10)
P.M, P.N = 128, 256; P.Pa = 1e-4
sysd = ao_opt.setup_system(P, rng); P.beta_su = sysd["beta_su"]
val, hist, W, (th_t, th_r) = ao_opt.run_ao(sysd, P, rng, n_outer=4)
np.savez("perfect_design.npz", th_t=th_t, th_r=th_r, W=W, val=val)
print("perfect-CSI design:", round(val, 3), "bps/Hz  hist:", [round(h,2) for h in hist])
