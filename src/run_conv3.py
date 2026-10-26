import numpy as np, time
import sim_skeleton as ss, ao_opt
rng = np.random.default_rng(7)
P = ss.Params(); P.Gs=10**(P.Gs/10); P.Gr=10**(P.Gr/10)
P.M, P.N = 128, 256; P.Pa = 1e-4
sysd = ao_opt.setup_system(P, rng); P.beta_su = sysd["beta_su"]
K = len(sysd["users"])
theta_t, theta_r = ss.make_star_ris_coeffs(P, rng)   # random phases
H, h_a = ao_opt.channels_from_theta(sysd, P, theta_t, theta_r)
norms = np.linalg.norm(H,axis=1); norms=np.where(norms<1e-30,1,norms)
W = H.T/norms*np.sqrt(0.8*P.Pmax/K)                  # MRT init
hist=[ao_opt.sum_secrecy_rate(H, W, h_a, P)]
print("init", round(hist[0],3), flush=True)
for it in range(8):
    W = ao_opt.sca_W_step(H, h_a, P, W)
    hist.append(ao_opt.sum_secrecy_rate(H, W, h_a, P))      # after W-step
    theta_t, theta_r = ao_opt.sca_theta_step(sysd, P, W, (theta_t, theta_r))
    H, h_a = ao_opt.channels_from_theta(sysd, P, theta_t, theta_r)
    hist.append(ao_opt.sum_secrecy_rate(H, W, h_a, P))      # after theta-step
    print(it, round(hist[-2],3), round(hist[-1],3), flush=True)
np.save("convergence.npy", np.array(hist))
print("saved", len(hist), "points")
