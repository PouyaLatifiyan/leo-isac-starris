"""
EKF tracking demo for the spoofer (Section II-D of the paper).

State:  xi = [x, y, vx, vy]^T  (constant-velocity model, ground plane)
Meas.:  near-field anchor observes range r = sqrt(x^2+y^2) and angle
        theta = atan2(y, x) with noise std set from the CRB (crb_analysis).
This is the tracking block whose output feeds the channel reconstruction
h_a(theta_a, r_a) used by the AO beamforming.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def run_ekf(seed=0, T=60, dt=0.5):
    rng = np.random.default_rng(seed)
    # true trajectory: spoofer moving with constant velocity + slight turn
    x0 = np.array([30.0, 20.0, 1.5, -0.8])       # m, m, m/s, m/s
    F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
    q = 0.05                                       # process noise (m/s^2 scaled)
    Q = q*np.array([[dt**4/4,0,dt**3/2,0],[0,dt**4/4,0,dt**3/2],
                    [dt**3/2,0,dt**2,0],[0,dt**3/2,0,dt**2]])
    sig_r, sig_th = 0.8, np.deg2rad(1.0)           # from CRB analysis (~0.8 m at 50 m)
    R = np.diag([sig_r**2, sig_th**2])

    # simulate truth and measurements
    xs, zs = [], []
    x = x0.copy()
    for _ in range(T):
        x = F@x + rng.multivariate_normal(np.zeros(4), Q)
        r, th = np.hypot(x[0],x[1]), np.arctan2(x[1],x[0])
        z = np.array([r,th]) + rng.multivariate_normal(np.zeros(2), R)
        xs.append(x.copy()); zs.append(z)
    xs, zs = np.array(xs), np.array(zs)

    # EKF
    xh = np.array([zs[0,0]*np.cos(zs[0,1]), zs[0,0]*np.sin(zs[0,1]), 0, 0])
    Pcov = np.diag([10.0,10.0,5.0,5.0])
    ests = []
    for z in zs:
        xh = F@xh; Pcov = F@Pcov@F.T + Q            # predict
        r, th = np.hypot(xh[0],xh[1]), np.arctan2(xh[1],xh[0])
        H = np.array([[xh[0]/r, xh[1]/r, 0, 0],
                      [-xh[1]/r**2, xh[0]/r**2, 0, 0]])
        S = H@Pcov@H.T + R
        K = Pcov@H.T@np.linalg.inv(S)
        innov = z - np.array([r,th]); innov[1] = np.angle(np.exp(1j*innov[1]))
        xh = xh + K@innov
        Pcov = (np.eye(4)-K@H)@Pcov
        ests.append(xh.copy())
    ests = np.array(ests)
    rmse = np.sqrt(np.mean((ests[:,:2]-xs[:,:2])**2))
    return xs, ests, rmse

if __name__ == "__main__":
    xs, ests, rmse = run_ekf()
    print(f"position RMSE: {rmse:.3f} m")
    plt.figure(figsize=(5,4))
    plt.plot(xs[:,0], xs[:,1], 'k-', label='True spoofer trajectory')
    plt.plot(ests[:,0], ests[:,1], 'b--', label='EKF estimate')
    plt.plot(0,0,'r^',ms=10,label='STAR-RIS / ISAC anchor')
    plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.legend(); plt.grid(alpha=.3)
    plt.title(f'Spoofer tracking (RMSE = {rmse:.2f} m)')
    plt.tight_layout(); plt.savefig("fig_ekf.png", dpi=200)
    print("fig_ekf.png saved")
