#!/usr/bin/python
"""
PID Control Algorithm - Dideriksen et al. (2010)

Implements the descending drive estimation algorithm described in:
"An integrative model of motor unit activity during sustained submaximal contractions"
Dideriksen et al., J Appl Physiol 108:1550-1562, 2010

Equations implemented:
- Eq. 12: Error function
- Eq. 13: Required excitatory input (REI)
- Eq. 14: Time-varying derivative gain (Kd)
- Eq. 15: Mean metabolite concentration (MMC)
"""
import numpy as np


class DescendingDriveController:
    """
    PID-based descending drive estimator from Dideriksen et al. (2010).

    Estimates the required excitatory input (REI) to the motoneuron pool
    to maintain a prescribed target force profile during fatiguing contractions.

    Parameters
    ----------
    target_force : float
        Target force as a fraction of MVC (e.g., 0.20 for 20% MVC)
    max_REI : float
        Maximum allowable REI output (Emax of largest motor unit)
    dt : float
        Simulation timestep in seconds (default: 0.001 s = 1 ms)
    update_interval : float
        Controller update interval in seconds (default: 1/3 s = 3 Hz)
    """

    def __init__(self, target_force, max_REI, dt=0.001, update_interval=1 / 3):

        # Fixed gains from Table 1 / Section "Estimation of descending drive"
        self.Kp = 2e-1 # Eq. 13
        self.Ki = 2.5e-1 # Eq. 13
        #self.Kd = 2.5e-4 # Eq. 14 - computed dynamically, starts near zero

        # Timing
        self.dt = dt
        self.update_interval = update_interval  # 3 Hz = ~333 ms
        self.time_since_update = 0.0  # tracks time since last 3 Hz update

        # Force targets
        self.target_force = target_force  # Ft in Eq. 12, in %MVC
        self.max_REI = max_REI  # output clamp upper bound

        self.clear()

    def clear(self):
        """Clears PID computations and resets all internal state."""

        # Error terms
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Integral windup guard (in %MVC units)
        self.windup_guard = 50 # 50% MVC max integral accumulation

        # Output
        self.output = 0.0  # current REI value

        # MMC state
        self.current_MMC = 0.0  # mean metabolite concentration (Eq. 15)
        self.time_since_update = 0.0


    def compute_MMC(self, MC_array, V_array):
        """
        Compute mean metabolite concentration across all motor units (Equation 15).

        Parameters
        ----------
        MC_array : ndarray
            Intracellular metabolite concentration for each motor unit (shape: n_motor_units,)
        V_array : ndarray
            Volume of each motor unit intracellular compartment (shape: n_motor_units,)

        Returns
        -------
        float
            Mean metabolite concentration (MMC)
        """
        # Eq. 15: MMC(t) = sum(MC_i(t)) / sum(V_i)
        return np.sum(MC_array) / np.sum(V_array)

    def compute_MMC_placeholder(self, n_motor_units):
        """
        Placeholder MMC computation when metabolite model is not available.
        Returns MMC = 0 (no fatigue).

        Parameters
        ----------
        n_motor_units : int
            Number of motor units

        Returns
        -------
        float
            MMC = 0.0
        """
        return 0.0

    def debug_state(self, actual_force):
        """
        Print detailed PID state for debugging.

        Parameters
        ----------
        actual_force : float
            Current force as fraction of MVC
        """
        error = self.target_force - actual_force

        print(f"\n=== PID DEBUG ===")
        print(f"Target force: {self.target_force * 100:.2f}% MVC")
        print(f"Actual force: {actual_force * 100:.2f}% MVC")
        print(f"Error: {error * 100:.2f}% MVC")
        print(f"\nPID Terms:")
        print(f"  P term: {self.PTerm:.6f}")
        print(f"  I term: {self.ITerm:.6f}")
        print(f"  D term: {self.DTerm:.6f}")
        print(f"\nGains:")
        print(f"  Kp: {self.Kp:.2e}")
        print(f"  Ki: {self.Ki:.2e}")
        print(f"  Kd: {self.Kd:.6f}")
        print(f"\nOutput:")
        print(f"  Raw: {self.PTerm + (self.Ki * self.ITerm) + self.DTerm:.6f}")
        print(f"  Clamped (REI): {self.output:.6f}")
        print(f"  MMC: {self.current_MMC:.2f}")
        print("=" * 40)

    def compute_Kd(self, MMC):
        """
        Compute time-varying derivative gain as a function of mean
        metabolite concentration (Equation 14).

        Parameters
        ----------
        MMC : float
            Mean metabolite concentration across all motor unit compartments

        Returns
        -------
        float
            Derivative gain Kd(t)

        Notes
        -----
        Kd grows exponentially with MMC, which progressively impairs
        the controller's ability to track the target force and produces
        the experimentally observed increase in force variability
        toward task failure.
        """
        # Eq. 14: Kd(t) = 1.6e-2 * exp(3.5e-8 * MMC(t)) - 5.9e-2
        return 1.6e-2 * np.exp(3.5e-3 * MMC) + 5.9e-2

    def update(self, actual_force, MC_array=None, V_array=None):
        """
        Update the PID controller and compute required excitatory input (Equation 13).

        Should be called every simulation timestep (1 ms), but only recomputes
        REI at 3 Hz intervals. Between updates, holds the last REI value.

        Parameters
        ----------
        actual_force : float
            Current force produced by the model as fraction of MVC
        sim_time : float
            Current simulation time in seconds
        MC_array : ndarray
            Intracellular metabolite concentration for each motor unit
        V_array : ndarray
            Volume of each motor unit intracellular compartment

        Returns
        -------
        float
            Required excitatory input (REI) to the motoneuron pool
        """
        # Accumulate time since last 3 Hz update
        self.time_since_update += self.dt

        # Only recompute at 3 Hz (every ~333 ms)
        if self.time_since_update < self.update_interval:
            return self.output

        # Reset update timer
        self.time_since_update = 0.0

        # Eq. 12: error = target force - actual force (both in %MVC)
        error = self.target_force - actual_force

        # Time elapsed since last update (should be ~333 ms)
        delta_time = self.update_interval
        delta_error = error - self.last_error

        # --- Proportional term ---
        self.PTerm = self.Kp * error

        # --- Integral term with windup guard ---
        self.ITerm += error * delta_time
        #self.ITerm = np.clip(self.ITerm, 0.0, self.windup_guard)
        self.ITerm = np.clip(self.ITerm, -self.windup_guard, self.windup_guard)

        # --- Derivative term ---
        # Update MMC and Kd from current metabolite state
        if MC_array is not None and V_array is not None:
            self.current_MMC = self.compute_MMC(MC_array, V_array)
        else:
            # Placeholder: no metabolite model available yet
            self.current_MMC = 0.0

        self.Kd = self.compute_Kd(self.current_MMC)

        if delta_time > 0:
            self.DTerm = self.Kd * (delta_error / delta_time)
        else:
            self.DTerm = 0.0

        # Store error for next update
        self.last_error = error

        # Eq. 13: REI(t) = Kp*e(t) + Ki*integral(e) + Kd*de/dt
        raw_output = self.PTerm + (self.Ki * self.ITerm) + self.DTerm

        # Clamp output to valid physiological range [0, max_REI]
        self.output = np.clip(raw_output, 0.0, self.max_REI)

        return self.output

    def set_target_force(self, target_force):
        """
        Update the target force setpoint.

        Parameters
        ----------
        target_force : float
            New target force as fraction of MVC (e.g., 0.30 for 30% MVC)
        """
        self.target_force = target_force

    def set_windup_guard(self, windup):
        """
        Update the integral windup guard.

        Parameters
        ----------
        windup : float
            Maximum absolute value of the integral term (in %MVC units)
        """
        self.windup_guard = windup

    def set_max_REI(self, max_REI):
        """
        Update the output clamp upper bound.

        Parameters
        ----------
        max_REI : float
            Maximum allowable REI (Emax of largest motor unit)
        """
        self.max_REI = max_REI

    def get_state(self):
        """
        Return current internal state for logging and debugging.

        Returns
        -------
        dict
            Dictionary containing current values of all internal state variables
        """
        return {
            'output': self.output,
            'PTerm': self.PTerm,
            'ITerm': self.ITerm,
            'DTerm': self.DTerm,
            'Kd': self.Kd,
            'MMC': self.current_MMC,
            'last_error': self.last_error
        }




'''
import numpy as np
import matplotlib.pyplot as plt
from metabolite_block import MetaboliteModel
from motorUnti import MotorNeuronPool
from pid_new import DescendingDriveController

# ── Setup ─────────────────────────────────────────────────────
met        = MetaboliteModel()
mnp        = MotorNeuronPool(n_motor_units=120, dt=0.001)
pid        = DescendingDriveController(20.0, np.max(mnp.Emax))

# ── Run 120s MVC simulation (metabolite block only) ───────────
duration      = 120.0
epoch_dur     = 0.5
n_epochs      = int(duration / epoch_dur)
spikes_per_ep = np.round(mnp.PDR * epoch_dur).astype(int)
epoch_times   = np.arange(1, n_epochs + 1) * epoch_dur

# Storage
selected_mus  = [1, 30, 60, 90, 120]
colors_mu     = ['blue', 'green', 'orange', 'red', 'black']
MC_i_hist     = {mu: np.zeros(n_epochs) for mu in selected_mus}
MMC_hist      = np.zeros(n_epochs)
sum_MCi_hist  = np.zeros(n_epochs)   # Σ MC_i (numerator)
Kd_hist       = np.zeros(n_epochs)

# Σ V_i is constant — compute once
sum_Vi = np.sum(met.V_i)
print(f"Σ V_i = {sum_Vi:.4f} au  (constant throughout simulation)")

for ep in range(n_epochs):
    out = met.step_epoch(spikes_per_ep, 100.0)

    MMC_hist[ep]     = out['MMC']
    sum_MCi_hist[ep] = np.sum(out['MC_i'])
    Kd_hist[ep]      = pid.compute_Kd(out['MMC'])

    for mu in selected_mus:
        MC_i_hist[mu][ep] = out['MC_i'][mu - 1]

# ── Single figure with 4 stacked panels ───────────────────────
fig, axes = plt.subplots(4, 1, figsize=(12, 14),
                          sharex=True)

# ── Panel 1: MC_i for selected MUs ───────────────────────────
ax1 = axes[0]
for mu, color in zip(selected_mus, colors_mu):
    ax1.plot(epoch_times, MC_i_hist[mu],
             color=color, linewidth=2,
             label=f'MC_{mu}(t)  (MU{mu})')

ax1.axhline(met.MC_ref, color='gray', linewidth=1.5,
            linestyle='--',
            label=f'MC_ref = {met.MC_ref}')
ax1.set_ylabel('MC_i (au)', fontsize=11)
ax1.set_title('MC_i(t) — Intracellular MC per selected MU',
              fontweight='bold')
ax1.legend(fontsize=8, loc='upper left', ncol=3)
ax1.grid(True, alpha=0.3)

# ── Panel 2: Σ MC_i and Σ V_i ────────────────────────────────
ax2 = axes[1]
ax2b = ax2.twinx()

ax2.plot(epoch_times, sum_MCi_hist,
         color='black', linewidth=2.5,
         label='Σ MC_i(t)  (numerator of MMC)')

# Σ V_i is constant — flat line
ax2b.axhline(sum_Vi, color='red', linewidth=2,
             linestyle='--',
             label=f'Σ V_i = {sum_Vi:.1f} au  (constant denominator)')

ax2.set_ylabel('Σ MC_i (au)', color='black', fontsize=11)
ax2b.set_ylabel('Σ V_i (au)', color='red', fontsize=11)
ax2.set_title('Σ MC_i(t) and Σ V_i — numerator and denominator of MMC',
              fontweight='bold')

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2,
           fontsize=9, loc='upper left')
ax2.grid(True, alpha=0.3)

# ── Panel 3: MMC = Σ MC_i / Σ V_i ────────────────────────────
ax3 = axes[2]
ax3.plot(epoch_times, MMC_hist,
         color='black', linewidth=2.5,
         label='MMC(t) = Σ MC_i / Σ V_i  (Eq. 15)')

# Verify: recompute MMC manually
MMC_recomputed = sum_MCi_hist / sum_Vi
ax3.plot(epoch_times, MMC_recomputed,
         color='red', linewidth=1.5,
         linestyle='--',
         label='Σ MC_i / Σ V_i  (manual — should overlay)')

ax3.set_ylabel('MMC (au)', fontsize=11)
ax3.set_title('MMC(t) = Σ MC_i(t) / Σ V_i  —  Eq. 15',
              fontweight='bold')
ax3.legend(fontsize=9, loc='upper left')
ax3.grid(True, alpha=0.3)

# ── Panel 4: Kd = f(MMC) ──────────────────────────────────────
ax4 = axes[3]
Kd_eff = np.maximum(Kd_hist, 0.0)

ax4.plot(epoch_times, Kd_hist,
         color='black', linewidth=2.5,
         label='Kd(t) = 1.6e-2·exp(3.5e-3·MMC) − 5.9e-2  (Eq. 14)')
ax4.plot(epoch_times, Kd_eff,
         color='red', linewidth=2,
         linestyle='--',
         label='Effective Kd (clipped at 0)')

ax4.axhline(0, color='gray', linewidth=1.2, linestyle=':')

# Mark crossover
crossover_MMC = np.log(5.9e-2 / 1.6e-2) / 3.5e-3
# Find when MMC crosses threshold
cross_idx = np.where(MMC_hist >= crossover_MMC)[0]
if len(cross_idx) > 0:
    t_cross = epoch_times[cross_idx[0]]
    ax4.axvline(t_cross, color='green', linewidth=1.5,
                linestyle='--',
                label=f'Kd becomes positive at t={t_cross:.1f}s\n'
                      f'(MMC={crossover_MMC:.1f} au)')

ax4.set_xlabel('Time (s)', fontsize=11)
ax4.set_ylabel('Kd', fontsize=11)
ax4.set_title('Kd(t) — derivative gain grows with fatigue  (Eq. 14)',
              fontweight='bold')
ax4.legend(fontsize=9, loc='upper left')
ax4.grid(True, alpha=0.3)

plt.suptitle('MMC components and Kd over time\n'
             'MVC sustained contraction — 120 seconds\n'
             'Dideriksen et al. (2010)',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.show()

# ── Sanity check table ────────────────────────────────────────
print(f"\n{'Time(s)':>8} | {'Σ MC_i':>10} | {'Σ V_i':>8} | "
      f"{'MMC':>8} | {'MMC check':>10} | {'Kd':>8}")
print("-" * 65)
for ep in [0, 10, 30, 60, 100, n_epochs-1]:
    if ep < n_epochs:
        mmc_check = sum_MCi_hist[ep] / sum_Vi
        print(f"{epoch_times[ep]:>8.1f} | "
              f"{sum_MCi_hist[ep]:>10.2f} | "
              f"{sum_Vi:>8.2f} | "
              f"{MMC_hist[ep]:>8.5f} | "
              f"{mmc_check:>10.5f} | "
              f"{Kd_hist[ep]:>8.4f}")

print(f"\n=== Key values ===")
print(f"Σ V_i (constant):         {sum_Vi:.4f} au")
print(f"MMC crossover (Kd=0):     {crossover_MMC:.2f} au")
print(f"Final MMC:                {MMC_hist[-1]:.5f} au")
print(f"Final Kd:                 {Kd_hist[-1]:.4f}")




met = MetaboliteModel()
mnp = MotorNeuronPool()
spikes = np.round(mnp.PDR * 0.5).astype(int)

for ep in range(240):  # 120s
    out = met.step_epoch(spikes, 100.0)

MC_i = out['MC_i']
print(f"Σ MC_i        = {np.sum(MC_i):.2f}")
print(f"Σ V_i         = {np.sum(met.V_i):.2f}")
print(f"n MUs         = {len(MC_i)}")
print(f"")
print(f"MMC (vol-weighted) = {np.sum(MC_i)/np.sum(met.V_i):.4f} au")
print(f"MMC (simple mean)  = {np.mean(MC_i):.4f} au")
print(f"MMC (paper out)    = {out['MMC']:.4f} au")
print(f"")
print(f"Kd crossover at MMC = 373 au")
print(f"Vol-weighted reaches 373? "
      f"{'YES' if np.sum(MC_i)/np.sum(met.V_i) > 373 else 'NO'}")
print(f"Simple mean reaches 373?  "
      f"{'YES' if np.mean(MC_i) > 373 else 'NO'}")



import numpy as np
import matplotlib.pyplot as plt
from pid_new import DescendingDriveController

dt      = 0.001
max_REI = 45.0
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ══════════════════════════════════════════════════════════════
# Panel 1 — Error e(t) = Ft - Fm  (Eq. 12)
# ══════════════════════════════════════════════════════════════
ax   = axes[0]
target = 20.0
pid    = DescendingDriveController(target, max_REI, dt=dt)
pid.warm_start((target / 100.0) * max_REI)

n_steps   = 9000   # 9 seconds
time_arr  = np.arange(n_steps) * dt
err_hist  = np.zeros(n_steps)
REI_hist  = np.zeros(n_steps)

# Simulate force: starts at target, drops at t=2s,
# recovers at t=5s, overshoots at t=7s
def synthetic_force(t):
    if t < 2.0:
        return target
    elif t < 5.0:
        return target - 4.0        # force drop → positive error
    elif t < 7.0:
        return target              # recovery
    else:
        return target + 3.0       # overshoot → negative error

for step in range(n_steps):
    t   = step * dt
    fm  = synthetic_force(t)
    err = target - fm
    REI = pid.update(actual_force=fm)
    err_hist[step] = err
    REI_hist[step] = REI

ax.plot(time_arr, err_hist,
        color='black', lw=2.5,
        label='e(t) = Ft − Fm')
ax.axhline(0, color='gray', lw=1.5, ls='--')

# Shade regions
ax.fill_between(time_arr, err_hist, 0,
                where=err_hist > 0,
                color='red', alpha=0.15,
                label='Force below target (PID increases REI)')
ax.fill_between(time_arr, err_hist, 0,
                where=err_hist < 0,
                color='blue', alpha=0.15,
                label='Force above target (PID decreases REI)')

ax.axvline(2.0, color='gray', lw=1.0, ls=':')
ax.axvline(5.0, color='gray', lw=1.0, ls=':')
ax.axvline(7.0, color='gray', lw=1.0, ls=':')

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Error e(t)  (% MVC)', fontsize=12)
ax.set_title('Eq. 12 — Error function\n'
             'e(t) = Ft − Fm', fontweight='bold', fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ══════════════════════════════════════════════════════════════
# Panel 2 — Kp and Ki in isolation
# ══════════════════════════════════════════════════════════════
ax = axes[1]

# Show what each gain contributes independently
# P term: Kp × e(t)  — instantaneous
# I term: Ki × ∫e dt — accumulated over time
errors     = np.linspace(-10, 10, 500)
P_contrib  = 2e-6  * errors          # Kp × e
I_t_1s     = 2.5e-4 * errors * 1.0  # Ki × e × 1s
I_t_5s     = 2.5e-4 * errors * 5.0  # Ki × e × 5s
I_t_30s    = 2.5e-4 * errors * 30.0 # Ki × e × 30s

ax.plot(errors, P_contrib,
        color='blue', lw=2.5,
        label=f'P term: Kp×e  (Kp=2×10⁻⁶)')
ax.plot(errors, I_t_1s,
        color='green', lw=2, ls='--',
        label=f'I term: Ki×e×1s  (Ki=2.5×10⁻⁴)')
ax.plot(errors, I_t_5s,
        color='orange', lw=2, ls='--',
        label=f'I term: Ki×e×5s')
ax.plot(errors, I_t_30s,
        color='red', lw=2, ls='--',
        label=f'I term: Ki×e×30s')

ax.axhline(0, color='gray', lw=1.2, ls=':')
ax.axvline(0, color='gray', lw=1.2, ls=':')

# Annotate: I dominates over time
ax.annotate('I term dominates\nover time',
            xy=(5, I_t_30s[375]), xytext=(2, 0.06),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=9)

ax.set_xlabel('Error e(t)  (% MVC)', fontsize=12)
ax.set_ylabel('Contribution to REI', fontsize=12)
ax.set_title('Kp and Ki in isolation\n'
             'I term grows with time → drives sustained tracking',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ══════════════════════════════════════════════════════════════
# Panel 3 — Kd in isolation vs MMC (Eq. 14)
# ══════════════════════════════════════════════════════════════
ax = axes[2]

pid_tmp   = DescendingDriveController(20.0, max_REI, dt=dt)
MMC_range = np.linspace(0, 2500, 1000)
Kd_vals   = np.array([pid_tmp.compute_Kd(m) for m in MMC_range])
Kd_eff    = np.maximum(Kd_vals, 0.0)

ax.plot(MMC_range, Kd_vals,
        color='black', lw=2.5,
        label='Kd(t) — Eq. 14')
ax.plot(MMC_range, Kd_eff,
        color='red', lw=2, ls='--',
        label='Effective Kd (clipped at 0)')
ax.axhline(0, color='gray', lw=1.2, ls=':')

# Key reference points
ref_pts = [
    (0,    'blue',   'Rest\n(no effect)'),
    (373,  'green',  'Crossover\n(Kd=0)'),
    (1150, 'orange', 'MCref'),
    (2000, 'red',    'High fatigue'),
]
for mmc, col, lbl in ref_pts:
    kd = pid_tmp.compute_Kd(mmc)
    ax.plot(mmc, kd, 'o', color=col, ms=10, zorder=5,
            label=f'{lbl}: Kd={kd:.3f}')
    ax.axvline(mmc, color=col, lw=0.8, ls=':', alpha=0.4)

# Annotate CV criteria from paper
ax.annotate('CV=3% at rest\n(Kd≈0)',
            xy=(373, 0), xytext=(600, -0.03),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=8, color='green')
ax.annotate('CV=8% at\ntask failure',
            xy=(2000, pid_tmp.compute_Kd(2000)),
            xytext=(1500, 10),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=8, color='red')

ax.set_xlabel('MMC (au)', fontsize=12)
ax.set_ylabel('Kd', fontsize=12)
ax.set_title('Eq. 14 — Kd grows with fatigue\n'
             'CV: 3% (rest) → 8% (task failure)',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 2500])

plt.suptitle('PID Controller Component Validation\n'
             'Dideriksen et al. (2010)  |  Eqs. 12–14',
             fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()

# ── Console checks ────────────────────────────────────────────
pid_chk = DescendingDriveController(20.0, max_REI, dt=dt)
print(f"\n{'='*50}")
print(f"PARAMETER CHECK vs Table 1 / Paper text")
print(f"{'='*50}")
print(f"Kp = {pid_chk.Kp:.2e}  "
      f"(paper: 2×10⁻⁶)  "
      f"{'✓' if pid_chk.Kp == 2e-6 else '✗'}")
print(f"Ki = {pid_chk.Ki:.2e}  "
      f"(paper: 2.5×10⁻⁴)  "
      f"{'✓' if pid_chk.Ki == 2.5e-4 else '✗'}")
print(f"Update = {1/pid_chk.update_interval:.1f} Hz  "
      f"(paper: 3 Hz)  "
      f"{'✓' if abs(1/pid_chk.update_interval-3)<0.01 else '✗'}")
print(f"\nKd at MMC=0:    {pid_chk.compute_Kd(0):.4f}  → clips to 0")
print(f"Kd crossover:   MMC ≈ 373 au")
print(f"Kd at MCref:    {pid_chk.compute_Kd(1150):.4f}")
print(f"\nMMC formula:    Σ MC_i / n  (simple mean, Eq. 15)  ✓")
'''


"""
fig_pid_validation.py

Generates the missing validation figure for Section 1.4 (Neural Control).

Three-panel figure:
    A — Kd(t) as a function of MMC (Equation 14 of Dideriksen 2010)
        Shows the exponential growth of derivative gain with metabolite
        accumulation — the physiological basis for increasing force variability
    B — REI(t) over time during a simulated fatiguing contraction
        Shows the controller compensating for fatigue by progressively
        increasing excitatory drive
    C — Force tracking: target vs simulated force over time
        Demonstrates that the controller successfully maintains the
        target force during the early phase and shows increasing
        variability toward task failure

Uses the SimpleExcitationController and the metabolite/force blocks directly.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pid_new   import DescendingDriveController
from metabolite_block  import MetaboliteModel

mpl.rcParams.update({
    'font.family':      'Times New Roman',
    'font.weight':      'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'font.size':         12,
    'axes.titlesize':    12,
    'axes.labelsize':    11,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':    9,
    'lines.linewidth':    2.0,
})

COLOR_PID  = '#6A1B9A'   # purple
COLOR_REI  = '#1565C0'   # blue
COLOR_FORCE= '#2E7D32'   # green
COLOR_KD   = '#C62828'   # red


# ══════════════════════════════════════════════════════════════════════
# Panel A — Kd as a function of MMC (standalone, no simulation needed)
# ══════════════════════════════════════════════════════════════════════
def compute_kd_curve():
    """Equation 14: Kd(t) = 1.6e-2 * exp(3.5e-3 * MMC) - 5.9e-2"""
    pid = DescendingDriveController(target_force=0.3, max_REI=45.0)
    mmc_range = np.linspace(0, 2000, 500)
    kd_values = np.array([pid.compute_Kd(mmc) for mmc in mmc_range])
    return mmc_range, kd_values


# ══════════════════════════════════════════════════════════════════════
# Panels B & C — REI and force tracking over simulated contraction
# Uses a simplified spike-rate model to drive metabolite accumulation
# without needing the full closed-loop simulation
# ══════════════════════════════════════════════════════════════════════
def run_pid_simulation(target_pct=30.0, duration=200.0,
                       dt=0.001, epoch_dt=0.5):
    """
    Simulate PID controller behaviour during a fatiguing contraction.

    Instead of running the full closed-loop, we:
    1. Use the metabolite model to accumulate MC over time
       (driven by fixed spike counts approximating 30% MVC)
    2. Feed MMC into the PID at each update
    3. Simulate force as: force = REI * scale_factor * (1 - fatigue_loss)
       where fatigue_loss grows with MMC — a simplified force model
       sufficient to demonstrate controller behaviour
    """
    met  = MetaboliteModel()
    pid  = DescendingDriveController(
        target_force    = target_pct,
        max_REI         = 45.0,
        dt              = dt,
        update_interval = 1.0 / 3.0
    )

    samples_per_epoch = int(epoch_dt / dt)
    n_steps           = int(duration / dt)

    # Storage
    time_arr  = np.zeros(n_steps)
    REI_arr   = np.zeros(n_steps)
    force_arr = np.zeros(n_steps)
    MMC_arr   = np.zeros(n_steps)
    Kd_arr    = np.zeros(n_steps)

    # Fixed spike counts per epoch (approximating 30% MVC, 85 MUs active)
    spike_counts = np.zeros(120, dtype=int)
    for i in range(85):
        frac = i / 84.0
        spike_counts[i] = max(1, int(round(6 * (1 - frac) + 3 * frac)))

    # Metabolite state
    MC_i  = np.zeros(120)
    MMC   = 0.0
    epoch_counter = 0
    force_epoch_buf = []

    # Simple force model: maps REI to force percentage
    # At REI=25, force ≈ 30% MVC (from calibration)
    # Force declines as MC increases (twitch depression)
    REI_to_force_scale = target_pct / 25.0   # calibrated ratio

    current_force = 0.0

    for step in range(n_steps):
        t = step * dt

        # Simple force model with fatigue depression
        # P_gain approximation: force_remaining ≈ 1 - 0.3*(MMC/1150)
        fatigue_loss  = min(0.5, 0.3 * (MMC / 1150.0))
        noise         = np.random.normal(0, target_pct * 0.015)   # ~1.5% noise
        current_force = (pid.output * REI_to_force_scale
                         * (1 - fatigue_loss) + noise)
        current_force = max(0, current_force)

        # PID update
        REI = pid.update(
            actual_force = current_force,
            MC_array     = MC_i,
            V_array      = met.V_i
        )

        force_epoch_buf.append(current_force)
        epoch_counter += 1

        # Metabolite epoch
        if epoch_counter >= samples_per_epoch:
            mean_force = float(np.mean(force_epoch_buf))
            met_out    = met.step_epoch(spike_counts, mean_force)
            MC_i       = met_out['MC_i'].copy()
            MMC        = float(met_out['MMC'])
            epoch_counter   = 0
            force_epoch_buf = []

        time_arr[step]  = t
        REI_arr[step]   = REI
        force_arr[step] = current_force
        MMC_arr[step]   = MMC
        Kd_arr[step]    = pid.Kd

    return time_arr, REI_arr, force_arr, MMC_arr, Kd_arr, target_pct


# ══════════════════════════════════════════════════════════════════════
# Main plotting function
# ══════════════════════════════════════════════════════════════════════
def plot_pid_validation(save=True):

    print("Computing Kd curve...")
    mmc_range, kd_values = compute_kd_curve()

    print("Running PID simulation...")
    time_arr, REI_arr, force_arr, MMC_arr, Kd_arr, target_pct = \
        run_pid_simulation(target_pct=30.0, duration=200.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        'Neural Control Block — Validation\n'
        'PID-Based Descending Drive Estimator '
        '(Dideriksen et al. 2010, Eqs. 12–15)',
        fontsize=13, fontweight='bold'
    )

    # ── Panel A: Kd vs MMC ───────────────────────────────────────────
    ax = axes[0]
    ax.plot(mmc_range, kd_values, color=COLOR_KD, lw=2.5)

    # Shade the physiological MMC range during a typical contraction
    ax.axvspan(0, 1150, alpha=0.07, color=COLOR_KD,
               label='Physiological range (0–$MC_{ref}$)')
    ax.axvline(1150, color='gray', lw=1.0, ls='--')
    ax.text(1160, kd_values.min() + 0.002,
            '$MC_{ref}$ = 1,150 au', fontsize=8, color='gray')

    # Annotate zero crossing
    zero_idx = np.argmin(np.abs(kd_values))
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.plot(mmc_range[zero_idx], 0, 'ko', ms=5)
    ax.text(mmc_range[zero_idx] + 30, 0.001,
            f'$K_d$ = 0\nat MMC ≈ {mmc_range[zero_idx]:.0f} au',
            fontsize=8)

    ax.set_xlabel('Mean Metabolite Concentration MMC (au)')
    ax.set_ylabel('Derivative Gain $K_d(t)$')
    ax.set_title('A — $K_d$(MMC): Equation 14', loc='left')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel B: REI over time ───────────────────────────────────────
    ax = axes[1]

    # Smooth REI for display (3 Hz updates produce staircase — show both)
    from scipy.ndimage import uniform_filter1d
    REI_smooth = uniform_filter1d(REI_arr, size=1000)

    ax.plot(time_arr, REI_arr,    color=COLOR_REI, lw=0.8, alpha=0.3)
    ax.plot(time_arr, REI_smooth, color=COLOR_REI, lw=2.2,
            label='REI (smoothed trend)')
    ax.axhline(25.3, color='gray', lw=1.0, ls='--')
    ax.text(5, 25.5, 'Initial REI (calibrated)', fontsize=8, color='gray')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Required Excitatory Input (au)')
    ax.set_title('B — REI(t): Controller Output Over Time', loc='left')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Annotate Kd increase
    ax2 = ax.twinx()
    ax2.plot(time_arr, Kd_arr, color=COLOR_KD, lw=1.5, ls='--',
             alpha=0.7, label='$K_d(t)$')
    ax2.set_ylabel('$K_d(t)$', color=COLOR_KD)
    ax2.tick_params(axis='y', labelcolor=COLOR_KD)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc='upper left', fontsize=8)

    # ── Panel C: Force tracking ───────────────────────────────────────
    ax = axes[2]

    # Downsample for display (every 10 ms)
    ds = 10
    ax.plot(time_arr[::ds], force_arr[::ds],
            color=COLOR_FORCE, lw=1.0, alpha=0.6,
            label='Simulated force')

    # Rolling mean to show trend
    force_trend = uniform_filter1d(force_arr, size=3000)
    ax.plot(time_arr[::ds], force_trend[::ds],
            color=COLOR_FORCE, lw=2.2,
            label='Force trend (rolling mean)')

    ax.axhline(target_pct,
               color='k', lw=1.3, ls=':',
               label=f'Target ({target_pct:.0f}% MVC)')
    ax.axhline(target_pct * 0.90,
               color='gray', lw=1.0, ls='-.',
               label='Task-failure threshold (90%)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (% MVC)')
    ax.set_title('C — Force Tracking Performance', loc='left')
    ax.set_ylim([0, target_pct * 1.8])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    # Annotate force variability increase
    # Compare CV at start vs end
    n_start = int(0.05 * len(force_arr))
    n_end   = int(0.10 * len(force_arr))
    cv_start = np.std(force_arr[:n_start]) / np.mean(force_arr[:n_start]) * 100
    cv_end   = np.std(force_arr[-n_end:])  / np.mean(force_arr[-n_end:])  * 100
    ax.text(0.02, 0.12,
            f'CV at start: {cv_start:.1f}%\nCV at end: {cv_end:.1f}%',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='gray', alpha=0.85))

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save:
        fig.savefig('fig_pid_validation.png', dpi=130, bbox_inches='tight')
        print('[Plot] Saved → fig_pid_validation.png')

    plt.show()
    return fig


if __name__ == '__main__':
    plot_pid_validation(save=True)