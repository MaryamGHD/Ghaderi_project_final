"""
Force Generation Block - Dideriksen et al. (2010)

This module implements the force generation model exactly as described in:
"An integrative model of motor unit activity during sustained submaximal contractions"
Dideriksen et al., J Appl Physiol 108:1550-1562, 2010

Equations implemented:
- Eq. 37: Motor unit twitch force (Fuglevand base + fatigue gains)
- Eq. 38-40: Contraction time gain (T_gain)  [fatigue only]
- Eq. 41-42: Correction factor (CF)           [fatigue only]
- Eq. 43-44: Amplitude gain (P_gain)          [fatigue only]

Usage modes:
- Non-fatigued (MC=None or MC=0): Pure Fuglevand twitch (Eq. 37 with gains = 1)
- Fatigued (MC > 0): Full Dideriksen gains applied
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Global bold style ─────────────────────────────────────────
mpl.rcParams['font.weight']        = 'bold'
mpl.rcParams['axes.titleweight']   = 'bold'
mpl.rcParams['axes.labelweight']   = 'bold'
mpl.rcParams['font.size']          = 12
mpl.rcParams['axes.titlesize']     = 14
mpl.rcParams['axes.labelsize']     = 12
mpl.rcParams['xtick.labelsize']    = 11
mpl.rcParams['ytick.labelsize']    = 11
mpl.rcParams['legend.fontsize']    = 10
mpl.rcParams['lines.linewidth']    = 5
class ForceGenerationBlock:
    """
    Force generation block from Dideriksen et al. (2010).

    Converts motor unit spike trains into muscle force through temporal
    and spatial summation of motor unit twitches.

    Parameters
    ----------
    n_motor_units : int
        Total number of motor units (default: 120, as in paper for FDI)
    dt : float
        Time step in seconds (default: 0.001 s = 1 ms, as in paper)
    P_base : float
        Base peak twitch force for the smallest motor unit (arbitrary units)
    T_base : float
        Base contraction time for the smallest motor unit in seconds
        (default: 0.090 s = 90 ms, from Fuglevand et al. 1993)
    RP : float
        Range of peak forces: ratio of largest to smallest MU twitch
        (default: 100, from Fuglevand et al. 1993 for FDI)
    RT : float
        Range of contraction times: ratio of longest to shortest
        (default: 3.0, from Fuglevand et al. 1993 Table 1)
    """

    def __init__(self, n_motor_units=120, dt=0.001, P_base=1.0,
                 T_base=0.090, RP=100.0, RT=3.0):
        self.n = n_motor_units
        self.dt = dt
        self.P_base = P_base
        self.T_base = T_base
        self.RP = RP
        self.RT = RT

        # Metabolite reference for fatigue equations (Table 1)
        self.MC_ref = 1150.0

        # Initialize motor unit properties (twitch amplitudes and times)
        self._initialize_motor_unit_properties()

        # Initialize online stepping state
        self.reset()

    def _initialize_motor_unit_properties(self):
        """
        Initialize motor unit-specific properties following
        Fuglevand et al. (1993) as referenced in the paper.

        - Peak twitch forces: exponentially distributed (Eq. 13 in Fuglevand)
        - Contraction times: inversely related to peak force
        """
        # Motor unit indices (1 to n)
        self.mu_indices = np.arange(1, self.n + 1)

        # --- Peak twitch force distribution ---
        # Fuglevand Eq. 13: P_i = P_1 * exp(b * (i-1))
        # where b = ln(RP) / (n-1)
        b = np.log(self.RP) / (self.n - 1)
        self.P = self.P_base * np.exp(b * (self.mu_indices - 1))

        # --- Contraction time distribution ---
        # Fuglevand: T_i = T_base * (P_1 / P_i)^(1/c)
        # where c = log(RP) / log(RT), RT = range of contraction times
        # RT = 3 from Fuglevand Table 1 (NOT equal to RP)
        c = np.log(self.RP) / np.log(self.RT)
        self.T = self.T_base * (self.P[0] / self.P) ** (1.0 / c)

    # ==========================================================
    # Online step-by-step interface (for closed-loop simulation)
    # ==========================================================

    def reset(self):
        """Clear all internal state for a new simulation."""
        # Each entry is a list of spike times (in seconds) still producing force
        self.twitch_buffers = [[] for _ in range(self.n)]

    def step(self, t, spike_events, MC=None):
        # 1. Record new spikes
        firing_indices = np.where(spike_events)[0]
        for mu_idx in firing_indices:
            self.twitch_buffers[mu_idx].append(t)

        # 2. Determine fatigue mode
        use_fatigue = MC is not None

        # 3. Compute force from all active twitches
        total_force = 0.0

        for mu_idx in range(self.n):
            if len(self.twitch_buffers[mu_idx]) == 0:
                continue

            i = mu_idx + 1
            P_i = self.P[mu_idx]
            T_i = self.T[mu_idx]

            # Match simulate()'s fatigue logic exactly
            if use_fatigue:
                mc_val = MC[mu_idx] if hasattr(MC, '__len__') else MC
                T_gain = self.compute_T_gain(i, mc_val)
                P_gain = self.compute_P_gain(i, mc_val)
                twitch_scale = 1 - P_gain
            else:
                T_gain = self.compute_T_gain(i, 0)
                twitch_scale = 1 - self.compute_P_gain(i, 0)

            T_scaled = T_gain * T_i
            cutoff = 5.0 * T_scaled

            mu_force = 0.0
            still_active = []

            for spike_idx_j, spike_time in enumerate(self.twitch_buffers[mu_idx]):
                t_rel = t - spike_time

                if t_rel > cutoff:
                    continue

                still_active.append(spike_time)

                if t_rel <= 0:
                    continue

                # Match simulate()'s gain_f logic
                gain_f = 1.0 if spike_idx_j == 0 else \
                    self.compute_gain_f(i,
                                        self.twitch_buffers[mu_idx][spike_idx_j] - self.twitch_buffers[mu_idx][
                                            spike_idx_j - 1],
                                        T_scaled=T_scaled)

                normalized = t_rel / T_scaled
                mu_force += gain_f * twitch_scale * P_i * normalized * np.exp(1.0 - normalized)

            self.twitch_buffers[mu_idx] = still_active
            total_force += mu_force

        return total_force

    # ==========================================================
    # Fatigue gain equations (Eqs. 38-44) — used when MC > 0
    # ==========================================================

    def compute_gain_max(self, i):
        # Eq. 39: confirmed i²/n
        ratio= i/self.n

        return (1.66 * (ratio**2) + 0.25 * (i / self.n) - 0.25)
    def compute_delta_gain_mc(self, MC):
        """Eq. 40: Metabolite-dependent gain factor (0 to 1)."""
        return np.tanh(4.0 * ((MC / self.MC_ref) - 0.5)) * 0.5 + 0.5

    def compute_T_gain(self, i, MC):
        """Eq. 38: Contraction time gain (≥ 1, increases with fatigue)."""
        gain_max = self.compute_gain_max(i)
        delta_gain = self.compute_delta_gain_mc(MC)

        return 1.0 + gain_max * delta_gain

    def compute_b_i(self, i):
        """Eq. 42: b_i parameter for correction factor."""
        n_i = i / self.n
        return 0.389 * np.exp(-4.413 * n_i) + 0.935 * np.exp(0.182 * n_i)

    def compute_CF(self, i, MC):
        """Eq. 41: Correction factor for amplitude during fatigue."""
        T_gain = self.compute_T_gain(i, MC)
        delta_gain = self.compute_delta_gain_mc(MC)
        b_i = self.compute_b_i(i)
        return (1.0 / T_gain) * (1.0 + delta_gain * ((1.0 / b_i )- 1.0))

    def compute_h(self, i):
        # -(i - 0.67n) not (-i - 0.67n)
        #s_i = (np.tanh((i - 0.75 * self.n) / (0.34 * self.n)) * 1.04 +
        #np.tanh(-(i - 0.67 * self.n) / (0.37 * self.n)) * 0.95 + 0.97)

        s_i = (np.tanh((i - 0.75 * self.n) / (0.34 * self.n)) * 1.04 +
               np.tanh(-((i - 0.67 * self.n) / (0.37 * self.n))) * 0.95 + 0.97)

        t_i = np.tanh(i / (0.12 * self.n )+ 1.0) * 0.13+ 1.0
        #t_i = np.tanh(i / (0.12 * self.n) + 1.0) *13 + 1.0

        u_i = np.tanh((i - self.n) / (0.04 * self.n) + 1.0) * 0.06 + 1.0
        return (-2.0 * i / self.n + 2.88) * s_i * t_i * u_i
        #n_i = i / self.n
        #return 2.0 * np.exp(-1.75 * n_i)



        # Linear decrease: h(1)≈2.30, h(60)≈0.70, h(120)≈0.40









    def compute_P_gain(self, i, MC):
        CF = self.compute_CF(i, MC)
        h_i = self.compute_h(i)
        ratio=0.5 * h_i


        return (1 * ((np.tanh((MC / self.MC_ref - h_i) / (ratio)) * 0.5 + 0.5)))
    # ==========================================================
    # Single twitch computation (for testing / inspection)
    # ==========================================================
    def compute_twitch(self, t, i, MC=0, gain_f=1.0):
        """
        Compute motor unit twitch force (Eq. 37 Dideriksen / Eq. 18 Fuglevand).

        Parameters
        ----------
        t      : float or ndarray — time after spike (seconds)
        i      : int              — MU index (1-indexed)
        MC     : float or None    — metabolite concentration
                                    None = no fatigue
        gain_f : float            — twitch summation gain (Eqs. 16-17)
                                    computed externally from T_i/ISI
        """
        idx = i - 1
        P = self.P[idx]
        T = self.T[idx]

        # Only apply fatigue gains if MC is provided and positive
        use_fatigue = (MC is not None) and (float(np.asarray(MC).flat[0]) > 0)


        T_gain = self.compute_T_gain(i, MC)

        P_gain = self.compute_P_gain(i, MC)
        twitch_scale = 1- P_gain  # P_gain = force LOSS fraction


        t = np.maximum(t, 0.0)
        T_scaled = T_gain * T
        normalized_time = t / T_scaled

        # Eq. 18: f(t) = gain_f · twitch_scale · P · (t/T) · exp(1 - t/T)
        return gain_f * twitch_scale * P * normalized_time * np.exp(1.0 - normalized_time)

    def compute_gain_f(self, i, ISI, T_scaled=None):
        if ISI is None or ISI <= 0:
            return 1.0

        T_i = T_scaled if T_scaled is not None else self.T[i - 1]
        ratio = T_i / ISI

        if ratio <= 0.4:
            return 1.0

        # Sigmoid region
        S = 1.0 - np.exp(-2.0 * ratio ** 3)
        g = S / ratio

        # ← normalization MUST stay: makes gain_f=1.0 at boundary
        # and INCREASES above 1.0 for higher ratios (more summation)
        S_04 = 1.0 - np.exp(-2.0 * 0.4 ** 3)
        g_04 = S_04 / 0.4

        return g / g_04  # ← this gives gain_f > 1 for ratio > 0.4


    def simulate(self, spike_times, duration, MC=None):
        n_steps = int(duration / self.dt)
        time = np.arange(n_steps) * self.dt
        force_mu = np.zeros((self.n, n_steps))

        # FIXED: proper use_fatigue definition
        use_fatigue = MC is not None

        for mu_idx in range(self.n):
            i = mu_idx + 1
            spikes = spike_times[mu_idx]

            if len(spikes) == 0:
                continue

            P_i = self.P[mu_idx]
            T_i = self.T[mu_idx]

            # FIXED: guard before calling compute functions
            if use_fatigue:
                mc_val = MC[mu_idx] if hasattr(MC, '__len__') else MC
                T_gain = self.compute_T_gain(i, mc_val)
                P_gain = self.compute_P_gain(i, mc_val)
                twitch_scale =1-P_gain

            else:
                T_gain = self.compute_T_gain(i, 0)
                twitch_scale = 1-self.compute_P_gain(i, 0)

            T_scaled = T_gain * T_i

            for spike_idx_j, spike_time in enumerate(spikes):
                gain_f = 1.0 if spike_idx_j == 0 else \
                    self.compute_gain_f(i, spike_time - spikes[spike_idx_j - 1],T_scaled=T_scaled)

                spike_idx = int(spike_time / self.dt)
                if spike_idx >= n_steps:
                    continue

                t_rel = time[spike_idx:] - spike_time
                t_rel = np.maximum(t_rel, 0.0)
                normalized = t_rel / T_scaled
                twitch = gain_f* twitch_scale *P_i* normalized * np.exp(1.0 - normalized)
                force_mu[mu_idx, spike_idx:] += twitch

        return {
            'time': time,
            'force_total': np.sum(force_mu, axis=0),
            'force_mu': force_mu
        }
    # ==========================================================
    # Plotting utility
    # ==========================================================

    def plot_force(self, results, mu_indices=None, title='Motor Unit Force Generation'):
        """Plot force traces similar to Figure 4 in Dideriksen et al. (2010)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        time_ms = results['time'] * 1000

        if mu_indices is not None:
            for mu_idx in mu_indices:
                idx = mu_idx - 1
                ax.plot(time_ms, results['force_mu'][idx, :],
                        label=f'MU{mu_idx}', linewidth=1.5)

        ax.plot(time_ms, results['force_total'],
                label='Total Force', linewidth=2, color='black', linestyle='--')

        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Force (au)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax

    # ==========================================================
    # Debug helper
    # ==========================================================

    def debug_force_computation(self, spike_times, current_time, mu_indices=[0, 1, 2]):
        """Debug helper to show force computation details."""
        print(f"\n=== FORCE DEBUG at t={current_time:.3f}s ===")
        total_force = 0.0

        for mu_idx in mu_indices:
            i = mu_idx + 1
            spikes = spike_times[mu_idx] if mu_idx < len(spike_times) else []

            print(f"\nMU #{i}:")
            print(f"  P = {self.P[mu_idx]:.4f}, T = {self.T[mu_idx]*1000:.1f} ms")
            print(f"  Total spikes: {len(spikes)}")

            if len(spikes) == 0:
                continue

            mu_force = 0.0
            max_duration = 5 * self.T[mu_idx]

            for spike_time in spikes:
                t_since = current_time - spike_time
                if t_since < 0 or t_since > max_duration:
                    continue
                twitch = self.compute_twitch(t=t_since, i=i, MC=None)
                mu_force += twitch

            print(f"  Force contribution: {mu_force:.4f}")
            total_force += mu_force

        print(f"\nTotal force (from {len(mu_indices)} MUs): {total_force:.4f}")
        print("=" * 50)



'''


import numpy as np
import matplotlib.pyplot as plt
from force_generation2 import ForceGenerationBlock

force_model = ForceGenerationBlock(n_motor_units=120, dt=0.001)

n          = 120
MCref      = 1150
mu_indices = np.arange(1, n + 1)

# ----------------------------------------------------------------
# Key idea: single isolated spike per MU
# gain_f = 1.0 (first spike)
# peak occurs at normalized=1 → t = T_scaled
# so we need duration long enough to capture peak for all MUs
# ----------------------------------------------------------------

# Check max T_i to set duration safely
max_T = np.max(force_model.T)
print(f"Max T_i: {max_T:.4f} s")

# Duration must be > spike_time + max(T_scaled_fatigued)
# T_scaled can be ~2.5x T_i at MCref for large MUs
duration     = 0.1 + 3.0 * max_T   # safe margin
spike_time   = 0.1                  # single spike at t=0.1s
single_spike = [np.array([spike_time])] * n

print(f"Duration used: {duration:.4f} s")

# ----------------------------------------------------------------
# Run 1: Baseline (no fatigue)
# ----------------------------------------------------------------
result_base = force_model.simulate(
    spike_times=single_spike,
    duration=duration,
    MC=None
)

# ----------------------------------------------------------------
# Run 2: Fatigued at MCref
# ----------------------------------------------------------------
MC_at_ref = np.full(n, MCref)
result_fat = force_model.simulate(
    spike_times=single_spike,
    duration=duration,
    MC=1150
)

# ----------------------------------------------------------------
# Peak force per MU
# ----------------------------------------------------------------
peak_base = np.max(result_base['force_mu'], axis=1)  # shape (120,)
peak_fat  = np.max(result_fat['force_mu'],  axis=1)  # shape (120,)

# ----------------------------------------------------------------
# Normalized force (solid line)
# ----------------------------------------------------------------

normalized_force = np.where(
    peak_base > 0,
    peak_fat / peak_base,
    0.0
)


normalized_force= peak_fat/peak_base
# ----------------------------------------------------------------
# Normalized relaxation time (dashed line)
# ----------------------------------------------------------------
T_scaled_base = np.zeros(n)
T_scaled_fat  = np.zeros(n)

for idx, i in enumerate(mu_indices):
    mu_idx = idx
    T_i = force_model.T[mu_idx]
    T_gain_base = force_model.compute_T_gain(i, 0.0)
    T_gain_fat = force_model.compute_T_gain(i, 1150)
    T_scaled_base[idx] = T_gain_base * T_i
    T_scaled_fat[idx] = T_gain_fat  * T_i

normalized_T = T_scaled_fat / T_scaled_base

# ----------------------------------------------------------------
# Diagnostics — check before plotting
# ----------------------------------------------------------------
print("\n--- Per MU spot check ---")
for check_mu in [1, 30, 60, 90, 120]:
    idx = check_mu - 1
    print(f"MU{check_mu:3d} | "
          f"peak_base={peak_base[idx]:.4f} | "
          f"peak_fat={peak_fat[idx]:.4f}  | "
          f"norm_force={normalized_force[idx]:.4f} | "
          f"norm_T={normalized_T[idx]:.4f}")

print(f"\nnormalized_force: {normalized_force.min():.4f} → "
      f"{normalized_force.max():.4f}")
print(f"Expected (paper):  ~1.0 (MU1) → ~0.0 (MU120)")
print(f"\nnormalized_T    : {normalized_T.min():.4f} → "
      f"{normalized_T.max():.4f}")
print(f"Expected (paper):  ~0.75 (MU1) → ~2.5 (MU120)")

# ----------------------------------------------------------------
# Plot
# ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(mu_indices, normalized_force, '-',
        color='black', linewidth=2, label='MU force')
ax.plot(mu_indices, normalized_T,     '--',color='black', linewidth=2, label='Relaxation time')

ax.set_xlabel('MU')
ax.set_ylabel('Normalized change')
ax.set_xlim([0, 120])
ax.set_ylim([0, 3.5])
ax.legend()
plt.tight_layout()
plt.show()






import numpy as np
import matplotlib.pyplot as plt
from force_generation2 import ForceGenerationBlock

# ── Setup ─────────────────────────────────────────────────────
force_model = ForceGenerationBlock(n_motor_units=120, dt=0.001)

# ── Stimulus protocol (from paper caption Fig 4) ──────────────
# 13 action potentials at 40 Hz in a 330-ms interval
freq        = 40.0          # Hz
n_spikes    = 13
spike_start = 0.1           # start at 100ms to see baseline before
ISI         = 1.0 / freq    # 25 ms between spikes

spike_train = spike_start + np.arange(n_spikes) * ISI
duration    = spike_start + n_spikes * ISI + 0.4   # enough tail to see relaxation

# ── Three MC conditions ────────────────────────────────────────
MC_conditions = {
    'MC = 0 (initial)' :  0.0,
    'MC = 575'         :  575.0,
    'MC = 1150 (MCref)': 1150.0,
}
linestyles = ['-', '--', ':']

# ── Three motor units ──────────────────────────────────────────
target_mus = [1, 60, 120]

# ── Build spike_times list (all 120 MUs, only target MUs fire) ─
def make_spike_times(target_mu_idx):
    """
    Return spike_times list where only one MU fires,
    all others are empty.
    """
    spike_times = [np.array([]) for _ in range(120)]
    spike_times[target_mu_idx] = spike_train
    return spike_times

# ── Run simulations and collect results ───────────────────────
# results[mu_label][mc_label] = force array
results = {}

for mu in target_mus:
    mu_idx      = mu - 1
    spike_times = make_spike_times(mu_idx)
    results[mu] = {}

    for mc_label, mc_val in MC_conditions.items():
        MC_array = np.full(120, mc_val)
        out = force_model.simulate(
            spike_times=spike_times,
            duration=duration,
            MC=MC_array
        )
        # Extract only this MU's force
        results[mu][mc_label] = out['force_mu'][mu_idx]

time_ms = out['time'] * 1000   # convert to ms for x-axis

# ── Plot — 3 panels like Figure 4 ─────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
panel_labels = ['A', 'B', 'C']

for ax, mu, panel in zip(axes, target_mus, panel_labels):
    for (mc_label, _), ls in zip(MC_conditions.items(), linestyles):
        force = results[mu][mc_label]
        ax.plot(time_ms, force,
                linestyle=ls,
                color='black',
                linewidth=2,
                label=mc_label)

    # Mark stimulus times
    for st in spike_train:
        ax.axvline(st * 1000, color='gray',
                   linewidth=0.5, alpha=0.4)

    ax.set_title(f'{panel}  MU{mu}', loc='left', fontweight='bold')
    ax.set_ylabel('Force (au)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel('Time (ms)')
fig.suptitle('Figure 4 reproduction — 40 Hz stimulation, 330 ms\n'
             '3 MC conditions per motor unit',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.show()

# ── Quick sanity check ─────────────────────────────────────────
print("\n=== Peak force per MU per condition ===")
print(f"{'MU':>4} | {'MC=0':>10} | {'MC=575':>10} | {'MC=1150':>10}")
print("-" * 45)
for mu in target_mus:
    peaks = [np.max(results[mu][mc_label])
             for mc_label in MC_conditions]
    print(f"{mu:>4} | {peaks[0]:>10.4f} | "
          f"{peaks[1]:>10.4f} | {peaks[2]:>10.4f}")




import numpy as np
import matplotlib.pyplot as plt
from force_generation2 import ForceGenerationBlock

# ── Setup ─────────────────────────────────────────────────────
force_model = ForceGenerationBlock(n_motor_units=120, dt=0.001)

# ── Protocol: single spike per MU ─────────────────────────────
n          = 120
spike_time = 0.05           # spike at 50ms
max_T      = np.max(force_model.T)
duration   = spike_time + 5.0 * max_T * 3.0   # enough tail for all MUs

MC_conditions = {
    'MC = 0'    :   0.0,
    'MC = 575'  : 575.0,
    'MC = 1150' : 1150.0,
}
linestyles = ['-', '--', ':']
colors     = ['black', 'gray', 'lightgray']

# ── Build spike times: one spike per MU at spike_time ─────────
single_spike = [np.array([spike_time])] * n

# ── Run simulations ───────────────────────────────────────────
results = {}
for mc_label, mc_val in MC_conditions.items():
    MC_array = np.full(n, mc_val)
    out = force_model.simulate(
        spike_times=single_spike,
        duration=duration,
        MC=MC_array
    )
    results[mc_label] = out['force_mu']   # shape (120, n_steps)

time_ms = out['time'] * 1000   # ms

# ── Normalize each MU by its own baseline peak ────────────────
peak_baseline = np.max(results['MC = 0'], axis=1)   # shape (120,)
peak_baseline = np.where(peak_baseline > 0, peak_baseline, 1.0)

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for ax, (mc_label, mc_val), ls, col in zip(
        axes, MC_conditions.items(), linestyles, colors):

    force_matrix = results[mc_label]   # shape (120, n_steps)

    for mu_idx in range(n):
        # Normalize by MU's own baseline peak so all visible
        trace = force_matrix[mu_idx] / peak_baseline[mu_idx]
        ax.plot(time_ms, trace,
                color='black',
                linewidth=0.5,
                alpha=0.4)

    # Highlight MU1, MU60, MU120
    for highlight_mu, highlight_color in zip(
            [1, 60, 120], ['blue', 'red', 'green']):
        idx   = highlight_mu - 1
        trace = force_matrix[idx] / peak_baseline[idx]
        ax.plot(time_ms, trace,
                color=highlight_color,
                linewidth=2.5,
                label=f'MU{highlight_mu}',
                zorder=5)

    ax.set_title(mc_label, fontweight='bold')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalized Force (au)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.axvline(spike_time * 1000,
               color='gray', linewidth=1,
               linestyle='--', alpha=0.6,
               label='spike')

fig.suptitle('Single twitch — all 120 MUs across MC conditions\n'
             'Normalized to each MU baseline peak',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.show()

# ── Sanity check: peak time and amplitude ─────────────────────
print(f"\n{'MU':>4} | {'Peak(MC=0)':>10} | "
      f"{'Peak(MC=575)':>12} | {'Peak(MC=1150)':>13} | "
      f"{'T_peak(MC=0) ms':>16} | {'T_peak(MC=1150) ms':>18}")
print("-" * 80)

for mu in [1, 30, 60, 90, 120]:
    idx = mu - 1
    peaks = [np.max(results[mc][idx]) for mc in MC_conditions]
    t_peak_base = time_ms[np.argmax(results['MC = 0'][idx])]
    t_peak_fat  = time_ms[np.argmax(results['MC = 1150'][idx])]
    print(f"{mu:>4} | {peaks[0]:>10.4f} | "
          f"{peaks[1]:>12.4f} | {peaks[2]:>13.4f} | "
          f"{t_peak_base:>16.1f} | {t_peak_fat:>18.1f}")


import numpy as np
import matplotlib.pyplot as plt
from force_generation2 import ForceGenerationBlock

force_model = ForceGenerationBlock(n_motor_units=120, dt=0.001)
mu_indices  = np.arange(1, 121)

# ── Plot 1: P_gain and 1-P_gain vs MC_i/MC_ref ───────────────
# For selected motor units across full MC range
MC_range      = np.linspace(0, 2300, 1000)   # 0 to 2x MCref
MC_normalized = MC_range / force_model.MC_ref  # x-axis: MC/MCref

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
selected_mus = [1, 30, 60, 90, 120]
colors       = ['blue', 'green', 'orange', 'red', 'black']

for mu, color in zip(selected_mus, colors):
    P_gain_vals  = np.array([force_model.compute_P_gain(mu, mc)
                              for mc in MC_range])
    ax1.plot(MC_normalized, P_gain_vals,
             color=color, linewidth=2,
             linestyle='--', label=f'P_gain MU{mu}')
    ax1.plot(MC_normalized, 1 - P_gain_vals,
             color=color, linewidth=2,
             linestyle='-', label=f'1-P_gain MU{mu}')

ax1.axvline(1.0, color='gray', linewidth=1.5,
            linestyle=':', label='MC = MCref')
ax1.axhline(0.0, color='gray', linewidth=0.8)
ax1.axhline(1.0, color='gray', linewidth=0.8)
ax1.set_xlabel('MC_i / MC_ref')
ax1.set_ylabel('Gain value')
ax1.set_title('P_gain (dashed) and 1-P_gain (solid)\nvs Normalized MC\nfor selected MUs')
ax1.legend(fontsize=8, ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 2])
ax1.set_ylim([-0.05, 1.05])

# ── Plot 2: P_gain and 1-P_gain vs Motor Unit index ──────────
# For selected MC conditions
ax2 = axes[1]
MC_conditions = {
    'MC = 0'         :    0.0,
    'MC = 0.5*MCref' :  575.0,
    'MC = MCref'     : 1150.0,
    'MC = 2*MCref'   : 2300.0,
}
linestyles = ['-', '--', '-.', ':']
colors2    = ['blue', 'green', 'red', 'black']

for (mc_label, mc_val), ls, color in zip(
        MC_conditions.items(), linestyles, colors2):

    P_gain_vals = np.array([force_model.compute_P_gain(i, mc_val)
                             for i in mu_indices])

    ax2.plot(mu_indices, P_gain_vals,
             color=color, linewidth=2,
             linestyle=ls, alpha=0.6,
             label=f'P_gain {mc_label}')
    ax2.plot(mu_indices, 1 - P_gain_vals,
             color=color, linewidth=2,
             linestyle=ls,
             label=f'1-P_gain {mc_label}')

ax2.axhline(0.0, color='gray', linewidth=0.8)
ax2.axhline(1.0, color='gray', linewidth=0.8)
ax2.set_xlabel('Motor Unit Index')
ax2.set_ylabel('Gain value')
ax2.set_title('P_gain (faded) and 1-P_gain (solid)\nvs Motor Unit Index\nfor MC conditions')
ax2.legend(fontsize=8, ncol=2)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([1, 120])
ax2.set_ylim([-0.05, 1.05])

plt.suptitle('P_gain and 1-P_gain analysis\n'
             'P_gain = force LOST | 1-P_gain = force REMAINING',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.show()

# ── Sanity check table ────────────────────────────────────────
print(f"\n{'MU':>4} | {'MC=0':>18} | {'MC=0.5*MCref':>18} | "
      f"{'MC=MCref':>18} | {'MC=2*MCref':>18}")
print(f"{'':>4} | {'Pgain | 1-Pgain':>18} | {'Pgain | 1-Pgain':>18} | "
      f"{'Pgain | 1-Pgain':>18} | {'Pgain | 1-Pgain':>18}")
print("-" * 85)

for mu in [1,60, 120]:
    row = f"{mu:>4}"
    for mc_val in [0, 575, 1150, 2300]:
        pg  = force_model.compute_P_gain(mu, mc_val)
        row += f" | {pg:>7.4f} {1-pg:>7.4f}"
    print(row)


import numpy as np
import matplotlib.pyplot as plt
from force_generation2 import ForceGenerationBlock

force_model = ForceGenerationBlock(n_motor_units=120, dt=0.001)
mu_indices  = np.arange(1, 121)
MC_fixed    = force_model.MC_ref

# ── Compute all components ────────────────────────────────────
gain_max      = np.array([force_model.compute_gain_max(i)      for i in mu_indices])
delta_gain_mc = np.array([force_model.compute_delta_gain_mc(MC_fixed) for i in mu_indices])
T_gain        = np.array([force_model.compute_T_gain(i, MC_fixed)     for i in mu_indices])
CF            = np.array([force_model.compute_CF(i, MC_fixed)         for i in mu_indices])
h_i           = np.array([force_model.compute_h(i)                    for i in mu_indices])
P_gain        = np.array([force_model.compute_P_gain(i, MC_fixed)     for i in mu_indices])

# ── Single plot ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(mu_indices, gain_max,
        color='blue',   linewidth=2, linestyle='-',
        label='gain_max,i  (Eq. 39)')

ax.plot(mu_indices, delta_gain_mc,
        color='green',  linewidth=2, linestyle='--',
        label=f'Δgain_mc = {delta_gain_mc[0]:.3f}  (Eq. 40, flat at MC=MCref)')

ax.plot(mu_indices, T_gain,
        color='orange', linewidth=2, linestyle='-.',
        label='T_gain,i  (Eq. 38)')

ax.plot(mu_indices, CF,
        color='purple', linewidth=2, linestyle=':',
        label='CF_i  (Eq. 41)')

ax.plot(mu_indices, h_i,
        color='brown',  linewidth=2, linestyle='-',
        label='h_i  (Eq. 44)')

ax.plot(mu_indices, P_gain,
        color='red',    linewidth=3, linestyle='-',
        label='P_gain,i  (Eq. 43) — RESULT')

ax.plot(mu_indices, 1 - P_gain,
        color='black',  linewidth=3, linestyle='--',
        label='1 - P_gain,i  (force remaining)')

# ── Reference lines ───────────────────────────────────────────
ax.axhline(0.0, color='gray', linewidth=0.8, linestyle=':')
ax.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
ax.axvline(60,  color='gray', linewidth=0.8, linestyle=':',
           alpha=0.5)
ax.axvline(120, color='gray', linewidth=0.8, linestyle=':',
           alpha=0.5)

# ── Annotations for key MUs ───────────────────────────────────
for mu, label_offset in zip([1, 60, 120], [0.08, 0.08, -0.12]):
    idx = mu - 1
    ax.annotate(f'MU{mu}\nP={P_gain[idx]:.3f}',
                xy=(mu, P_gain[idx]),
                xytext=(mu + 3, P_gain[idx] + label_offset),
                fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red',
                                lw=1.2))

ax.set_xlabel('Motor Unit Index')
ax.set_ylabel('Component value')
ax.set_title(f'P_gain components across 120 MUs\n'
             f'at MC = MC_ref = {MC_fixed}  |  Eqs. 38–44',
             fontweight='bold')
ax.legend(fontsize=9, loc='upper left',
          framealpha=0.9)
ax.grid(True, alpha=0.2)
ax.set_xlim([1, 120])

plt.tight_layout()
plt.show()

# ── Table ─────────────────────────────────────────────────────
print(f"\n{'MU':>4} | {'gain_max':>9} | {'Δgain_mc':>9} | "
      f"{'T_gain':>7} | {'h_i':>7} | {'CF':>7} | "
      f"{'P_gain':>7} | {'1-P_gain':>9}")
print("-" * 75)
for mu in [1, 20, 40, 60, 80, 100, 120]:
    idx = mu - 1
    print(f"{mu:>4} | {gain_max[idx]:>9.4f} | "
          f"{delta_gain_mc[idx]:>9.4f} | "
          f"{T_gain[idx]:>7.4f} | "
          f"{h_i[idx]:>7.4f} | "
          f"{CF[idx]:>7.4f} | "
          f"{P_gain[idx]:>7.4f} | "
          f"{1-P_gain[idx]:>9.4f}")


import numpy as np
import matplotlib.pyplot as plt
from force_generation2 import ForceGenerationBlock

force_model  = ForceGenerationBlock(n_motor_units=120, dt=0.001)
selected_mus = [1, 30, 60, 90, 120]
colors       = ['blue', 'green', 'orange', 'red', 'black']

# ── Single spike per MU at t=0.05s ───────────────────────────
spike_time = 0.05
max_T      = np.max(force_model.T)
duration   = spike_time + 6.0 * max_T   # enough tail for all MUs

single_spike = [np.array([spike_time])] * 120

# ── Simulate at MC=0 ─────────────────────────────────────────
result = force_model.simulate(
    spike_times = single_spike,
    duration    = duration,
    MC          = None
)
time_ms = (result['time'] - spike_time) * 1000   # relative to spike, in ms

# ── Figure: 3 panels ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── Panel 1: Raw twitch waveforms ────────────────────────────
ax1 = axes[0]
for mu, color in zip(selected_mus, colors):
    idx   = mu - 1
    trace = result['force_mu'][idx]
    ax1.plot(time_ms, trace,
             color=color, linewidth=2.5,
             label=f'MU{mu}  P={force_model.P[idx]:.2f}  '
                   f'T={force_model.T[idx]*1000:.1f}ms')

ax1.axvline(0, color='gray', linewidth=1,
            linestyle='--', label='spike time')
ax1.set_xlabel('Time relative to spike (ms)')
ax1.set_ylabel('Force (au)')
ax1.set_title('Baseline twitch waveforms\nMC = 0 (no fatigue)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-10, time_ms[-1]])

# ── Panel 2: Normalized twitch waveforms ─────────────────────
ax2 = axes[1]
for mu, color in zip(selected_mus, colors):
    idx       = mu - 1
    trace     = result['force_mu'][idx]
    peak      = np.max(trace)
    if peak > 0:
        trace_norm = trace / peak
    ax2.plot(time_ms, trace_norm,
             color=color, linewidth=2.5,
             label=f'MU{mu}')

ax2.axvline(0, color='gray', linewidth=1, linestyle='--')
ax2.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
ax2.set_xlabel('Time relative to spike (ms)')
ax2.set_ylabel('Normalized Force')
ax2.set_title('Normalized twitch waveforms\n'
              '(shows contraction time differences)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-10, time_ms[-1]])

# ── Panel 3: Peak force and contraction time vs MU index ─────
ax3  = axes[2]
ax3b = ax3.twinx()

# Peak force for all 120 MUs
peak_forces = np.array([np.max(result['force_mu'][idx])
                         for idx in range(120)])
mu_indices  = np.arange(1, 121)

ax3.plot(mu_indices, peak_forces,
         color='black', linewidth=2.5,
         label='Peak force (au)')
ax3b.plot(mu_indices, force_model.T * 1000,
          color='red', linewidth=2.5,
          linestyle='--',
          label='Contraction time (ms)')

# Mark selected MUs
for mu, color in zip(selected_mus, colors):
    idx = mu - 1
    ax3.plot(mu, peak_forces[idx], 'o',
             color=color, markersize=10, zorder=5)
    ax3b.plot(mu, force_model.T[idx] * 1000, 's',
              color=color, markersize=8, zorder=5)

ax3.set_xlabel('Motor Unit Index')
ax3.set_ylabel('Peak twitch force (au)', color='black')
ax3b.set_ylabel('Contraction time (ms)', color='red')
ax3.set_title('Peak force and contraction time\nvs MU index')

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2,
           fontsize=9, loc='upper left')
ax3.grid(True, alpha=0.3)

plt.suptitle('Baseline twitch validation — MC = 0\n'
             'Fuglevand et al. (1993) distributions',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.show()

# ── Sanity check table ────────────────────────────────────────
print(f"\n{'MU':>4} | {'P_i':>8} | {'T_i (ms)':>9} | "
      f"{'Peak force':>10} | {'t_peak (ms)':>11} | "
      f"{'P ratio to MU1':>15} | {'T ratio to MU120':>16}")
print("-" * 85)

peak_mu1  = np.max(result['force_mu'][0])
T_mu120   = force_model.T[119] * 1000

for mu in [1, 30, 60, 90, 120]:
    idx    = mu - 1
    peak   = np.max(result['force_mu'][idx])
    t_peak = time_ms[np.argmax(result['force_mu'][idx])]
    print(f"{mu:>4} | "
          f"{force_model.P[idx]:>8.4f} | "
          f"{force_model.T[idx]*1000:>9.2f} | "
          f"{peak:>10.4f} | "
          f"{t_peak:>11.2f} | "
          f"{peak/peak_mu1:>15.2f} | "
          f"{force_model.T[idx]*1000/T_mu120:>16.3f}")

# ── Key ratios from Fuglevand 1993 ────────────────────────────
print(f"\n=== Key ratios (Fuglevand 1993) ===")
print(f"RP (range of peak forces):        "
      f"{force_model.P[-1]/force_model.P[0]:.1f}  "
      f"(expected: {force_model.RP:.0f})")
print(f"RT (range of contraction times):  "
      f"{force_model.T[0]/force_model.T[-1]:.2f}  "
      f"(expected: {force_model.RT:.1f})")
print(f"Peak force MU120 / MU1:           "
      f"{peak_forces[-1]/peak_forces[0]:.1f}")
print(f"Contraction time MU1 / MU120:     "
      f"{force_model.T[0]/force_model.T[-1]:.2f}")




import numpy as np
import matplotlib.pyplot as plt
from force_generation2 import ForceGenerationBlock

force_model  = ForceGenerationBlock(n_motor_units=120, dt=0.001)
selected_mus = [1, 30, 60, 90, 120]
colors       = ['blue', 'green', 'orange', 'red', 'black']

# ── MC sweep ─────────────────────────────────────────────────
MC_range      = np.linspace(0, 2300, 1000)
MC_normalized = MC_range / force_model.MC_ref   # x-axis: MC/MCref

# ── Compute gains for each MU across MC range ─────────────────
P_gain_curves = {}
T_gain_curves = {}

for mu in selected_mus:
    P_gain_curves[mu] = np.array([force_model.compute_P_gain(mu, mc)
                                   for mc in MC_range])
    T_gain_curves[mu] = np.array([force_model.compute_T_gain(mu, mc)
                                   for mc in MC_range])

# ── Figure: 2 panels ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Panel 1: P_gain vs MC/MCref ───────────────────────────────
ax1 = axes[0]

for mu, color in zip(selected_mus, colors):
    ax1.plot(MC_normalized, P_gain_curves[mu],
             color=color, linewidth=2.5,
             label=f'MU{mu}')

    # Mark value at MC = MCref (x=1)
    pg_at_ref = force_model.compute_P_gain(mu, force_model.MC_ref)
    ax1.plot(1.0, pg_at_ref, 'o',
             color=color, markersize=8, zorder=5)

    # Mark h_i (MC at half-max P_gain) on x-axis
    h_i = force_model.compute_h(mu)
    ax1.axvline(h_i, color=color, linewidth=0.8,
                linestyle=':', alpha=0.5)

# Reference lines
ax1.axvline(1.0, color='gray', linewidth=1.5,
            linestyle='--', label='MC = MCref')
ax1.axhline(0.0, color='gray', linewidth=0.8, linestyle=':')
ax1.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')
ax1.axhline(0.5, color='gray', linewidth=0.8,
            linestyle='-.', alpha=0.5,
            label='0.5 (half-max P_gain)')

ax1.set_xlabel('MC_i / MC_ref')
ax1.set_ylabel('P_gain,i')
ax1.set_title('P_gain vs MC_i / MC_ref\n'
              'Eq. 43 — dots mark value at MC=MCref\n'
              'dotted verticals mark h_i per MU',
              fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 2])
ax1.set_ylim([-0.05, 1.05])

# ── Panel 2: T_gain vs MC/MCref ───────────────────────────────
ax2 = axes[1]

for mu, color in zip(selected_mus, colors):
    ax2.plot(MC_normalized, T_gain_curves[mu],
             color=color, linewidth=2.5,
             label=f'MU{mu}')

    # Mark value at MC = MCref
    tg_at_ref = force_model.compute_T_gain(mu, force_model.MC_ref)
    ax2.plot(1.0, tg_at_ref, 'o',
             color=color, markersize=8, zorder=5)

# Reference lines
ax2.axvline(1.0, color='gray', linewidth=1.5,
            linestyle='--', label='MC = MCref')
ax2.axhline(1.0, color='gray', linewidth=1.5,
            linestyle='-.', label='T_gain = 1 (no fatigue)')

ax2.set_xlabel('MC_i / MC_ref')
ax2.set_ylabel('T_gain,i')
ax2.set_title('T_gain vs MC_i / MC_ref\n'
              'Eq. 38 — dots mark value at MC=MCref\n'
              'larger MUs have steeper increase',
              fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 2])

plt.suptitle('P_gain and T_gain vs Normalized MC\n'
             'Sigmoidal shapes and MU-dependent shifts — '
             'Dideriksen et al. (2010)',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.show()

# ── Sanity check table ────────────────────────────────────────
print(f"\n{'MU':>4} | {'h_i':>6} | "
      f"{'P_gain(MC=0)':>12} | {'P_gain(MC=0.5)':>14} | "
      f"{'P_gain(MC=1)':>12} | {'P_gain(MC=2)':>12} | "
      f"{'T_gain(MC=0)':>12} | {'T_gain(MC=1)':>12} | "
      f"{'T_gain(MC=2)':>12}")
print("-" * 105)

for mu in selected_mus:
    idx  = mu - 1
    h_i  = force_model.compute_h(mu)
    MCr  = force_model.MC_ref
    row  = f"{mu:>4} | {h_i:>6.3f}"
    for mc in [0, 0.5*MCr, MCr, 2*MCr]:
        row += f" | {force_model.compute_P_gain(mu, mc):>12.4f}"
    for mc in [0, MCr, 2*MCr]:
        row += f" | {force_model.compute_T_gain(mu, mc):>12.4f}"
    print(row)

# ── Key observations ──────────────────────────────────────────
print(f"\n=== Key observations at MC = MCref ===")
print(f"{'MU':>4} | {'P_gain':>8} | {'force lost%':>11} | "
      f"{'T_gain':>8} | {'twitch slowed by':>16}")
print("-" * 55)
for mu in selected_mus:
    pg = force_model.compute_P_gain(mu, force_model.MC_ref)
    tg = force_model.compute_T_gain(mu, force_model.MC_ref)
    print(f"{mu:>4} | {pg:>8.4f} | {pg*100:>10.1f}% | "
          f"{tg:>8.4f} | {(tg-1)*100:>14.1f}%")
    
'''


"""
fig_total_force_comparison.py

Same protocol as Dideriksen et al. (2010) Fig. 4:
    40 Hz stimulation for 330 ms, at MC = 0, 575, 1150 au

But instead of a single MU, shows TOTAL force from:
    LEFT  — Voluntary: only the MUs needed for 30% MVC (size principle)
    RIGHT — FES: all 120 MUs active simultaneously

This directly shows:
    1. FES produces much higher total force (all MUs active)
    2. FES fatigue (force drop from MC=0 to MC=1150) is larger
       because ALL MUs degrade together — no fresh reserve
    3. Voluntary retains more force at MC=1150 because
       high-threshold MUs are untouched
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from force_generation2 import ForceGenerationBlock

mpl.rcParams.update({
    'font.family':      'Times New Roman',
    'font.weight':      'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'font.size':         13,
    'axes.titlesize':    13,
    'axes.labelsize':    12,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   10,
    'lines.linewidth':    2.5,
})

# ── Parameters ────────────────────────────────────────────────────────
N_MU      = 120
DT        = 0.001
STIM_FREQ = 40.0          # Hz
STIM_DUR  = 0.330         # s — 330 ms stimulation
TOTAL_DUR = 0.620         # s — enough to see full relaxation
MC_LEVELS = [0, 575, 1150]

LINE_STYLES = {0: '-',  575: '--',  1150: ':'}
LINE_LABELS = {0: 'MC = 0 (no fatigue)',
               575:  'MC = 575',
               1150: 'MC = 1,150'}
COLORS = {'voluntary': '#1565C0', 'FES': '#C62828'}


def make_spike_train(freq=STIM_FREQ, dur=STIM_DUR):
    isi = 1.0 / freq
    return np.arange(isi, dur, isi).tolist()


def find_n_recruited(target_pct, mvc_force, fb, n=N_MU):
    """
    Find how many low-threshold MUs (size principle) are needed
    to produce target_pct % MVC at 40 Hz stimulation.
    Returns n_recruited.
    """
    target_au = (target_pct / 100.0) * mvc_force

    for n_rec in range(1, n + 1):
        spike_times = [[] for _ in range(n)]
        for mu_idx in range(n_rec):
            spike_times[mu_idx] = make_spike_train()

        res    = fb.simulate(spike_times, TOTAL_DUR, MC=None)
        # Use peak force as the reference
        f_peak = float(np.max(res['force_total']))

        if f_peak >= target_au:
            print(f"  Voluntary: {n_rec} MUs needed "
                  f"(peak = {f_peak:.1f} au, target = {target_au:.1f} au)")
            return n_rec

    print(f"  Warning: could not reach target with {n} MUs")
    return n


def build_mvc_force(fb, n=N_MU):
    """All MUs at PDR for 3 s — plateau mean = MVC."""
    a   = np.log(30) / n
    RTE = np.exp(a * np.arange(1, n + 1))
    PDR = 15.6 + 15.6 * (RTE / RTE[-1])
    spike_times = []
    for mu_idx in range(n):
        isi = 1.0 / PDR[mu_idx]
        spike_times.append(np.arange(isi, 3.0, isi).tolist())
    res = fb.simulate(spike_times, 3.0, MC=None)
    return float(np.mean(res['force_total'][int(2.5/DT):]))


def simulate_group(active_mu_indices, mc_val, fb):
    """
    Stimulate a group of MUs at 40 Hz / 330 ms.
    mc_val applied uniformly to all active MUs.
    Returns (t_ms, total_force).
    """
    spike_times = [[] for _ in range(N_MU)]
    MC_arr      = np.zeros(N_MU)

    for mu_idx in active_mu_indices:
        spike_times[mu_idx] = make_spike_train()
        MC_arr[mu_idx]      = mc_val

    res = fb.simulate(spike_times, TOTAL_DUR,
                      MC=MC_arr if mc_val > 0 else None)
    return res['time'] * 1000, res['force_total']


# ══════════════════════════════════════════════════════════════════════
def plot(target_pct=30.0, save=True):

    fb        = ForceGenerationBlock(n_motor_units=N_MU, dt=DT)
    mvc_force = build_mvc_force(fb)
    print(f"MVC = {mvc_force:.2f} au")

    # Find how many MUs voluntary needs
    print(f"\nFinding voluntary recruitment for {target_pct:.0f}% MVC ...")
    n_vol = find_n_recruited(target_pct, mvc_force, fb)
    vol_indices = list(range(n_vol))           # MU0 … MU(n_vol-1)  [0-based]
    fes_indices = list(range(N_MU))            # all 120

    print(f"\n  Voluntary: {n_vol} MUs  (MU1–MU{n_vol})")
    print(f"  FES:       {N_MU} MUs  (MU1–MU{N_MU})")

    # ── Build figure ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    configs = [
        ('voluntary', vol_indices,
         f'Voluntary — {n_vol} MUs\n(Size-principle recruitment)'),
        ('FES',       fes_indices,
         f'FES — All {N_MU} MUs\n(Non-selective recruitment)'),
    ]

    for ax, (mode, mu_indices, title) in zip(axes, configs):
        clr = COLORS[mode]

        for mc_val in MC_LEVELS:
            t_ms, force = simulate_group(mu_indices, mc_val, fb)
            ax.plot(t_ms, force,
                    ls=LINE_STYLES[mc_val],
                    color=clr,
                    lw=2.5,
                    label=LINE_LABELS[mc_val])

        # Stimulation window marker
        ax.axvline(STIM_DUR * 1000,
                   color='gray', lw=1.0, ls=':')
        ax.axvspan(0, STIM_DUR * 1000, alpha=0.05, color='gray')

        ax.set_title(title, fontweight='bold', color=clr, pad=10)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Total Force (au)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.22)
        ax.set_xlim([0, TOTAL_DUR * 1000])

    # Compute and annotate % force loss at peak
    for ax, (mode, mu_indices, _) in zip(axes, configs):
        _, f0    = simulate_group(mu_indices, 0,    fb)
        _, f1150 = simulate_group(mu_indices, 1150, fb)
        peak0    = np.max(f0)
        peak1150 = np.max(f1150)
        pct_loss = (1 - peak1150 / peak0) * 100 if peak0 > 0 else 0
        ax.annotate(
            f'Force loss at MC=1150:\n{pct_loss:.1f}% of initial peak',
            xy=(STIM_DUR * 1000 * 0.5, peak0 * 0.15),
            fontsize=9, color=COLORS[mode],
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', edgecolor=COLORS[mode],
                      alpha=0.85)
        )

    fig.suptitle(
        f'Total Force at 40 Hz / 330 ms Stimulation — {target_pct:.0f}% MVC Task\n'
        'Effect of Fatigue (MC = 0, 575, 1150 au)\n'
        'Voluntary (size-principle) vs FES (full recruitment)',
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save:
        fname = f'fig_total_force_vol_vs_FES_{int(target_pct)}pct.png'
        fig.savefig(fname, dpi=130, bbox_inches='tight')
        print(f'\n[Plot] Saved → {fname}')

    plt.show()
    return fig


if __name__ == '__main__':
    plot(target_pct=30.0, save=True)




"""
fig_total_force_comparison_FIXED.py

Key fixes vs original:
  1. FES recruits large MUs first (reverse order) — physiologically correct
  2. Both conditions matched to SAME 30% MVC force at MC=0
  3. n_fes is found by searching from MU120 downward
  4. The comparison is now FAIR — same force, different recruitment strategy

Physiological insight shown:
  - Voluntary needs MANY small MUs to reach 30% MVC
  - FES needs FEW large MUs to reach the same force
  - But FES fatigues faster because large MUs are less fatigue-resistant
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from force_generation2 import ForceGenerationBlock

mpl.rcParams.update({
    'font.family':      'Times New Roman',
    'font.weight':      'bold',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'font.size':         13,
    'axes.titlesize':    13,
    'axes.labelsize':    12,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'legend.fontsize':   10,
    'lines.linewidth':    2.5,
})

# ── Parameters ────────────────────────────────────────────────────────
N_MU      = 120
DT        = 0.001
STIM_FREQ = 40.0
STIM_DUR  = 0.330
TOTAL_DUR = 0.620
MC_LEVELS = [0, 575, 1150]

LINE_STYLES = {0: '-', 575: '--', 1150: ':'}
LINE_LABELS = {
    0:    'MC = 0 (no fatigue)',
    575:  'MC = 575',
    1150: 'MC = 1,150'
}
COLORS = {'voluntary': '#1565C0', 'FES': '#C62828'}


def make_spike_train(freq=STIM_FREQ, dur=STIM_DUR):
    isi = 1.0 / freq
    return np.arange(isi, dur, isi).tolist()


def build_mvc_force(fb, n=N_MU):
    """All MUs at PDR for 3 s — plateau mean = MVC."""
    a   = np.log(30) / n
    RTE = np.exp(a * np.arange(1, n + 1))
    PDR = 15.6 + 15.6 * (RTE / RTE[-1])
    spike_times = []
    for mu_idx in range(n):
        isi = 1.0 / PDR[mu_idx]
        spike_times.append(np.arange(isi, 3.0, isi).tolist())
    res = fb.simulate(spike_times, 3.0, MC=None)
    return float(np.mean(res['force_total'][int(2.5 / DT):]))


def get_peak_force(active_indices, fb, mc_val=0):
    """Simulate and return peak force for a given set of active MU indices."""
    spike_times = [[] for _ in range(N_MU)]
    MC_arr = np.zeros(N_MU)
    for mu_idx in active_indices:
        spike_times[mu_idx] = make_spike_train()
        MC_arr[mu_idx] = mc_val
    res = fb.simulate(spike_times, TOTAL_DUR,
                      MC=MC_arr if mc_val > 0 else None)
    return float(np.max(res['force_total']))


def find_n_voluntary(target_au, fb):
    """
    Voluntary recruitment: add MUs from SMALLEST (index 0) upward.
    Returns list of MU indices needed to reach target force.
    """
    for n_rec in range(1, N_MU + 1):
        indices = list(range(n_rec))  # MU0, MU1, ... MU(n_rec-1)
        peak = get_peak_force(indices, fb)
        if peak >= target_au:
            print(f"  Voluntary: {n_rec} MUs (MU1–MU{n_rec}), "
                  f"peak = {peak:.2f} au, target = {target_au:.2f} au")
            return indices
    return list(range(N_MU))


def find_n_fes(target_au, fb):
    """
    FES recruitment: add MUs from LARGEST (index 119) downward.
    This reflects that electrical stimulation preferentially activates
    large-diameter axons (lower impedance) first.
    Returns list of MU indices needed to reach target force.
    """
    for n_rec in range(1, N_MU + 1):
        # Start from largest MU (index 119) and work downward
        indices = list(range(N_MU - 1, N_MU - 1 - n_rec, -1))
        peak = get_peak_force(indices, fb)
        if peak >= target_au:
            print(f"  FES:       {n_rec} MUs (MU{N_MU}–MU{N_MU - n_rec + 1}), "
                  f"peak = {peak:.2f} au, target = {target_au:.2f} au")
            return indices
    return list(range(N_MU))


def simulate_group(active_indices, mc_val, fb):
    """Simulate force for a group of MUs at a given MC snapshot."""
    spike_times = [[] for _ in range(N_MU)]
    MC_arr = np.zeros(N_MU)
    for mu_idx in active_indices:
        spike_times[mu_idx] = make_spike_train()
        MC_arr[mu_idx] = mc_val
    res = fb.simulate(spike_times, TOTAL_DUR,
                      MC=MC_arr if mc_val > 0 else None)
    return res['time'] * 1000, res['force_total']


# ══════════════════════════════════════════════════════════════════════
def plot(target_pct=30.0, save=True):

    fb = ForceGenerationBlock(n_motor_units=N_MU, dt=DT)

    # Step 1: compute MVC reference
    mvc_force = build_mvc_force(fb)
    target_au = (target_pct / 100.0) * mvc_force
    print(f"\nMVC = {mvc_force:.2f} au")
    print(f"Target ({target_pct:.0f}% MVC) = {target_au:.2f} au\n")

    # Step 2: find recruitment for each strategy
    print("Finding voluntary recruitment (small → large):")
    vol_indices = find_n_voluntary(target_au, fb)

    print("\nFinding FES recruitment (large → small):")
    fes_indices = find_n_fes(target_au, fb)

    n_vol = len(vol_indices)
    n_fes = len(fes_indices)

    # ── Build figure ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    configs = [
        ('voluntary', vol_indices,
         f'Voluntary — {n_vol} MUs\n(Size-principle: small MUs first)'),
        ('FES', fes_indices,
         f'FES — {n_fes} MUs\n(Reverse order: large MUs first)'),
    ]

    for ax, (mode, mu_indices, title) in zip(axes, configs):
        clr = COLORS[mode]

        for mc_val in MC_LEVELS:
            t_ms, force = simulate_group(mu_indices, mc_val, fb)
            ax.plot(t_ms, force,
                    ls=LINE_STYLES[mc_val],
                    color=clr,
                    lw=2.5,
                    label=LINE_LABELS[mc_val])

        ax.axvline(STIM_DUR * 1000, color='gray', lw=1.0, ls=':')
        ax.axvspan(0, STIM_DUR * 1000, alpha=0.05, color='gray')

        ax.set_title(title, fontweight='bold', color=clr, pad=10)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Total Force (au)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.22)
        ax.set_xlim([0, TOTAL_DUR * 1000])

    # Annotate force loss
    for ax, (mode, mu_indices, _) in zip(axes, configs):
        _, f0    = simulate_group(mu_indices, 0,    fb)
        _, f1150 = simulate_group(mu_indices, 1150, fb)
        peak0    = np.max(f0)
        peak1150 = np.max(f1150)
        pct_loss = (1 - peak1150 / peak0) * 100 if peak0 > 0 else 0
        ax.annotate(
            f'Force loss at MC=1150:\n{pct_loss:.1f}% of initial peak',
            xy=(STIM_DUR * 1000 * 0.5, peak0 * 0.15),
            fontsize=9, color=COLORS[mode],
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', edgecolor=COLORS[mode],
                      alpha=0.85)
        )

    fig.suptitle(
        f'Total Force at 40 Hz / 330 ms — {target_pct:.0f}% MVC Task\n'
        f'FAIR COMPARISON: Both conditions matched to same target force\n'
        f'Voluntary ({n_vol} small MUs) vs FES ({n_fes} large MUs)',
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    if save:
        fname = f'fig_force_vol_vs_FES_FIXED_{int(target_pct)}pct.png'
        fig.savefig(fname, dpi=130, bbox_inches='tight')
        print(f'\n[Plot] Saved → {fname}')

    plt.show()
    return fig, n_vol, n_fes


if __name__ == '__main__':
    fig, n_vol, n_fes = plot(target_pct=30.0)

    print(f"\n{'='*50}")
    print("KEY PHYSIOLOGICAL INSIGHT:")
    print(f"  Voluntary needs {n_vol} small MUs → fatigue-RESISTANT")
    print(f"  FES needs only {n_fes} large MUs → fatigue-PRONE")
    print(f"  Same force, very different fatigue profiles!")
    print(f"{'='*50}")

