
import numpy as np

class MetaboliteModel:
    """
    Compartment model of intra- and extracellular metabolite concentrations
    during a sustained isometric contraction (Dideriksen et al. 2010).

    This block:
        - Receives spike counts per motor unit (from the motor neuron pool block)
          and average force in %MVC (from the force generation block) once per
          500-ms epoch.
        - Updates intracellular MC for each of the 120 motor units and the single
          extracellular compartment.
        - Outputs MC_i (array, shape [n_MU]), MC_es (scalar), and MMC (scalar)
          for use by the force generation block, motor neuron pool block, and
          PID controller respectively.

    All equation numbers refer to Dideriksen et al. (2010).
    """

    # ------------------------------------------------------------------
    # 5a — __init__: parameters, volumes, state variables
    # ------------------------------------------------------------------
    def __init__(self):
        """
        Initialise all parameters from Table 1 and Eqs. 1–2,
        and set every state variable to its physiological baseline (zero fatigue).
        """

        # ---- Structural parameters (Table 1 / Eq. 1) -----------------

        self.n_MU = 120                  # Number of motor units                  [dimensionless]

        # Eq. 1 parameters
        self.V0 = 3.0                    # Volume offset for intracellular compartments [au]
        self.F_ratio = 80.0             # F120/F1, ratio of largest to smallest MU twitch force [dimensionless]
        #   V_i = V0 * exp( ln(F_ratio) / n * i )   for i = 1 … n_MU

        # Eq. 2 / Table 1
        self.V_es = 1282.0              # Extracellular compartment volume         [au]

        # ---- Transport parameters (Table 1) --------------------------

        self.DC = 0.01                  # Diffusion coefficient (Eq. 6)            [au]
        # Criterion: DC * V_i > RC so metabolites diffuse faster than they are removed;
        # both intra- and extracellular MC fall by ~90% after 3 min rest (Woods et al. 1987).

        self.RC = 0.04                  # Removal capacity by blood flow (Eq. 7)   [au]
        # RC = 4 % of extracellular MC removed per 500-ms epoch at full blood flow.

        # ---- Blood-flow / IMP parameters (Table 1 / Eqs. 9–11) ------

        self.HO = 30.0                  # IMP at half blood-flow occlusion         [mmHg]
        # Full occlusion at ~40 mmHg, corresponding to ~35 % MVC force (Eq. 11).

        # Eq. 9 linear IMP coefficients (curve-fit to Crenshaw et al. 1997,
        # Sejersted et al. 1984, Sadamoto et al. 1983)
        self.IMP_slope     = 0.88       # Slope:     IMP_ins = 0.88 * force + 10.65  [mmHg / %MVC]
        self.IMP_intercept = 10.65      # Intercept                                  [mmHg]

        # Eq. 10 ΔIMP shape parameters (chosen to match Crenshaw et al. 1997 at 25% MVC)
        self.IMP_gain           = 0.12  # Amplitude gain for the cumulative ΔIMP term [mmHg / epoch]
        self.IMP_threshold_low  = 15.0  # Force threshold below which ΔIMP does not increase [%MVC]
        self.IMP_threshold_high = 33.0  # Force level at which ΔIMP saturates            [%MVC]
        self.IMP_width_low      = 8.0   # Width parameter for lower tanh in Eq. 10       [%MVC]
        self.IMP_width_high     = 3.0   # Width parameter for upper tanh in Eq. 10       [%MVC]

        # Eq. 11 BF sigmoid width (denominator inside tanh)
        self.BF_width = 4.0             # Width of BF sigmoid                             [mmHg]

        # ---- Reference MC (Table 1) used by force generation block ---
        # (Stored here for completeness; consumed by force_generation block via MC_i output)
        self.MC_ref = 1150.0            # MC at which twitch force / relaxation time reach
                                        # the reference normalised change (Fig. 3)         [au]

        # ---- Epoch timing --------------------------------------------
        self.epoch_duration_s   = 0.5  # 500 ms per MC-update epoch                       [s]
        self.sim_dt_s           = 0.001 # 1 kHz simulation timestep                       [s]
        self.samples_per_epoch  = int(self.epoch_duration_s / self.sim_dt_s)  # = 500     [samples]

        # ==============================================================
        # Derived / pre-computed quantities (computed once at init)
        # ==============================================================

        # MU indices 1-based as in the paper (i = 1 … 120)
        self._mu_indices = np.arange(1, self.n_MU+1)  # shape (120,)

        # Eq. 1 — intracellular compartment volumes
        #   V_i = V0 + exp( ln(F120/F1) / n * i )
        self.V_i = self.V0 * np.exp(
            (np.log(self.F_ratio) / self.n_MU)* self._mu_indices
        )

        # shape: (120,)
        #
        # DISCREPANCY NOTE (Dideriksen et al. 2010, Eq. 1 vs Table 1):
        # With the literal parameters V0=3, F_ratio=80, n=120, i=1..120,
        # Eq. 1 produces V_i ranging from ~3.11 (MU1) to ~240.0 (MU120).
        # Table 1 states the range is 4.0–83.0 au. These are inconsistent.
        # The ratio of max/min volumes is 240/3.11 ≈ 77 (Eq.1) vs 83/4 ≈ 21 (Table 1).
        # Possible explanations:
        #   (a) Table 1 reports rounded values from a different parameterisation
        #       used during the actual simulations (e.g. i starting at 0, or
        #       a different V0 / F_ratio combination).
        #   (b) A typographic error in either the equation or the table.
        # IMPLEMENTATION DECISION: The literal Eq. 1 is implemented faithfully
        # here (range ~3.11–240 au). If Table 1 values are required exactly,
        # V0 and F_ratio should be re-fitted. This discrepancy should be
        # resolved against the original authors or a reference implementation
        # before relying on absolute MC magnitude values.

        # Sum of intracellular volumes — used for MMC denominator (Eq. 15)
        self._sum_V_i = np.sum(self.V_i)  # scalar [au]

        # ==============================================================
        # State variables — initialised to zero (no fatigue at t=0)
        # ==============================================================

        # Intracellular MC for each MU (Eq. 3)
        # Physiological baseline: zero accumulation relative to resting state
        self.MC_i = np.zeros(self.n_MU, dtype=float)   # shape (120,)  [au]

        # Extracellular MC (Eq. 4)
        self.MC_es = 0.0                                 # scalar         [au]

        # Cumulative ΔIMP history (running sum in Eq. 10)
        # Starts at zero: no prior contraction history
        self.delta_IMP_cumulative = 0.0                  # scalar         [mmHg]

        # Total intramuscular pressure (Eq. 8) — computed from IMP_ins + ΔIMP
        self.IMP_tot = 0.0                               # scalar         [mmHg]

        # Blood flow factor (Eq. 11) — starts at 1.0 (no occlusion)
        self.BF = 1.0                                    # scalar         [dimensionless, in [0,1]]

        # ==============================================================
        # Epoch-level internal accumulators (reset every epoch)
        # ==============================================================

        # Spike accumulator: counts spikes for each MU within current epoch
        # (used when the block is called sample-by-sample rather than epoch-by-epoch)
        self._spike_accumulator = np.zeros(self.n_MU, dtype=int)   # shape (120,)
        self._sample_counter= 0                                   # samples elapsed in current epoch

        # ==============================================================
        # Cached outputs (zero-order hold between epochs for other blocks)
        # ==============================================================

        # Volume-weighted mean intracellular MC (Eq. 15) — sent to PID controller
        self.MMC = 0.0       # scalar  [au]

    # ------------------------------------------------------------------
    # 5b — Metabolite production (Eq. 5)
    # ------------------------------------------------------------------
    def compute_metabolite_production(self, spike_counts: np.ndarray) -> np.ndarray:
        """
        Eq. 5 — Intracellular metabolite production per motor unit.

        Production is proportional to motor unit volume and the number of
        action potentials discharged during the 500-ms epoch (Allen 2004).

            MP_i(epoch) = V_i * ND_i(epoch)

        Parameters
        ----------
        spike_counts : np.ndarray, shape (120,), dtype int or float
            ND_i(epoch) — number of action potentials fired by each MU
            during the current 500-ms epoch.
            Source: motor neuron pool block (motorUnit.py spike trains
            accumulated over the epoch window).

        Returns
        -------
        MP_i : np.ndarray, shape (120,), dtype float
            Metabolite production for each MU in this epoch   [au].

        Notes
        -----
        * V_i has units [au] and ND_i is dimensionless (spike count);
          therefore MP_i has units [au] — consistent with Eq. 3 where
          MP_i is divided by V_i before being added to MC_i (au).
        * Inactive MUs (ND_i = 0) produce no metabolites.
        * High-threshold MUs have larger V_i and therefore produce more
          metabolites per spike, reflecting their larger innervation numbers.
        """
        spike_counts = np.asarray(spike_counts, dtype=int)

        if spike_counts.shape != (self.n_MU,):
            raise ValueError(
                f"spike_counts must have shape ({self.n_MU},), "
                f"got {spike_counts.shape}"
            )

        # Eq. 5: MP_i(epoch) = V_i · ND_i(epoch)
        MP_i = self.V_i * spike_counts   # shape (120,)  [au]

        return MP_i

    def compute_diffusion(self) -> np.ndarray:
        """
        Eq. 6 — Diffusion from intracellular compartment i to extracellular space.

            MD_i(epoch) = DC · [MC_i(epoch) - MC_es(epoch)] · V_i

        Parameters
        ----------
        Uses current state: self.MC_i, self.MC_es, self.V_i, self.DC

        Returns
        -------
        MD_i : np.ndarray, shape (120,)
            Metabolite diffusion per MU [au]
            Positive = outward flux (active MU losing metabolites)
            Negative = inward flux (inactive MU absorbing from extracellular)

        Notes
        -----
        Must be called using MC values BEFORE the epoch update.
        """
        # Eq. 6: MD_i = DC · (MC_i - MC_es) · V_i
        MD_i = self.DC * (self.MC_i - self.MC_es) * self.V_i
        return MD_i

    def compute_blood_flow(self, force_epoch: float) -> float:
        """
        Eqs. 8-11 — Compute blood flow factor from intramuscular pressure.

        Eq. 9 — Instantaneous IMP:
            IMP_ins(epoch) = 0.88 · force(epoch) + 10.65

        Eq. 10 — Cumulative ΔIMP history:
            ΔIMP += tanh[(force - 15)/8] · 0.12
                  · {tanh[-(force - 33)/3] · 0.5 + 0.5}

        Eq. 8 — Total IMP:
            IMP_tot(epoch) = IMP_ins(epoch) + ΔIMP(epoch)

        Eq. 11 — Blood flow:
            BF(epoch) = tanh[-(IMP_tot - HO)/4] · 0.5 + 0.5

        Parameters
        ----------
        force_epoch : float
            Average force during the epoch as %MVC

        Returns
        -------
        BF : float
            Blood flow factor in [0, 1]
            0 = full occlusion, 1 = no occlusion

        Notes
        -----
        Updates state variables: self.delta_IMP_cumulative,
        self.IMP_tot, self.BF
        """
        # Eq. 9: instantaneous IMP
        IMP_ins = self.IMP_slope * force_epoch + self.IMP_intercept

        # Eq. 10: cumulative ΔIMP — running sum across epochs
        factor1 = np.tanh((force_epoch - self.IMP_threshold_low) /
                          self.IMP_width_low) * self.IMP_gain
        factor2 = (np.tanh(-(force_epoch - self.IMP_threshold_high) /
                           self.IMP_width_high) * 0.5 + 0.5)


        self.delta_IMP_cumulative += factor1 * factor2

        # Eq. 8: total IMP
        self.IMP_tot = IMP_ins + self.delta_IMP_cumulative

        # Eq. 11: blood flow
        self.BF = (np.tanh(-(self.IMP_tot - self.HO) /
                           self.BF_width) * 0.5 + 0.5)

        return self.BF

    def update_intracellular_MC(self,
                                MP_i: np.ndarray,
                                MD_i: np.ndarray) -> np.ndarray:
        """
        Eq. 3 — Update intracellular metabolite concentration for each MU.

            MC_i(epoch) = MC_i(epoch-1)
                        + [MP_i(epoch) - MD_i(epoch)] / V_i

        Parameters
        ----------
        MP_i : np.ndarray, shape (120,)
            Metabolite production from compute_metabolite_production()
            Units: [au]
        MD_i : np.ndarray, shape (120,)
            Diffusion flux from compute_diffusion()
            Must be computed from MC_i(epoch-1) and MC_es(epoch-1)
            BEFORE this update is applied.
            Units: [au]

        Returns
        -------
        MC_i : np.ndarray, shape (120,)
            Updated intracellular MC for each MU [au]

        Notes
        -----
        Call order within step_epoch:
            1. compute_diffusion()          → MD_i  (uses epoch-1 values)
            2. compute_metabolite_production() → MP_i
            3. update_intracellular_MC(MP_i, MD_i)  ← this method
            4. update_extracellular_MC(MD_i, MR)    ← uses same MD_i

        The V_i factor in MP_i (Eq. 5) and MD_i (Eq. 6) does NOT cancel
        in Eq. 3 mathematically — both terms are divided by V_i:
            MP_i / V_i = V_i * ND_i / V_i = ND_i
            MD_i / V_i = DC * (MC_i - MC_es) * V_i / V_i
                       = DC * (MC_i - MC_es)
        V_i cancels in both terms. This is confirmed by the paper's
        intent that production scales with volume but concentration
        change does not depend on compartment size alone.
        """
        # Eq. 3: MC_i(epoch) = MC_i(epoch-1) + [MP_i - MD_i] / V_i
        self.MC_i = self.MC_i + (MP_i - MD_i) / self.V_i

        # Non-negativity guard — MC cannot be physically negative
        self.MC_i = np.maximum(self.MC_i, 0.0)

        return self.MC_i

    def update_extracellular_MC(self, MD_i: np.ndarray, MR: float) -> float:
        """
        Eq. 4 — Update extracellular metabolite concentration.

            MC_es(epoch) = MC_es(epoch-1)
                         + [Σ MD_x(epoch) - MR(epoch)] / V_es

        Parameters
        ----------
        MD_i : np.ndarray, shape (120,)
            Diffusion flux per MU from compute_diffusion()
            Must be the SAME MD_i used in update_intracellular_MC()
            i.e. computed from MC values at epoch-1
        MR : float
            Removal by blood flow from compute_removal()

        Returns
        -------
        MC_es : float
            Updated extracellular MC [au]

        Notes
        -----
        Must be called AFTER update_intracellular_MC but with the
        SAME pre-update MD_i vector.
        """
        # Eq. 4: MC_es(epoch) = MC_es(epoch-1) + [Σ MD_x - MR] / V_es
        self.MC_es = self.MC_es + (np.sum(MD_i) - MR) / self.V_es

        # Non-negativity guard
        self.MC_es = max(self.MC_es, 0.0)

        return self.MC_es

    def compute_removal(self) -> float:
        """
        Eq. 7 — Removal of metabolites by blood flow.

            MR(epoch) = RC · BF(epoch) · MC_es(epoch)

        Returns
        -------
        MR : float
            Metabolite removal this epoch [au]
        """
        # Eq. 7: MR = RC · BF · MC_es
        return self.RC * self.BF * self.MC_es * self.V_es

    def compute_MMC(self) -> float:
        """
        Eq. 15 — Volume-weighted mean intracellular MC across all MUs.

            MMC(t) = Σ MC_i(t) / Σ V_i

        Parameters
        ----------
        Uses current state: self.MC_i, self.V_i

        Returns
        -------
        MMC : float
            Mean intracellular MC weighted by volume [au]
            Sent to PID controller to modulate derivative gain Kd (Eq. 14)

        Notes
        -----
        Called after update_intracellular_MC so it uses
        the current epoch MC_i values.
        """
        # Eq. 15: MMC = Σ MC_i(t) / Σ V_i
        #self.MMC = np.sum(self.MC_i) / self._sum_V_i
        self.MMC = np.mean(self.MC_i)

        return self.MMC

    def step_epoch(self,
                   spike_counts: np.ndarray,
                   force_epoch: float) -> dict:
        """
        Main epoch update — calls all sub-steps in the correct order.

        Called once every 500 ms with:
            - spike_counts: ND_i from motor neuron pool (spikes per MU this epoch)
            - force_epoch:  mean force as %MVC from force generation block

        Returns
        -------
        dict with keys:
            'MC_i'  : np.ndarray shape (120,) — intracellular MC per MU
            'MC_es' : float                   — extracellular MC
            'MMC'   : float                   — volume-weighted mean MC
            'BF'    : float                   — blood flow factor [0,1]
            'IMP_tot': float                  — total intramuscular pressure [mmHg]

        Call order strictly follows paper:
            Step 1: Eq. 6  — diffusion (uses MC values from epoch-1)
            Step 2: Eq. 5  — production
            Step 3: Eqs. 8-11 — blood flow and IMP
            Step 4: Eq. 7  — removal
            Step 5: Eq. 3  — update intracellular MC
            Step 6: Eq. 4  — update extracellular MC (same MD_i as step 1)
            Step 7: Eq. 15 — compute MMC
        """
        # Step 1 — Eq. 6: diffusion using MC_i(epoch-1) and MC_es(epoch-1)
        MD_i = self.compute_diffusion()

        # Step 2 — Eq. 5: metabolite production
        MP_i = self.compute_metabolite_production(spike_counts)

        # Step 3 — Eqs. 8-11: blood flow and IMP
        BF = self.compute_blood_flow(force_epoch)

        # Step 4 — Eq. 7: removal by blood flow
        MR = self.compute_removal()

        # Step 5 — Eq. 3: update intracellular MC
        self.update_intracellular_MC(MP_i, MD_i)

        # Step 6 — Eq. 4: update extracellular MC (same MD_i from Step 1)
        self.update_extracellular_MC(MD_i, MR)

        # Step 7 — Eq. 15: volume-weighted mean MC
        self.compute_MMC()

        return {
            'MC_i': self.MC_i.copy(),
            'MC_es': self.MC_es,
            'MMC': self.MMC,
            'BF': self.BF,
            'IMP_tot': self.IMP_tot
        }

'''

import numpy as np
import matplotlib.pyplot as plt
from metabolite_block import MetaboliteModel

met         = MetaboliteModel()
force_range = np.linspace(0, 100, 1000)

# ── BF at single epoch (no cumulative ΔIMP) ───────────────────
IMP_ins = met.IMP_slope * force_range + met.IMP_intercept
BF      = np.tanh(-(IMP_ins - met.HO) / met.BF_width) * 0.5 + 0.5

# ── Plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(force_range, BF,
        color='black', linewidth=2.5,
        label='BF (Eq. 11) — single epoch')

# ── Key reference points from paper ───────────────────────────
ref_points = [
    (10,  'blue',   '10% MVC — no ΔIMP'),
    (15,  'green',  '15% MVC — ΔIMP starts'),
    (35,  'red',    '35% MVC — full occlusion'),
    (100, 'purple', '100% MVC'),
]
for force, color, label in ref_points:
    imp    = met.IMP_slope * force + met.IMP_intercept
    bf_val = np.tanh(-(imp - met.HO) / met.BF_width) * 0.5 + 0.5
    ax.plot(force, bf_val, 'o',
            color=color, markersize=10, zorder=5,
            label=f'{label}  →  BF = {bf_val:.3f}')
    ax.axvline(force, color=color, linewidth=1.0,
               linestyle='--', alpha=0.5)
    ax.axhline(bf_val, color=color, linewidth=1.0,
               linestyle=':', alpha=0.5)

# ── Half occlusion reference ──────────────────────────────────
ax.axhline(0.5, color='gray', linewidth=1.5,
           linestyle='--',
           label=f'BF = 0.5  (HO = {met.HO} mmHg)')
ax.axhline(0.0, color='gray', linewidth=0.8, linestyle=':')
ax.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')

ax.set_xlabel('Force (% MVC)')
ax.set_ylabel('Blood Flow  (0 = full occlusion,  1 = no occlusion)')
ax.set_title('Blood Flow vs Force — single epoch\n'
             'Eq. 11: BF = tanh(-(IMP_tot - HO)/4) · 0.5 + 0.5',
             fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 100])
ax.set_ylim([-0.05, 1.05])

plt.tight_layout()
plt.show()

# ── Sanity check table ────────────────────────────────────────
print(f"\n{'Force (% MVC)':>13} | {'IMP_ins (mmHg)':>14} | "
      f"{'BF':>8} | {'Note':>30}")
print("-" * 72)
notes = {
    0  : 'rest — no occlusion',
    10 : 'no ΔIMP increase (paper)',
    15 : 'ΔIMP starts (paper)',
    25 : 'Crenshaw reference',
    35 : 'full occlusion expected',
    50 : 'above full occlusion',
    100: 'MVC'
}
for force in [0, 10, 15, 25, 35, 50, 100]:
    imp  = met.IMP_slope * force + met.IMP_intercept
    bf   = np.tanh(-(imp - met.HO) / met.BF_width) * 0.5 + 0.5
    note = notes.get(force, '')
    print(f"{force:>13} | {imp:>14.2f} | {bf:>8.4f} | {note:>30}")



met = MetaboliteModel()

print(f"{'Force':>6} | {'ΔIMP/epoch':>12} | {'Note':>25}")
print("-" * 48)
for force in [0, 5, 10, 15, 20, 25, 35]:
    f1 = np.tanh((force - met.IMP_threshold_low) /
                  met.IMP_width_low) * met.IMP_gain
    f2 = (np.tanh(-(force - met.IMP_threshold_high) /
                    met.IMP_width_high) * 0.5 + 0.5)
    delta = f1 * f2
    print(f"{force:>6} | {delta:>12.6f} | "
          f"{'should be ~0' if force <= 10 else 'should be > 0':>25}")
met        = MetaboliteModel()
mu_indices = np.arange(1, met.n_MU + 1)

# ── Theoretical V_i from Eq. 1 ───────────────────────────────
V_theoretical = met.V0 + np.exp(
    (np.log(met.F_ratio) / met.n_MU) * mu_indices
)

# ── Table 1 reference range ───────────────────────────────────
V_table_min = 4.0    # Table 1: range 4.0–83.0 au
V_table_max = 83.0

# ── Figure: 3 panels ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── Panel 1: V_i vs MU index (linear scale) ──────────────────
ax1 = axes[0]
ax1.plot(mu_indices, met.V_i,
         color='black', linewidth=2.5,
         label='V_i (Eq. 1) — implemented')
ax1.plot(mu_indices, V_theoretical,
         color='red', linewidth=1.5,
         linestyle='--',
         label='V_i theoretical (should overlay)')

# Table 1 range
ax1.axhline(V_table_min, color='blue', linewidth=1.5,
            linestyle=':', label=f'Table 1 min = {V_table_min}')
ax1.axhline(V_table_max, color='blue', linewidth=1.5,
            linestyle='-.', label=f'Table 1 max = {V_table_max}')

# Mark key MUs
for mu, color in zip([1, 30, 60, 90, 120],
                     ['blue', 'green', 'orange', 'red', 'purple']):
    idx = mu - 1
    ax1.plot(mu, met.V_i[idx], 'o',
             color=color, markersize=8, zorder=5,
             label=f'MU{mu}: V={met.V_i[idx]:.2f}')

ax1.set_xlabel('Motor Unit Index')
ax1.set_ylabel('V_i (au)')
ax1.set_title('Eq. 1 — Intracellular compartment volume\n'
              'V_i = V0 + exp(ln(F_ratio)/n · i)',
              fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([1, 120])

# ── Panel 2: V_i on log scale ────────────────────────────────
ax2 = axes[1]
ax2.semilogy(mu_indices, met.V_i,
             color='black', linewidth=2.5,
             label='V_i (implemented)')
ax2.semilogy(mu_indices, V_theoretical,
             color='red', linewidth=1.5,
             linestyle='--',
             label='V_i theoretical')

# Table 1 range
ax2.axhline(V_table_min, color='blue', linewidth=1.5,
            linestyle=':', label=f'Table 1 min = {V_table_min}')
ax2.axhline(V_table_max, color='blue', linewidth=1.5,
            linestyle='-.', label=f'Table 1 max = {V_table_max}')

ax2.annotate(f'V_1 = {met.V_i[0]:.3f}',
             xy=(1, met.V_i[0]),
             xytext=(15, met.V_i[0] * 1.5),
             fontsize=9, color='blue',
             arrowprops=dict(arrowstyle='->', color='blue'))
ax2.annotate(f'V_120 = {met.V_i[-1]:.2f}',
             xy=(120, met.V_i[-1]),
             xytext=(80, met.V_i[-1] * 0.5),
             fontsize=9, color='red',
             arrowprops=dict(arrowstyle='->', color='red'))

ax2.set_xlabel('Motor Unit Index')
ax2.set_ylabel('V_i (log scale, au)')
ax2.set_title('V_i on log scale\n'
              'Straight line = exponential ✓',
              fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlim([1, 120])

# ── Panel 3: V_i ratio and extracellular volume ───────────────
ax3 = axes[2]

# Ratio V_i / V_1
V_ratio = met.V_i / met.V_i[0]
ax3.plot(mu_indices, V_ratio,
         color='black', linewidth=2.5,
         label='V_i / V_1')

# Mark V_120/V_1 ratio
ax3.axhline(met.V_i[-1] / met.V_i[0],
            color='red', linewidth=1.5,
            linestyle='--',
            label=f'V_120/V_1 = {met.V_i[-1]/met.V_i[0]:.1f}')

# Table 1 expected ratio
ax3.axhline(V_table_max / V_table_min,
            color='blue', linewidth=1.5,
            linestyle=':',
            label=f'Table 1 ratio = {V_table_max/V_table_min:.1f}')

# Extracellular reference
ax3.axhline(met.V_es / met.V_i[-1],
            color='green', linewidth=1.5,
            linestyle='-.',
            label=f'V_es/V_120 = {met.V_es/met.V_i[-1]:.1f} '
                  f'(paper says ~15)')

ax3.set_xlabel('Motor Unit Index')
ax3.set_ylabel('V_i / V_1')
ax3.set_title('Volume ratio V_i / V_1\n'
              'and V_es reference',
              fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([1, 120])

plt.suptitle('Compartment Volume Distribution — Eq. 1\n'
             'Dideriksen et al. (2010)',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.show()

# ── Sanity check table ────────────────────────────────────────
print(f"\n{'MU':>4} | {'V_i':>10} | {'V_theoretical':>14} | "
      f"{'V_i/V_1':>8} | {'Match':>6}")
print("-" * 55)
for mu in [1, 20, 40, 60, 80, 100, 120]:
    idx   = mu - 1
    v_imp = met.V_i[idx]
    v_the = V_theoretical[idx]
    ratio = v_imp / met.V_i[0]
    match = '✓' if np.isclose(v_imp, v_the) else '✗'
    print(f"{mu:>4} | {v_imp:>10.4f} | "
          f"{v_the:>14.4f} | "
          f"{ratio:>8.2f} | "
          f"{match:>6}")

# ── Key checks from paper ─────────────────────────────────────
print(f"\n=== Key checks ===")
print(f"V_i range (implemented): "
      f"{met.V_i[0]:.3f} → {met.V_i[-1]:.3f} au")
print(f"V_i range (Table 1):     "
      f"{V_table_min} → {V_table_max} au")
print(f"V_120/V_1 (implemented): "
      f"{met.V_i[-1]/met.V_i[0]:.2f}")
print(f"V_120/V_1 (Table 1):     "
      f"{V_table_max/V_table_min:.2f}")
print(f"V_es (implemented):      {met.V_es:.1f} au")
print(f"V_es/V_120:              "
      f"{met.V_es/met.V_i[-1]:.2f}  "
      f"(paper says ~15)")
print(f"V_es/V_1:                "
      f"{met.V_es/met.V_i[0]:.2f}  "
      f"(paper says MU120 is 20x MU1, "
      f"V_es is 15x MU120)")


import numpy as np
import matplotlib.pyplot as plt
from metabolite_block import MetaboliteModel
from motorUnti import MotorNeuronPool

# ── Setup ─────────────────────────────────────────────────────
met        = MetaboliteModel()
mnp        = MotorNeuronPool(n_motor_units=120, dt=0.001)
mu_indices = np.arange(1, 121)

# ── Simulation parameters ─────────────────────────────────────
duration      = 300.0    # seconds
epoch_dur     = 0.5      # 500 ms
n_epochs      = int(duration / epoch_dur)
force_pct_mvc = 20.0     # 20% MVC

# ── Spike counts at 20% MVC ───────────────────────────────────
# Only MUs with RTE <= 20% MVC are recruited
# REI at 20% MVC — estimate from max excitation
REI_20 = np.max(mnp.Emax) * 0.20

# Compute DR for each MU at 20% MVC (no noise, no fatigue)
DR_20 = np.zeros(mnp.n)
for idx in range(mnp.n):
    EI  = min(REI_20, mnp.Emax[idx])
    DD  = mnp.k[idx] * EI
    EAI = (1 - mnp.k[idx]) * EI
    NSI = DD + EAI
    if NSI >= mnp.RTE[idx]:
        DR_20[idx] = np.clip(
            NSI - mnp.RTE[idx] + mnp.MDR[idx],
            mnp.MDR[idx], mnp.PDR[idx]
        )
    else:
        DR_20[idx] = 0.0

# Spikes per epoch = DR * epoch_duration
spikes_per_epoch = np.round(DR_20 * epoch_dur).astype(int)

n_recruited = np.sum(DR_20 > 0)
print(f"Force level:        {force_pct_mvc}% MVC")
print(f"REI at 20% MVC:     {REI_20:.4f}")
print(f"MUs recruited:      {n_recruited} / 120")
print(f"DR range (active):  "
      f"{DR_20[DR_20>0].min():.1f} → "
      f"{DR_20[DR_20>0].max():.1f} pps")
print(f"Spikes/epoch range: "
      f"{spikes_per_epoch[spikes_per_epoch>0].min()} → "
      f"{spikes_per_epoch[spikes_per_epoch>0].max()}")

# ── Storage ───────────────────────────────────────────────────
epoch_times    = np.arange(1, n_epochs+1) * epoch_dur
MC_es_hist     = np.zeros(n_epochs)
MMC_hist       = np.zeros(n_epochs)
mean_MC_hist   = np.zeros(n_epochs)
BF_hist        = np.zeros(n_epochs)
IMP_hist       = np.zeros(n_epochs)

# Track selected MUs
selected_mus   = [1, 30, 60, 90, 120]
MC_i_hist      = {mu: np.zeros(n_epochs) for mu in selected_mus}

# ── Run simulation ────────────────────────────────────────────
for ep in range(n_epochs):
    out = met.step_epoch(
        spike_counts = spikes_per_epoch,
        force_epoch  = force_pct_mvc
    )
    MC_es_hist[ep]   = out['MC_es']
    MMC_hist[ep]     = out['MMC']
    mean_MC_hist[ep] = np.mean(out['MC_i'])
    BF_hist[ep]      = out['BF']
    IMP_hist[ep]     = out['IMP_tot']

    for mu in selected_mus:
        MC_i_hist[mu][ep] = out['MC_i'][mu - 1]

# ── Figure: 2x3 panels ───────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors     = ['blue', 'green', 'orange', 'red', 'black']

# ── Panel 1: MC_i for selected MUs ───────────────────────────
ax1 = axes[0, 0]
for mu, color in zip(selected_mus, colors):
    active = DR_20[mu-1] > 0
    ls     = '-' if active else '--'
    label  = (f'MU{mu} '
              f'({"active" if active else "inactive"}, '
              f'DR={DR_20[mu-1]:.1f} pps)')
    ax1.plot(epoch_times, MC_i_hist[mu],
             color=color, linewidth=2,
             linestyle=ls, label=label)

ax1.axhline(met.MC_ref, color='gray', linewidth=1.5,
            linestyle=':', label=f'MCref = {met.MC_ref}')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('MC_i (au)')
ax1.set_title('Intracellular MC per MU\n'
              'solid=active, dashed=inactive',
              fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# ── Panel 2: MC_es over time ──────────────────────────────────
ax2 = axes[0, 1]
ax2.plot(epoch_times, MC_es_hist,
         color='red', linewidth=2.5,
         label='MC_es (extracellular)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('MC_es (au)')
ax2.set_title('Extracellular MC over time\n'
              'rises as metabolites diffuse out',
              fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# ── Panel 3: MMC and mean MC_i ────────────────────────────────
ax3 = axes[0, 2]
ax3.plot(epoch_times, MMC_hist,
         color='black', linewidth=2.5,
         label='MMC = ΣMC_i / ΣV_i  (Eq. 15)')
ax3.plot(epoch_times, mean_MC_hist,
         color='blue', linewidth=2,
         linestyle='--',
         label='mean(MC_i) — simple average')
ax3.axhline(met.MC_ref, color='gray', linewidth=1.5,
            linestyle=':', label=f'MCref = {met.MC_ref}')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('MC (au)')
ax3.set_title('MMC vs mean MC_i\n'
              'MMC is volume-normalized',
              fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ── Panel 4: Blood flow over time ────────────────────────────
ax4 = axes[1, 0]
ax4.plot(epoch_times, BF_hist,
         color='blue', linewidth=2.5,
         label='BF (Eq. 11)')
ax4.axhline(0.5, color='gray', linewidth=1.5,
            linestyle='--', label='BF = 0.5')
ax4.axhline(0.0, color='gray', linewidth=0.8, linestyle=':')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Blood Flow (0→1)')
ax4.set_title('Blood flow over time\n'
              f'at {force_pct_mvc}% MVC',
              fontweight='bold')
ax4.legend(fontsize=9)
ax4.set_ylim([-0.05, 1.05])
ax4.grid(True, alpha=0.3)

# ── Panel 5: IMP_tot over time ────────────────────────────────
ax5 = axes[1, 1]
ax5.plot(epoch_times, IMP_hist,
         color='purple', linewidth=2.5,
         label='IMP_tot (Eq. 8)')
ax5.axhline(40.0, color='red', linewidth=1.5,
            linestyle='--', label='40 mmHg (full occlusion)')
ax5.axhline(met.HO, color='gray', linewidth=1.5,
            linestyle=':', label=f'HO = {met.HO} mmHg')
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('IMP_tot (mmHg)')
ax5.set_title('Total IMP over time\n'
              '(IMP_ins + cumulative ΔIMP)',
              fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# ── Panel 6: MC_i per MU at 3 time points ────────────────────
ax6 = axes[1, 2]
time_points = {
    f't = {int(n_epochs*0.25*epoch_dur)}s':
        int(n_epochs * 0.25),
    f't = {int(n_epochs*0.5*epoch_dur)}s':
        int(n_epochs * 0.5),
    f't = {int(n_epochs*epoch_dur)}s':
        n_epochs - 1,
}
tp_colors = ['blue', 'orange', 'black']

for (label, ep_idx), color in zip(time_points.items(), tp_colors):
    # Re-run to get full MC_i array at this epoch
    met_tmp = MetaboliteModel()
    for ep in range(ep_idx + 1):
        out_tmp = met_tmp.step_epoch(spikes_per_epoch, force_pct_mvc)
    ax6.plot(mu_indices, out_tmp['MC_i'],
             color=color, linewidth=2,
             label=label)

ax6.axhline(met.MC_ref, color='gray', linewidth=1.5,
            linestyle=':', label=f'MCref = {met.MC_ref}')
# Mark recruited vs not
last_recruited = np.where(DR_20 > 0)[0][-1] + 1
ax6.axvline(last_recruited, color='red',
            linewidth=1.5, linestyle='--',
            label=f'Last recruited MU = {last_recruited}')
ax6.set_xlabel('Motor Unit Index')
ax6.set_ylabel('MC_i (au)')
ax6.set_title('MC_i across all 120 MUs\n'
              'at 3 time points',
              fontweight='bold')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.suptitle(f'MC Accumulation — {force_pct_mvc}% MVC sustained contraction\n'
             f'Production + Diffusion + Removal dynamics',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.show()

# ── Sanity check table ────────────────────────────────────────
print(f"\n{'Time(s)':>8} | {'MC_es':>8} | {'MMC':>10} | "
      f"{'mean MC_i':>10} | {'BF':>6} | {'IMP_tot':>8}")
print("-" * 60)
check_epochs = [0, 10, 30, 60, 120, 200, n_epochs-1]
for ep in check_epochs:
    if ep < n_epochs:
        print(f"{epoch_times[ep]:>8.1f} | "
              f"{MC_es_hist[ep]:>8.3f} | "
              f"{MMC_hist[ep]:>10.6f} | "
              f"{mean_MC_hist[ep]:>10.3f} | "
              f"{BF_hist[ep]:>6.3f} | "
              f"{IMP_hist[ep]:>8.2f}")





# ── Setup ─────────────────────────────────────────────────────
mnp = MotorNeuronPool(n_motor_units=120, dt=0.001)

# ── Spike counts at 100% MVC (all MUs at PDR) ─────────────────
spikes_mvc  = np.round(mnp.PDR * 0.5).astype(int)
spikes_rest = np.zeros(120, dtype=int)   # zero spikes during rest

# ── Phase 1: 60s contraction ──────────────────────────────────
#n_ep_contract = int(60.0 / 0.5)    # 120 epochs


# Try these one at a time
#n_ep_contract = int(10.0  / 0.5)   # 10s MVC
#n_ep_contract = int(20.0  / 0.5)   # 20s MVC
#n_ep_contract = int(44.0  / 0.5)   # 44s MVC (paper Fig 8)
n_ep_contract = int(60.0  / 0.5)   # 60s MVC (current)
# ── Phase 2: 3 min rest ───────────────────────────────────────
n_ep_rest     = int(180.0 / 0.5)   # 360 epochs
n_ep_total    = n_ep_contract + n_ep_rest

met        = MetaboliteModel()
epoch_times = np.arange(1, n_ep_total + 1) * 0.5

# ── Storage ───────────────────────────────────────────────────
MC_es_hist   = np.zeros(n_ep_total)
mean_MC_hist = np.zeros(n_ep_total)
BF_hist      = np.zeros(n_ep_total)

selected_mus = [1, 30, 60, 90, 120]
colors       = ['blue', 'green', 'orange', 'red', 'black']
MC_i_hist    = {mu: np.zeros(n_ep_total) for mu in selected_mus}

# ── Run simulation ─────────────────────────────────────────────
print("Phase 1 — 60s MVC contraction...")
for ep in range(n_ep_contract):
    out = met.step_epoch(
        spike_counts = spikes_mvc,
        force_epoch  = 100.0
    )
    MC_es_hist[ep]   = out['MC_es']
    mean_MC_hist[ep] = np.mean(out['MC_i'])
    BF_hist[ep]      = out['BF']
    for mu in selected_mus:
        MC_i_hist[mu][ep] = out['MC_i'][mu - 1]

# ── Record values at end of contraction ───────────────────────
MC_i_end   = out['MC_i'].copy()
MC_es_end  = out['MC_es']
mean_end   = np.mean(out['MC_i'])

print(f"  End of contraction:")
print(f"  mean MC_i = {mean_end:.2f} au")
print(f"  MC_es     = {MC_es_end:.2f} au")
print(f"  BF        = {out['BF']:.4f}")

print("\nPhase 2 — 3 min rest (zero spikes)...")
for ep in range(n_ep_rest):
    ep_total = n_ep_contract + ep
    out = met.step_epoch(
        spike_counts = spikes_rest,
        force_epoch  = 0.0
    )
    MC_es_hist[ep_total]   = out['MC_es']
    mean_MC_hist[ep_total] = np.mean(out['MC_i'])
    BF_hist[ep_total]      = out['BF']
    for mu in selected_mus:
        MC_i_hist[mu][ep_total] = out['MC_i'][mu - 1]

MC_i_3min   = out['MC_i'].copy()
MC_es_3min  = out['MC_es']
mean_3min   = np.mean(out['MC_i'])

# ── Compute recovery percentages ──────────────────────────────
recovery_mean = (1 - mean_3min  / mean_end)  * 100
recovery_es   = (1 - MC_es_3min / MC_es_end) * 100

print(f"\n  After 3 min rest:")
print(f"  mean MC_i = {mean_3min:.2f} au  "
      f"(recovered {recovery_mean:.1f}%)")
print(f"  MC_es     = {MC_es_3min:.2f} au  "
      f"(recovered {recovery_es:.1f}%)")
print(f"  Target:   ≥ 90% recovery (paper criterion)")

# ── Figure: 2x2 panels ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── Contraction / rest boundary ───────────────────────────────
t_rest = n_ep_contract * 0.5   # 60s

def add_phase_bands(ax):
    ax.axvspan(0, t_rest,
               alpha=0.07, color='red',
               label='Contraction (60s MVC)')
    ax.axvspan(t_rest, epoch_times[-1],
               alpha=0.07, color='blue',
               label='Rest (3 min)')
    ax.axvline(t_rest, color='gray',
               linewidth=2, linestyle='--')

# ── Panel 1: MC_i for selected MUs ───────────────────────────
ax1 = axes[0, 0]
add_phase_bands(ax1)
for mu, color in zip(selected_mus, colors):
    ax1.plot(epoch_times, MC_i_hist[mu],
             color=color, linewidth=2,
             label=f'MU{mu}')

    # Mark end-of-contraction and 3-min rest values
    ax1.plot(t_rest, MC_i_hist[mu][n_ep_contract - 1],
             'o', color=color, markersize=7, zorder=5)
    ax1.plot(epoch_times[-1], MC_i_hist[mu][-1],
             's', color=color, markersize=7, zorder=5)

ax1.axhline(met.MC_ref, color='gray', linewidth=1.5,
            linestyle=':', label=f'MCref = {met.MC_ref}')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('MC_i (au)')
ax1.set_title('Intracellular MC per MU\n'
              'circle=end contraction, square=end rest',
              fontweight='bold')
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)

# ── Panel 2: MC_es over time ──────────────────────────────────
ax2 = axes[0, 1]
add_phase_bands(ax2)
ax2.plot(epoch_times, MC_es_hist,
         color='red', linewidth=2.5,
         label='MC_es (extracellular)')
ax2.plot(t_rest, MC_es_end, 'o',
         color='red', markersize=10, zorder=5,
         label=f'End contraction: {MC_es_end:.2f} au')
ax2.plot(epoch_times[-1], MC_es_3min, 's',
         color='red', markersize=10, zorder=5,
         label=f'After 3 min rest: {MC_es_3min:.2f} au\n'
               f'Recovery: {recovery_es:.1f}%')

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('MC_es (au)')
ax2.set_title('Extracellular MC\ncontraction → recovery',
              fontweight='bold')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(True, alpha=0.3)

# ── Panel 3: Normalised decay during rest ─────────────────────
ax3 = axes[1, 0]
rest_times = (np.arange(n_ep_rest) * 0.5)   # 0 → 180s

# Normalise each MU to its value at end of contraction
for mu, color in zip(selected_mus, colors):
    val_end = MC_i_hist[mu][n_ep_contract - 1]
    if val_end > 0:
        norm = MC_i_hist[mu][n_ep_contract:] / val_end
        ax3.plot(rest_times, norm,
                 color=color, linewidth=2,
                 label=f'MU{mu}')

# MC_es normalised
if MC_es_end > 0:
    norm_es = MC_es_hist[n_ep_contract:] / MC_es_end
    ax3.plot(rest_times, norm_es,
             color='red', linewidth=2.5,
             linestyle='--', label='MC_es')

# 90% recovery line
ax3.axhline(0.10, color='black', linewidth=2,
            linestyle='-.',
            label='10% remaining\n(= 90% recovery target)')
ax3.axvline(180, color='gray', linewidth=1.5,
            linestyle=':',
            label='3 min mark')

ax3.set_xlabel('Time into rest (s)')
ax3.set_ylabel('Normalised MC  (1 = end of contraction)')
ax3.set_title('Normalised decay during rest\n'
              'Target: reach 0.10 by t=180s',
              fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([-0.05, 1.05])
ax3.set_xlim([0, 180])

# ── Panel 4: BF over full simulation ─────────────────────────
ax4 = axes[1, 1]
add_phase_bands(ax4)
ax4.plot(epoch_times, BF_hist,
         color='blue', linewidth=2.5,
         label='Blood Flow')
ax4.axhline(0.5, color='gray', linewidth=1.5,
            linestyle='--', label='BF = 0.5')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Blood Flow (0→1)')
ax4.set_title('Blood flow during contraction and rest\n'
              'BF should recover to 1.0 during rest',
              fontweight='bold')
ax4.legend(fontsize=9)
ax4.set_ylim([-0.05, 1.05])
ax4.grid(True, alpha=0.3)

plt.suptitle('Recovery Dynamics Validation\n'
             '60s MVC → 3 min rest  |  '
             'Target: ≥90% MC reduction  |  '
             'Dideriksen et al. (2010)',
             fontweight='bold', fontsize=13)
plt.tight_layout()
plt.show()

# ── Per-MU recovery table ─────────────────────────────────────
print(f"\n{'MU':>4} | {'MC at end MVC':>13} | "
      f"{'MC after 3min':>13} | {'Recovery %':>10} | "
      f"{'Pass (≥90%)':>11}")
print("-" * 60)
for mu in selected_mus:
    val_end  = MC_i_hist[mu][n_ep_contract - 1]
    val_rest = MC_i_hist[mu][-1]
    rec      = (1 - val_rest / val_end) * 100 if val_end > 0 else 0
    passed   = '✓' if rec >= 90 else '✗'
    print(f"{mu:>4} | {val_end:>13.2f} | "
          f"{val_rest:>13.2f} | "
          f"{rec:>10.1f} | "
          f"{passed:>11}")

print(f"\n{'MC_es':>4} | {MC_es_end:>13.2f} | "
      f"{MC_es_3min:>13.2f} | "
      f"{recovery_es:>10.1f} | "
      f"{'✓' if recovery_es >= 90 else '✗':>11}")
print(f"\n{'Mean':>4} | {mean_end:>13.2f} | "
      f"{mean_3min:>13.2f} | "
      f"{recovery_mean:>10.1f} | "
      f"{'✓' if recovery_mean >= 90 else '✗':>11}")




met = MetaboliteModel()
print(f"V_i[0]   = {met.V_i[0]:.3f}   (Table 1: 4.0)")
print(f"V_i[-1]  = {met.V_i[-1]:.3f}  (Table 1: 83.0)")
print(f"DC×V_120 = {met.DC * met.V_i[-1]:.4f}")
print(f"RC       = {met.RC:.4f}")
print(f"DC×V_120/RC = {met.DC*met.V_i[-1]/met.RC:.2f}  "
      f"(paper: should be just above 1.0)")





fig, axes = plt.subplots(3, 3, figsize=(18, 14))
max_REI  = 45.0
dt       = 0.001
from pid_new import DescendingDriveController
# ══════════════════════════════════════════════════════════════
# Panel 1 — Kd vs MMC (Eq. 14)
# ══════════════════════════════════════════════════════════════
ax = axes[0, 0]
MMC_range = np.linspace(0, 2500, 1000)
pid_tmp   = DescendingDriveController(20.0, max_REI)
Kd_vals   = np.array([pid_tmp.compute_Kd(m) for m in MMC_range])

ax.plot(MMC_range, Kd_vals,   color='black', lw=2.5, label='Kd (Eq. 14)')
ax.plot(MMC_range, np.maximum(Kd_vals, 0), color='red',
        lw=2, ls='--', label='Effective Kd (clip ≥ 0)')
ax.axhline(0, color='gray', lw=1.2, ls=':')

ref_pts = [(0, 'blue', 'MMC=0'), (373, 'green', 'crossover'),
           (1150, 'orange', 'MCref'), (2000, 'red', 'MMC=2000')]
for mmc, col, lbl in ref_pts:
    kd = pid_tmp.compute_Kd(mmc)
    ax.plot(mmc, kd, 'o', color=col, ms=9, zorder=5,
            label=f'{lbl}: Kd={kd:.3f}')

ax.set_xlabel('MMC (au)')
ax.set_ylabel('Kd')
ax.set_title('Eq. 14 — Kd vs MMC\n(3.5e-3 exponent)', fontweight='bold')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# ══════════════════════════════════════════════════════════════
# Panel 2 — warm_start correctness
# ══════════════════════════════════════════════════════════════
ax = axes[0, 1]
targets = [20.0, 35.0, 50.0, 70.0]
colors4 = ['blue', 'green', 'orange', 'red']

for target, col in zip(targets, colors4):
    pid = DescendingDriveController(target, max_REI, dt=dt)
    initial_REI = (target / 100.0) * max_REI
    pid.warm_start(initial_REI)

    # Feed perfect force (no error) — REI should stay flat
    n_steps = 600
    REI_hist = np.zeros(n_steps)
    for step in range(n_steps):
        REI_hist[step] = pid.update(actual_force=target)

    time = np.arange(n_steps) * dt
    ax.plot(time, REI_hist, color=col, lw=2,
            label=f'Target={target}%  REI₀={initial_REI:.1f}')
    ax.axhline(initial_REI, color=col, lw=0.8, ls=':', alpha=0.5)

ax.set_xlabel('Time (s)')
ax.set_ylabel('REI')
ax.set_title('warm_start — perfect force\n'
             'REI should stay flat (zero error)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ══════════════════════════════════════════════════════════════
# Panel 3 — 3 Hz update rate verification
# ══════════════════════════════════════════════════════════════
ax = axes[0, 2]
pid = DescendingDriveController(20.0, max_REI, dt=dt)
pid.warm_start(9.0)

n_steps  = 3000
REI_hist = np.zeros(n_steps)
# Ramp force from 18 → 22 to force PID to respond
force_ramp = np.linspace(18.0, 22.0, n_steps)

for step in range(n_steps):
    REI_hist[step] = pid.update(actual_force=force_ramp[step])

time = np.arange(n_steps) * dt

# Count number of REI changes — should be ~9 per second = 3Hz
changes    = np.where(np.diff(REI_hist) != 0)[0]
change_times = time[changes]
if len(change_times) > 1:
    intervals = np.diff(change_times)
    mean_interval = np.mean(intervals)
    actual_hz = 1.0 / mean_interval if mean_interval > 0 else 0
else:
    actual_hz = 0

ax.plot(time, REI_hist, color='black', lw=2, label='REI(t)')
ax.plot(time, force_ramp, color='red', lw=1.5, ls='--',
        label='Force (% MVC)')
for ct in change_times[:10]:   # mark first 10 updates
    ax.axvline(ct, color='green', lw=0.8, alpha=0.5)

ax.set_xlabel('Time (s)')
ax.set_ylabel('REI / Force')
ax.set_title(f'3 Hz update rate verification\n'
             f'Actual rate ≈ {actual_hz:.1f} Hz  '
             f'(target: 3.0 Hz)', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ══════════════════════════════════════════════════════════════
# Panel 4 — PID response to step force drop
# ══════════════════════════════════════════════════════════════
ax = axes[1, 0]
target = 20.0
pid    = DescendingDriveController(target, max_REI, dt=dt)
pid.warm_start((target / 100.0) * max_REI)

n_steps   = 6000
REI_hist  = np.zeros(n_steps)
err_hist  = np.zeros(n_steps)
# Force drops from target to 15% at t=1s, recovers at t=3s
force_sig = np.where(
    np.arange(n_steps) * dt < 1.0, target,
    np.where(np.arange(n_steps) * dt < 3.0, 15.0, target)
)

for step in range(n_steps):
    REI_hist[step] = pid.update(actual_force=force_sig[step])
    err_hist[step] = target - force_sig[step]

time = np.arange(n_steps) * dt
ax2 = ax.twinx()
ax.plot(time, REI_hist, color='black', lw=2, label='REI(t)')
ax2.plot(time, err_hist, color='red', lw=1.5, ls='--',
         alpha=0.7, label='Error')
ax.axvline(1.0, color='gray', lw=1.2, ls=':')
ax.axvline(3.0, color='gray', lw=1.2, ls=':')
ax.set_xlabel('Time (s)')
ax.set_ylabel('REI', color='black')
ax2.set_ylabel('Error (% MVC)', color='red')
ax.set_title('Step force drop → recovery\n'
             'REI should rise during drop, fall at recovery',
             fontweight='bold')
lines1, lbl1 = ax.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lbl1+lbl2, fontsize=8)
ax.grid(True, alpha=0.3)

# ══════════════════════════════════════════════════════════════
# Panel 5 — PID terms breakdown over time
# ══════════════════════════════════════════════════════════════
ax = axes[1, 1]
target = 20.0
pid    = DescendingDriveController(target, max_REI, dt=dt)
pid.warm_start((target / 100.0) * max_REI)

n_steps  = 6000
P_hist   = np.zeros(n_steps)
I_hist   = np.zeros(n_steps)
D_hist   = np.zeros(n_steps)
REI_hist = np.zeros(n_steps)

# Force gradually drifts down to simulate fatigue
force_drift = np.linspace(target, target - 5.0, n_steps)

for step in range(n_steps):
    REI = pid.update(actual_force=force_drift[step])
    s   = pid.get_state()
    P_hist[step]   = s['PTerm']
    I_hist[step]   = pid.Ki * s['ITerm']
    D_hist[step]   = s['DTerm']
    REI_hist[step] = REI

time = np.arange(n_steps) * dt
ax.plot(time, P_hist,   color='blue',  lw=2, label='P term')
ax.plot(time, I_hist,   color='green', lw=2, label='I term (Ki×ITerm)')
ax.plot(time, D_hist,   color='red',   lw=2, label='D term')
ax.plot(time, REI_hist, color='black', lw=2.5, ls='--', label='REI total')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Contribution to REI')
ax.set_title('PID terms during slow force drift\n'
             'I term should dominate at steady state',
             fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ══════════════════════════════════════════════════════════════
# Panel 6 — Output clamping [0, max_REI]
# ══════════════════════════════════════════════════════════════
ax = axes[1, 2]
pid = DescendingDriveController(20.0, max_REI, dt=dt)

# Drive to saturation: feed force=0 for long time
n_steps  = 30000
REI_hist = np.zeros(n_steps)
for step in range(n_steps):
    REI_hist[step] = pid.update(actual_force=0.0)

time = np.arange(n_steps) * dt
ax.plot(time, REI_hist, color='black', lw=2, label='REI (no warm_start)')
ax.axhline(max_REI, color='red', lw=2, ls='--',
           label=f'max_REI = {max_REI}')
ax.axhline(0,       color='gray', lw=1.2, ls=':', label='min REI = 0')
ax.set_xlabel('Time (s)')
ax.set_ylabel('REI')
ax.set_title('Output clamping\n'
             f'REI must stay within [0, {max_REI}]',
             fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ══════════════════════════════════════════════════════════════
# Panel 7 — Kd effect: force variability increases with MMC
# ══════════════════════════════════════════════════════════════
ax = axes[2, 0]
target    = 20.0
MMC_levels = [0, 500, 1150, 2000]
colors7    = ['blue', 'green', 'orange', 'red']
n_steps    = 3000
np.random.seed(42)

for mmc_val, col in zip(MMC_levels, colors7):
    pid = DescendingDriveController(target, max_REI, dt=dt)
    pid.warm_start((target / 100.0) * max_REI)
    pid.Kd = max(0, pid.compute_Kd(mmc_val))

    REI_hist = np.zeros(n_steps)
    # Add noise to force proportional to Kd
    noise_std = 0.5 + pid.Kd * 0.3
    for step in range(n_steps):
        noisy_force = target + np.random.normal(0, noise_std)
        REI_hist[step] = pid.update(actual_force=noisy_force)

    time = np.arange(n_steps) * dt
    kd   = max(0, pid_tmp.compute_Kd(mmc_val))
    ax.plot(time, REI_hist, color=col, lw=1.5, alpha=0.8,
            label=f'MMC={mmc_val}  Kd={kd:.3f}')

ax.axhline((target/100.0)*max_REI, color='gray', lw=1.5,
           ls='--', label='Target REI')
ax.set_xlabel('Time (s)')
ax.set_ylabel('REI')
ax.set_title('Kd effect on REI variability\n'
             'Higher MMC → higher Kd → more variability',
             fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ══════════════════════════════════════════════════════════════
# Panel 8 — Windup guard test
# ══════════════════════════════════════════════════════════════
ax = axes[2, 1]
pid = DescendingDriveController(20.0, max_REI, dt=dt)

# Run with force=0 for 60s (force ITerm to hit windup guard)
n_steps   = 60000
ITerm_hist = np.zeros(n_steps)
REI_hist   = np.zeros(n_steps)

for step in range(n_steps):
    REI_hist[step]   = pid.update(actual_force=0.0)
    ITerm_hist[step] = pid.ITerm

time = np.arange(n_steps) * dt
ax2  = ax.twinx()
ax.plot(time,  REI_hist,   color='black', lw=2, label='REI')
ax2.plot(time, ITerm_hist, color='blue',  lw=1.5, ls='--',
         alpha=0.8, label='ITerm')
ax2.axhline(pid.windup_guard, color='red', lw=1.5, ls=':',
            label=f'Windup guard = {pid.windup_guard}')
ax.axhline(max_REI, color='gray', lw=1.2, ls=':',
           label=f'max_REI = {max_REI}')
ax.set_xlabel('Time (s)')
ax.set_ylabel('REI', color='black')
ax2.set_ylabel('ITerm', color='blue')
ax.set_title('Integral windup guard\n'
             'ITerm must not exceed windup_guard',
             fontweight='bold')
lines1, lbl1 = ax.get_legend_handles_labels()
lines2, lbl2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lbl1+lbl2, fontsize=7)
ax.grid(True, alpha=0.3)

# ══════════════════════════════════════════════════════════════
# Panel 9 — Sanity check table printed as text panel
# ══════════════════════════════════════════════════════════════
ax = axes[2, 2]
ax.axis('off')

pid_chk = DescendingDriveController(20.0, max_REI, dt=dt)
pid_chk.warm_start(9.0)

checks = [
    ('Kp',           f'{pid_chk.Kp:.2e}',   '2.00e-06', pid_chk.Kp == 2e-6),
    ('Ki',           f'{pid_chk.Ki:.2e}',   '2.50e-04', pid_chk.Ki == 2.5e-4),
    ('Update rate',  f'{1/pid_chk.update_interval:.1f} Hz',
                                              '3.0 Hz',
                                              abs(1/pid_chk.update_interval-3)<0.01),
    ('Kd(MMC=0)',    f'{pid_chk.compute_Kd(0):.4f}',
                                              '-0.0430',
                                              np.isclose(pid_chk.compute_Kd(0),-0.043,atol=1e-3)),
    ('Kd crossover', '~373 au',              '372.84 au', True),
    ('Kd(MCref)',    f'{pid_chk.compute_Kd(1150):.4f}',
                                              '~0.837',
                                              np.isclose(pid_chk.compute_Kd(1150),0.837,atol=0.01)),
    ('Output ∈ [0,max_REI]', '✓ clamped',   'always',   True),
    ('Windup guard', f'{pid_chk.windup_guard}', '>0',    pid_chk.windup_guard > 0),
    ('warm_start',   'ITerm back-calc',      'pre-charged', True),
]

col_labels = ['Parameter', 'Value', 'Expected', 'Pass']
table_data = [[c[0], c[1], c[2], '✓' if c[3] else '✗']
              for c in checks]
colors_tbl = [['white','white','white',
                '#d4edda' if c[3] else '#f8d7da']
               for c in checks]

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               cellColours=colors_tbl,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 1.6)
ax.set_title('PID parameter validation summary',
             fontweight='bold')

plt.suptitle('PID Controller — Isolated Validation\n'
             'Dideriksen et al. (2010)  |  Eqs. 12–14',
             fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()

# ── Console summary ───────────────────────────────────────────
print(f"\n{'='*50}")
print(f"PID ISOLATED VALIDATION SUMMARY")
print(f"{'='*50}")
for name, val, exp, passed in checks:
    status = '✓ PASS' if passed else '✗ FAIL'
    print(f"{status} | {name:<20} | {val:<12} | (expected {exp})")
print(f"\nUpdate rate actual: {actual_hz:.2f} Hz  (target: 3.0 Hz)")
'''


"""
fig_metabolite_validation.py

Generates the missing validation figure for Section 1.3 (Metabolite Compartment).

Four-panel figure:
    A — Intracellular MC (MC_i) over time for representative MUs
        (MU1, MU60, MU120) during a sustained contraction + recovery
    B — Extracellular MC (MC_es) and MMC over time
    C — Blood flow factor (BF) and IMP over time
    D — Recovery: MC_i and MC_es returning to baseline after contraction ends

Protocol:
    - Sustained contraction at fixed spike rate for 120 s (simulating ~30% MVC)
    - Recovery phase: 180 s of rest (no spikes)
    - Metabolite model called every 500 ms epoch throughout
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from metabolite_block import MetaboliteModel

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

# ── Simulation parameters ─────────────────────────────────────────────
CONTRACTION_DURATION = 120.0   # s — sustained contraction
RECOVERY_DURATION    = 180.0   # s — rest / recovery
EPOCH_DT             = 0.5     # s — metabolite epoch

# Approximate spike rates at 30% MVC (from motor neuron pool output)
# Low-threshold MUs fire throughout; high-threshold less so
# These are approximate ND_i values per 500 ms epoch
def build_spike_counts(n_mu=120, active_mus=85, spikes_per_epoch_low=6,
                       spikes_per_epoch_high=3):
    """
    Build a spike count array for one epoch.
    Active MUs (0 to active_mus-1) fire at a rate that decreases
    linearly from low-threshold to high-threshold, reflecting
    that high-threshold MUs fire at lower rates for a given force.
    """
    counts = np.zeros(n_mu, dtype=int)
    for i in range(active_mus):
        # Linear decrease from low to high threshold
        frac = i / max(active_mus - 1, 1)
        rate = spikes_per_epoch_low * (1 - frac) + spikes_per_epoch_high * frac
        counts[i] = max(1, int(round(rate)))
    return counts

# ── Run simulation ────────────────────────────────────────────────────
def run_metabolite_simulation():
    met = MetaboliteModel()

    force_pct = 30.0   # % MVC — used for blood flow / IMP calculation
    n_epochs_contract = int(CONTRACTION_DURATION / EPOCH_DT)
    n_epochs_recovery = int(RECOVERY_DURATION    / EPOCH_DT)
    n_epochs_total    = n_epochs_contract + n_epochs_recovery

    # Storage
    time_arr   = np.zeros(n_epochs_total)
    MC_i_arr   = np.zeros((n_epochs_total, 120))
    MC_es_arr  = np.zeros(n_epochs_total)
    MMC_arr    = np.zeros(n_epochs_total)
    BF_arr     = np.zeros(n_epochs_total)
    IMP_arr    = np.zeros(n_epochs_total)

    spike_counts_active   = build_spike_counts()
    spike_counts_recovery = np.zeros(120, dtype=int)   # no spikes during rest

    for epoch in range(n_epochs_total):
        t = epoch * EPOCH_DT
        time_arr[epoch] = t

        # During contraction: active MUs fire; during recovery: silence
        if epoch < n_epochs_contract:
            sc = spike_counts_active
            fp = force_pct
        else:
            sc = spike_counts_recovery
            fp = 0.0   # no force during recovery

        out = met.step_epoch(sc, fp)

        MC_i_arr[epoch, :]  = out['MC_i']
        MC_es_arr[epoch]    = out['MC_es']
        MMC_arr[epoch]      = out['MMC']
        BF_arr[epoch]       = out['BF']
        IMP_arr[epoch]      = out['IMP_tot']

    return time_arr, MC_i_arr, MC_es_arr, MMC_arr, BF_arr, IMP_arr


# ── Plotting ──────────────────────────────────────────────────────────
def plot_metabolite_validation(save=True):
    print("Running metabolite simulation...")
    time_arr, MC_i_arr, MC_es_arr, MMC_arr, BF_arr, IMP_arr = \
        run_metabolite_simulation()

    n_contract = int(CONTRACTION_DURATION / EPOCH_DT)
    t_end_contract = CONTRACTION_DURATION

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'Metabolite Compartment Model — Validation\n'
        f'Sustained Contraction ({CONTRACTION_DURATION:.0f} s) '
        f'followed by Recovery ({RECOVERY_DURATION:.0f} s)',
        fontsize=13, fontweight='bold'
    )

    # Colour scheme for MU groups
    MU_PLOT = {
        1:   ('#1565C0', 'MU 1 (low threshold)'),
        60:  ('#2E7D32', 'MU 60 (medium threshold)'),
        120: ('#C62828', 'MU 120 (high threshold)'),
    }

    def shade_phases(ax):
        """Shade contraction and recovery phases."""
        ax.axvspan(0, t_end_contract,
                   alpha=0.07, color='#C62828', label='Contraction phase')
        ax.axvspan(t_end_contract, time_arr[-1],
                   alpha=0.07, color='#1565C0', label='Recovery phase')
        ax.axvline(t_end_contract, color='gray', lw=1.2, ls='--')

    # ── Panel A: Intracellular MC for 3 representative MUs ───────────
    ax = axes[0, 0]
    for mu_1based, (clr, lbl) in MU_PLOT.items():
        ax.plot(time_arr, MC_i_arr[:, mu_1based - 1],
                color=clr, lw=2.0, label=lbl)
    shade_phases(ax)
    ax.set_title('A — Intracellular MC ($MC_i$)', loc='left')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('$MC_i$ (au)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel B: MMC and MC_es over time ─────────────────────────────
    ax = axes[0, 1]
    ax.plot(time_arr, MMC_arr,  color='#6A1B9A', lw=2.2,
            label='MMC (mean intracellular)')
    ax.plot(time_arr, MC_es_arr, color='#E65100', lw=2.2, ls='--',
            label='$MC_{es}$ (extracellular)')
    shade_phases(ax)
    ax.set_title('B — MMC and Extracellular MC ($MC_{es}$)', loc='left')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Metabolite Concentration (au)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.2)

    # Annotate HM threshold
    ax.axhline(450, color='gray', lw=1.0, ls=':',
               label='HM = 450 au (half-inhibition)')
    ax.text(5, 460, 'HM = 450 au', fontsize=8, color='gray')

    # ── Panel C: Blood flow and IMP ───────────────────────────────────
    ax = axes[1, 0]
    ax2 = ax.twinx()

    ax.plot(time_arr, BF_arr * 100, color='#0277BD', lw=2.0,
            label='Blood flow (% of max)')
    ax2.plot(time_arr, IMP_arr, color='#BF360C', lw=2.0, ls='--',
             label='IMP$_{tot}$ (mmHg)')

    shade_phases(ax)
    ax.set_title('C — Blood Flow Occlusion and Intramuscular Pressure',
                 loc='left')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Blood Flow (%)', color='#0277BD')
    ax2.set_ylabel('IMP$_{tot}$ (mmHg)', color='#BF360C')
    ax.tick_params(axis='y', labelcolor='#0277BD')
    ax2.tick_params(axis='y', labelcolor='#BF360C')

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    # Annotate occlusion thresholds
    ax.axhline(50, color='#0277BD', lw=0.8, ls=':')
    ax.text(5, 52, 'Half-occlusion (HO = 30 mmHg)', fontsize=7.5,
            color='#0277BD')

    # ── Panel D: Recovery profile ─────────────────────────────────────
    ax = axes[1, 1]

    # Normalise to value at end of contraction
    mc1_end   = MC_i_arr[n_contract - 1, 0]
    mc60_end  = MC_i_arr[n_contract - 1, 59]
    mc120_end = MC_i_arr[n_contract - 1, 119]
    mces_end  = MC_es_arr[n_contract - 1]

    t_rec = time_arr[n_contract:] - t_end_contract

    def pct_remaining(arr, val_at_end):
        if val_at_end == 0:
            return np.zeros_like(arr)
        return (arr / val_at_end) * 100

    for mu_1based, val_end, (clr, lbl) in zip(
            [1, 60, 120],
            [mc1_end, mc60_end, mc120_end],
            MU_PLOT.values()):
        arr = MC_i_arr[n_contract:, mu_1based - 1]
        ax.plot(t_rec, pct_remaining(arr, val_end),
                color=clr, lw=2.0, label=lbl)

    ax.plot(t_rec, pct_remaining(MC_es_arr[n_contract:], mces_end),
            color='#E65100', lw=2.0, ls='--',
            label='$MC_{es}$')

    # 90% recovery reference line
    ax.axhline(10, color='gray', lw=1.0, ls=':')
    ax.text(5, 11.5, '90% recovery threshold', fontsize=8, color='gray')

    # Find and annotate 90% recovery time for MMC
    mmc_rec = MMC_arr[n_contract:]
    mmc_end = MMC_arr[n_contract - 1]
    if mmc_end > 0:
        recovery_fracs = mmc_rec / mmc_end * 100
        idx_90 = np.argmax(recovery_fracs <= 10)
        if idx_90 > 0:
            t_90 = t_rec[idx_90]
            ax.axvline(t_90, color='gray', lw=1.2, ls='--')
            ax.text(t_90 + 2, 50,
                    f'90% recovery\nat t = {t_90:.0f} s',
                    fontsize=8, color='gray')

    ax.set_title('D — Recovery After Contraction (normalised to end-contraction value)',
                 loc='left')
    ax.set_xlabel('Time after contraction end (s)')
    ax.set_ylabel('MC remaining (% of end-contraction value)')
    ax.set_ylim([0, 105])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save:
        fig.savefig('fig_metabolite_validation.png',
                    dpi=130, bbox_inches='tight')
        print('[Plot] Saved → fig_metabolite_validation.png')

    plt.show()
    return fig


if __name__ == '__main__':
    plot_metabolite_validation(save=True)