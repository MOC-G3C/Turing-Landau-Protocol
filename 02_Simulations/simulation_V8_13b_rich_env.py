import numpy as np
import copy
import math

# ============================================================
# simulation_V8_13b_rich_env.py
#
# Moteur enrichi pour Direction A — objectif : briser le
# plafond R2=0.33 de V8.22 en augmentant la complexité
# structurée de l'environnement.
#
# Enrichissements vs V8.13 :
#   1. Hotspots MOBILES  — trajectoires orbitales + drift lent
#   2. Couplage non-linéaire — h_comb² et tanh(h_f·h_s)
#   3. Troisième canal de mémoire (tau_ultra_slow)
#   4. Turbulence spatiale corrélée (bruit non-blanc)
#   5. Événements rares de couplage global (cross-talk)
#
# Interface identique à V8.13 — drop-in pour V8.22 :
#   eval_individu_raw(adn, steps, return_traces)
#   creer_individu_aleatoire()
#   mutate_individu(ind)
# ============================================================

np.random.seed(42)

kB, T_env, LN2 = 1.38e-23, 300.0, math.log(2.0)
SPIKE_PROB_FAST  = 0.10
SPIKE_PROB_SLOW  = 0.02
CROSSTALK_PROB   = 0.008   # événement rare de couplage global
INFO_TO_BITS     = 0.2
SEUIL_REACTION   = 0.08

# Grille 16×16
NX, NY = 16, 16
XS = np.arange(NX)[:, None]
YS = np.arange(NY)[None, :]

# Seuils anti-divergence
THETA_CLIP = 10.0
H_CLIP     = 50.0

META_BOUNDS = {
    "kappa_gain":      (0.01, 2.0),
    "mutation_rate":   (0.01, 0.30),
    "env_gain_spike":  (0.005, 0.20),
    "env_gain_calm":   (0.0001, 0.02),
    "nl_gain":         (0.0, 0.5),   # gain du terme non-linéaire h²
    "cross_sensitivity":(0.0, 1.0),  # sensibilité aux événements cross-talk
}

# ============================================================
# HOTSPOTS MOBILES
# Chaque hotspot a : position initiale (x0,y0), amplitude,
# phase, vitesse orbitale (omega), rayon orbital (r_orb),
# et drift lent (dx_drift, dy_drift par step)
# ============================================================
HOTSPOT_DEFS = [
    # x0,  y0, amp,  phase,    omega,   r_orb, dx_drift, dy_drift
    (  3,   3, 10.0,  0.0,    0.012,    2.0,   0.003,   0.001),
    ( 12,  12,  8.0,  np.pi/2, 0.018,  1.5,  -0.002,   0.004),
    (  7,   3,  6.0,  np.pi,   0.025,  2.5,   0.001,  -0.003),  # nouveau
]

def hotspot_position(hdef, t):
    """Position instantanée d'un hotspot au step t."""
    x0, y0, amp, phase, omega, r_orb, dx, dy = hdef
    # Orbite elliptique + drift lent
    x = x0 + r_orb * np.cos(omega * t + phase) + dx * t
    y = y0 + r_orb * np.sin(omega * t + phase * 0.7) + dy * t
    # Rebond sur les bords (toroïdal modulo)
    x = x % NX
    y = y % NY
    return x, y, amp, phase


def build_env_field(t, f_comp, s_comp, turb):
    """
    Construit le champ environnemental à l'instant t.
    turb : bruit spatial corrélé (16,16) pré-calculé.
    """
    env = np.zeros((NX, NY))
    for hdef in HOTSPOT_DEFS:
        x, y, amp, ph = hotspot_position(hdef, t)
        dist2 = (XS - x)**2 + (YS - y)**2
        env += amp * np.exp(-dist2 / 9.0) * (
            f_comp + s_comp + np.sin(t / 10.0 + ph) * 0.2
        )
    # Turbulence spatiale corrélée
    env += turb * 1.5
    return env


# ============================================================
# BRUIT SPATIAL CORRÉLÉ (noyau gaussien)
# ============================================================
_TURB_KERNEL = None

def _make_turb_kernel(sigma=2.5):
    global _TURB_KERNEL
    if _TURB_KERNEL is not None:
        return _TURB_KERNEL
    k = np.zeros((NX, NY))
    cx, cy = NX // 2, NY // 2
    for i in range(NX):
        for j in range(NY):
            k[i, j] = np.exp(-((i-cx)**2 + (j-cy)**2) / (2*sigma**2))
    k /= k.sum()
    _TURB_KERNEL = k
    return k

def sample_turbulence():
    """Bruit blanc filtré spatialement → corrélation spatiale ~sigma=2.5 cells."""
    from numpy.fft import fft2, ifft2, fftshift
    white = np.random.randn(NX, NY)
    kernel = _make_turb_kernel()
    # Convolution rapide via FFT
    turb = np.real(ifft2(fft2(white) * fft2(kernel)))
    return turb


# ============================================================
# INDIVIDU
# ============================================================

def creer_individu_aleatoire():
    return {
        "theta":         np.random.randn(NX, NY) * 0.1,
        "tau_fast":      np.random.uniform(0.15, 0.40, (NX, NY)),
        "tau_slow":      np.random.uniform(0.70, 0.98, (NX, NY)),
        "tau_ultra":     np.random.uniform(0.990, 0.999, (NX, NY)),  # nouveau
        "w_fast_mat":    np.random.uniform(0.2, 0.8, (NX, NY)),
        "w_ultra_mat":   np.random.uniform(0.0, 0.2, (NX, NY)),      # nouveau
        "meta": {k: np.random.uniform(v[0], v[1]) for k, v in META_BOUNDS.items()}
    }

def mutate_individu(ind):
    child = copy.deepcopy(ind)
    child["w_fast_mat"]  = np.clip(child["w_fast_mat"]  + np.random.randn(NX,NY)*0.03, 0.0, 1.0)
    child["w_ultra_mat"] = np.clip(child["w_ultra_mat"] + np.random.randn(NX,NY)*0.02, 0.0, 0.5)
    child["tau_fast"]    = np.clip(child["tau_fast"]    + np.random.randn(NX,NY)*0.02, 0.15, 0.40)
    child["tau_slow"]    = np.clip(child["tau_slow"]    + np.random.randn(NX,NY)*0.02, 0.70, 0.98)
    child["tau_ultra"]   = np.clip(child["tau_ultra"]   + np.random.randn(NX,NY)*0.002, 0.990, 0.999)
    for k, (lo, hi) in META_BOUNDS.items():
        child["meta"][k] = float(np.clip(
            child["meta"][k] * (1.0 + np.random.randn() * 0.07), lo, hi
        ))
    return child


# ============================================================
# ÉVALUATION
# ============================================================

def eval_individu_raw(adn, steps=200, return_traces=False):
    """
    Interface identique à V8.13 + return_traces pour V8.22.
    Enrichissements actifs :
      - Hotspots mobiles à chaque step
      - 3ème canal mémoire tau_ultra
      - Couplage non-linéaire h_comb² (nl_gain)
      - Turbulence spatiale corrélée
      - Cross-talk global aléatoire (cross_sensitivity)
    """
    _make_turb_kernel()   # pré-calcul kernel si besoin

    trials     = []
    hs_trace   = []
    envs_trace = []

    for trial_idx in range(3):
        theta = adn["theta"].copy()
        h_f   = np.zeros((NX, NY))
        h_s   = np.zeros((NX, NY))
        h_u   = np.zeros((NX, NY))   # ultra-slow
        Q_heat, lucidity, inaction = 0.0, 0.0, 0.0
        collect = return_traces and (trial_idx == 0)

        # État turbulence : AR(1) pour continuité temporelle
        turb = sample_turbulence()

        for t in range(1, steps + 1):

            # ── Turbulence AR(1) ────────────────────────────────────
            turb = 0.85 * turb + 0.15 * sample_turbulence()

            # ── Composantes temporelles ──────────────────────────────
            f_comp = (1.0 if np.random.rand() < SPIKE_PROB_FAST else 0.1) * np.sin(t / 8.0)
            s_comp = (1.0 if np.random.rand() < SPIKE_PROB_SLOW else 0.1) * np.sin(t / 200.0)

            # ── Champ environnemental (hotspots mobiles + turbulence) ─
            env_field = build_env_field(t, f_comp, s_comp, turb)

            # ── Cross-talk global : activation simultanée de tous les ─
            #    hotspots avec un offset de phase aléatoire (événement rare)
            if np.random.rand() < CROSSTALK_PROB:
                cross_amp = np.random.uniform(2.0, 5.0) * adn["meta"]["cross_sensitivity"]
                env_field += cross_amp * np.sin(t / 3.0 + np.random.uniform(0, 2*np.pi))

            # ── Triple canal mémoire ─────────────────────────────────
            h_f = adn["tau_fast"]  * h_f + (1.0 - adn["tau_fast"])  * env_field
            h_s = adn["tau_slow"]  * h_s + (1.0 - adn["tau_slow"])  * env_field
            h_u = adn["tau_ultra"] * h_u + (1.0 - adn["tau_ultra"]) * env_field

            # ── Mix spatial ──────────────────────────────────────────
            w_f = adn["w_fast_mat"]
            w_u = adn["w_ultra_mat"]
            w_s = np.clip(1.0 - w_f - w_u, 0.0, 1.0)
            h_comb = w_f * h_f + w_s * h_s + w_u * h_u

            # ── Couplage non-linéaire ────────────────────────────────
            nl = adn["meta"]["nl_gain"] * (
                np.tanh(h_f * h_s * 0.1)          # interaction fast×slow
                + 0.3 * np.sign(h_comb) * h_comb**2 * 0.01  # saturation douce
            )
            h_comb = np.clip(h_comb + nl, -H_CLIP, H_CLIP)

            # ── Dynamique Theta ──────────────────────────────────────
            is_spike = float(np.max(np.abs(env_field))) > 4.0
            gain     = adn["meta"]["env_gain_spike"] if is_spike else adn["meta"]["env_gain_calm"]
            theta   += np.random.randn(NX, NY) * 0.01 + h_comb * gain
            theta    = np.clip(theta, -THETA_CLIP, THETA_CLIP)

            Q_heat  += kB * T_env * LN2 * np.sum(np.abs(theta * 0.5)) * INFO_TO_BITS
            theta   *= 0.8

            if is_spike:
                if np.max(np.abs(theta)) < SEUIL_REACTION:
                    inaction += 1
                else:
                    lucidity += 1

            if collect:
                hs_trace.append(h_comb.copy())
                envs_trace.append(env_field.copy())

        trials.append({"luc": lucidity, "ina": inaction, "q": Q_heat})

    result = {
        "luc_avg": float(np.mean([t["luc"] for t in trials])),
        "q_avg":   float(np.mean([t["q"]   for t in trials])),
        "ina_avg": float(np.mean([t["ina"] for t in trials])),
    }

    if return_traces:
        result["hs"]   = np.array(hs_trace)    # (T, 16, 16)
        result["envs"] = np.array(envs_trace)  # (T, 16, 16)

    return result


# ============================================================
# ÉVOLUTION STANDALONE (test rapide)
# ============================================================

def run_evolution(pop_size=20, generations=10):
    population = [creer_individu_aleatoire() for _ in range(pop_size)]
    for gen in range(generations):
        evals = [eval_individu_raw(ind) for ind in population]
        fits  = np.array([e["luc_avg"] / (e["q_avg"] * 1e18 + 1) for e in evals])
        idx   = np.argsort(fits)[::-1]
        print(f"Gen {gen:02d} | Luc: {np.mean([e['luc_avg'] for e in evals]):.2f} "
              f"| Q: {np.mean([e['q_avg'] for e in evals]):.2e}")
        next_pop = [population[i] for i in idx[:4]]
        while len(next_pop) < pop_size:
            next_pop.append(mutate_individu(population[np.random.choice(idx[:10])]))
        population = next_pop
    np.savez_compressed("final_population_v8_13b.npz",
                        population=np.array(population, dtype=object))
    print("\n💾 V8.13b sauvegardée.")


if __name__ == "__main__":
    run_evolution()
