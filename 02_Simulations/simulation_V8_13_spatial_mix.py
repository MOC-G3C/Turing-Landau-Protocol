import numpy as np
import copy
import math

# ============================================================
# 1. CONSTANTES V8.13 (SPATIAL MIXING)
# ============================================================
np.random.seed(42)
kB, T_env, LN2 = 1.38e-23, 300.0, math.log(2.0)
SPIKE_PROB_FAST, SPIKE_PROB_SLOW = 0.12, 0.02
INFO_TO_BITS, Q_MAX_LANDAUER = 0.2, 8e-19
SEUIL_REACTION, MAX_INACTION = 0.08, 3

META_BOUNDS = {
    "kappa_gain":     (0.01, 2.0),
    "mutation_rate":  (0.01, 0.30),
    "env_gain_spike": (0.005, 0.20),
    "env_gain_calm":  (0.0001, 0.02),
}

HOTSPOTS = [(3, 3, 10.0, 0.0), (12, 12, 8.0, np.pi/2)]
XS, YS = np.arange(16)[:, None], np.arange(16)[None, :]

# Seuils anti-divergence (NaN dans Ridge)
THETA_CLIP = 10.0
H_CLIP     = 50.0

# ============================================================
# 2. FONCTIONS
# ============================================================

def creer_individu_aleatoire():
    return {
        "theta":      np.random.randn(16, 16) * 0.1,
        "tau_fast":   np.random.uniform(0.15, 0.40, (16, 16)),
        "tau_slow":   np.random.uniform(0.70, 0.98, (16, 16)),
        "w_fast_mat": np.random.uniform(0.2,  0.8,  (16, 16)),
        "meta":       {k: np.random.uniform(v[0], v[1]) for k, v in META_BOUNDS.items()}
    }

def mutate_individu(ind):
    child = copy.deepcopy(ind)
    child["w_fast_mat"] = np.clip(child["w_fast_mat"] + np.random.randn(16,16)*0.03, 0.0, 1.0)
    child["tau_fast"]   = np.clip(child["tau_fast"]   + np.random.randn(16,16)*0.02, 0.15, 0.40)
    child["tau_slow"]   = np.clip(child["tau_slow"]   + np.random.randn(16,16)*0.02, 0.70, 0.98)
    for k, (lo, hi) in META_BOUNDS.items():
        child["meta"][k] = float(np.clip(
            child["meta"][k] * (1.0 + np.random.randn() * 0.07), lo, hi
        ))
    return child

def eval_individu_raw(adn, steps=200, return_traces=False):
    trials     = []
    hs_trace   = []
    envs_trace = []

    for trial_idx in range(3):
        theta = adn["theta"].copy()
        h_f   = np.zeros((16, 16))
        h_s   = np.zeros((16, 16))
        Q_heat, lucidity, inaction = 0.0, 0.0, 0.0
        collect = return_traces and (trial_idx == 0)

        for t in range(1, steps + 1):
            env_field = np.zeros((16, 16))
            f_comp = (1.0 if np.random.rand() < SPIKE_PROB_FAST else 0.1) * np.sin(t / 8.0)
            s_comp = (1.0 if np.random.rand() < SPIKE_PROB_SLOW else 0.1) * np.sin(t / 200.0)
            for (x, y, amp, ph) in HOTSPOTS:
                dist2 = (XS - x)**2 + (YS - y)**2
                env_field += amp * np.exp(-dist2 / 9.0) * (
                    f_comp + s_comp + np.sin(t / 10.0 + ph) * 0.2
                )

            h_f = adn["tau_fast"] * h_f + (1.0 - adn["tau_fast"]) * env_field
            h_s = adn["tau_slow"] * h_s + (1.0 - adn["tau_slow"]) * env_field
            h_comb = adn["w_fast_mat"] * h_f + (1.0 - adn["w_fast_mat"]) * h_s

            # FIX anti-NaN : clipping h_comb et theta
            h_comb = np.clip(h_comb, -H_CLIP, H_CLIP)

            is_spike = float(np.max(np.abs(env_field))) > 4.0
            gain     = adn["meta"]["env_gain_spike"] if is_spike else adn["meta"]["env_gain_calm"]
            theta   += np.random.randn(16, 16) * 0.01 + h_comb * gain
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
        result["hs"]   = np.array(hs_trace)
        result["envs"] = np.array(envs_trace)

    return result

# ============================================================
# 3. ÉVOLUTION STANDALONE
# ============================================================

def run_evolution(pop_size=40, generations=20):
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
    np.savez_compressed("final_population_v8_13.npz",
                        population=np.array(population, dtype=object))
    print("\n💾 V8.13 sauvegardée.")

if __name__ == "__main__":
    run_evolution()
