#!/usr/bin/env python3
"""
V8.31 — Bassin HOF + RIDGE fine scan + weights longs-lags x3 + NLOC réduit + alpha-trend save
"""

import numpy as np
import copy
import warnings
import os
from collections import Counter, defaultdict

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import simulation_V8_13_spatial_mix as engine

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===============================
# PARAMÈTRES (modifiable)
# ===============================
GENS = 200
POP = 40

PRED_LAGS = [110, 90, 70, 50, 30, 15, 5]
# sweet-spot fine scan
RIDGE_ALPHAS = [0.3, 0.1, 0.03, 0.01]

# resolution / bruit
VALID_STEPS = 1200          # ↑ plus long pour stabiliser R2 (suggested)
BASE_ADAPT_NOISE = 0.05
NOISE_INCREASE = 0.0008

# fitness weights
W_LUC = 0.13
W_PRED = 0.87
Q_SCALE = 1e19
COMPLEXITY_PENALTY = 0.15
DIVERSITY_BONUS = 0.055
EXPLORATION_REWARD = 0.25   # augmenté comme demandé

# mutation / exploration
BASE_MUT_RATE = 0.035       # mode raffinement
MUT_DECAY = 0.0005

# genetic meta bounds (same as before)
META_BOUNDS = {
    "kappa_gain":     (0.01, 2.0),
    "mutation_rate":  (0.01, 0.30),
    "env_gain_spike": (0.005, 0.20),
    "env_gain_calm":  (0.0001, 0.02),
}

# speed/resolution tradeoffs
NLOC = 40                  # réduit à 40 → plus rapide
ELITE_SIZE = 8             # top8 gardés en élite
HOF_CLONES = 3             # clonage HOF ×3
HOF_CLONE_MUT_SCALE = 0.1  # mutation très faible sur les clones

np.random.seed(42)

# Try to enable plotting
MATPLOTLIB_AVAILABLE = True
try:
    import matplotlib.pyplot as plt
except Exception:
    MATPLOTLIB_AVAILABLE = False

# ===============================
# SEED
# ===============================

def load_previous_population():
    for fname in ["final_population_v8_30.npz", "final_population_v8_29.npz", "final_population_v8_28.npz",
                  "final_population_v8_27.npz", "final_population_v8_26.npz"]:
        try:
            data = np.load(fname, allow_pickle=True)
            key = 'population' if 'population' in data.files else list(data.files)[0]
            pop = list(data[key])
            print(f"   ✅ Seed chargé : {fname} ({len(pop)} individus)")
            return pop
        except Exception:
            continue
    print("   ⚠️  Aucun seed → nouvelle pop")
    return [engine.creer_individu_aleatoire() for _ in range(POP)]

def create_population(pop_size):
    seed_pop = load_previous_population()
    return seed_pop[:pop_size]

# ===============================
# UTILITAIRES
# ===============================

def normalize(v):
    v = np.array(v)
    if v.max() == v.min():
        return np.zeros_like(v)
    return (v - v.min()) / (v.max() - v.min() + 1e-9)

def complexity_penalty(ind):
    tau_var = np.var(ind["tau_fast"]) + np.var(ind["tau_slow"])
    w_var = np.var(ind["w_fast_mat"])
    return (tau_var + w_var) * COMPLEXITY_PENALTY

def individual_distance(ind1, ind2):
    dist = 0.0
    dist += np.linalg.norm(ind1["tau_fast"] - ind2["tau_fast"])
    dist += np.linalg.norm(ind1["tau_slow"] - ind2["tau_slow"])
    dist += np.linalg.norm(ind1["w_fast_mat"] - ind2["w_fast_mat"])
    for k in ind1["meta"]:
        dist += abs(ind1["meta"][k] - ind2["meta"][k])
    return dist

def diversity_bonus(pop):
    dists = []
    for i in range(len(pop)):
        for j in range(i+1, len(pop)):
            dists.append(individual_distance(pop[i], pop[j]))
    return np.mean(dists) * DIVERSITY_BONUS if dists else 0.0

def crossover_meta(p1, p2):
    child = copy.deepcopy(p1)
    for k in p1["meta"]:
        alpha = np.random.uniform(0.3, 0.7)
        child["meta"][k] = alpha * p1["meta"][k] + (1 - alpha) * p2["meta"][k]
    return child

def adaptive_mutate(child, gen, mut_scale=1.0):
    """mut_scale allows applying much smaller mutation to HOF clones"""
    mut_rate = BASE_MUT_RATE * (1 - MUT_DECAY * gen) * mut_scale
    child["w_fast_mat"] = np.clip(child["w_fast_mat"] + np.random.randn(*child["w_fast_mat"].shape) * mut_rate, 0.0, 1.0)
    child["tau_fast"]   = np.clip(child["tau_fast"]   + np.random.randn(*child["tau_fast"].shape) * mut_rate * 0.4, 0.15, 0.40)
    child["tau_slow"]   = np.clip(child["tau_slow"]   + np.random.randn(*child["tau_slow"].shape) * mut_rate * 0.4, 0.70, 0.98)
    for k, (lo, hi) in META_BOUNDS.items():
        child["meta"][k] = float(np.clip(child["meta"][k] + np.random.randn() * mut_rate, lo, hi))
    return child

def clone_hof_copies(hof, n_copies, gen):
    clones = []
    for _ in range(n_copies):
        c = copy.deepcopy(hof)
        # apply tiny mutation to each clone
        c = adaptive_mutate(c, gen, mut_scale=HOF_CLONE_MUT_SCALE)
        clones.append(c)
    return clones

# ===============================
# PREDICTIVE R2 (avec tracking alpha winners)
# ===============================

def compute_predictive_r2(ind, gen):
    """
    Retourne (mean_r2, alpha_counter) :
    - mean_r2 : score agrégé pondéré
    - alpha_counter : Counter des alphas gagnants (par lag)
    """
    try:
        res = engine.eval_individu_raw(ind, steps=VALID_STEPS, return_traces=True)
        if "hs" not in res or "envs" not in res:
            print("   ❌ Pas de traces hs/envs")
            return 0.0, Counter()

        hs = np.array(res["hs"])
        envs = np.array(res["envs"])
        T = hs.shape[0]

        if T < max(PRED_LAGS) + 60:
            print(f"   ⚠️ Traces trop courtes ({T}) → R2 désactivé")
            return 0.0, Counter()

        hs = np.clip(hs, -100, 100)
        envs = np.clip(envs, -100, 100)

        r2s_all = []
        adapt_noise = BASE_ADAPT_NOISE + NOISE_INCREASE * gen
        alpha_counter = Counter()

        for lag in PRED_LAGS:
            X = hs[:-lag].reshape(T - lag, -1)
            Y = envs[lag:].reshape(T - lag, -1)

            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            Xn = scaler_x.fit_transform(X)
            Yn = scaler_y.fit_transform(Y)

            Ya = Y + np.random.normal(0, adapt_noise, Y.shape)
            Ya = scaler_y.fit_transform(Ya)

            best_r2_for_lag = -1.0
            best_alpha_for_lag = None

            for Yv in [Yn, Ya]:
                # train/test split
                Xtr, Xte, Ytr, Yte = train_test_split(Xn, Yv, test_size=0.3, random_state=0)
                nloc = min(NLOC, Ytr.shape[1])
                if nloc <= 0:
                    continue
                idx = np.random.choice(Ytr.shape[1], nloc, replace=False)

                # test each alpha, compute mean r2 across selected target dims
                for alpha in RIDGE_ALPHAS:
                    r2_local = []
                    for k in idx:
                        ytr = Ytr[:, k]
                        yte = Yte[:, k]
                        if np.var(ytr) < 1e-8:
                            continue
                        try:
                            m = Ridge(alpha=float(alpha), fit_intercept=True)
                            m.fit(Xtr, ytr)
                            yp = m.predict(Xte)
                            r2_local.append(max(-1.0, r2_score(yte, yp)))
                        except Exception:
                            continue
                    if r2_local:
                        mean_alpha_r2 = float(np.mean(r2_local))
                        # keep best across alphas and Yv
                        if mean_alpha_r2 > best_r2_for_lag:
                            best_r2_for_lag = mean_alpha_r2
                            best_alpha_for_lag = alpha

            if best_alpha_for_lag is not None:
                alpha_counter[best_alpha_for_lag] += 1
                r2s_all.append(best_r2_for_lag)
            else:
                r2s_all.append(0.0)

        # Weighted mean with longs (110 & 90) x3
        r2_array = np.nan_to_num(np.array(r2s_all), nan=0.0)
        weights = np.array([3 if lag in (110, 90) else 1 for lag in PRED_LAGS])
        mean_r2 = float(np.sum(r2_array * weights) / np.sum(weights))

        # exploration bonus if long lag significantly better than short average
        long_vals = [r for lag, r in zip(PRED_LAGS, r2s_all) if lag in (110, 90)]
        long_r2 = np.mean(long_vals) if long_vals else 0.0
        short_avg = np.mean([r for lag, r in zip(PRED_LAGS, r2s_all) if lag <= 30]) if r2s_all else 0.0
        exp_bonus = EXPLORATION_REWARD if long_r2 > short_avg * 1.12 else 0.0

        # debug
        print(f"   DEBUG → mean_r2={mean_r2:.4f} + exp={exp_bonus:.4f}  (best alpha per lag counted)")

        return mean_r2 + exp_bonus, alpha_counter

    except Exception as e:
        print(f"   ❌ Exception dans compute_predictive_r2 : {e}")
        return 0.0, Counter()

# ===============================
# EVOLUTION (Hall of Fame + cloning)
# ===============================

def run_evolution():
    print("🚀 LANCEMENT V8.31 — Bassin HOF + RIDGE fine scan + longs x3 + NLOC réduit")
    print(f"   PRED_LAGS={PRED_LAGS} | RIDGE_ALPHAS={RIDGE_ALPHAS} | NLOC={NLOC}\n")

    pop = create_population(POP)
    history = []
    global_alpha_counter = Counter()
    per_gen_alpha_counts = []

    hall_of_fame = None
    best_fit_hof = -np.inf

    for gen in range(GENS):
        evals = []
        r2_list = []
        success_r2 = 0
        gen_alpha_count = Counter()

        print(f"--- Gen {gen:02d} -----------------------------------")

        for ind in pop:
            res = engine.eval_individu_raw(ind)
            luc = res.get("luc_avg", 0.0)
            q = res.get("q_avg", 0.0)
            r2, alpha_counter = compute_predictive_r2(ind, gen)

            comp_pen = complexity_penalty(ind)
            evals.append((ind, luc, q, r2, comp_pen))
            r2_list.append(r2)
            gen_alpha_count.update(alpha_counter)
            if r2 > 0.01:
                success_r2 += 1

        per_gen_alpha_counts.append(gen_alpha_count)
        global_alpha_counter.update(gen_alpha_count)

        lucs = np.array([e[1] for e in evals])
        qs = np.array([e[2] for e in evals])
        r2s = np.array(r2_list)
        comps = np.array([e[4] for e in evals])

        luc_n = normalize(lucs)
        q_scaled = qs * Q_SCALE

        div_bonus = diversity_bonus([e[0] for e in evals])

        fitness = (W_LUC * (luc_n / (1 + q_scaled))) + (W_PRED * r2s) - comps + div_bonus

        max_fit_idx = np.argmax(fitness)
        if fitness[max_fit_idx] > best_fit_hof:
            best_fit_hof = fitness[max_fit_idx]
            hall_of_fame = copy.deepcopy(evals[max_fit_idx][0])
            print(f"   🏅 Hall of Fame mis à jour ! BestFit = {best_fit_hof:.4f}")

        order = np.argsort(fitness)[::-1]
        best_fit = fitness[order[0]]

        print(f"BestFit {best_fit:.4f} | Luc_n {luc_n.mean():.3f} | "
              f"R2_avg {r2s.mean():.4f} (max={r2s.max():.4f}) | "
              f"Success R2>0.01: {success_r2}/{POP} | "
              f"DivBonus {div_bonus:.4f} | CompPen {comps.mean():.4f} | Q {qs.mean():.2e}")

        # save top-5 régulièrement
        if gen % 6 == 0 and gen > 0:
            top5 = [evals[i][0] for i in order[:5]]
            np.savez(f"top5_gen_{gen:02d}.npz", top5=top5)
            print(f"   💾 Top-5 sauvegardés : top5_gen_{gen:02d}.npz")

        # immigration
        if gen % 18 == 0 and gen > 0:
            immigrants = [engine.creer_individu_aleatoire() for _ in range(5)]
            for i in range(5):
                pop[order[-1-i]] = copy.deepcopy(immigrants[i])
            print(f"   🌍 Immigration : 5 nouveaux individus injectés")

        # SELECTION TOURNAMENT
        survivors = []
        for _ in range(POP // 2):
            candidates = np.random.choice(range(POP), 4, replace=False)
            winner = candidates[np.argmax(fitness[candidates])]
            survivors.append(copy.deepcopy(evals[winner][0]))

        # ELITISM: top ELITE_SIZE carried directly (to maintain good genes)
        newpop = [copy.deepcopy(evals[i][0]) for i in order[:ELITE_SIZE]]

        # Add HOF and HOF clones into the new population BEFORE filling the rest
        if hall_of_fame is not None:
            # add one exact HOF (kept as-is)
            newpop.append(copy.deepcopy(hall_of_fame))
            # add HOF clones with tiny mutation to explore local basin
            hof_clones = clone_hof_copies(hall_of_fame, HOF_CLONES, gen)
            for c in hof_clones:
                newpop.append(c)
            print(f"   ♻️ HOF cloné ×{HOF_CLONES} (bassin local)")

        # Fill remaining slots with crossover / survivors + small mutation
        while len(newpop) < POP:
            if np.random.rand() < 0.20:
                p1 = survivors[np.random.randint(len(survivors))]
                p2 = survivors[np.random.randint(len(survivors))]
                child = copy.deepcopy(p1)
                alpha = np.random.uniform(0.3, 0.7)
                child["tau_fast"] = alpha * p1["tau_fast"] + (1-alpha) * p2["tau_fast"]
                child["tau_slow"] = alpha * p1["tau_slow"] + (1-alpha) * p2["tau_slow"]
                child["w_fast_mat"] = alpha * p1["w_fast_mat"] + (1-alpha) * p2["w_fast_mat"]
                child = crossover_meta(child, p2)
            else:
                child = copy.deepcopy(survivors[np.random.randint(len(survivors))])

            child = adaptive_mutate(child, gen)
            newpop.append(child)

        # trim just in case (should be exactly POP)
        pop = newpop[:POP]

        # save per-gen summary to history (light)
        history.append({
            "gen": gen,
            "best_fit": float(best_fit),
            "r2_avg": float(r2s.mean()),
            "r2_max": float(r2s.max()),
            "alpha_counts": dict(gen_alpha_count)
        })

    # end for gens
    np.savez("final_population_v8_31.npz", population=pop, history=history)
    print("\n✅ V8.31 TERMINÉ")
    print("   Fichier final : final_population_v8_31.npz")

    # ===== alpha trend save / plot =====
    # aggregate per generation counts into arrays
    alphalist = sorted(set([a for genc in per_gen_alpha_counts for a in genc.keys()]))
    gen_indices = [h["gen"] for h in history]
    alpha_matrix = {a: [] for a in alphalist}
    for genc in per_gen_alpha_counts:
        for a in alphalist:
            alpha_matrix[a].append(genc.get(a, 0))

    # try plotting if matplotlib available
    if MATPLOTLIB_AVAILABLE:
        try:
            plt.figure(figsize=(8, 4))
            for a in alphalist:
                plt.plot(gen_indices, alpha_matrix[a], label=str(a))
            plt.xlabel("Generation")
            plt.ylabel("alpha win count (per gen)")
            plt.title("Alpha trend per generation (V8.31)")
            plt.legend(title="alpha")
            plt.tight_layout()
            plt.savefig("alpha_trend_v8_31.png", dpi=150)
            plt.close()
            print("   💾 alpha_trend_v8_31.png sauvegardé")
        except Exception as e:
            print("   ⚠️ Erreur plot matplotlib :", e)
            # fallback to txt
            with open("alpha_trend_v8_31.txt", "w") as f:
                f.write("gen\t" + "\t".join(map(str, alphalist)) + "\n")
                for i, g in enumerate(gen_indices):
                    row = [str(alpha_matrix[a][i]) for a in alphalist]
                    f.write(f"{g}\t" + "\t".join(row) + "\n")
            print("   💾 alpha_trend_v8_31.txt sauvegardé (fallback)")
    else:
        # matplotlib not available -> write txt
        with open("alpha_trend_v8_31.txt", "w") as f:
            f.write("gen\t" + "\t".join(map(str, alphalist)) + "\n")
            for i, g in enumerate(gen_indices):
                row = [str(alpha_matrix[a][i]) for a in alphalist]
                f.write(f"{g}\t" + "\t".join(row) + "\n")
        print("   💾 alpha_trend_v8_31.txt sauvegardé (matplotlib absent)")

if __name__ == "__main__":
    run_evolution()