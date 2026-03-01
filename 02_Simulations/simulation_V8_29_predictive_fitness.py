#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V8.29 — RIDGE ALPHA EXTENDED SCAN + ALPHA USAGE TRACKING
Remplace simulation_V8_28_predictive_fitness.py par ce fichier.
"""

import numpy as np
import copy
import warnings
import time
from collections import Counter, defaultdict

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import simulation_V8_13_spatial_mix as engine

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===============================
# PARAMÈTRES
# ===============================

GENS = 200
POP = 40

PRED_LAGS = [110, 90, 70, 50, 30, 15, 5]
BASE_ADAPT_NOISE = 0.05
NOISE_INCREASE = 0.0008
VALID_STEPS = 850

W_LUC = 0.13
W_PRED = 0.87
Q_SCALE = 1e19
COMPLEXITY_PENALTY = 0.15
DIVERSITY_BONUS = 0.055
EXPLORATION_REWARD = 0.18

# --- Nouveauté V8.29 : scan plus fin
RIDGE_ALPHAS = [30, 10, 3, 1, 0.3, 0.1]

BASE_MUT_RATE = 0.05
MUT_DECAY = 0.0005

META_BOUNDS = {
    "kappa_gain":     (0.01, 2.0),
    "mutation_rate":  (0.01, 0.30),
    "env_gain_spike": (0.005, 0.20),
    "env_gain_calm":  (0.0001, 0.02),
}

SAVE_TOP5_EVERY = 6
IMMIGRATION_EVERY = 18
IMMIGRANTS_COUNT = 5
ALPHA_SUMMARY_SAVE_EVERY = 10

np.random.seed(42)


# ===============================
# UTIL / SEED
# ===============================

def load_previous_population():
    for fname in ["final_population_v8_28.npz", "final_population_v8_27.npz", "final_population_v8_26.npz"]:
        try:
            data = np.load(fname, allow_pickle=True)
            if "population" in data:
                pop = list(data["population"])
                print(f"   ✅ Seed chargé : {fname} ({len(pop)} individus)")
                return pop
        except Exception:
            continue
    print("   ⚠️  Aucun seed trouvé → création nouvelle population")
    return [engine.creer_individu_aleatoire() for _ in range(POP)]

def create_population(pop_size):
    seed_pop = load_previous_population()
    return seed_pop[:pop_size]

def normalize(v):
    v = np.array(v)
    if v.max() == v.min():
        return np.zeros_like(v, dtype=float)
    return (v - v.min()) / (v.max() - v.min() + 1e-12)


# ===============================
# MESURES & MUTATION
# ===============================

def complexity_penalty(ind):
    try:
        tau_var = np.var(ind["tau_fast"]) + np.var(ind["tau_slow"])
        w_var = np.var(ind["w_fast_mat"])
        return (tau_var + w_var) * COMPLEXITY_PENALTY
    except Exception:
        return 0.0

def individual_distance(ind1, ind2):
    dist = 0.0
    dist += np.linalg.norm(ind1["tau_fast"] - ind2["tau_fast"])
    dist += np.linalg.norm(ind1["tau_slow"] - ind2["tau_slow"])
    dist += np.linalg.norm(ind1["w_fast_mat"] - ind2["w_fast_mat"])
    for k in ind1.get("meta", {}):
        dist += abs(ind1["meta"].get(k, 0.0) - ind2["meta"].get(k, 0.0))
    return dist

def diversity_bonus(pop):
    dists = []
    for i in range(len(pop)):
        for j in range(i+1, len(pop)):
            dists.append(individual_distance(pop[i], pop[j]))
    return np.mean(dists) * DIVERSITY_BONUS if dists else 0.0

def crossover_meta(p1, p2):
    child = copy.deepcopy(p1)
    for k in p1.get("meta", {}):
        alpha = np.random.uniform(0.3, 0.7)
        child["meta"][k] = alpha * p1["meta"][k] + (1 - alpha) * p2["meta"][k]
    return child

def adaptive_mutate(child, gen):
    mut_rate = max(1e-4, BASE_MUT_RATE * (1 - MUT_DECAY * gen))
    # assume shapes known (16x16 in previous versions)
    child["w_fast_mat"] = np.clip(child["w_fast_mat"] + np.random.randn(*child["w_fast_mat"].shape) * mut_rate, 0.0, 1.0)
    child["tau_fast"]   = np.clip(child["tau_fast"]   + np.random.randn(*child["tau_fast"].shape) * mut_rate * 0.4, 0.15, 0.40)
    child["tau_slow"]   = np.clip(child["tau_slow"]   + np.random.randn(*child["tau_slow"].shape) * mut_rate * 0.4, 0.70, 0.98)
    for k, (lo, hi) in META_BOUNDS.items():
        child["meta"][k] = float(np.clip(child["meta"].get(k, (lo+hi)/2) + np.random.randn() * mut_rate, lo, hi))
    return child


# ===============================
# PREDICTION R2 + ALPHA TRACKING
# ===============================

def compute_predictive_r2(ind, gen, alpha_usage, alpha_usage_per_gen, debug=False):
    """
    Evaluer R2 prédictif en scannant RIDGE_ALPHAS. 
    Met à jour alpha_usage (Counter) et alpha_usage_per_gen (defaultdict) pour diagnostics.
    """
    try:
        res = engine.eval_individu_raw(ind, steps=VALID_STEPS, return_traces=True)
        if "hs" not in res or "envs" not in res:
            if debug: print("   ❌ Pas de traces hs/envs")
            return 0.0

        hs = np.array(res["hs"])
        envs = np.array(res["envs"])
        T = hs.shape[0]

        if T < max(PRED_LAGS) + 60:
            if debug: print(f"   ⚠️ Traces trop courtes ({T}) → R2 désactivé")
            return 0.0

        hs = np.clip(hs, -100, 100)
        envs = np.clip(envs, -100, 100)

        r2s_all = []
        per_lag_best_alpha = []
        adapt_noise = BASE_ADAPT_NOISE + NOISE_INCREASE * gen

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

            # Test both normal and noisy target variants
            for Yv in (Yn, Ya):
                Xtr, Xte, Ytr, Yte = train_test_split(Xn, Yv, test_size=0.3, random_state=0)
                # choose subset of output channels to speed: keep min(80, n_out)
                nloc = min(80, Ytr.shape[1])
                idx = np.random.choice(Ytr.shape[1], nloc, replace=False)

                # scan alphas
                for alpha in RIDGE_ALPHAS:
                    r2_local = []
                    for k in idx:
                        ytr = Ytr[:, k]
                        yte = Yte[:, k]
                        if np.var(ytr) < 1e-8:
                            continue
                        try:
                            m = Ridge(alpha=alpha, fit_intercept=True)
                            m.fit(Xtr, ytr)
                            yp = m.predict(Xte)
                            r2_local.append(max(-1.0, r2_score(yte, yp)))
                        except Exception:
                            continue
                    if r2_local:
                        mean_r2_alpha = float(np.mean(r2_local))
                        # si ce alpha est meilleur pour ce lag, on le garde
                        if mean_r2_alpha > best_r2_for_lag:
                            best_r2_for_lag = mean_r2_alpha
                            best_alpha_for_lag = alpha

            # Enregistrer alpha choisi pour ce lag
            if best_alpha_for_lag is not None:
                alpha_usage[best_alpha_for_lag] += 1
                alpha_usage_per_gen[best_alpha_for_lag] += 1
                per_lag_best_alpha.append(best_alpha_for_lag)
            else:
                per_lag_best_alpha.append(None)

            r2s_all.append(best_r2_for_lag if best_r2_for_lag > -1 else 0.0)

        # Weighted mean (long lags pondérés x2)
        r2_array = np.nan_to_num(np.array(r2s_all), nan=0.0)
        weights = np.array([2 if lag >= 90 else 1 for lag in PRED_LAGS])
        mean_r2 = float(np.sum(r2_array * weights) / np.sum(weights))

        long_r2 = r2s_all[0] if len(r2s_all) > 0 else 0.0
        short_avg = np.mean(r2s_all[-3:]) if len(r2s_all) >= 3 else 0.0
        exp_bonus = EXPLORATION_REWARD if long_r2 > short_avg * 1.12 else 0.0

        if debug:
            print(f"   DEBUG → mean_r2={mean_r2:.4f} + exp={exp_bonus:.4f}  (alphas per lag: {per_lag_best_alpha})")

        # On retourne aussi la liste des alphas choisis pour ce calcul si besoin (non utilisé en fitness)
        return mean_r2 + exp_bonus

    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        print(f"   ❌ Exception compute_predictive_r2 : {e}")
        return 0.0


# ===============================
# EVOLUTION (Hall of Fame + Tournament)
# ===============================

def run_evolution():
    print("🚀 LANCEMENT V8.29 — Ridge Alpha extended scan + alpha usage tracking")
    print(f"   PRED_LAGS={PRED_LAGS} | RIDGE_ALPHAS={RIDGE_ALPHAS}\n")

    pop = create_population(POP)
    history = []

    # global alpha usage counters (persist across gens)
    alpha_usage = Counter()
    # alpha usage per generation (reset each gen, then aggregated)
    alpha_usage_per_gen = None

    hall_of_fame = None
    best_fit_hof = -np.inf

    for gen in range(GENS):
        gen_start = time.time()
        evals = []
        r2_list = []
        success_r2 = 0
        alpha_usage_per_gen = Counter()

        print(f"--- Gen {gen:02d} -----------------------------------")

        for ind_idx, ind in enumerate(pop):
            # engine eval rapide (sans traces) pour luc/q (plus rapide) ; compute_predictive_r2 fait traces détaillées
            try:
                res = engine.eval_individu_raw(ind)
            except Exception as e:
                print(f"   ❌ engine.eval_individu_raw failed for ind {ind_idx} : {e}")
                res = {"luc_avg": 0.0, "q_avg": 0.0}

            luc = res.get("luc_avg", 0.0)
            q = res.get("q_avg", 0.0)

            # compute predictive r2 and update alpha_usage counters
            r2 = compute_predictive_r2(ind, gen, alpha_usage, alpha_usage_per_gen, debug=False)

            comp_pen = complexity_penalty(ind)
            evals.append((ind, luc, q, r2, comp_pen))
            r2_list.append(r2)
            if r2 > 0.01:
                success_r2 += 1

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

        # résumé alpha usage de la génération
        gen_alpha_summary = alpha_usage_per_gen.most_common()
        if gen_alpha_summary:
            top_alpha, top_count = gen_alpha_summary[0]
        else:
            top_alpha, top_count = (None, 0)

        print(f"BestFit {best_fit:.4f} | Luc_n {luc_n.mean():.3f} | "
              f"R2_avg {r2s.mean():.4f} (max={r2s.max():.4f}) | "
              f"Success R2>0.01: {success_r2}/{POP} | "
              f"DivBonus {div_bonus:.4f} | CompPen {comps.mean():.4f} | Q {qs.mean():.2e}")
        print(f"   → α winner gen {gen:02d}: {top_alpha} (count {top_count}) | α cumulatif top3: {alpha_usage.most_common(3)}")

        # save top5 occasionally
        if gen % SAVE_TOP5_EVERY == 0 and gen > 0:
            top5 = [evals[i][0] for i in order[:5]]
            np.savez(f"top5_gen_{gen:02d}.npz", top5=top5)
            print(f"   💾 Top-5 sauvegardés : top5_gen_{gen:02d}.npz")

        # save alpha usage summary occasionally
        if gen % ALPHA_SUMMARY_SAVE_EVERY == 0 and gen > 0:
            # save cumulative alpha usage and per-gen snapshot
            np.savez(f"alpha_usage_summary_gen_{gen:02d}.npz",
                     alpha_usage=dict(alpha_usage),
                     alpha_usage_this_gen=dict(alpha_usage_per_gen))
            print(f"   💾 Alpha usage résumé sauvegardé : alpha_usage_summary_gen_{gen:02d}.npz")

        # immigration
        if gen % IMMIGRATION_EVERY == 0 and gen > 0:
            immigrants = [engine.creer_individu_aleatoire() for _ in range(IMMIGRANTS_COUNT)]
            for i in range(IMMIGRANTS_COUNT):
                pop[order[-1 - i]] = copy.deepcopy(immigrants[i])
            print(f"   🌍 Immigration : {IMMIGRANTS_COUNT} nouveaux individus injectés")

        # tournament selection (4-way) and reproduction
        survivors = []
        for _ in range(POP // 2):
            candidates = np.random.choice(range(POP), 4, replace=False)
            winner = candidates[np.argmax(fitness[candidates])]
            survivors.append(copy.deepcopy(evals[winner][0]))

        # elitism: keep top 5, add hall_of_fame
        newpop = [copy.deepcopy(evals[i][0]) for i in order[:5]]
        if hall_of_fame is not None:
            newpop.append(copy.deepcopy(hall_of_fame))

        # fill rest by crossover / mutation
        while len(newpop) < POP:
            if np.random.rand() < 0.20 and len(survivors) >= 2:
                p1 = survivors[np.random.randint(len(survivors))]
                p2 = survivors[np.random.randint(len(survivors))]
                child = copy.deepcopy(p1)
                alpha = np.random.uniform(0.3, 0.7)
                child["tau_fast"] = alpha * p1["tau_fast"] + (1 - alpha) * p2["tau_fast"]
                child["tau_slow"] = alpha * p1["tau_slow"] + (1 - alpha) * p2["tau_slow"]
                child["w_fast_mat"] = alpha * p1["w_fast_mat"] + (1 - alpha) * p2["w_fast_mat"]
                child = crossover_meta(child, p2)
            else:
                child = copy.deepcopy(survivors[np.random.randint(len(survivors))])

            child = adaptive_mutate(child, gen)
            newpop.append(child)

        pop = newpop
        gen_time = time.time() - gen_start
        print(f"   gen {gen:02d} done in {gen_time:.1f}s\n")

    # final saves
    np.savez("final_population_v8_29.npz", population=pop, history=history)
    np.savez("alpha_usage_final_v8_29.npz", alpha_usage=dict(alpha_usage))
    print("\n✅ V8.29 TERMINÉ")
    print("   Fichier final : final_population_v8_29.npz")
    print("   Alpha usage final (top5):", alpha_usage.most_common(5))


if __name__ == "__main__":
    try:
        run_evolution()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving partial state...")
        # attempt a graceful save (if variables exist)
        try:
            np.savez("final_population_v8_29_partial.npz")
            print("Partial state saved.")
        except Exception:
            pass
        raise