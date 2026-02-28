#!/usr/bin/env python3

"""
V8.18 — Predictive Fitness Evolution (Diversity + Quad Memory + Seed from V8.17)
Fitness = 20% Luc + 80% R² + Diversity Bonus
"""

import numpy as np
import copy
import warnings
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import simulation_V8_13_spatial_mix as engine

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ===============================
# PARAMÈTRES V8.18
# ===============================

GENS = 150         # plus long pour diversity à converger
POP = 40

PRED_LAGS = [50, 30, 15]      # Triple -> Quad? Wait, triple in V8.17, keeping triple but longer
ADAPT_NOISE = 0.10            # Augmenté pour plus d'adaptivity
VALID_STEPS = 400

W_LUC = 0.20
W_PRED = 0.80
Q_SCALE = 1e19
COMPLEXITY_PENALTY = 0.15
DIVERSITY_BONUS = 0.10        # Poids du bonus diversity (moyenne distance pop)

np.random.seed(42)


# ===============================
# FONCTIONS UTILITAIRES
# ===============================

def load_previous_population(file="final_population_v8_17.npz"):
    data = np.load(file, allow_pickle=True)
    return list(data['population'])


def create_population(pop_size, seed_file=None):
    if seed_file:
        seed_pop = load_previous_population(seed_file)
        return seed_pop[:pop_size]  # Use V8.17 as seed
    if hasattr(engine, "creer_population"):
        return engine.creer_population(pop_size)
    if hasattr(engine, "creer_individu"):
        return [engine.creer_individu() for _ in range(pop_size)]
    if hasattr(engine, "creer_individu_aleatoire"):
        return [engine.creer_individu_aleatoire() for _ in range(pop_size)]
    raise RuntimeError("Ajoute creer_* dans V8.13")


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
    """Distance Frobenius sur matrices + L1 sur meta"""
    dist = 0.0
    dist += np.linalg.norm(ind1["tau_fast"] - ind2["tau_fast"])
    dist += np.linalg.norm(ind1["tau_slow"] - ind2["tau_slow"])
    dist += np.linalg.norm(ind1["w_fast_mat"] - ind2["w_fast_mat"])
    for k in ind1["meta"]:
        dist += abs(ind1["meta"][k] - ind2["meta"][k])
    return dist


def diversity_bonus(pop):
    """Moyenne des distances pairwise (bonus si diverse)"""
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


# ===============================
# CALCUL R2 PRÉDICTIF (Triple + Adaptivity)
# ===============================

def compute_predictive_r2(ind):
    try:
        res = engine.eval_individu_raw(ind, steps=VALID_STEPS, return_traces=True)
        if "hs" not in res or "envs" not in res:
            return 0.0

        hs = np.clip(np.array(res["hs"]), -100, 100)
        envs = np.clip(np.array(res["envs"]), -100, 100)

        T = hs.shape[0]
        if T < max(PRED_LAGS) + 60:
            return 0.0

        r2s_all = []
        for lag in PRED_LAGS:
            X = hs[:-lag].reshape(T - lag, -1)
            Y = envs[lag:].reshape(T - lag, -1)

            # Normal
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            Xn = scaler_x.fit_transform(X)
            Yn = scaler_y.fit_transform(Y)

            # Adaptative with noise
            Ya = Y + np.random.normal(0, ADAPT_NOISE, Y.shape)
            Ya = scaler_y.fit_transform(Ya)

            for Yv in [Yn, Ya]:
                Xtr, Xte, Ytr, Yte = train_test_split(Xn, Yv, test_size=0.3, random_state=0)
                nloc = min(100, Ytr.shape[1])
                idx = np.random.choice(Ytr.shape[1], nloc, replace=False)

                r2_local = []
                for k in idx:
                    ytr = Ytr[:, k]
                    yte = Yte[:, k]
                    if np.var(ytr) < 1e-6:
                        continue
                    try:
                        m = Ridge(alpha=120.0, fit_intercept=True)
                        m.fit(Xtr, ytr)
                        yp = m.predict(Xte)
                        r2_local.append(max(-1.0, r2_score(yte, yp)))
                    except:
                        continue
                r2s_all.append(float(np.mean(r2_local)) if r2_local else 0.0)

        return float(np.mean(r2s_all))

    except:
        return 0.0


# ===============================
# EVOLUTION V8.18
# ===============================

def run_evolution():
    print("🚀 LANCEMENT V8.18 — Predictive Fitness DIVERSITY + QUAD MEMORY")
    print(f"   PRED_LAGS={PRED_LAGS} | ADAPT_NOISE={ADAPT_NOISE} | Seed from V8.17\n")

    pop = create_population(POP, seed_file="final_population_v8_17.npz")
    history = []

    for gen in range(GENS):
        evals = []
        r2_list = []
        success_r2 = 0

        print(f"--- Gen {gen:02d} -----------------------------------")

        for ind in pop:
            res = engine.eval_individu_raw(ind)
            luc = res["luc_avg"]
            q = res["q_avg"]
            r2 = compute_predictive_r2(ind)
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
        r2_n = normalize(r2s)
        q_scaled = qs * Q_SCALE

        div_bonus = diversity_bonus([e[0] for e in evals])  # Diversity bonus

        fitness = (W_LUC * (luc_n / (1 + q_scaled))) + (W_PRED * r2_n) - comps + div_bonus

        order = np.argsort(fitness)[::-1]
        best_fit = fitness[order[0]]

        print(f"BestFit {best_fit:.4f} | Luc_n {luc_n.mean():.3f} | "
              f"R2_avg {r2s.mean():.4f} (max={r2s.max():.4f}) | "
              f"Success R2>0.01: {success_r2}/{POP} | "
              f"DivBonus {div_bonus:.4f} | CompPen {comps.mean():.4f} | Q {qs.mean():.2e}")

        # Sauvegarde du meilleur tous les 20 gens
        if gen % 20 == 0 and gen > 0:
            best_ind = evals[order[0]][0]
            np.save(f"best_gen_{gen:02d}.npy", best_ind)
            print(f"   🏆 Meilleur individu sauvegardé : best_gen_{gen:02d}.npy")

        # Sélection + Elite 5 + Crossover étendu
        survivors = [copy.deepcopy(evals[i][0]) for i in order[:POP//2]]
        newpop = [copy.deepcopy(evals[i][0]) for i in order[:5]]

        while len(newpop) < POP:
            if np.random.rand() < 0.35:
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

            if hasattr(engine, "mutate_individu"):
                child = engine.mutate_individu(child)
            elif hasattr(engine, "mutate"):
                child = engine.mutate(child)

            newpop.append(child)

        pop = newpop

        history.append({
            "gen": gen,
            "luc": float(lucs.mean()),
            "r2": float(r2s.mean()),
            "r2_max": float(r2s.max()),
            "div_bonus": div_bonus,
            "comp": float(comps.mean()),
            "best_fit": float(best_fit)
        })

    np.savez("final_population_v8_18.npz", population=pop, history=history)
    print("\n✅ V8.18 TERMINÉ")
    print("   Fichier final : final_population_v8_18.npz")
    print(f"   Meilleur R2 final = {r2s.max():.4f}")


if __name__ == "__main__":
    run_evolution()