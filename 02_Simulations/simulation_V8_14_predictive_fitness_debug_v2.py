#!/usr/bin/env python3

"""
V8.14 — Predictive Fitness Evolution (version finale propre)
Compatible avec simulation_V8_13_spatial_mix.py
Fitness = Lucidité efficace (40%) + Bonus Prédictif R² (60%)
"""

import numpy as np
import copy
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import simulation_V8_13_spatial_mix as engine


# ===============================
# PARAMÈTRES
# ===============================

GENS = 60
POP = 40

PRED_LAG = 12          # ← augmenté (meilleure prédiction à long terme)
VALID_STEPS = 200      # ← plus de données pour Ridge

W_LUC = 0.40
W_PRED = 0.60
Q_SCALE = 1e19

np.random.seed(42)


# ===============================
# CRÉATION POPULATION
# ===============================

def create_population(pop_size):
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


# ===============================
# CALCUL R2 PRÉDICTIF (version propre)
# ===============================

def compute_predictive_r2(ind):
    try:
        res = engine.eval_individu_raw(
            ind,
            steps=VALID_STEPS,
            return_traces=True
        )

        if "hs" not in res or "envs" not in res:
            return 0.0

        hs = np.array(res["hs"])
        envs = np.array(res["envs"])

        T, nx, ny = hs.shape
        if T < PRED_LAG + 30:
            return 0.0

        X = hs[:-PRED_LAG].reshape(T - PRED_LAG, -1)
        Y = envs[PRED_LAG:].reshape(T - PRED_LAG, -1)

        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=0)

        nloc = min(100, Ytr.shape[1])
        idx = np.random.choice(Ytr.shape[1], nloc, replace=False)

        r2s = []
        for k in idx:
            ytr = Ytr[:, k]
            yte = Yte[:, k]
            if np.var(ytr) < 1e-8:
                continue
            m = Ridge(alpha=1.0)
            m.fit(Xtr, ytr)
            yp = m.predict(Xte)
            r2s.append(max(-1.0, r2_score(yte, yp)))

        return float(np.mean(r2s)) if r2s else 0.0

    except Exception:
        return 0.0


# ===============================
# EVOLUTION
# ===============================

def run_evolution():
    print("🚀 LANCEMENT V8.14 — Predictive Fitness (version finale)")
    print(f"   PRED_LAG={PRED_LAG} | VALID_STEPS={VALID_STEPS} | POP={POP}\n")

    pop = create_population(POP)
    history = []

    for gen in range(GENS):
        evals = []
        r2_list = []
        success_r2 = 0

        print(f"--- Gen {gen:02d} -----------------------------------")

        for ind in pop:
            # Éval normale (luc + q)
            res = engine.eval_individu_raw(ind)
            luc = res["luc_avg"]
            q = res["q_avg"]

            # Éval prédictive
            r2 = compute_predictive_r2(ind)

            evals.append((ind, luc, q, r2))
            r2_list.append(r2)
            if r2 > 0.01:
                success_r2 += 1

        # Calcul fitness
        lucs = np.array([e[1] for e in evals])
        qs = np.array([e[2] for e in evals])
        r2s = np.array(r2_list)

        luc_n = normalize(lucs)
        r2_n = normalize(r2s)
        q_scaled = qs * Q_SCALE

        fitness = W_LUC * (luc_n / (1 + q_scaled)) + W_PRED * r2_n

        order = np.argsort(fitness)[::-1]
        best_fit = fitness[order[0]]

        print(f"BestFit {best_fit:.4f} | "
              f"Luc_n {luc_n.mean():.3f} | "
              f"R2_avg {r2s.mean():.4f} (max={r2s.max():.4f}) | "
              f"Success R2>0.01: {success_r2}/{POP} | "
              f"Q {qs.mean():.2e}")

        # Sélection + mutation
        survivors = [copy.deepcopy(evals[i][0]) for i in order[:POP//2]]
        newpop = survivors.copy()

        while len(newpop) < POP:
            p = copy.deepcopy(survivors[np.random.randint(len(survivors))])
            if hasattr(engine, "mutate_individu"):
                p = engine.mutate_individu(p)
            elif hasattr(engine, "mutate"):
                p = engine.mutate(p)
            newpop.append(p)

        pop = newpop

        history.append({
            "gen": gen,
            "luc": float(lucs.mean()),
            "r2": float(r2s.mean()),
            "r2_max": float(r2s.max()),
            "q": float(qs.mean()),
            "best_fit": float(best_fit)
        })

    # Sauvegarde finale
    np.savez(
        "final_population_v8_14.npz",
        population=pop,
        history=history
    )

    print("\n✅ V8.14 TERMINÉ")
    print("   Fichier sauvegardé : final_population_v8_14.npz")
    print(f"   Meilleur R2 final = {r2s.max():.4f}")


if __name__ == "__main__":
    run_evolution()