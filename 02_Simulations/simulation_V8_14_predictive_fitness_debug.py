#!/usr/bin/env python3

"""
V8.14-DEBUG — Predictive Fitness Evolution + Diagnostics complets
Compatible avec simulation_V8_13_spatial_mix.py

Objectif : comprendre pourquoi R2 restait à 0.000 et le faire monter
"""

import numpy as np
import copy
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import simulation_V8_13_spatial_mix as engine


# ===============================
# PARAMÈTRES DEBUG (faciles à changer)
# ===============================

GENS = 60
POP = 40

PRED_LAG = 8          # ← réduit (25 → 8) pour que la prédiction soit plus facile au début
VALID_STEPS = 150     # ← réduit (300 → 150) pour aller plus vite

W_LUC = 0.45
W_PRED = 0.55

Q_SCALE = 1e19

DEBUG = True
DEBUG_INDIV = False   # ← Mets True si tu veux TOUT voir (très verbeux)

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
    raise RuntimeError(
        "Impossible de créer population.\n"
        "Ajoute creer_population() ou creer_individu() dans V8.13"
    )


# ===============================
# NORMALISATION
# ===============================

def normalize(v):
    v = np.array(v)
    if v.max() == v.min():
        return np.zeros_like(v)
    return (v - v.min()) / (v.max() - v.min() + 1e-9)


# ===============================
# CALCUL R2 PRÉDICTIF — VERSION DEBUG
# ===============================

def compute_predictive_r2(ind, gen=0, idx_ind=0):
    try:
        res = engine.eval_individu_raw(
            ind,
            steps=VALID_STEPS,
            return_traces=True
        )

        # Debug shapes une fois par génération
        if DEBUG and idx_ind == 0:
            print(f"  [DEBUG Gen {gen:02d}] hs shape = {np.array(res.get('hs', [])).shape} | "
                  f"envs shape = {np.array(res.get('envs', [])).shape}")

        if "hs" not in res or "envs" not in res:
            if DEBUG and idx_ind == 0:
                print(f"  [!] hs/envs MANQUANTS dans V8.13 → R2 forcé à 0")
            return 0.0

        hs = np.array(res["hs"])
        envs = np.array(res["envs"])

        T, nx, ny = hs.shape
        if T < PRED_LAG + 20:
            if DEBUG and idx_ind == 0:
                print(f"  [!] Pas assez de steps (T={T})")
            return 0.0

        # Préparation données
        X = hs[:-PRED_LAG].reshape(T - PRED_LAG, -1)
        Y = envs[PRED_LAG:].reshape(T - PRED_LAG, -1)

        Xtr, Xte, Ytr, Yte = train_test_split(
            X, Y, test_size=0.3, random_state=0
        )

        nloc = min(80, Ytr.shape[1])
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
            r2 = r2_score(yte, yp)
            r2s.append(max(-1.0, r2))

        mean_r2 = float(np.mean(r2s)) if r2s else 0.0

        if DEBUG and idx_ind == 0 and mean_r2 > 0:
            print(f"  [DEBUG] Premier R2 positif détecté : {mean_r2:.4f}")

        return mean_r2

    except Exception as e:
        if DEBUG and idx_ind == 0:
            print(f"  [ERROR Gen {gen:02d}] Exception dans compute_predictive_r2 : {e}")
        return 0.0


# ===============================
# EVOLUTION
# ===============================

def run_evolution():
    print("🚀 LANCEMENT V8.14-DEBUG")
    print(f"   PRED_LAG={PRED_LAG} | VALID_STEPS={VALID_STEPS} | DEBUG={DEBUG}\n")

    pop = create_population(POP)
    history = []

    for gen in range(GENS):
        evals = []
        r2_list = []
        success_r2 = 0

        print(f"--- Gen {gen:02d} -----------------------------------")

        for i, ind in enumerate(pop):
            # Éval normale (sans traces)
            res = engine.eval_individu_raw(ind)
            luc = res["luc_avg"]
            q = res["q_avg"]

            # Éval prédictive
            r2 = compute_predictive_r2(ind, gen=gen, idx_ind=i)

            evals.append((ind, luc, q, r2))
            r2_list.append(r2)

            if r2 > 0.001:
                success_r2 += 1

            if DEBUG_INDIV:
                print(f"  Ind {i:2d} | Luc {luc:.3f} | R2 {r2:.4f} | Q {q:.2e}")

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
        best_r2 = r2s[order[0]]

        print(f"BestFit {best_fit:.4f} | "
              f"Luc_n {luc_n.mean():.3f} | "
              f"R2_avg {r2s.mean():.4f} (max={r2s.max():.4f}) | "
              f"Success R2: {success_r2}/{POP} | "
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
            "success_r2": success_r2,
            "q": float(qs.mean()),
            "best_fit": float(best_fit)
        })

    # Sauvegarde
    np.savez(
        "final_population_v8_14_debug.npz",
        population=pop,
        history=history
    )

    print("\n✅ V8.14-DEBUG TERMINÉ")
    print("   Fichier : final_population_v8_14_debug.npz")
    print(f"   Meilleur R2 final = {r2s.max():.4f}")
    print("   Colle-moi les 10-15 premières lignes de sortie !")


if __name__ == "__main__":
    run_evolution()