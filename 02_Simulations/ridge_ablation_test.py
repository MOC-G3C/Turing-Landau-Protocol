import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor


# ==========================
# PARAMÈTRES
# ==========================

STEPS = 600
LAG = 20


# ==========================
# LOAD CHAMPION
# ==========================

pop = np.load("final_population_v8_15.npz",allow_pickle=True)

champion = pop["population"][0]


# ==========================
# ENVIRONNEMENT
# ==========================

def generate_environment(n):

    t = np.arange(n)

    env = (
        np.sin(t*0.03)
        + 0.5*np.sin(t*0.007)
        + 0.2*np.sin(t*0.11)
    )

    return env



# ==========================
# SIMULATION CORTEX
# ==========================

def simulate(agent,fast_off=False,slow_off=False):

    tau_fast = np.array(agent["tau_fast"]).reshape(-1)
    tau_slow = np.array(agent["tau_slow"]).reshape(-1)


    N = tau_fast.shape[0]

    state_fast = np.zeros(N)
    state_slow = np.zeros(N)


    E = generate_environment(STEPS)

    cortex = np.zeros((STEPS,N))


    for t in range(STEPS):

        inp = E[t]


        if fast_off:
            tauF = np.ones_like(tau_fast)*1e9
        else:
            tauF = tau_fast


        if slow_off:
            tauS = np.ones_like(tau_slow)*1e9
        else:
            tauS = tau_slow


        # dynamique stable

        state_fast += (inp - state_fast)/tauF
        state_slow += (inp - state_slow)/tauS


        # clamp sécurité

        state_fast = np.clip(state_fast,-5,5)
        state_slow = np.clip(state_slow,-5,5)


        cortex[t] = state_fast + state_slow


    return cortex,E



# ==========================
# PREDICTIVE R2 ROBUSTE
# ==========================

def predictive_R2(agent,fast_off=False,slow_off=False):

    cortex,E = simulate(agent,fast_off,slow_off)


    X = cortex[:-LAG]
    Y = E[LAG:]


    split = int(0.7*len(X))

    Xtrain = X[:split]
    Ytrain = Y[:split]

    Xtest = X[split:]
    Ytest = Y[split:]


    # ==========================
    # NORMALISATION ROBUSTE
    # ==========================

    mean = np.mean(Xtrain,axis=0)

    std = np.std(Xtrain,axis=0)

    mask = std > 1e-3


    if np.sum(mask) < 5:
        return 0


    Xtrain = (Xtrain[:,mask]-mean[mask])/(std[mask]+1e-6)
    Xtest  = (Xtest[:,mask]-mean[mask])/(std[mask]+1e-6)


    Xtrain = np.clip(Xtrain,-10,10)
    Xtest  = np.clip(Xtest,-10,10)


    Xtrain = np.nan_to_num(Xtrain)
    Xtest  = np.nan_to_num(Xtest)


    # ==========================
    # PCA STABLE
    # ==========================

    dim = min(15,Xtrain.shape[1])

    pca = PCA(
        n_components=dim,
        svd_solver="randomized"
    )

    Xtrain = pca.fit_transform(Xtrain)
    Xtest  = pca.transform(Xtest)


    Xtrain = np.nan_to_num(Xtrain)
    Xtest  = np.nan_to_num(Xtest)


    # ==========================
    # RANDOM FOREST
    # ==========================

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        n_jobs=-1
    )

    model.fit(Xtrain,Ytrain)

    pred = model.predict(Xtest)


    var = np.var(Ytest)

    if var < 1e-8:
        return 0


    return 1 - np.mean((pred-Ytest)**2)/var



# ==========================
# TESTS
# ==========================

print("\nBaseline RF R2")

baseline = predictive_R2(champion)

print("R2 =",baseline)


print("\nFAST OFF")

fast = predictive_R2(champion,fast_off=True)

print("R2 =",fast)


print("\nSLOW OFF")

slow = predictive_R2(champion,slow_off=True)

print("R2 =",slow)



print("\nSUMMARY\n")

print("Baseline :",baseline)
print("FastOff  :",fast)
print("SlowOff  :",slow)

print()

print("Loss Fast =",baseline-fast)
print("Loss Slow =",baseline-slow)