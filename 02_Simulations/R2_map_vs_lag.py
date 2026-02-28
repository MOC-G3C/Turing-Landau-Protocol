import numpy as np

POP_FILE = "final_population_v8_15.npz"

VALID_STEPS = 500
MAX_LAG = 30


# -------------------------
# ENVIRONNEMENT

def generate_environment(T):

    E = np.random.normal(0,0.05,T)

    for t in range(20,T,40):

        width = np.random.randint(3,8)
        amp = np.random.uniform(0.8,1.5)

        shape = np.exp(-np.linspace(0,2,width))

        end = min(T,t+width)

        E[t:end] += amp*shape[:end-t]

    return E


# -------------------------

def compute_R2(a,b):

    a = np.nan_to_num(a)
    b = np.nan_to_num(b)

    var = np.var(a)

    if var < 1e-12:
        return 0.0

    mse = np.mean((a-b)**2)

    if not np.isfinite(mse):
        return 0.0

    return 1 - mse/var


# -------------------------

def simulate(agent):

    E = generate_environment(VALID_STEPS)

    tau_fast = agent["tau_fast"]
    tau_slow = agent["tau_slow"]

    H,W = tau_fast.shape

    fast = np.zeros((VALID_STEPS,H,W))
    slow = np.zeros((VALID_STEPS,H,W))

    state_fast = np.zeros((H,W))
    state_slow = np.zeros((H,W))

    for t in range(VALID_STEPS):

        inp = E[t]

        state_fast += (inp-state_fast)/tau_fast
        state_slow += (inp-state_slow)/tau_slow

        state_fast = np.clip(state_fast,-5,5)
        state_slow = np.clip(state_slow,-5,5)

        fast[t] = state_fast
        slow[t] = state_slow

    cortex = fast + slow

    return cortex,E


# -------------------------

data = np.load(POP_FILE,allow_pickle=True)

champion = data["population"][0]


print("Simulating champion...")

cortex,E = simulate(champion)


H,W = cortex.shape[1:]


print("Computing R2 maps...")

R2_short = np.zeros((H,W))
R2_long = np.zeros((H,W))


SHORT_LAG = 3
LONG_LAG = 20


for i in range(H):
    for j in range(W):

        signal = cortex[:,i,j]

        R2_short[i,j] = compute_R2(
            signal[:-SHORT_LAG],
            E[SHORT_LAG:]
        )

        R2_long[i,j] = compute_R2(
            signal[:-LONG_LAG],
            E[LONG_LAG:]
        )


print("\nR2 short lag mean:",np.mean(R2_short))
print("R2 long lag mean:",np.mean(R2_long))


print("\nShort-lag max:",np.max(R2_short))
print("Long-lag max:",np.max(R2_long))


# cellules les plus prédictives

best_short = np.unravel_index(
    np.argmax(R2_short),
    R2_short.shape
)

best_long = np.unravel_index(
    np.argmax(R2_long),
    R2_long.shape
)


print("\nBest short cell:",best_short,
      "R2=",R2_short[best_short])

print("Best long cell:",best_long,
      "R2=",R2_long[best_long])


print("\nTau slow short:",
      champion["tau_slow"][best_short])

print("Tau slow long:",
      champion["tau_slow"][best_long])