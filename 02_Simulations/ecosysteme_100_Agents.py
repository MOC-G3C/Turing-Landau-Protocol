import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
from scipy.io import wavfile

# --- 1. SETTINGS: THE SUPER-ARENA (100 AGENTS) ---
num_agents = 100
L = 16
steps = 8000 
dt = 0.01
thermal_capacity = 5000       
cooling_rate = 500            
CORE_DIR = "../03_Core"
DNA_FILE = "hybrid_champion.npz"

def cognitive_load(step):
    return 0.5 + 1.0 * np.sin(2 * np.pi * step / 1500)**2

def get_tesla_bonus(variance):
    v_scaled = variance * 100
    b3 = np.exp(-(v_scaled - 3)**2 / 0.5)
    b6 = np.exp(-(v_scaled - 6)**2 / 0.5)
    b9 = np.exp(-(v_scaled - 9)**2 / 0.5)
    return b3, b6, b9

# --- 2. INITIALIZATION & ALPHA INJECTION ---
agents = []
DNA_PATH = os.path.join(CORE_DIR, DNA_FILE)

if os.path.exists(DNA_PATH):
    print("👑 ALPHA DETECTED: Injecting the Hybrid Predator...")
    dna = np.load(DNA_PATH)
    agents.append({
        "id": 0, "Theta": dna['Theta'], "kappa": 1.5,
        "heat": 0.0, "adapt": 0.0001, "color": "gold", 
        "name": "ALPHA_HYBRID", "alive": True,
        "kappa_history": [], "heat_history": [], "sound_data": [], 
        "birth_step": 0, "death_step": None
    })

for i in range(len(agents), num_agents):
    agents.append({
        "id": i, "Theta": 0.001 * np.random.randn(L, L), "kappa": 1.0,
        "heat": 0.0, "adapt": 0.005, "color": "purple",
        "name": f"R_{i}", "alive": True,
        "kappa_history": [], "heat_history": [], "sound_data": [], 
        "birth_step": 0, "death_step": None
    })

# --- 3. MASSIVE SIMULATION ---
print(f"🚀 Arena Launching: {num_agents} agents in collision...")

for step in range(steps):
    active = [a for a in agents if a["alive"]]
    if not active: break
    
    for a in active:
        Theta, kappa = a["Theta"], a["kappa"]
        Omega = cognitive_load(step)
        D = 0.05 + 0.1 * Omega
        
        lap = (np.roll(Theta,1,0) + np.roll(Theta,-1,0) + np.roll(Theta,1,1) + np.roll(Theta,-1,1) - 4*Theta)
        Theta -= 1.0 * (Theta**3 - kappa*lap - Omega) * dt + np.sqrt(2*D*dt) * np.random.randn(L,L)
        
        # Tesla 3-6-9 Resonance
        b3, b6, b9 = get_tesla_bonus(np.var(Theta))
        t_total = max(b3, b6, b9)
        eff_cooling = cooling_rate * (1.0 + t_total)
        
        a["heat"] = max(0, a["heat"] + ((D + np.var(lap)) * (L**2 * 4.0) - eff_cooling) * dt)
        
        # Landauer Amnesia Protocol
        if a["heat"] > 4200:
            Theta *= 0.6 
            a["heat"] *= 0.4
        
        if a["heat"] > thermal_capacity:
            a["alive"] = False
            a["death_step"] = step
            continue

        kappa = np.clip(kappa + a["adapt"] * (0.2 - np.var(Theta)), 0.5, 2.5)
        a.update({"Theta": Theta, "kappa": kappa})
        
        if a["id"] == 0: # Performance: only log Alpha's detailed stats
            a["kappa_history"].append(kappa)
            a["heat_history"].append(a["heat"])
            a["sound_data"].append([b3, b6, b9, (a["heat"] / thermal_capacity)**2])

# --- 4. SONIFICATION: LIFE, BIRTH, DEATH ---
print("🎵 Synthesizing the 100-Agent Symphony...")
fs = 44100
t_step = np.linspace(0, 0.001, int(fs * 0.001))
audio_buffer = np.zeros(steps * len(t_step))

# Voice of the Alpha
for s_idx, s_data in enumerate(agents[0]["sound_data"]):
    b3, b6, b9, vol = s_data
    wave = vol * (b3 * np.sin(2 * np.pi * 396 * t_step) + 
                  b6 * np.sin(2 * np.pi * 639 * t_step) + 
                  b9 * np.sin(2 * np.pi * 963 * t_step))
    audio_buffer[s_idx*len(t_step) : (s_idx+1)*len(t_step)] = wave

# Injected Birth/Death Sounds
t_event = np.linspace(0, 0.2, int(fs * 0.2))
for a in agents:
    if a["death_step"]:
        freq_down = np.linspace(800, 100, len(t_event))
        cry = 0.5 * np.sin(2 * np.pi * freq_down * t_event) * np.exp(-t_event * 10)
        start = int(a["death_step"] * len(t_step))
        if start + len(t_event) < len(audio_buffer): audio_buffer[start:start+len(t_event)] += cry
    if a["birth_step"] == 0:
        freq_up = np.linspace(100, 1000, len(t_event))
        pulse = 0.3 * np.sin(2 * np.pi * freq_up * t_event) * np.exp(-t_event * 5)
        audio_buffer[0:len(t_event)] += pulse

audio_final = (audio_buffer / (np.max(np.abs(audio_buffer)) + 1e-6) * 32767).astype(np.int16)
wavfile.write("super_arena.wav", fs, audio_final)

# --- 5. GITHUB SYNC & DASHBOARD ---
np.savez(os.path.join(CORE_DIR, "alpha_post_arena.npz"), Theta=agents[0]["Theta"], kappa=agents[0]["kappa"])

try:
    subprocess.run(["git", "add", "."], check=False)
    subprocess.run(["git", "commit", "-m", "ALPHA_ARENA: Survival test 100 agents"], check=False)
    subprocess.run(["git", "push", "origin", "main"], check=False)
    print("✅ GitHub MOC-G3C Updated.")
except: pass

subprocess.Popen(["afplay", "super_arena.wav"])

plt.figure(figsize=(12, 8))
plt.subplot(2,1,1); plt.plot(agents[0]["heat_history"], color="gold", label="Alpha Heat")
plt.axhline(y=4200, color='orange', linestyle='--'); plt.title("Alpha's Thermal Resistance")
plt.subplot(2,1,2); plt.imshow(agents[0]["Theta"], cmap='magma'); plt.title("Alpha Signature Post-Arena")
plt.tight_layout(); plt.show()