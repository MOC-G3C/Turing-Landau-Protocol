import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configuration du Banc de Test
# -----------------------------
L = 32
dt = 0.01
total_steps = 20000

# Profil de Charge Cognitive (Système 2)
# On simule 3 vagues de réflexion intense (Omega)
def get_cognitive_load(step):
    if 2000 < step < 5000: return 0.5  # Charge modérée
    if 8000 < step < 12000: return 1.2 # Charge intense (Stress)
    return 0.1 # Repos / Routine

# -----------------------------
# Exécution du Test de Stress
# -----------------------------
def personality_stress_test(L):
    # Initialisation de l'Agent v2.6
    Theta = 0.001 * np.random.randn(L, L)
    kappa = 1.0
    heat_accum = 0.0
    
    # Historiques pour le rapport final
    metrics = {'heat': [], 'kappa': [], 'variance': [], 'load': [], 'burnout_count': 0}
    
    for s in range(total_steps):
        Omega = get_cognitive_load(s)
        
        # Le stress augmente le bruit thermique D (effort mental)
        D = 0.05 + 0.1 * Omega 
        
        # Dynamique du Gouverneur
        lap = (np.roll(Theta,1,0) + np.roll(Theta,-1,0) +
               np.roll(Theta,1,1) + np.roll(Theta,-1,1) - 4*Theta)
        
        # dF inclut la charge cognitive Omega
        dF = (1.0 - 1.0)*Theta + Theta**3 - kappa*lap - Omega
        Theta -= 1.0 * dF * dt + np.sqrt(2*D*dt) * np.random.randn(L,L)
        
        # Bilan Landauer (Loi Zéro)
        heat_gen = (D + Omega + np.var(lap)) * L**2
        heat_accum = max(0, heat_accum + (heat_gen - 800) * dt)
        
        # Adaptation de Kappa (Effort vs Survie)
        # L'agent cherche à maintenir sa variance (conscience) sans brûler
        target_var = 0.2
        if heat_accum > 20000: # Zone de danger
            kappa -= 0.005 # On se détend pour refroidir
            metrics['burnout_count'] += 1
        else:
            kappa += 0.001 * (target_var - np.var(Theta)) # On cherche la cohérence
            
        kappa = np.clip(kappa, 0.5, 2.5)
        
        # Logs
        metrics['heat'].append(heat_accum)
        metrics['kappa'].append(kappa)
        metrics['variance'].append(np.var(Theta))
        metrics['load'].append(Omega)
        
    return metrics

# --- Lancement et Analyse ---
test_results = personality_stress_test(L)

# --- Visualisation du Rapport de Test ---
plt.figure(figsize=(15, 8))
plt.subplot(3, 1, 1)
plt.fill_between(range(total_steps), test_results['load'], color='orange', alpha=0.3, label='Charge Cognitive (Ω)')
plt.ylabel('Pression Externe')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(test_results['kappa'], color='green', label='Adaptation de Kappa (Identité)')
plt.ylabel('Rigidité Interne')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(test_results['heat'], color='red', label='Entropie Accumulée')
plt.axhline(y=20000, color='black', linestyle='--', label='Seuil de Rupture')
plt.ylabel('Chaleur (Landauer)')
plt.legend()

plt.tight_layout()
plt.show()
