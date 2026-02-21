import time
import math

# Simulation de la connexion au Moteur Vortex
# (Dans une version future, on importera directement le module VORTEX)

def calculate_gravity(mass, energy_state):
    # La gravité standard
    G_CONST = 9.81
    
    # Si l'état énergétique est une clé de Tesla (3, 6, 9), la gravité s'allège
    if energy_state in [3, 6]:
        # Flux magnétique : La gravité est moins dense
        modifier = 0.8
        status = "FLUX STATE (Levitation Potential)"
    elif energy_state == 9:
        # Point Zéro : Connexion pure
        modifier = 0.0
        status = "ZERO POINT (Singularity)"
    else:
        # Matière normale (1, 2, 4, 5, 7, 8)
        modifier = 1.0
        status = "SOLID STATE"
        
    current_gravity = G_CONST * modifier
    return current_gravity, status

def run_simulation():
    print("--- DIGITAL GRAVITY BRIDGE INITIALIZED ---")
    print("Synchronization with VORTEX ENGINE... OK.")
    
    # Séquence de test synchronisée
    cycle_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    object_mass = 75 # kg (Poids moyen humain)
    
    try:
        while True:
            for phase in cycle_sequence:
                g_force, state = calculate_gravity(object_mass, phase)
                weight = object_mass * g_force
                
                print(f"[PHASE {phase}] Gravity: {g_force:.2f} m/s² | Weight: {weight:.1f} N | [{state}]")
                time.sleep(0.8)
                
    except KeyboardInterrupt:
        print("\nGravity Simulation Decoupled.")

if __name__ == "__main__":
    run_simulation()