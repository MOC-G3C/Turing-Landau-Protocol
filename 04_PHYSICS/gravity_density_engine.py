import math

def calculate_dilation(bpm):
    """
    Calcule la dilatation du temps basée sur la densité biotique.
    Plus le BPM est élevé, plus la 'masse' augmente, plus le temps ralentit.
    """
    # Constante de Tesla (3-6-9) pour l'ancrage
    TESLA_FACTOR = 3.69
    
    # Calcul de la densité (rho)
    # On normalise : un BPM de 60 = densité 1.0
    density = (bpm * TESLA_FACTOR) / 221.4 # 221.4 = 60 * 3.69
    
    # Facteur de dilatation : si densité > 1.5, le temps commence à se courber
    if density > 1.6: # Environ 100 BPM
        # Formule simplifiée de type Lorentz pour la dilatation
        dilation_factor = math.sqrt(1 + (density / 10))
    else:
        dilation_factor = 1.0
        
    return dilation_factor, density