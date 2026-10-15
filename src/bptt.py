"""
Module : bptt.py
Description : Retropropagation a travers le temps (Backpropagation Through Time - BPTT).
Calcule les gradients de l'erreur par rapport a tous les parametres du RNN :
  dWxh, dWhh, dWhy, dbh, dby
en remontant la sequence temporelle de t = T-1 a 0.
"""

import os
import sys

# Ajout du dossier racine au sys.path si execute directement
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.maths_utils import (
    creer_matrice_zeros,
    creer_vecteur_zeros,
    produit_matrice_vecteur,
    produit_exterieur,
    somme_vecteurs,
    soustraction_vecteurs,
    transposee,
    norme_vecteur,
    norme_globale_matrices
)


def ajouter_matrice_en_place(cible, source):
    """
    Ajoute la matrice 'source' a la matrice 'cible' en place : cible += source.
    """
    for i in range(len(cible)):
        for j in range(len(cible[i])):
            cible[i][j] += source[i][j]


def ajouter_vecteur_en_place(cible, source):
    """
    Ajoute le vecteur 'source' au vecteur 'cible' en place : cible += source.
    """
    for i in range(len(cible)):
        cible[i] += source[i]


def retropropagation_bptt(X_sequence, E_sequence, H, Y, parametres):
    """
    Concept du cours : Algorithme BPTT (Backpropagation Through Time).
    Remonte le temps de t = T-1 a t = 0 pour accumuler les gradients des poids partages.
    
    Equations du cours :
      1. Sortie : dzy_t = Yt - Et
      2. Couche de sortie :
           dWhy += dzy_t * (Ht)^T
           dby  += dzy_t
      3. Etat cache :
           dHt = (Why)^T * dzy_t + dH_suivant
      4. Activation tanh :
           dzh_t = dHt * (1 - (Ht)^2)
      5. Poids recurrents et d'entree :
           dWhh += dzh_t * (H(t-1))^T   <-- Accumulation du gradient de Whh
           dWxh += dzh_t * (Xt)^T
           dbh  += dzh_t
      6. Transmission temporelle vers le passe :
           dH_suivant = (Whh)^T * dzh_t
           
    Arguments :
      - X_sequence : liste des vecteurs One-Hot d'entree [X0, ..., XT-1]
      - E_sequence : liste des vecteurs One-Hot cibles  [E0, ..., ET-1]
      - H : dictionnaire de tous les etats caches forward { -1: H_init, 0: H0, ..., T-1: HT-1 }
      - Y : dictionnaire de toutes les distributions forward { 0: Y0, ..., T-1: YT-1 }
      - parametres : dictionnaire des poids actuels (Wxh, Whh, Why, bh, by)
    Retourne :
      - gradients : dictionnaire contenant { "dWxh": ..., "dWhh": ..., "dWhy": ..., "dbh": ..., "dby": ... }
    """
    Why = parametres["Why"]
    Whh = parametres["Whh"]
    H_dim = parametres["taille_cachee"]
    V = parametres["taille_vocab"]
    T = len(X_sequence)

    # 1. Initialisation des accumulateurs de gradients a zero
    dWxh = creer_matrice_zeros(H_dim, V)
    dWhh = creer_matrice_zeros(H_dim, H_dim)
    dWhy = creer_matrice_zeros(V, H_dim)
    dbh = creer_vecteur_zeros(H_dim)
    dby = creer_vecteur_zeros(V)

    # A l'instant terminal t = T-1, aucun gradient ne provient du futur
    dH_suivant = creer_vecteur_zeros(H_dim)

    # Pre-calcul des matrices transposees pour la retropropagation
    Why_T = transposee(Why)
    Whh_T = transposee(Whh)

    # 2. Boucle temporelle remontante : de t = T-1 jusqu'a t = 0
    for t in range(T - 1, -1, -1):
        Xt = X_sequence[t]
        Et = E_sequence[t]
        Ht = H[t]
        Ht_precedent = H[t - 1]
        Yt = Y[t]

        # --- Etape A : Gradient sur la couche de sortie (Softmax + Entropie Croisee) ---
        # Derivee analytique directe : dzy_t = Yt - Et
        dzy_t = soustraction_vecteurs(Yt, Et)

        # Accumulation de dWhy : produit exterieur dzy_t * Ht^T
        dWhy_t = produit_exterieur(dzy_t, Ht)
        ajouter_matrice_en_place(dWhy, dWhy_t)

        # Accumulation de dby
        ajouter_vecteur_en_place(dby, dzy_t)

        # --- Etape B : Gradient arrivant a l'etat cache Ht ---
        # Provient de deux chemins : l'erreur a l'instant courant (Why^T * dzy_t)
        # et l'erreur propagee depuis les pas futurs (dH_suivant)
        grad_sortie_vers_h = produit_matrice_vecteur(Why_T, dzy_t)
        dHt = somme_vecteurs(grad_sortie_vers_h, dH_suivant)

        # --- Etape C : Traversee de la derivee de tanh ---
        # dzh_t = dHt * (1 - Ht^2)
        dzh_t = []
        for i in range(H_dim):
            derivee_tanh_i = 1.0 - Ht[i] * Ht[i]
            dzh_t.append(dHt[i] * derivee_tanh_i)

        # --- Etape D : Accumulation des gradients dWhh, dWxh et dbh ---
        # Gradient de Whh : produit exterieur dzh_t * (Ht_precedent)^T
        # C'est ici exactement que le gradient recurrent de Whh s'accumule dans le temps !
        dWhh_t = produit_exterieur(dzh_t, Ht_precedent)
        ajouter_matrice_en_place(dWhh, dWhh_t)

        # Gradient de Wxh : produit exterieur dzh_t * (Xt)^T
        dWxh_t = produit_exterieur(dzh_t, Xt)
        ajouter_matrice_en_place(dWxh, dWxh_t)

        # Gradient de bh : dzh_t
        ajouter_vecteur_en_place(dbh, dzh_t)

        # --- Etape E : Propagation du gradient vers le pas de temps precedent (t-1) ---
        # dH_suivant pour t-1 devient : Whh^T * dzh_t
        dH_suivant = produit_matrice_vecteur(Whh_T, dzh_t)

    gradients = {
        "dWxh": dWxh,
        "dWhh": dWhh,
        "dWhy": dWhy,
        "dbh": dbh,
        "dby": dby
    }
    return gradients


if __name__ == "__main__":
    from src.donnees import preparer_donnees
    from src.rnn import initialiser_parametres, propagation_avant

    chemin_corpus = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")
    exemples, c2i, i2c, V = preparer_donnees(chemin_corpus)

    # Initialisation
    H_dim = 32
    parametres = initialiser_parametres(taille_vocab=V, taille_cachee=H_dim, echelle=0.01)

    # Test BPTT sur le 1er exemple
    premier = exemples[0]
    X_seq = premier["X_one_hot"]
    E_seq = premier["E_one_hot"]
    T = premier["T"]

    print(f"Lancement de la propagation avant sur sequence (longueur T={T})...")
    H, Y = propagation_avant(X_seq, parametres)

    print("Lancement de la retropropagation a travers le temps (BPTT)...")
    gradients = retropropagation_bptt(X_seq, E_seq, H, Y, parametres)

    # Normes de chaque gradient
    norme_Wxh = norme_globale_matrices([gradients["dWxh"]])
    norme_Whh = norme_globale_matrices([gradients["dWhh"]])
    norme_Why = norme_globale_matrices([gradients["dWhy"]])
    norme_bh = norme_vecteur(gradients["dbh"])
    norme_by = norme_vecteur(gradients["dby"])
    norme_totale = norme_globale_matrices([gradients["dWxh"], gradients["dWhh"], gradients["dWhy"]])

    print("\n=== Verification des Gradients BPTT Calcules ===")
    print(f"Norme ||dWxh|| : {norme_Wxh:.6f} (dimensions {len(gradients['dWxh'])} x {len(gradients['dWxh'][0])})")
    print(f"Norme ||dWhh|| : {norme_Whh:.6f} (dimensions {len(gradients['dWhh'])} x {len(gradients['dWhh'][0])})")
    print(f"Norme ||dWhy|| : {norme_Why:.6f} (dimensions {len(gradients['dWhy'])} x {len(gradients['dWhy'][0])})")
    print(f"Norme ||dbh||  : {norme_bh:.6f} (taille {len(gradients['dbh'])})")
    print(f"Norme ||dby||  : {norme_by:.6f} (taille {len(gradients['dby'])})")
    print(f"Norme globale poids : {norme_totale:.6f}")

    print("\n[OK] Retropropagation a travers le temps (BPTT) executee avec succes !")
