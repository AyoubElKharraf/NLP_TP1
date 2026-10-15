"""
Module : bptt.py
Description : Retropropagation a travers le temps (BPTT) du Tanh-RNN.
Equations du cours :
  - dzy_t = Yt - Et
  - dWhy += dzy_t * (Ht)^T,  dby += dzy_t
  - dHt = (Why)^T * dzy_t + dH_suivant
  - dzh_t = dHt * (1 - (Ht)^2)
  - dWhh += dzh_t * (H(t-1))^T,  dWxh += dzh_t * (Xt)^T,  dbh += dzh_t
  - dH_suivant = (Whh)^T * dzh_t
Toutes les fonctions comptent strictement moins de 40 lignes.
"""

from src.maths_utils import (
    creer_matrice_zeros,
    creer_vecteur_zeros,
    produit_matrice_vecteur,
    produit_exterieur,
    somme_vecteurs,
    soustraction_vecteurs,
    transposee
)


def accumuler_matrice(cible, source):
    """
    Ajoute la matrice 'source' a la matrice 'cible' element par element.
    """
    for i in range(len(cible)):
        for j in range(len(cible[i])):
            cible[i][j] += source[i][j]


def accumuler_vecteur(cible, source):
    """
    Ajoute le vecteur 'source' au vecteur 'cible' element par element.
    """
    for i in range(len(cible)):
        cible[i] += source[i]


def etape_retropropagation_pas(t, X_sequence, E_sequence, H, Y, Why_T, dH_suivant):
    """
    Calcule les gradients locaux a l'instant t pour la sortie et l'etat cache.
    """
    Xt = X_sequence[t]
    Et = E_sequence[t]
    Ht = H[t]
    Yt = Y[t]

    # 1. Gradient de sortie : dzy_t = Yt - Et
    dzy_t = soustraction_vecteurs(Yt, Et)

    # 2. Gradient arrivant au cache : dHt = Why^T * dzy_t + dH_suivant
    grad_vers_h = produit_matrice_vecteur(Why_T, dzy_t)
    dHt = somme_vecteurs(grad_vers_h, dH_suivant)

    # 3. Traversee de la derivee de tanh : dzh_t = dHt * (1 - Ht^2)
    dzh_t = []
    for i in range(len(Ht)):
        dzh_t.append(dHt[i] * (1.0 - Ht[i] * Ht[i]))

    return dzy_t, dzh_t, Xt, Ht


def retropropagation_bptt(X_sequence, E_sequence, H, Y, parametres):
    """
    Concept du cours : Algorithme BPTT complet remontant de t = T-1 a 0.
    Accumule les gradients pour les matrices partagées Wxh, Whh, Why et biais.
    """
    Why = parametres["Why"]
    Whh = parametres["Whh"]
    H_dim = parametres["taille_cachee"]
    V = parametres["taille_vocab"]
    T = len(X_sequence)

    dWxh = creer_matrice_zeros(H_dim, V)
    dWhh = creer_matrice_zeros(H_dim, H_dim)
    dWhy = creer_matrice_zeros(V, H_dim)
    dbh = creer_vecteur_zeros(H_dim)
    dby = creer_vecteur_zeros(V)

    dH_suivant = creer_vecteur_zeros(H_dim)
    Why_T = transposee(Why)
    Whh_T = transposee(Whh)

    for t in range(T - 1, -1, -1):
        dzy_t, dzh_t, Xt, Ht = etape_retropropagation_pas(
            t, X_sequence, E_sequence, H, Y, Why_T, dH_suivant
        )
        accumuler_matrice(dWhy, produit_exterieur(dzy_t, Ht))
        accumuler_vecteur(dby, dzy_t)

        accumuler_matrice(dWhh, produit_exterieur(dzh_t, H[t - 1]))
        accumuler_matrice(dWxh, produit_exterieur(dzh_t, Xt))
        accumuler_vecteur(dbh, dzh_t)

        dH_suivant = produit_matrice_vecteur(Whh_T, dzh_t)

    return {"dWxh": dWxh, "dWhh": dWhh, "dWhy": dWhy, "dbh": dbh, "dby": dby}
