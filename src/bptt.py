"""
Module : bptt.py
Description : Retropropagation a travers le temps (Backpropagation Through Time - BPTT).
Calcule les gradients dWxh, dWhh, dWhy, dbh, dby en remontant la sequence temporelle.
"""

from src.maths_utils import (
    creer_matrice_zeros,
    creer_vecteur_zeros,
    produit_matrice_vecteur,
    accumuler_produit_exterieur,
    somme_vecteurs,
    soustraction_vecteurs,
    transposee
)


def ajouter_vecteur_en_place(cible, source):
    """Equation : cible = cible + source (addition vectorielle en place)."""
    for i in range(len(cible)):
        cible[i] += source[i]


def retropropager_un_pas(Xt, Ht, H_prec, Et, Yt, Why_T, Whh_T, dH_suivant, H_dim, dWxh, dWhh, dWhy, dbh, dby):
    """
    Equations du cours :
      1. dzy_t = Yt - Et
      2. dHt = Why_T * dzy_t + dH_suivant
      3. dzh_t = dHt * (1 - Ht^2)
      4. Accumulation : dWhy += dzy_t*Ht^T, dWhh += dzh_t*H_prec^T, dWxh += dzh_t*Xt^T
      5. dH_suivant = Whh_T * dzh_t
    """
    dzy_t = soustraction_vecteurs(Yt, Et)
    dHt = somme_vecteurs(produit_matrice_vecteur(Why_T, dzy_t), dH_suivant)
    dzh_t = []
    for i in range(H_dim):
        dzh_t.append(dHt[i] * (1.0 - Ht[i] * Ht[i]))

    accumuler_produit_exterieur(dWhy, dzy_t, Ht)
    accumuler_produit_exterieur(dWhh, dzh_t, H_prec)
    accumuler_produit_exterieur(dWxh, dzh_t, Xt)
    ajouter_vecteur_en_place(dbh, dzh_t)
    ajouter_vecteur_en_place(dby, dzy_t)

    return produit_matrice_vecteur(Whh_T, dzh_t)


def retropropagation_bptt(X_sequence, E_sequence, H, Y, parametres):
    """
    Concept du cours : Algorithme BPTT complet remontant de t = T-1 a 0.
    Accumule les gradients pour les parametres partages Wxh, Whh, Why, dbh, dby.
    """
    H_dim = parametres["taille_cachee"]
    V = parametres["taille_vocab"]
    T = len(X_sequence)

    dWxh = creer_matrice_zeros(H_dim, V)
    dWhh = creer_matrice_zeros(H_dim, H_dim)
    dWhy = creer_matrice_zeros(V, H_dim)
    dbh = creer_vecteur_zeros(H_dim)
    dby = creer_vecteur_zeros(V)
    dH_suivant = creer_vecteur_zeros(H_dim)

    Why_T = transposee(parametres["Why"])
    Whh_T = transposee(parametres["Whh"])

    for t in range(T - 1, -1, -1):
        dH_suivant = retropropager_un_pas(
            X_sequence[t], H[t], H[t - 1], E_sequence[t], Y[t],
            Why_T, Whh_T, dH_suivant, H_dim,
            dWxh, dWhh, dWhy, dbh, dby
        )

    return {"dWxh": dWxh, "dWhh": dWhh, "dWhy": dWhy, "dbh": dbh, "dby": dby}
