"""
Module : rnn.py
Description : Initialisation des poids et propagation avant du Tanh-RNN.
Equations du cours :
  - Ht = tanh(Whh * H(t-1) + Wxh * Xt + bh)
  - Yt = softmax(Why * Ht + by)
"""

from src.maths_utils import (
    creer_matrice_aleatoire,
    creer_vecteur_zeros,
    produit_matrice_vecteur,
    somme_vecteurs,
    tanh_vecteur,
    softmax
)


def initialiser_parametres(taille_vocab, taille_cachee=32, echelle=0.1):
    """
    Concept du cours : Matrices de poids Wxh, Whh, Why et biais bh, by partages dans le temps.
    Poids initialises uniformement dans [-echelle, echelle].
    """
    V = taille_vocab
    H = taille_cachee
    return {
        "Wxh": creer_matrice_aleatoire(H, V, echelle),
        "Whh": creer_matrice_aleatoire(H, H, echelle),
        "Why": creer_matrice_aleatoire(V, H, echelle),
        "bh": creer_vecteur_zeros(H),
        "by": creer_vecteur_zeros(V),
        "taille_vocab": V,
        "taille_cachee": H
    }


def propagation_avant_pas(Xt, H_prec, Wxh, Whh, Why, bh, by):
    """
    Execute un seul pas temporel :
    Ht = tanh(Whh * H(t-1) + Wxh * Xt + bh)
    Yt = softmax(Why * Ht + by)
    """
    rec = produit_matrice_vecteur(Whh, H_prec)
    ent = produit_matrice_vecteur(Wxh, Xt)
    zh = somme_vecteurs(somme_vecteurs(rec, ent), bh)
    Ht = tanh_vecteur(zh)
    zy = somme_vecteurs(produit_matrice_vecteur(Why, Ht), by)
    Yt = softmax(zy)
    return Ht, zy, Yt


def propagation_avant(X_sequence, parametres):
    """
    Concept du cours : Propagation avant a travers le temps (Forward Pass).
    Garde en memoire tous les Ht dans H pour la retropropagation (BPTT).
    """
    Wxh = parametres["Wxh"]
    Whh = parametres["Whh"]
    Why = parametres["Why"]
    bh = parametres["bh"]
    by = parametres["by"]
    H_dim = parametres["taille_cachee"]

    H = {-1: creer_vecteur_zeros(H_dim)}
    Y = {}
    T = len(X_sequence)

    for t in range(T):
        Xt = X_sequence[t]
        Ht, _, Yt = propagation_avant_pas(Xt, H[t - 1], Wxh, Whh, Why, bh, by)
        H[t] = Ht
        Y[t] = Yt

    return H, Y


def predire_caractere_argmax(Yt, idx_vers_car):
    """Decodage glouton : argmax_k Yt[k]."""
    max_p = Yt[0]
    idx_max = 0
    for i in range(1, len(Yt)):
        if Yt[i] > max_p:
            max_p = Yt[i]
            idx_max = i
    return idx_vers_car[idx_max], max_p
