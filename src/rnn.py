"""
Module : rnn.py
Description : Initialisation et propagation avant (Forward Pass) du Tanh-RNN.
Equations du cours :
  - Ht = tanh(Whh * H(t-1) + Wxh * Xt + bh)
  - Yt = softmax(Why * Ht + by)
Chaque fonction compte moins de 40 lignes.
"""

from src.config import TAILLE_CACHEE, BORNE_INIT
from src.maths_utils import (
    creer_matrice_aleatoire,
    creer_vecteur_zeros,
    produit_matrice_vecteur,
    somme_vecteurs,
    tanh_vecteur,
    softmax
)


def initialiser_parametres(taille_vocab, taille_cachee=TAILLE_CACHEE, borne=BORNE_INIT):
    """
    Concept du cours : Initialisation des parametres partages dans le temps.
    Initialise Wxh, Whh, Why uniformement dans [-borne, borne] et les biais a 0.
    """
    V = taille_vocab
    H = taille_cachee

    Wxh = creer_matrice_aleatoire(H, V, borne)
    Whh = creer_matrice_aleatoire(H, H, borne)
    Why = creer_matrice_aleatoire(V, H, borne)

    bh = creer_vecteur_zeros(H)
    by = creer_vecteur_zeros(V)

    return {
        "Wxh": Wxh, "Whh": Whh, "Why": Why,
        "bh": bh, "by": by,
        "taille_vocab": V, "taille_cachee": H
    }


def propagation_avant(X_sequence, parametres):
    """
    Concept du cours : Propagation avant a travers le temps (Forward Pass).
    Equations :
      Ht = tanh(Whh * H(t-1) + Wxh * Xt + bh)
      Yt = softmax(Why * Ht + by)
    Garde tous les Ht en memoire pour eviter tout recalcul lors de BPTT.
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
        rec = produit_matrice_vecteur(Whh, H[t - 1])
        ent = produit_matrice_vecteur(Wxh, Xt)
        zh = somme_vecteurs(somme_vecteurs(rec, ent), bh)
        Ht = tanh_vecteur(zh)
        H[t] = Ht

        zy = somme_vecteurs(produit_matrice_vecteur(Why, Ht), by)
        Yt = softmax(zy)
        Y[t] = Yt

    return H, Y


def predire_caractere_argmax(Yt, idx_vers_car):
    """
    Selection gloutonne (argmax) du caractere le plus probable.
    """
    max_p = Yt[0]
    idx_max = 0
    for i in range(1, len(Yt)):
        if Yt[i] > max_p:
            max_p = Yt[i]
            idx_max = i
    return idx_vers_car[idx_max], max_p
