"""
Module : generation.py
Description : Generation autoregressive de texte caractere par caractere.
Prend en charge le decodage glouton (argmax) et l'echantillonnage avec temperature.
"""

from random import random
from src.maths_utils import (
    creer_vecteur_zeros,
    produit_matrice_vecteur,
    somme_vecteurs,
    tanh_vecteur,
    softmax
)
from src.vocabulaire import tokeniser, one_hot


def un_pas_avant(Xt, H_prec, parametres):
    """
    Equation : Ht = tanh(Whh*H(t-1) + Wxh*Xt + bh), Yt = softmax(Why*Ht + by)
    Calcule un seul pas pour la generation autoregressive.
    """
    rec = produit_matrice_vecteur(parametres["Whh"], H_prec)
    ent = produit_matrice_vecteur(parametres["Wxh"], Xt)
    zh = somme_vecteurs(somme_vecteurs(rec, ent), parametres["bh"])
    Ht = tanh_vecteur(zh)
    zy = somme_vecteurs(produit_matrice_vecteur(parametres["Why"], Ht), parametres["by"])
    Yt = softmax(zy)
    return Ht, zy, Yt


def selectionner_caractere(zy, Yt, idx_vers_car, temperature=0.0):
    """
    Selectionne le caractere :
      - temperature == 0.0 : argmax_k Yt[k] (glouton deterministe)
      - temperature > 0.0  : echantillonnage pondere apres softmax(zy / T)
    """
    V = len(Yt)
    if temperature <= 0.0:
        idx_max = 0
        max_p = Yt[0]
        for i in range(1, V):
            if Yt[i] > max_p:
                max_p = Yt[i]
                idx_max = i
        return idx_vers_car[idx_max], idx_max

    zy_temp = []
    for i in range(V):
        zy_temp.append(zy[i] / temperature)
    Y_temp = softmax(zy_temp)

    tirage = random()
    cumul = 0.0
    idx_choisi = V - 1
    for i in range(V):
        cumul += Y_temp[i]
        if tirage <= cumul:
            idx_choisi = i
            break
    return idx_vers_car[idx_choisi], idx_choisi


def generer_texte(amorce, parametres, car_vers_idx, idx_vers_car, max_caracteres=100, temperature=0.8):
    """
    Concept du cours : Decodage autoregressif pas a pas conditionne par l'amorce.
    S'arrete des l'apparition du jeton terminal '<fin>'.
    """
    V = parametres["taille_vocab"]
    Ht = creer_vecteur_zeros(parametres["taille_cachee"])
    zy, Yt = None, None

    # Conditionnement sur l'amorce (question)
    for tok in tokeniser(amorce):
        idx = car_vers_idx.get(tok, car_vers_idx.get(" ", 0))
        Ht, zy, Yt = un_pas_avant(one_hot(idx, V), Ht, parametres)

    # Generation pas a pas
    texte_genere = ""
    for _ in range(max_caracteres):
        car, idx = selectionner_caractere(zy, Yt, idx_vers_car, temperature)
        if car == "<fin>":
            break
        texte_genere += car
        Ht, zy, Yt = un_pas_avant(one_hot(idx, V), Ht, parametres)

    return texte_genere
