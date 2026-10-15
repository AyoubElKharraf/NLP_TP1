"""
Module : generation.py
Description : Generation autoregressive de texte caractere par caractere.
Supporte le decodage glouton (argmax) et le tirage probabiliste avec temperature.
Toutes les fonctions comptent strictement moins de 40 lignes.
"""

from random import random
from src.config import TEMPERATURE, LONGUEUR_MAX_GENERATION, CHEMIN_EXEMPLES_TXT
from src.maths_utils import (
    creer_vecteur_zeros,
    produit_matrice_vecteur,
    somme_vecteurs,
    tanh_vecteur,
    softmax
)
from src.vocabulaire import tokeniser, one_hot


def un_pas_avant(Xt, H_precedent, parametres):
    """
    Execute un seul pas de propagation avant pour le decodage autoregressif.
    """
    Wxh = parametres["Wxh"]
    Whh = parametres["Whh"]
    Why = parametres["Why"]
    bh = parametres["bh"]
    by = parametres["by"]

    rec = produit_matrice_vecteur(Whh, H_precedent)
    ent = produit_matrice_vecteur(Wxh, Xt)
    zh = somme_vecteurs(somme_vecteurs(rec, ent), bh)
    Ht = tanh_vecteur(zh)

    zy = somme_vecteurs(produit_matrice_vecteur(Why, Ht), by)
    Yt = softmax(zy)
    return Ht, zy, Yt


def selectionner_caractere(zy, Yt, idx_vers_car, temperature=0.0):
    """
    Selectionne le caractere suivant : Argmax pur si temp=0.0, tirage pondere sinon.
    """
    V = len(Yt)
    if temperature <= 0.0:
        max_p = Yt[0]
        idx_choisi = 0
        for i in range(1, V):
            if Yt[i] > max_p:
                max_p = Yt[i]
                idx_choisi = i
        return idx_vers_car[idx_choisi], idx_choisi

    zy_ajuste = [z / temperature for z in zy]
    Yt_temp = softmax(zy_ajuste)
    tirage = random()
    cumul = 0.0
    idx_choisi = V - 1
    for i in range(V):
        cumul += Yt_temp[i]
        if tirage <= cumul:
            idx_choisi = i
            break
    return idx_vers_car[idx_choisi], idx_choisi


def generer_texte(amorce, parametres, car_vers_idx, idx_vers_car, max_car=LONGUEUR_MAX_GENERATION, temperature=0.0):
    """
    Concept du cours : Decodage autoregressif conditionne par l'amorce.
    Reinjecte la sortie predite a l'instant t comme entree du pas t+1.
    """
    V = parametres["taille_vocab"]
    Ht = creer_vecteur_zeros(parametres["taille_cachee"])
    zy, Yt = None, None

    # Phase 1 : Ingestion de l'amorce
    for tok in tokeniser(amorce):
        idx = car_vers_idx.get(tok, car_vers_idx.get(" ", 0))
        Ht, zy, Yt = un_pas_avant(one_hot(idx, V), Ht, parametres)

    # Phase 2 : Generation autoregressive jusqu'a '<fin>'
    reponse = ""
    for _ in range(max_car):
        car_pred, idx_pred = selectionner_caractere(zy, Yt, idx_vers_car, temperature)
        if car_pred == "<fin>":
            break
        reponse += car_pred
        Ht, zy, Yt = un_pas_avant(one_hot(idx_pred, V), Ht, parametres)

    return reponse


def evaluer_et_sauvegarder_exemples(parametres, c2i, i2c, chemin=CHEMIN_EXEMPLES_TXT):
    """
    Genere les reponses pour 10 questions (5 presentes, 5 absentes) et les enregistre.
    """
    questions_corpus = [
        "comment tu t'appelles ?", "qui es-tu ?", "quel est ton cours ?",
        "quelle est ton activation ?", "bonjour !"
    ]
    questions_hors_corpus = [
        "quel temps fait-il ?", "ou habites-tu ?", "quelle heure est-il ?",
        "qui a gagne le match ?", "peux-tu resoudre une equation ?"
    ]

    lignes = ["=== EVALUATION COMPARATIVE DU MODELE (ARGMAX vs TEMPERATURE 0.8) ===\n\n"]
    lignes.append("--- 1. QUESTIONS PRESENTES DANS LE CORPUS (IN-DISTRIBUTION) ---\n")
    for q in questions_corpus:
        amorce = f"Q: {q}\nR:"
        rep_argmax = generer_texte(amorce, parametres, c2i, i2c, temperature=0.0).strip()
        rep_temp = generer_texte(amorce, parametres, c2i, i2c, temperature=TEMPERATURE).strip()
        lignes.append(f"Q: {q}\n   [Argmax]       : {rep_argmax}\n   [Temp {TEMPERATURE}] : {rep_temp}\n\n")

    lignes.append("--- 2. QUESTIONS ABSENTES DU CORPUS (OUT-OF-DISTRIBUTION) ---\n")
    for q in questions_hors_corpus:
        amorce = f"Q: {q}\nR:"
        rep_argmax = generer_texte(amorce, parametres, c2i, i2c, temperature=0.0).strip()
        rep_temp = generer_texte(amorce, parametres, c2i, i2c, temperature=TEMPERATURE).strip()
        lignes.append(f"Q: {q}\n   [Argmax]       : {rep_argmax}\n   [Temp {TEMPERATURE}] : {rep_temp}\n\n")

    with open(chemin, "w", encoding="utf-8") as f:
        f.writelines(lignes)
