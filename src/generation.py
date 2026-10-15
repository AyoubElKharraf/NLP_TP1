"""
Module : generation.py
Description : Generation autoregressive de texte caractere par caractere.
A partir d'une amorce (prompt), le RNN conditionne son etat cache initial,
puis predit iterativement le caractere suivant jusqu'au token '<fin>'.
Supporte le decodage glouton (argmax) et l'echantillonnage avec temperature.
"""

import os
import sys

# Ajout du dossier racine au sys.path si execute directement
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.maths_utils import (
    creer_vecteur_zeros,
    produit_matrice_vecteur,
    somme_vecteurs,
    tanh_vecteur,
    softmax,
    aleatoire_uniforme
)
from src.vocabulaire import tokeniser, one_hot


def un_pas_avant(Xt, H_precedent, parametres):
    """
    Execute un seul pas temporel de propagation avant :
      Ht = tanh(Whh * H(t-1) + Wxh * Xt + bh)
      zy = Why * Ht + by
      Yt = softmax(zy)
    Retourne :
      - Ht : nouvel etat cache
      - zy : pre-activations (logits) avant softmax (utiles pour la temperature)
      - Yt : distribution de probabilites
    """
    Wxh = parametres["Wxh"]
    Whh = parametres["Whh"]
    Why = parametres["Why"]
    bh = parametres["bh"]
    by = parametres["by"]

    rec = produit_matrice_vecteur(Whh, H_precedent)
    ent = produit_matrice_vecteur(Wxh, Xt)
    zh = somme_vecteurs(rec, ent)
    zh = somme_vecteurs(zh, bh)
    Ht = tanh_vecteur(zh)

    zy = produit_matrice_vecteur(Why, Ht)
    zy = somme_vecteurs(zy, by)
    Yt = softmax(zy)

    return Ht, zy, Yt


def selectionner_caractere(zy, Yt, idx_vers_car, temperature=0.0):
    """
    Selectionne l'indice du caractere suivant selon la strategie choisie :
      - Si temperature == 0.0 : decodage glouton (argmax)
      - Si temperature > 0.0 : softmax(zy / temperature) puis tirage aleatoire pondere
    """
    V = len(Yt)

    # 1. Decodage glouton (Argmax pur)
    if temperature <= 0.0:
        max_p = Yt[0]
        idx_choisi = 0
        for i in range(1, V):
            if Yt[i] > max_p:
                max_p = Yt[i]
                idx_choisi = i
        return idx_vers_car[idx_choisi], idx_choisi

    # 2. Tirage avec Temperature : z_ajuste = zy / temperature
    zy_ajuste = []
    for i in range(V):
        zy_ajuste.append(zy[i] / temperature)

    Yt_temp = softmax(zy_ajuste)

    # Tirage aleatoire d'apres la distribution cumulative (Roulette)
    tirage = aleatoire_uniforme(0.0, 1.0)
    cumul = 0.0
    idx_choisi = V - 1

    for i in range(V):
        cumul += Yt_temp[i]
        if tirage <= cumul:
            idx_choisi = i
            break

    return idx_vers_car[idx_choisi], idx_choisi


def generer_texte(amorce_texte, parametres, car_vers_idx, idx_vers_car, max_caracteres=100, temperature=0.0):
    """
    Concept du cours : Decodage autoregressif.
    1. Conditionnement : alimente le RNN avec les caracteres de l'amorce pour construire l'etat cache H.
    2. Generation boucle : predit le caractere suivant, l'ajoute a la reponse,
       et le reinjecte comme entree au pas suivant jusqu'a emettre '<fin>'.
    """
    V = parametres["taille_vocab"]
    H_dim = parametres["taille_cachee"]

    # Initialisation de l'etat cache a zero (H0 nul)
    Ht = creer_vecteur_zeros(H_dim)

    # Tokenisation de l'amorce
    tokens_amorce = tokeniser(amorce_texte)

    # Phase 1 : Alimentation de l'amorce
    zy = None
    Yt = None
    for tok in tokens_amorce:
        # Gestion securisee des caracteres hors vocabulaire
        if tok in car_vers_idx:
            idx = car_vers_idx[tok]
        else:
            # Remplacement de secours par l'espace si caractere inconnu
            idx = car_vers_idx.get(" ", 0)

        Xt = one_hot(idx, V)
        Ht, zy, Yt = un_pas_avant(Xt, Ht, parametres)

    # Phase 2 : Generation autoregressive
    texte_genere = ""
    for _ in range(max_caracteres):
        car_predit, idx_predit = selectionner_caractere(zy, Yt, idx_vers_car, temperature)

        # Arret des que le RNN produit le token terminal '<fin>'
        if car_predit == "<fin>":
            break

        texte_genere += car_predit

        # Le caractere genere devient l'entree du pas suivant
        Xt = one_hot(idx_predit, V)
        Ht, zy, Yt = un_pas_avant(Xt, Ht, parametres)

    return texte_genere


if __name__ == "__main__":
    from src.donnees import preparer_donnees
    from src.rnn import initialiser_parametres
    from src.entrainement import entrainer_sgd

    chemin_corpus = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")
    exemples, c2i, i2c, V = preparer_donnees(chemin_corpus)

    # Initialisation et entrainement rapide pour tester la generation
    parametres = initialiser_parametres(taille_vocab=V, taille_cachee=32, echelle=0.01)
    print("Entrainement rapide du RNN (100 epoques)...")
    entrainer_sgd(exemples, parametres, nb_epoques=100, delta=0.1, seuil=5.0, afficher_chaque=50)

    # Test de generation pour chaque question
    questions_test = [
        "Q: Bonjour\nR:",
        "Q: Qui es-tu?\nR:",
        "Q: Comment vas-tu?\nR:",
        "Q: Que fais-tu?\nR:"
    ]

    print("\n=== Test de generation autoregressive (Argmax) ===")
    for q in questions_test:
        reponse = generer_texte(q, parametres, c2i, i2c, max_caracteres=50, temperature=0.0)
        print(f"Amorce : {repr(q)}")
        print(f"Reponse generee : {repr(reponse.strip())}\n")
