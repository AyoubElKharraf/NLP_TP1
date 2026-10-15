"""
Module : rnn.py
Description : Initialisation des parametres et propagation avant (forward pass)
d'un reseau de neurones recurrent standard (Tanh-RNN) a temps discret.
Equations du cours :
  - Ht = tanh(Whh * H(t-1) + Wxh * Xt + bh)
  - Yt = softmax(Why * Ht + by)
"""

import os
import sys

# Ajout du dossier racine au sys.path si execute directement
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.maths_utils import (
    creer_matrice_aleatoire,
    creer_vecteur_zeros,
    produit_matrice_vecteur,
    somme_vecteurs,
    tanh_vecteur,
    softmax
)


def initialiser_parametres(taille_vocab, taille_cachee=32, echelle=0.01):
    """
    Concept du cours : Initialisation des parametres partages dans le temps.
    Arguments :
      - taille_vocab (V) : dimension de l'entree Xt et de la sortie Yt
      - taille_cachee (H) : dimension du vecteur d'etat cache Ht (defaut: 32)
      - echelle : amplitude des petites valeurs initiales (evite la saturation de tanh)
    Retourne :
      Dictionnaire contenant les matrices de poids et vecteurs de biais :
      - Wxh : matrice (H x V) reliant l'entree au cache
      - Whh : matrice (H x H) reliant l'etat precedent au cache courant
      - Why : matrice (V x H) reliant l'etat cache a la sortie
      - bh  : vecteur (H) de biais pour l'etat cache
      - by  : vecteur (V) de biais pour la sortie
    """
    V = taille_vocab
    H = taille_cachee

    # Matrices de poids initialisees avec de petites valeurs aleatoires
    Wxh = creer_matrice_aleatoire(H, V, echelle)
    Whh = creer_matrice_aleatoire(H, H, echelle)
    Why = creer_matrice_aleatoire(V, H, echelle)

    # Biais initialises a zero
    bh = creer_vecteur_zeros(H)
    by = creer_vecteur_zeros(V)

    parametres = {
        "Wxh": Wxh,
        "Whh": Whh,
        "Why": Why,
        "bh": bh,
        "by": by,
        "taille_vocab": V,
        "taille_cachee": H
    }
    return parametres


def propagation_avant(X_sequence, parametres):
    """
    Concept du cours : Propagation avant a travers le temps (Forward Pass).
    Parcourt la sequence pas a pas de t = 0 a T-1.
    Garde en memoire tous les Ht car ils sont indispensables pour la retropropagation (BPTT).
    
    Equations :
      Ht = tanh(Whh * H(t-1) + Wxh * Xt + bh)
      Yt = softmax(Why * Ht + by)
      
    Arguments :
      - X_sequence : liste des vecteurs One-Hot [X0, X1, ..., XT-1]
      - parametres : dictionnaire des poids (Wxh, Whh, Why, bh, by)
    Retourne :
      - H : dictionnaire des etats caches { -1: H0_nul, 0: H0, 1: H1, ..., T-1: HT-1 }
      - Y : dictionnaire des sorties { 0: Y0, 1: Y1, ..., T-1: YT-1 }
    """
    Wxh = parametres["Wxh"]
    Whh = parametres["Whh"]
    Why = parametres["Why"]
    bh = parametres["bh"]
    by = parametres["by"]
    H_dim = parametres["taille_cachee"]

    # H[-1] represente l'etat initial H0 impose par le cours (vecteur nul)
    H = {}
    H[-1] = creer_vecteur_zeros(H_dim)

    Y = {}
    T = len(X_sequence)

    for t in range(T):
        Xt = X_sequence[t]
        H_precedent = H[t - 1]

        # 1. Calcul de la composante recurrente : Whh * H(t-1)
        rec = produit_matrice_vecteur(Whh, H_precedent)

        # 2. Calcul de la composante de l'entree courante : Wxh * Xt
        ent = produit_matrice_vecteur(Wxh, Xt)

        # 3. Somme affine : Whh * H(t-1) + Wxh * Xt + bh
        pre_activation_h = somme_vecteurs(rec, ent)
        pre_activation_h = somme_vecteurs(pre_activation_h, bh)

        # 4. Activation non-lineaire par tangente hyperbolique : Ht = tanh(...)
        Ht = tanh_vecteur(pre_activation_h)
        H[t] = Ht

        # 5. Calcul de la sortie affine : Why * Ht + by
        pre_activation_y = produit_matrice_vecteur(Why, Ht)
        pre_activation_y = somme_vecteurs(pre_activation_y, by)

        # 6. Distribution de probabilites sur le vocabulaire : Yt = softmax(...)
        Yt = softmax(pre_activation_y)
        Y[t] = Yt

    return H, Y


def predire_caractere_argmax(Yt, idx_vers_car):
    """
    Selectionne l'indice ayant la plus grande probabilite (argmax explicite)
    et retourne le caractere correspondant.
    """
    max_p = Yt[0]
    idx_max = 0
    for i in range(1, len(Yt)):
        if Yt[i] > max_p:
            max_p = Yt[i]
            idx_max = i
    return idx_vers_car[idx_max], max_p


if __name__ == "__main__":
    from src.donnees import preparer_donnees

    chemin_corpus = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")
    exemples, c2i, i2c, V = preparer_donnees(chemin_corpus)

    # Initialisation du RNN avec H=32 (valeur par defaut recommandee)
    H_dim = 32
    print(f"Initialisation du RNN : Vocabulaire V = {V}, Etat cache H = {H_dim}")
    parametres = initialiser_parametres(taille_vocab=V, taille_cachee=H_dim, echelle=0.01)

    # Test de propagation avant sur le premier exemple
    premier_exemple = exemples[0]
    X_seq = premier_exemple["X_one_hot"]
    T = premier_exemple["T"]

    print(f"\nExecution de la propagation avant sur une sequence de longueur T = {T}...")
    H, Y = propagation_avant(X_seq, parametres)

    print("\n--- Etats Caches Ht (extraits des 5 premieres composantes) ---")
    print("H[-1] (vecteur nul initial) :", [round(val, 4) for val in H[-1][:5]])
    print("H[0]  (apres le 1er caractere) :", [round(val, 4) for val in H[0][:5]])
    print(f"H[{T-1}] (etat cache final)     :", [round(val, 4) for val in H[T-1][:5]])

    print("\n--- Sorties Yt (Probabilites de prediction) ---")
    car_pred_0, p_max_0 = predire_caractere_argmax(Y[0], i2c)
    car_attendu_0 = premier_exemple["E_tokens"][0]
    print(f"t = 0  | Entree: '{premier_exemple['X_tokens'][0]}' | Pred: '{car_pred_0}' (p={p_max_0:.4f}) | Cible attendue: '{car_attendu_0}'")

    car_pred_fin, p_max_fin = predire_caractere_argmax(Y[T-1], i2c)
    car_attendu_fin = premier_exemple["E_tokens"][-1]
    print(f"t = {T-1} | Entree: '{premier_exemple['X_tokens'][T-1]}' | Pred: '{car_pred_fin}' (p={p_max_fin:.4f}) | Cible attendue: '{car_attendu_fin}'")

    print("\n[OK] Propagation avant validee avec succes !")
