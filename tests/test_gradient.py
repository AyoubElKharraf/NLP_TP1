"""
Module : test_gradient.py
Description : Verification numerique des gradients (Gradient Checking) par differences finies.
Compare pour chaque parametre (Wxh, Whh, Why, bh, by) le gradient analytique calcule par BPTT
avec l'approximation numerique :
  grad_num(theta) = [ Err(theta + epsilon) - Err(theta - epsilon) ] / (2 * epsilon)
"""

import os
import sys

# Ajout du dossier racine au sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.maths_utils import creer_matrice_zeros, creer_vecteur_zeros
from src.rnn import initialiser_parametres, propagation_avant
from src.perte import erreur_sequence
from src.bptt import retropropagation_bptt
from src.vocabulaire import one_hot


def calculer_gradient_numerique_matrice(nom_param, parametres, X_seq, E_seq, epsilon=1e-5):
    """
    Calcule le gradient numerique par differences finies pour une matrice de poids.
    """
    M = parametres[nom_param]
    nb_lignes = len(M)
    nb_colonnes = len(M[0])
    grad_num = creer_matrice_zeros(nb_lignes, nb_colonnes)

    for i in range(nb_lignes):
        for j in range(nb_colonnes):
            val_orig = M[i][j]

            # f(theta + epsilon)
            M[i][j] = val_orig + epsilon
            _, Y_plus = propagation_avant(X_seq, parametres)
            perte_plus, _ = erreur_sequence(Y_plus, E_seq)

            # f(theta - epsilon)
            M[i][j] = val_orig - epsilon
            _, Y_moins = propagation_avant(X_seq, parametres)
            perte_moins, _ = erreur_sequence(Y_moins, E_seq)

            # Restauration de la valeur originale
            M[i][j] = val_orig

            # Difference finie centree
            grad_num[i][j] = (perte_plus - perte_moins) / (2.0 * epsilon)

    return grad_num


def calculer_gradient_numerique_vecteur(nom_param, parametres, X_seq, E_seq, epsilon=1e-5):
    """
    Calcule le gradient numerique par differences finies pour un vecteur de biais.
    """
    v = parametres[nom_param]
    taille = len(v)
    grad_num = creer_vecteur_zeros(taille)

    for i in range(taille):
        val_orig = v[i]

        # f(theta + epsilon)
        v[i] = val_orig + epsilon
        _, Y_plus = propagation_avant(X_seq, parametres)
        perte_plus, _ = erreur_sequence(Y_plus, E_seq)

        # f(theta - epsilon)
        v[i] = val_orig - epsilon
        _, Y_moins = propagation_avant(X_seq, parametres)
        perte_moins, _ = erreur_sequence(Y_moins, E_seq)

        # Restauration
        v[i] = val_orig

        # Difference finie centree
        grad_num[i] = (perte_plus - perte_moins) / (2.0 * epsilon)

    return grad_num


def comparer_gradients_matrice(grad_analytique, grad_numerique, nom):
    """
    Compare deux matrices et calcule l'erreur relative maximale element par element :
    erreur_relative = |analytique - numerique| / (max(|analytique|, |numerique|) + 1e-12)
    """
    nb_lignes = len(grad_analytique)
    nb_colonnes = len(grad_analytique[0])
    erreur_max = 0.0

    for i in range(nb_lignes):
        for j in range(nb_colonnes):
            ga = grad_analytique[i][j]
            gn = grad_numerique[i][j]
            denom = abs(ga) if abs(ga) > abs(gn) else abs(gn)
            denom += 1e-12
            diff_relative = abs(ga - gn) / denom
            if diff_relative > erreur_max:
                erreur_max = diff_relative

    return erreur_max


def comparer_gradients_vecteur(grad_analytique, grad_numerique, nom):
    """
    Compare deux vecteurs et calcule l'erreur relative maximale element par element.
    """
    erreur_max = 0.0
    for i in range(len(grad_analytique)):
        ga = grad_analytique[i]
        gn = grad_numerique[i]
        denom = abs(ga) if abs(ga) > abs(gn) else abs(gn)
        denom += 1e-12
        diff_relative = abs(ga - gn) / denom
        if diff_relative > erreur_max:
            erreur_max = diff_relative

    return erreur_max


def lancer_verification_gradient():
    """
    Point d'entree du test unitaire de Gradient Checking.
    Construit une sequence synthetique courte (T=4, V=5, H=4) pour tester l'ensemble
    exhaustif de tous les parametres du reseau.
    """
    print("=== Lancement de la verification numerique du gradient (Gradient Checking) ===")
    V = 5
    H_dim = 4
    T = 4

    # Initialisation deterministe des parametres
    parametres = initialiser_parametres(taille_vocab=V, taille_cachee=H_dim, echelle=0.05)

    # Sequence de test : indices 0 -> 1 -> 2 -> 3
    indices_X = [0, 1, 2, 3]
    indices_E = [1, 2, 3, 4]

    X_seq = []
    E_seq = []
    for idx in indices_X:
        X_seq.append(one_hot(idx, V))
    for idx in indices_E:
        E_seq.append(one_hot(idx, V))

    # 1. Calcul analytique par BPTT
    H, Y = propagation_avant(X_seq, parametres)
    grad_analytiques = retropropagation_bptt(X_seq, E_seq, H, Y, parametres)

    # 2. Calcul numerique par differences finies pour chaque groupe de parametres
    epsilon = 1e-5
    parametres_matrices = [("Wxh", "dWxh"), ("Whh", "dWhh"), ("Why", "dWhy")]
    parametres_vecteurs = [("bh", "dbh"), ("by", "dby")]

    erreurs = {}

    # Verification des matrices
    for nom_p, nom_g in parametres_matrices:
        grad_num = calculer_gradient_numerique_matrice(nom_p, parametres, X_seq, E_seq, epsilon)
        err = comparer_gradients_matrice(grad_analytiques[nom_g], grad_num, nom_p)
        erreurs[nom_p] = err
        print(f"Parametre {nom_p:<4} | Erreur relative max : {err:.2e}")
        assert err < 1e-5, f"Echec de verification pour {nom_p} : erreur relative = {err}"

    # Verification des vecteurs
    for nom_p, nom_g in parametres_vecteurs:
        grad_num = calculer_gradient_numerique_vecteur(nom_p, parametres, X_seq, E_seq, epsilon)
        err = comparer_gradients_vecteur(grad_analytiques[nom_g], grad_num, nom_p)
        erreurs[nom_p] = err
        print(f"Parametre {nom_p:<4} | Erreur relative max : {err:.2e}")
        assert err < 1e-5, f"Echec de verification pour {nom_p} : erreur relative = {err}"

    print("\n--> TOUS LES GRADIENTS BPTT SONT MATHEMATIQUEMENT EXACTS (Erreur < 1e-5) !")


if __name__ == "__main__":
    lancer_verification_gradient()
