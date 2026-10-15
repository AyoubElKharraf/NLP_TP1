"""
Module : test_gradient.py
Description : Gradient Checking par differences finies centrees sur un mini-modele.
Contraintes specifiees : Vocabulaire de 4 symboles, H = 3, sequence de 4 caracteres.
Equation : grad_num(theta) = [ Err(theta + eps) - Err(theta - eps) ] / (2 * eps)
Critere de succes : ecart relatif < 1e-4 pour chaque parametre.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.maths_utils import creer_matrice_zeros, creer_vecteur_zeros
from src.rnn import initialiser_parametres, propagation_avant
from src.perte import erreur_sequence
from src.bptt import retropropagation_bptt
from src.vocabulaire import one_hot


def grad_num_matrice(nom_param, params, X_seq, E_seq, eps=1e-5):
    """Calcule le gradient numerique par difference finie centree pour une matrice."""
    M = params[nom_param]
    grad = creer_matrice_zeros(len(M), len(M[0]))
    for i in range(len(M)):
        for j in range(len(M[0])):
            orig = M[i][j]
            M[i][j] = orig + eps
            _, Y_plus = propagation_avant(X_seq, params)
            p_plus, _ = erreur_sequence(Y_plus, E_seq)

            M[i][j] = orig - eps
            _, Y_moins = propagation_avant(X_seq, params)
            p_moins, _ = erreur_sequence(Y_moins, E_seq)

            M[i][j] = orig
            grad[i][j] = (p_plus - p_moins) / (2.0 * eps)
    return grad


def grad_num_vecteur(nom_param, params, X_seq, E_seq, eps=1e-5):
    """Calcule le gradient numerique par difference finie centree pour un vecteur."""
    v = params[nom_param]
    grad = creer_vecteur_zeros(len(v))
    for i in range(len(v)):
        orig = v[i]
        v[i] = orig + eps
        _, Y_plus = propagation_avant(X_seq, params)
        p_plus, _ = erreur_sequence(Y_plus, E_seq)

        v[i] = orig - eps
        _, Y_moins = propagation_avant(X_seq, params)
        p_moins, _ = erreur_sequence(Y_moins, E_seq)

        v[i] = orig
        grad[i] = (p_plus - p_moins) / (2.0 * eps)
    return grad


def ecart_relatif_max(g_ana, g_num):
    """Ecart relatif = |ana - num| / (max(|ana|, |num|) + 1e-12)."""
    max_err = 0.0
    for i in range(len(g_ana)):
        l_ana = g_ana[i] if isinstance(g_ana[i], list) else [g_ana[i]]
        l_num = g_num[i] if isinstance(g_num[i], list) else [g_num[i]]
        for j in range(len(l_ana)):
            a, n = l_ana[j], l_num[j]
            denom = max(abs(a), abs(n)) + 1e-12
            diff = abs(a - n) / denom
            if diff > max_err:
                max_err = diff
    return max_err


def tester_mini_modele():
    """
    Mini-modele conforme au cahier des charges : V = 4, H = 3, sequence de 4 caracteres.
    """
    print("=== Verification numerique du gradient (Mini-modele: V=4, H=3, T=4) ===")
    V, H_dim, T = 4, 3, 4
    params = initialiser_parametres(taille_vocab=V, taille_cachee=H_dim, echelle=0.1)

    X_seq = [one_hot(i, V) for i in [0, 1, 2, 3]]
    E_seq = [one_hot(i, V) for i in [1, 2, 3, 0]]

    H, Y = propagation_avant(X_seq, params)
    grads = retropropagation_bptt(X_seq, E_seq, H, Y, params)

    params_liste = [
        ("Wxh", "dWxh", True), ("Whh", "dWhh", True), ("Why", "dWhy", True),
        ("bh", "dbh", False), ("by", "dby", False)
    ]
    for nom_p, nom_g, est_mat in params_liste:
        gn = grad_num_matrice(nom_p, params, X_seq, E_seq) if est_mat else grad_num_vecteur(nom_p, params, X_seq, E_seq)
        err = ecart_relatif_max(grads[nom_g], gn)
        print(f"Parametre {nom_p:<4} | Ecart relatif max : {err:.2e}")
        assert err < 1e-4, f"Echec gradient checking pour {nom_p} : ecart {err} >= 1e-4"

    print("--> TOUS LES GRADIENTS SONT EXACTS (Ecart < 1e-4) !")


if __name__ == "__main__":
    tester_mini_modele()
