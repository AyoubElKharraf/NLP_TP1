"""
Module : optimiseur.py
Description : Ecretage du gradient (Gradient Clipping) et mise a jour des parametres
par descente de gradient stochastique (SGD).
Equations du cours :
  - Ecretage : si norme(grad) > seuil, alors grad = (seuil / norme(grad)) * grad
  - Mise a jour : P(k+1) = P(k) - delta * grad(Err(P(k)))
"""

import os
import sys

# Ajout du dossier racine au sys.path si execute directement
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def calculer_norme_globale_gradients(gradients):
    """
    Calcule la norme L2 globale (Frobenius) de l'ensemble des gradients :
    norme = racine_carree( somme(g_i^2) pour tous les elements de dWxh, dWhh, dWhy, dbh, dby )
    """
    somme_carres = 0.0

    # Gradients matriciels
    for nom_mat in ["dWxh", "dWhh", "dWhy"]:
        M = gradients[nom_mat]
        for i in range(len(M)):
            for j in range(len(M[i])):
                somme_carres += M[i][j] * M[i][j]

    # Gradients vectoriels (biais)
    for nom_vec in ["dbh", "dby"]:
        v = gradients[nom_vec]
        for i in range(len(v)):
            somme_carres += v[i] * v[i]

    return somme_carres ** 0.5


def ecreter_gradients(gradients, seuil=5.0):
    """
    Concept du cours : Ecretage du gradient (Gradient Clipping).
    Empeche l'explosion du gradient dans les RNN lors de longues retropropagations temporelles.
    Equation :
      si norme(grad) > seuil :
          grad = (seuil / norme(grad)) * grad
    Modifie les gradients en place et retourne la norme avant ecretage.
    """
    norme = calculer_norme_globale_gradients(gradients)

    if norme > seuil:
        facteur = seuil / norme

        # Ecretage des matrices
        for nom_mat in ["dWxh", "dWhh", "dWhy"]:
            M = gradients[nom_mat]
            for i in range(len(M)):
                for j in range(len(M[i])):
                    M[i][j] *= facteur

        # Ecretage des vecteurs
        for nom_vec in ["dbh", "dby"]:
            v = gradients[nom_vec]
            for i in range(len(v)):
                v[i] *= facteur

    return norme


def mettre_a_jour_parametres(parametres, gradients, delta=0.05):
    """
    Concept du cours : Descente de gradient stochastique (SGD).
    Equation : P(k+1) = P(k) - delta * grad(Err(P(k)))
    ou delta est le taux d'apprentissage (learning rate).
    """
    # Mise a jour des matrices de poids Wxh, Whh, Why
    for nom in ["Wxh", "Whh", "Why"]:
        M = parametres[nom]
        dM = gradients["d" + nom]
        for i in range(len(M)):
            for j in range(len(M[i])):
                M[i][j] -= delta * dM[i][j]

    # Mise a jour des vecteurs de biais bh, by
    for nom in ["bh", "by"]:
        v = parametres[nom]
        dv = gradients["d" + nom]
        for i in range(len(v)):
            v[i] -= delta * dv[i]
