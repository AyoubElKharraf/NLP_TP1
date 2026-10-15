"""
Module : optimiseur.py
Description : Ecretage du gradient (Gradient Clipping) et optimisation par SGD.
Equations du cours :
  - Ecretage : si ||grad|| > seuil, alors grad = (seuil / ||grad||) * grad
  - Mise a jour : P(k+1) = P(k) - delta * grad(Err(P(k)))
Toutes les fonctions comptent strictement moins de 40 lignes.
"""

from src.config import DELTA, SEUIL_CLIPPING


def calculer_norme_globale_gradients(gradients):
    """
    Calcule la norme L2 globale sur l'ensemble de tous les gradients matriciels et vectoriels.
    """
    somme_carres = 0.0
    for nom in ["dWxh", "dWhh", "dWhy"]:
        M = gradients[nom]
        for i in range(len(M)):
            for j in range(len(M[i])):
                somme_carres += M[i][j] * M[i][j]
    for nom in ["dbh", "dby"]:
        v = gradients[nom]
        for x in v:
            somme_carres += x * x
    return somme_carres ** 0.5


def ecreter_gradients(gradients, seuil=SEUIL_CLIPPING):
    """
    Concept du cours : Ecretage du gradient (Gradient Clipping) sur la norme globale.
    Equation : grad = (seuil / norme) * grad  si norme > seuil.
    """
    norme = calculer_norme_globale_gradients(gradients)
    if norme > seuil:
        facteur = seuil / norme
        for nom in ["dWxh", "dWhh", "dWhy"]:
            M = gradients[nom]
            for i in range(len(M)):
                for j in range(len(M[i])):
                    M[i][j] *= facteur
        for nom in ["dbh", "dby"]:
            v = gradients[nom]
            for i in range(len(v)):
                v[i] *= facteur
    return norme


def mettre_a_jour_parametres(parametres, gradients, delta=DELTA):
    """
    Concept du cours : Descente de gradient stochastique (SGD).
    Equation : P(k+1) = P(k) - delta * grad(Err(P(k)))
    """
    for nom in ["Wxh", "Whh", "Why"]:
        M = parametres[nom]
        dM = gradients["d" + nom]
        for i in range(len(M)):
            for j in range(len(M[i])):
                M[i][j] -= delta * dM[i][j]

    for nom in ["bh", "by"]:
        v = parametres[nom]
        dv = gradients["d" + nom]
        for i in range(len(v)):
            v[i] -= delta * dv[i]
