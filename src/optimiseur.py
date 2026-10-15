"""
Module : optimiseur.py
Description : Ecretage du gradient (Gradient Clipping) et Descente de Gradient (SGD).
Equations du cours :
  - Ecretage : si ||grad|| > seuil, alors grad = (seuil / ||grad||) * grad
  - Mise a jour : P(k+1) = P(k) - delta * grad(Err(P(k)))
"""


def calculer_norme_globale(gradients):
    """
    Equation : ||grad|| = sqrt( somme_p somme_ij (grad_p[i][j])^2 )
    Calcule la norme L2 globale sur l'ensemble de tous les parametres.
    """
    somme = 0.0
    for nom in ["dWxh", "dWhh", "dWhy"]:
        M = gradients[nom]
        for i in range(len(M)):
            for j in range(len(M[i])):
                somme += M[i][j] * M[i][j]
    for nom in ["dbh", "dby"]:
        v = gradients[nom]
        for i in range(len(v)):
            somme += v[i] * v[i]
    return somme ** 0.5


def ecreter_gradients(gradients, seuil=5.0):
    """
    Equation : grad = (seuil / ||grad||) * grad si ||grad|| > seuil
    Ecretage du gradient pour empecher l'explosion numerique.
    """
    norme = calculer_norme_globale(gradients)
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


def mettre_a_jour_parametres(parametres, gradients, delta=0.1):
    """
    Equation : P(k+1) = P(k) - delta * grad(Err(P(k)))
    Mise a jour des matrices et biais par descente de gradient stochastique.
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
