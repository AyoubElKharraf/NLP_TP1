"""
Module : maths_utils.py
Description : Fonctions mathematiques et d'algebre lineaire en Python pur.
Modules autorises : math (exp, log) et random (seed, uniform, random).
Toutes les operations matricielles sont implementees via des boucles explicites.
"""

from math import exp, log
from random import uniform, seed


def fixer_graine(graine):
    """Fixe la graine aleatoire pour la reproductibilite."""
    seed(graine)


def aleatoire_uniforme(val_min, val_max):
    """Tire un flottant uniforme dans [val_min, val_max] via random.uniform."""
    return uniform(val_min, val_max)


def exponentielle(x):
    """Calcule exp(x) avec protection contre le debordement."""
    if x > 50.0:
        return exp(50.0)
    if x < -50.0:
        return exp(-50.0)
    return exp(x)


def logarithme(x):
    """
    Equation : log(x + 1e-12)
    Calcule le logarithme naturel avec un terme regulateur pour eviter log(0).
    """
    if x < 1e-12:
        return log(1e-12)
    return log(x)


def tanh_scalaire(x):
    """
    Equation : tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Fonction d'activation de l'etat cache Ht bornee dans [-1, 1].
    """
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return -1.0
    e_pos = exp(x)
    e_neg = exp(-x)
    return (e_pos - e_neg) / (e_pos + e_neg)


def tanh_vecteur(v):
    """Applique tanh element par element sur un vecteur."""
    resultat = []
    for i in range(len(v)):
        resultat.append(tanh_scalaire(v[i]))
    return resultat


def derivee_tanh(y_tanh):
    """
    Equation : tanh'(x) = 1 - tanh(x)^2 = 1 - y^2
    Derivee de tanh exprimee directement a partir de sa sortie y.
    """
    return 1.0 - y_tanh * y_tanh


def softmax(v):
    """
    Equation : Yt[k] = exp(v[k] - max(v)) / somme_j exp(v[j] - max(v))
    Softmax numeriquement stable par soustraction du maximum.
    """
    max_v = v[0]
    for i in range(1, len(v)):
        if v[i] > max_v:
            max_v = v[i]
    exps = []
    somme_exp = 0.0
    for i in range(len(v)):
        e = exp(v[i] - max_v)
        exps.append(e)
        somme_exp += e
    probabilites = []
    for i in range(len(exps)):
        probabilites.append(exps[i] / somme_exp)
    return probabilites


def creer_vecteur_zeros(taille):
    """Cree un vecteur nul de dimension donnee (ex: H0 = vecteur nul)."""
    vecteur = []
    for _ in range(taille):
        vecteur.append(0.0)
    return vecteur


def creer_matrice_zeros(nb_lignes, nb_colonnes):
    """Cree une matrice nulle (liste de listes) de dimensions donnees."""
    matrice = []
    for _ in range(nb_lignes):
        ligne = []
        for _ in range(nb_colonnes):
            ligne.append(0.0)
        matrice.append(ligne)
    return matrice


def creer_matrice_aleatoire(nb_lignes, nb_colonnes, echelle=0.1):
    """Initialise une matrice avec des valeurs uniformes dans [-echelle, echelle]."""
    matrice = []
    for _ in range(nb_lignes):
        ligne = []
        for _ in range(nb_colonnes):
            ligne.append(uniform(-echelle, echelle))
        matrice.append(ligne)
    return matrice


def somme_vecteurs(u, v):
    """Equation : u + v (somme terme a terme de deux vecteurs)."""
    res = []
    for i in range(len(u)):
        res.append(u[i] + v[i])
    return res


def soustraction_vecteurs(u, v):
    """Equation : u - v (difference terme a terme de deux vecteurs)."""
    res = []
    for i in range(len(u)):
        res.append(u[i] - v[i])
    return res


def produit_matrice_vecteur(M, v):
    """
    Equation : res[i] = somme_j ( M[i][j] * v[j] )
    Multiplication matrice-vecteur (utilisee pour Wxh*Xt, Whh*H(t-1), Why*Ht).
    """
    res = []
    for i in range(len(M)):
        ligne = M[i]
        somme = 0.0
        for j in range(len(v)):
            somme += ligne[j] * v[j]
        res.append(somme)
    return res


def produit_matrice_one_hot(M, indice):
    """
    Equation : M * e_k = colonne k de M (propriete fondamentale du One-Hot).
    """
    res = []
    for i in range(len(M)):
        res.append(M[i][indice])
    return res


def produit_exterieur(u, v):
    """
    Equation : M[i][j] = u[i] * v[j]
    Produit exterieur produisant une nouvelle matrice.
    """
    matrice = []
    for i in range(len(u)):
        ligne = []
        ui = u[i]
        for j in range(len(v)):
            ligne.append(ui * v[j])
        matrice.append(ligne)
    return matrice


def accumuler_produit_exterieur(cible, u, v):
    """
    Equation : cible[i][j] += u[i] * v[j]
    Accumule directement le produit exterieur sans allocation memoire intermediaire.
    """
    for i in range(len(u)):
        ui = u[i]
        ligne = cible[i]
        for j in range(len(v)):
            ligne[j] += ui * v[j]


def accumuler_produit_exterieur_one_hot(cible, u, indice):
    """
    Equation : cible[i][indice] += u[i] (produit exterieur avec vecteur One-Hot e_k).
    """
    for i in range(len(u)):
        cible[i][indice] += u[i]


def transposee(M):
    """Equation : M_T[j][i] = M[i][j] (matrice transposee)."""
    nb_l = len(M)
    nb_c = len(M[0])
    M_T = []
    for j in range(nb_c):
        ligne = []
        for i in range(nb_l):
            ligne.append(M[i][j])
        M_T.append(ligne)
    return M_T


def norme_vecteur(v):
    """Equation : ||v|| = sqrt(somme_i v[i]^2) (norme euclidienne L2)."""
    somme = 0.0
    for x in v:
        somme += x * x
    return somme ** 0.5


def norme_globale_matrices(liste_matrices):
    """
    Equation : ||M||_F = sqrt(somme_m somme_ij M[i][j]^2)
    Norme de Frobenius globale utilisee pour l'ecretage du gradient.
    """
    somme = 0.0
    for M in liste_matrices:
        for i in range(len(M)):
            ligne = M[i]
            for j in range(len(ligne)):
                somme += ligne[j] * ligne[j]
    return somme ** 0.5
