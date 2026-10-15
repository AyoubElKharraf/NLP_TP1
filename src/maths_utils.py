"""
Module : maths_utils.py
Description : Fonctions mathematiques et d'algebre lineaire en Python pur.
Seuls modules autorises : math (exp, log) et random (seed, uniform, random).
Toutes les fonctions respectent la contrainte de taille (< 40 lignes)
et disposent d'une docstring associant la fonction a l'equation du cours.
"""

from math import exp, log
from random import uniform, random, seed


def fixer_graine(graine=42):
    """
    Fixe la graine pour garantir la reproductibilite parfaite des experiences.
    """
    seed(graine)


def tanh_scalaire(x):
    """
    Concept du cours : Activation par tangente hyperbolique.
    Equation : tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    if x > 20.0:
        return 1.0
    if x < -20.0:
        return -1.0
    exp_pos = exp(x)
    exp_neg = exp(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)


def tanh_vecteur(v):
    """
    Concept du cours : Application de tanh composante par composante.
    Equation : Ht = tanh(Whh * H(t-1) + Wxh * Xt + bh)
    """
    resultat = []
    for val in v:
        resultat.append(tanh_scalaire(val))
    return resultat


def derivee_tanh(y_tanh):
    """
    Concept du cours : Derivee de tanh exprimee par sa valeur de sortie y = tanh(x).
    Equation : tanh'(x) = 1 - tanh(x)^2 = 1 - y^2
    """
    return 1.0 - y_tanh * y_tanh


def softmax(v):
    """
    Concept du cours : Normalisation probabiliste multi-classes.
    Equation : Yt[k] = exp(z_k - max(z)) / somme_j exp(z_j - max(z))
    La soustraction du maximum assure une stabilite numerique contre l'overflow.
    """
    val_max = v[0]
    for x in v:
        if x > val_max:
            val_max = x

    exps = []
    somme_exp = 0.0
    for x in v:
        e = exp(x - val_max)
        exps.append(e)
        somme_exp += e

    probabilites = []
    for e in exps:
        probabilites.append(e / somme_exp)
    return probabilites


def creer_vecteur_zeros(taille):
    """
    Cree un vecteur nul de dimension donnee (ex: H0 = vecteur nul).
    """
    vecteur = []
    for _ in range(taille):
        vecteur.append(0.0)
    return vecteur


def creer_matrice_zeros(nb_lignes, nb_colonnes):
    """
    Cree une matrice nulle (lignes x colonnes) pour accumuler les gradients.
    """
    matrice = []
    for _ in range(nb_lignes):
        ligne = []
        for _ in range(nb_colonnes):
            ligne.append(0.0)
        matrice.append(ligne)
    return matrice


def creer_matrice_aleatoire(nb_lignes, nb_colonnes, borne=0.1):
    """
    Initialise une matrice de poids uniformement dans [-borne, borne].
    """
    matrice = []
    for _ in range(nb_lignes):
        ligne = []
        for _ in range(nb_colonnes):
            ligne.append(uniform(-borne, borne))
        matrice.append(ligne)
    return matrice


def somme_vecteurs(u, v):
    """
    Addition element par element de deux vecteurs de meme taille : u + v.
    """
    resultat = []
    for i in range(len(u)):
        resultat.append(u[i] + v[i])
    return resultat


def soustraction_vecteurs(u, v):
    """
    Soustraction element par element : u - v (utilisee pour dzy = Yt - Et).
    """
    resultat = []
    for i in range(len(u)):
        resultat.append(u[i] - v[i])
    return resultat


def produit_matrice_vecteur(M, v):
    """
    Concept du cours : Produit matrice-vecteur.
    Equation : res[i] = somme_j ( M[i][j] * v[j] )
    Utilise pour Whh * H(t-1), Wxh * Xt et Why * Ht.
    """
    nb_lignes = len(M)
    nb_colonnes = len(v)
    resultat = []
    for i in range(nb_lignes):
        somme = 0.0
        for j in range(nb_colonnes):
            somme += M[i][j] * v[j]
        resultat.append(somme)
    return resultat


def produit_exterieur(u, v):
    """
    Concept du cours : Produit exterieur entre deux vecteurs u et v.
    Equation : M[i][j] = u[i] * v[j]
    Utilise pour accumuler dWhy = dzy * Ht^T, dWhh = dzh * H(t-1)^T, dWxh = dzh * Xt^T.
    """
    matrice = []
    for i in range(len(u)):
        ligne = []
        for j in range(len(v)):
            ligne.append(u[i] * v[j])
        matrice.append(ligne)
    return matrice


def transposee(M):
    """
    Calcule la transposee M^T d'une matrice M pour retropropager l'erreur.
    """
    nb_lignes = len(M)
    nb_colonnes = len(M[0])
    matrice_t = []
    for j in range(nb_colonnes):
        ligne = []
        for i in range(nb_lignes):
            ligne.append(M[i][j])
        matrice_t.append(ligne)
    return matrice_t


def norme_vecteur(v):
    """
    Calcule la norme L2 d'un vecteur : sqrt(somme(v_i^2)).
    """
    somme_carres = 0.0
    for x in v:
        somme_carres += x * x
    return somme_carres ** 0.5


def norme_globale(liste_matrices, liste_vecteurs=None):
    """
    Concept du cours : Norme L2 globale sur l'ensemble de tous les parametres/gradients.
    Equation : norme = sqrt( somme(theta_ij^2) pour tous Wxh, Whh, Why, bh, by )
    """
    somme_carres = 0.0
    for M in liste_matrices:
        for i in range(len(M)):
            for j in range(len(M[i])):
                somme_carres += M[i][j] * M[i][j]
    if liste_vecteurs:
        for v in liste_vecteurs:
            for x in v:
                somme_carres += x * x
    return somme_carres ** 0.5
