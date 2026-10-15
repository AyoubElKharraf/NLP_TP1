"""
Module : maths_utils.py
Description : Fonctions mathematiques et d'algebre lineaire en Python pur.
Aucune bibliotheque externe autorisee, seul le module math (exp, log) est utilise.
Tous les calculs matriciels et vectoriels sont implementes avec des listes simples
et des boucles explicites pour une comprehension pedagogique maximale.
"""

from math import exp, log


# =====================================================================
# 1. Generateur pseudo-aleatoire (LCG) en Python pur
# =====================================================================
# Permet d'initialiser les poids sans importer le module 'random'.
_graine = 42

def fixer_graine(nouvelle_graine):
    """
    Fixe la graine pour rendre l'initialisation des poids reproductible.
    """
    global _graine
    _graine = nouvelle_graine

def aleatoire_uniforme(val_min, val_max):
    """
    Genere un nombre flottant pseudo-aleatoire dans l'intervalle [val_min, val_max].
    Utilise un generateur congruentiel lineaire (LCG).
    """
    global _graine
    # Constantes standards de generateur lineaire congruentiel (POSIX)
    _graine = (1103515245 * _graine + 12345) % (2**31)
    ratio = _graine / (2**31)
    return val_min + (val_max - val_min) * ratio


# =====================================================================
# 2. Fonctions d'activation et elementaires
# =====================================================================

def exponentielle(x):
    """
    Calcule exp(x). Protege contre le debordement numerique (overflow).
    """
    if x > 50.0:
        x = 50.0
    elif x < -50.0:
        x = -50.0
    return exp(x)

def logarithme(x):
    """
    Calcule log(x) pour la perte par entropie croisee.
    Ajoute un petit epsilon pour eviter log(0).
    """
    epsilon = 1e-15
    if x < epsilon:
        x = epsilon
    return log(x)

def tanh_scalaire(x):
    """
    Concept du cours : Fonction d'activation tangente hyperbolique.
    Equation : tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Sortie dans l'intervalle [-1, 1].
    """
    # Protection contre le debordement de exp(x)
    if x > 20.0:
        return 1.0
    elif x < -20.0:
        return -1.0
    
    exp_pos = exp(x)
    exp_neg = exp(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)

def tanh_vecteur(v):
    """
    Applique tanh element par element sur un vecteur d'entree.
    Utilise pour calculer l'etat cache Ht = tanh(Whh * H(t-1) + Wxh * Xt + bh).
    """
    resultat = []
    for i in range(len(v)):
        resultat.append(tanh_scalaire(v[i]))
    return resultat

def derivee_tanh(y_tanh):
    """
    Concept du cours : Derivee de tanh(x) exprimee en fonction de sa sortie y = tanh(x).
    Equation : tanh'(x) = 1 - tanh(x)^2 = 1 - y^2
    Utilisee lors de la retropropagation (BPTT).
    """
    return 1.0 - y_tanh * y_tanh

def softmax(v):
    """
    Concept du cours : Distribution de probabilites sur les caracteres de sortie.
    Equation : Yt[k] = exp(z_k) / somme_j(exp(z_j))
    Pour la stabilite numerique, on soustrait max(v) avant l'exponentielle.
    """
    # 1. Recherche explicite du maximum pour eviter un overflow dans exp()
    val_max = v[0]
    for i in range(1, len(v)):
        if v[i] > val_max:
            val_max = v[i]

    # 2. Calcul des exponentielles stables et de leur somme
    exps = []
    somme_exp = 0.0
    for i in range(len(v)):
        val_exp = exp(v[i] - val_max)
        exps.append(val_exp)
        somme_exp += val_exp

    # 3. Normalisation pour que la somme valle 1
    probabilites = []
    for i in range(len(exps)):
        probabilites.append(exps[i] / somme_exp)

    return probabilites


# =====================================================================
# 3. Operations d'algebre lineaire (Vecteurs et Matrices)
# =====================================================================

def creer_vecteur_zeros(taille):
    """
    Cree un vecteur de zeros de taille donnee (ex: H0 = vecteur nul).
    """
    vecteur = []
    for _ in range(taille):
        vecteur.append(0.0)
    return vecteur

def creer_matrice_zeros(nb_lignes, nb_colonnes):
    """
    Cree une matrice (liste de listes) remplie de zeros.
    Utilise pour accumuler les gradients (dWxh, dWhh, dWhy).
    """
    matrice = []
    for _ in range(nb_lignes):
        ligne = []
        for _ in range(nb_colonnes):
            ligne.append(0.0)
        matrice.append(ligne)
    return matrice

def creer_matrice_aleatoire(nb_lignes, nb_colonnes, echelle=0.01):
    """
    Initialise une matrice avec de petites valeurs uniformes dans [-echelle, echelle].
    """
    matrice = []
    for _ in range(nb_lignes):
        ligne = []
        for _ in range(nb_colonnes):
            valeur = aleatoire_uniforme(-echelle, echelle)
            ligne.append(valeur)
        matrice.append(ligne)
    return matrice

def somme_vecteurs(u, v):
    """
    Addition element par element de deux vecteurs de meme taille : res = u + v.
    """
    resultat = []
    for i in range(len(u)):
        resultat.append(u[i] + v[i])
    return resultat

def soustraction_vecteurs(u, v):
    """
    Soustraction element par element : res = u - v.
    """
    resultat = []
    for i in range(len(u)):
        resultat.append(u[i] - v[i])
    return resultat

def produit_matrice_vecteur(M, v):
    """
    Concept du cours : Multiplication d'une matrice M (taille N x P)
    par un vecteur v (taille P) donnant un vecteur de taille N.
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
    Concept du cours : Produit exterieur entre deux vecteurs u (taille N) et v (taille P).
    Equation : M[i][j] = u[i] * v[j], produisant une matrice de taille N x P.
    Fondamental en retropropagation pour calculer les gradients des poids :
    dWhy = dYt * Ht^T, dWhh = dHt * H(t-1)^T, dWxh = dHt * Xt^T.
    """
    nb_lignes = len(u)
    nb_colonnes = len(v)
    matrice = []
    for i in range(nb_lignes):
        ligne = []
        for j in range(nb_colonnes):
            ligne.append(u[i] * v[j])
        matrice.append(ligne)
    return matrice

def transposee(M):
    """
    Calcule la transposee M^T d'une matrice M.
    Utilisee en retropropagation pour propager l'erreur vers l'arriere :
    dHt = Why^T * dYt + Whh^T * dH(t+1)
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
    Calcule la norme euclidienne (L2) d'un vecteur : racine_carree(somme(v_i^2)).
    """
    somme_carres = 0.0
    for i in range(len(v)):
        somme_carres += v[i] * v[i]
    return somme_carres ** 0.5

def norme_globale_matrices(liste_matrices):
    """
    Concept du cours : Norme Frobenius globale de l'ensemble des matrices de gradients.
    Utilisee pour l'ecretage du gradient (gradient clipping) :
    norme = racine(somme de tous les carres des elements de dWxh, dWhh, dWhy, etc.).
    """
    somme_totale = 0.0
    for M in liste_matrices:
        for i in range(len(M)):
            for j in range(len(M[i])):
                somme_totale += M[i][j] * M[i][j]
    return somme_totale ** 0.5
