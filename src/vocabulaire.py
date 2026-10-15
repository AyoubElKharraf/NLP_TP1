"""
Module : vocabulaire.py
Description : Gestion du vocabulaire au niveau caractere et encodage One-Hot.
Chaque caractere est associe a un indice entier unique entre 0 et V-1 (ou V est la taille du vocabulaire).
Le token special '<fin>' est traite comme un caractere atomique unique.
"""


def tokeniser(texte):
    """
    Decoupe une chaine de caracteres en liste de tokens.
    Traite '<fin>' comme un seul token atomique, et chaque autre caractere individuellement.
    """
    tokens = []
    i = 0
    n = len(texte)
    while i < n:
        # Verification si le token special '<fin>' commence a l'indice i
        if i + 5 <= n and texte[i:i+5] == "<fin>":
            tokens.append("<fin>")
            i += 5
        else:
            tokens.append(texte[i])
            i += 1
    return tokens


def construire_vocab(tous_les_tokens):
    """
    Concept du cours : Construction du dictionnaire/vocabulaire du corpus.
    Identifie tous les caracteres distincts et leur assigne un indice unique de 0 a V-1.
    Retourne :
      - car_vers_idx : dictionnaire caractere -> indice entier
      - idx_vers_car : dictionnaire indice entier -> caractere
    """
    # 1. Extraction des tokens uniques sans utiliser set() (boucle explicite)
    tokens_uniques = []
    for tok in tous_les_tokens:
        deja_present = False
        for u in tokens_uniques:
            if u == tok:
                deja_present = True
                break
        if not deja_present:
            tokens_uniques.append(tok)

    # 2. Tri explicite (tri a bulles) pour avoir un ordre fixe et reproductible
    for i in range(len(tokens_uniques)):
        for j in range(i + 1, len(tokens_uniques)):
            if tokens_uniques[i] > tokens_uniques[j]:
                temp = tokens_uniques[i]
                tokens_uniques[i] = tokens_uniques[j]
                tokens_uniques[j] = temp

    # 3. Creation des tables de correspondance
    car_vers_idx = {}
    idx_vers_car = {}
    for idx in range(len(tokens_uniques)):
        car = tokens_uniques[idx]
        car_vers_idx[car] = idx
        idx_vers_car[idx] = car

    return car_vers_idx, idx_vers_car


def one_hot(indice, taille_vocab):
    """
    Concept du cours : Representation One-Hot d'un caractere Xt.
    Xt est un vecteur de taille V ou toutes les composantes sont nulles (0.0),
    sauf celle correspondant a l'indice du caractere qui vaut 1.0.
    """
    vecteur = []
    for i in range(taille_vocab):
        if i == indice:
            vecteur.append(1.0)
        else:
            vecteur.append(0.0)
    return vecteur


def encoder_sequence(tokens, car_vers_idx):
    """
    Transforme une liste de tokens en une liste d'indices entiers.
    """
    indices = []
    for tok in tokens:
        indices.append(car_vers_idx[tok])
    return indices


def decoder_sequence(indices, idx_vers_car):
    """
    Reconstitue le texte lisible a partir d'une liste d'indices entiers.
    """
    texte = ""
    for idx in indices:
        texte += idx_vers_car[idx]
    return texte


def encoder_one_hot_sequence(indices, taille_vocab):
    """
    Convertit une sequence d'indices en une sequence de vecteurs One-Hot [X0, X1, ..., XT-1].
    """
    vecteurs_one_hot = []
    for idx in indices:
        vecteurs_one_hot.append(one_hot(idx, taille_vocab))
    return vecteurs_one_hot
