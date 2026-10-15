"""
Module : vocabulaire.py
Description : Gestion du vocabulaire et encodage One-Hot au niveau caractere.
Chaque caractere est indexe de 0 a V-1, avec '<fin>' traite comme token atomique.
"""


def tokeniser(texte):
    """
    Decoupe une chaine en liste de caracteres en preservant '<fin>' comme un seul jeton.
    """
    tokens = []
    i = 0
    n = len(texte)
    while i < n:
        if i + 5 <= n and texte[i:i+5] == "<fin>":
            tokens.append("<fin>")
            i += 5
        else:
            tokens.append(texte[i])
            i += 1
    return tokens


def construire_vocab(tous_les_tokens):
    """
    Concept du cours : Table de correspondance bijective entre caracteres et indices entiers.
    Retourne (car_vers_idx, idx_vers_car).
    """
    tokens_uniques = []
    for tok in tous_les_tokens:
        deja_vu = False
        for u in tokens_uniques:
            if u == tok:
                deja_vu = True
                break
        if not deja_vu:
            tokens_uniques.append(tok)

    for i in range(len(tokens_uniques)):
        for j in range(i + 1, len(tokens_uniques)):
            if tokens_uniques[i] > tokens_uniques[j]:
                tmp = tokens_uniques[i]
                tokens_uniques[i] = tokens_uniques[j]
                tokens_uniques[j] = tmp

    car_vers_idx = {}
    idx_vers_car = {}
    for idx in range(len(tokens_uniques)):
        car = tokens_uniques[idx]
        car_vers_idx[car] = idx
        idx_vers_car[idx] = car
    return car_vers_idx, idx_vers_car


def one_hot(indice, taille_vocab):
    """
    Equation : Xt[k] = 1 si k == indice sinon 0
    Vecteur binaire de dimension V representant un caractere.
    """
    vecteur = []
    for i in range(taille_vocab):
        if i == indice:
            vecteur.append(1.0)
        else:
            vecteur.append(0.0)
    return vecteur


def encoder_sequence(tokens, car_vers_idx):
    """Convertit une liste de tokens en une liste d'indices entiers."""
    indices = []
    for tok in tokens:
        indices.append(car_vers_idx[tok])
    return indices


def decoder_sequence(indices, idx_vers_car):
    """Reconstitue le texte lisible a partir d'une liste d'indices."""
    texte = ""
    for idx in indices:
        texte += idx_vers_car[idx]
    return texte


def encoder_one_hot_sequence(indices, taille_vocab):
    """Convertit une suite d'indices en une suite temporelle de vecteurs One-Hot."""
    vecteurs = []
    for idx in indices:
        vecteurs.append(one_hot(idx, taille_vocab))
    return vecteurs
