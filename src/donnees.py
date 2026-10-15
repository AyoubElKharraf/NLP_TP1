"""
Module : donnees.py
Description : Chargement du corpus textuel et decoupage en sequences temporelles (X, E).
La cible E est la sequence d'entree decalee d'un cran vers l'avant, se terminant par '<fin>'.
"""

from src.vocabulaire import (
    tokeniser,
    construire_vocab,
    encoder_sequence,
    encoder_one_hot_sequence
)


def charger_corpus(chemin_fichier):
    """Charge le contenu brut du corpus textuel."""
    with open(chemin_fichier, "r", encoding="utf-8") as f:
        return f.read()


def decouper_en_exemples(texte_corpus):
    """
    Decoupe le texte en paires Question/Reponse distinctes grace au jeton '<fin>'.
    """
    blocs = texte_corpus.split("<fin>")
    exemples = []
    for bloc in blocs:
        nettoye = bloc.strip()
        if len(nettoye) > 0:
            exemples.append(nettoye + "\n<fin>")
    return exemples


def preparer_donnees(chemin_fichier):
    """
    Concept du cours : Preparation des sequences temporelles supervisées.
    Entree Xt = tokens[t], Cible Et = tokens[t+1] (decalage temporel).
    Retourne (exemples, car_vers_idx, idx_vers_car, V).
    """
    texte = charger_corpus(chemin_fichier)
    liste_textes = decouper_en_exemples(texte)
    tous_tokens = tokeniser(texte)
    c2i, i2c = construire_vocab(tous_tokens)
    V = len(c2i)

    exemples = []
    for txt in liste_textes:
        tokens_seq = tokeniser(txt)
        if len(tokens_seq) < 2:
            continue
        indices = encoder_sequence(tokens_seq, c2i)
        X_ind = indices[:-1]
        E_ind = indices[1:]
        exemples.append({
            "texte": txt,
            "X_tokens": tokens_seq[:-1],
            "E_tokens": tokens_seq[1:],
            "X_indices": X_ind,
            "E_indices": E_ind,
            "X_one_hot": encoder_one_hot_sequence(X_ind, V),
            "E_one_hot": encoder_one_hot_sequence(E_ind, V),
            "T": len(X_ind)
        })
    return exemples, c2i, i2c, V
