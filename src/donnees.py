"""
Module : donnees.py
Description : Chargement du corpus textuel et preparation des sequences supervisées.
Chaque fonction est dediee a une tache et compte moins de 40 lignes.
"""

from src.config import CHEMIN_CORPUS
from src.vocabulaire import (
    tokeniser,
    construire_vocab,
    encoder_sequence,
    encoder_one_hot_sequence
)


def charger_corpus(chemin_fichier=CHEMIN_CORPUS):
    """
    Lit le fichier texte du corpus brut et retourne son contenu.
    """
    with open(chemin_fichier, "r", encoding="utf-8") as f:
        contenu = f.read()
    return contenu


def decouper_en_exemples(texte_complet):
    """
    Decoupe le corpus en paires Q/R individuelles se terminant par '<fin>'.
    """
    blocs = texte_complet.split("<fin>")
    exemples = []
    for bloc in blocs:
        nettoye = bloc.strip()
        if len(nettoye) > 0:
            exemples.append(nettoye + "<fin>")
    return exemples


def preparer_donnees(chemin_fichier=CHEMIN_CORPUS):
    """
    Concept du cours : Preparation des paires temporelles supervisées (X, E).
    Pour chaque exemple :
      - Xt = caractere courant (one-hot)
      - Et = caractere suivant decale d'un pas (cible one-hot)
    """
    texte_corpus = charger_corpus(chemin_fichier)
    liste_textes = decouper_en_exemples(texte_corpus)

    tous_les_tokens = tokeniser(texte_corpus)
    car_vers_idx, idx_vers_car = construire_vocab(tous_les_tokens)
    V = len(car_vers_idx)

    exemples = []
    for texte in liste_textes:
        tokens_seq = tokeniser(texte)
        if len(tokens_seq) < 2:
            continue

        indices_seq = encoder_sequence(tokens_seq, car_vers_idx)
        X_indices = indices_seq[:-1]
        E_indices = indices_seq[1:]

        X_one_hot = encoder_one_hot_sequence(X_indices, V)
        E_one_hot = encoder_one_hot_sequence(E_indices, V)

        exemples.append({
            "texte_original": texte,
            "X_tokens": tokens_seq[:-1],
            "E_tokens": tokens_seq[1:],
            "X_indices": X_indices,
            "E_indices": E_indices,
            "X_one_hot": X_one_hot,
            "E_one_hot": E_one_hot,
            "T": len(X_indices)
        })

    return exemples, car_vers_idx, idx_vers_car, V
