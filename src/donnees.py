"""
Module : donnees.py
Description : Chargement du corpus textuel, decoupage en exemples d'apprentissage,
et generation des paires (X, E) avec cibles decalees d'un cran vers l'avant.
"""

import os
import sys

# Ajout du dossier racine au sys.path si execute directement
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vocabulaire import (
    tokeniser,
    construire_vocab,
    encoder_sequence,
    encoder_one_hot_sequence,
    decoder_sequence
)


def charger_corpus(chemin_fichier):
    """
    Lit le fichier texte du corpus brut et le retourne sous forme de chaine.
    """
    with open(chemin_fichier, "r", encoding="utf-8") as f:
        contenu = f.read()
    return contenu


def decouper_en_exemples(texte_complet):
    """
    Decoupe le texte complet en une liste d'exemples Question/Reponse distincts.
    Chaque exemple se termine par le token atomique '<fin>'.
    """
    # Recherche manuelle ou decoupage par '<fin>'
    blocs_bruts = texte_complet.split("<fin>")
    exemples_textes = []
    
    for bloc in blocs_bruts:
        bloc_nettoye = bloc.strip()
        if len(bloc_nettoye) > 0:
            # On rajoute '<fin>' a la fin de l'exemple comme symbole terminal
            exemples_textes.append(bloc_nettoye + "\n<fin>")
            
    return exemples_textes


def preparer_donnees(chemin_fichier):
    """
    Concept du cours : Preparation des sequences temporelles d'apprentissage.
    Pour chaque exemple :
      - Entree X = [tok_0, tok_1, ..., tok_{T-1}]
      - Cible  E = [tok_1, tok_2, ..., tok_T]   (decalee d'un cran, tok_T etant '<fin>')
    Retourne :
      - exemples : liste de dictionnaires contenant X et E (sous forme d'indices et One-Hot)
      - car_vers_idx : table caractere -> indice
      - idx_vers_car : table indice -> caractere
      - V : taille du vocabulaire
    """
    texte_corpus = charger_corpus(chemin_fichier)
    liste_textes = decouper_en_exemples(texte_corpus)

    # 1. Extraction de tous les tokens pour etablir le vocabulaire global
    tous_les_tokens = tokeniser(texte_corpus)
    car_vers_idx, idx_vers_car = construire_vocab(tous_les_tokens)
    V = len(car_vers_idx)

    # 2. Construction des paires (X, E) pour chaque sequence
    exemples = []
    for texte in liste_textes:
        tokens_seq = tokeniser(texte)
        # Il faut au moins 2 tokens pour faire une paire (entree, cible suivante)
        if len(tokens_seq) < 2:
            continue

        indices_seq = encoder_sequence(tokens_seq, car_vers_idx)

        # X : de l'indice 0 jusqu'a l'avant-dernier (longueur T)
        X_indices = indices_seq[:-1]
        # E : du 1er indice jusqu'au dernier (longueur T)
        E_indices = indices_seq[1:]

        # Encodage One-Hot pour la propagation avant et la retropropagation
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


if __name__ == "__main__":
    chemin = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")
    print(f"Chargement et preparation des donnees depuis : {chemin}")
    exemples, c2i, i2c, V = preparer_donnees(chemin)
    
    print(f"\n[OK] Taille du vocabulaire V = {V} caracteres uniques.")
    print("Vocabulaire :", [i2c[i] for i in range(V)])
    print(f"Nombre total d'exemples de sequences : {len(exemples)}")

    premier = exemples[0]
    print("\n--- Exemple 1 ---")
    print(f"Longueur de sequence T = {premier['T']}")
    print("Sequence d'entree X (texte) :", repr(decoder_sequence(premier["X_indices"], i2c)))
    print("Sequence cible   E (texte) :", repr(decoder_sequence(premier["E_indices"], i2c)))
    print(f"Dernier token cible : {premier['E_tokens'][-1]} (doit etre '<fin>')")
    print(f"Dimensions X_one_hot[0] : longueur {len(premier['X_one_hot'][0])} (doit etre V={V})")
    print("--> Verification des donnees et du vocabulaire terminee avec succes !")
