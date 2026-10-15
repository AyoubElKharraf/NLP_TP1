"""
Module : entrainement.py
Description : Boucle d'entrainement par descente de gradient stochastique (SGD)
sur les sequences temporelles avec suivi de la perte.
"""

import os
import sys

# Ajout du dossier racine au sys.path si execute directement
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rnn import initialiser_parametres, propagation_avant, predire_caractere_argmax
from src.perte import erreur_sequence, erreur_globale_corpus
from src.bptt import retropropagation_bptt
from src.optimiseur import ecreter_gradients, mettre_a_jour_parametres
from src.vocabulaire import (
    tokeniser,
    construire_vocab,
    encoder_sequence,
    encoder_one_hot_sequence,
    decoder_sequence
)
from src.donnees import preparer_donnees


def entrainer_sgd(exemples, parametres, nb_epoques=100, delta=0.05, seuil=5.0, afficher_chaque=10):
    """
    Concept du cours : Entrainement stochastique (SGD) sequence par sequence.
    Pour chaque epoque :
      Pour chaque sequence (X, E) du corpus :
        1. Propagation avant : calcul de Ht et Yt
        2. Retropropagation temporelle (BPTT) : calcul des gradients dWxh, dWhh, dWhy, dbh, dby
        3. Ecretage du gradient : si ||grad|| > seuil
        4. Mise a jour SGD : P = P - delta * grad
    """
    historique_perte = []

    for epoque in range(1, nb_epoques + 1):
        perte_totale_epoque = 0.0
        nb_caracteres_epoque = 0

        for ex in exemples:
            X_seq = ex["X_one_hot"]
            E_seq = ex["E_one_hot"]
            T = ex["T"]

            # 1. Forward
            H, Y = propagation_avant(X_seq, parametres)

            # Perte sur la sequence
            perte_seq, _ = erreur_sequence(Y, E_seq)
            perte_totale_epoque += perte_seq
            nb_caracteres_epoque += T

            # 2. Backward (BPTT)
            gradients = retropropagation_bptt(X_seq, E_seq, H, Y, parametres)

            # 3. Ecretage (Gradient Clipping)
            ecreter_gradients(gradients, seuil)

            # 4. Descente de gradient (SGD)
            mettre_a_jour_parametres(parametres, gradients, delta)

        perte_moyenne = perte_totale_epoque / nb_caracteres_epoque
        historique_perte.append(perte_moyenne)

        if epoque == 1 or epoque % afficher_chaque == 0 or epoque == nb_epoques:
            print(f"Epoque {epoque:3d}/{nb_epoques} | Perte moyenne par caractere : {perte_moyenne:.4f}")

    return historique_perte


def tester_surapprentissage_mot_unique():
    """
    Concept du cours / Validation methodologique :
    Avant d'entrainer sur tout le corpus, on teste le RNN sur un seul mot/sequence court.
    Le modele doit sur-apprendre (overfit) rapidement et voir sa perte s'effondrer vers 0.
    """
    print("\n--- TEST DE SUR-APPRENTISSAGE SUR UNE SEQUENCE UNIQUE ---")
    mot_cible = "Bonjour<fin>"
    tokens = tokeniser(mot_cible)
    c2i, i2c = construire_vocab(tokens)
    V = len(c2i)
    H_dim = 16

    indices = encoder_sequence(tokens, c2i)
    X_one_hot = encoder_one_hot_sequence(indices[:-1], V)
    E_one_hot = encoder_one_hot_sequence(indices[1:], V)
    T = len(X_one_hot)

    exemple_unique = [{
        "X_one_hot": X_one_hot,
        "E_one_hot": E_one_hot,
        "T": T
    }]

    # Initialisation
    params_test = initialiser_parametres(taille_vocab=V, taille_cachee=H_dim, echelle=0.01)

    # Entrainement intensif sur ce seul mot
    print(f"Sequence cible a memoriser : '{mot_cible}' (T={T}, V={V})")
    historique = entrainer_sgd(exemple_unique, params_test, nb_epoques=60, delta=0.2, seuil=5.0, afficher_chaque=15)

    # Verification de la generation
    H, Y = propagation_avant(X_one_hot, params_test)
    texte_predit = ""
    for t in range(T):
        car_pred, _ = predire_caractere_argmax(Y[t], i2c)
        texte_predit += car_pred

    print(f"Texte attendu : '{''.join(tokens[1:])}'")
    print(f"Texte predit  : '{texte_predit}'")

    perte_initiale = historique[0]
    perte_finale = historique[-1]
    print(f"Baisse de perte : {perte_initiale:.4f} --> {perte_finale:.4f}")
    assert perte_finale < 0.2, f"Le sur-apprentissage a echoue (perte finale {perte_finale:.4f})"
    print("[OK] Test de sur-apprentissage reussi avec succes !")


if __name__ == "__main__":
    # 1. Test du sur-apprentissage sur une sequence courte
    tester_surapprentissage_mot_unique()

    # 2. Entrainement sur le corpus complet
    print("\n--- ENTRAINEMENT COMPLET SUR LE CORPUS ---")
    chemin_corpus = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")
    exemples, c2i, i2c, V = preparer_donnees(chemin_corpus)

    H_dim = 32
    parametres_corpus = initialiser_parametres(taille_vocab=V, taille_cachee=H_dim, echelle=0.01)

    print(f"Demarrage de l'entrainement sur {len(exemples)} exemples (V={V}, H={H_dim})...")
    # Entrainement sur 120 epoques avec delta=0.1
    entrainer_sgd(exemples, parametres_corpus, nb_epoques=120, delta=0.1, seuil=5.0, afficher_chaque=20)

    print("\n[OK] Entrainement termine avec succes !")
