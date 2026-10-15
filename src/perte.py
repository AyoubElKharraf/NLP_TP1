"""
Module : perte.py
Description : Calcul de la fonction de perte (entropie croisee).
Equations du cours :
  - Err(Yt, Et) = - somme_k ( Et[k] * log(Yt[k]) )   (entropie croisee a l'instant t)
  - Err(X) = somme_t ( Err(Yt, Et) )                 (erreur cumulative d'une sequence)
  - Err globale = moyenne de Err(X) sur le corpus
"""

import os
import sys
from math import log

# Ajout du dossier racine au sys.path si execute directement
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.maths_utils import logarithme
from src.rnn import propagation_avant


def entropie_croisee(Yt, Et_one_hot):
    """
    Concept du cours : Entropie croisee entre la sortie predite Yt et la cible Et.
    Equation : Err(Yt, Et) = - somme_{k=0}^{V-1} ( Et[k] * log(Yt[k]) )
    Puisque Et est un vecteur One-Hot, seul le terme de la classe cible est non nul.
    """
    perte_t = 0.0
    for k in range(len(Yt)):
        if Et_one_hot[k] > 0.0:
            perte_t -= Et_one_hot[k] * logarithme(Yt[k])
    return perte_t


def erreur_sequence(Y, E_one_hot_sequence):
    """
    Concept du cours : Erreur totale sur une sequence X de longueur T.
    Equation : Err(X) = somme_{t=0}^{T-1} Err(Yt, Et)
    
    Arguments :
      - Y : dictionnaire des sorties { 0: Y0, 1: Y1, ..., T-1: YT-1 }
      - E_one_hot_sequence : liste des vecteurs cibles [E0, E1, ..., ET-1]
    Retourne :
      - perte_totale : somme des pertes sur toute la sequence
      - perte_moyenne_par_caractere : perte_totale / T
    """
    T = len(E_one_hot_sequence)
    perte_totale = 0.0

    for t in range(T):
        perte_t = entropie_croisee(Y[t], E_one_hot_sequence[t])
        perte_totale += perte_t

    perte_moyenne = perte_totale / T if T > 0 else 0.0
    return perte_totale, perte_moyenne


def erreur_globale_corpus(exemples, parametres):
    """
    Concept du cours : Erreur globale moyenne sur l'ensemble du corpus.
    Equation : Err_globale = (1 / N) * somme_{i=1}^N Err(X^(i))
    
    Arguments :
      - exemples : liste des exemples de donnees
      - parametres : dictionnaire des poids du RNN (Wxh, Whh, Why, bh, by)
    Retourne :
      - perte_moyenne_sequence : erreur moyenne par sequence complete
      - perte_moyenne_caractere : erreur moyenne par caractere predit
    """
    nb_exemples = len(exemples)
    somme_erreurs_sequences = 0.0
    somme_erreurs_caracteres = 0.0
    nombre_total_caracteres = 0

    for ex in exemples:
        X_seq = ex["X_one_hot"]
        E_seq = ex["E_one_hot"]
        T = ex["T"]

        # Propagation avant
        H, Y = propagation_avant(X_seq, parametres)

        # Calcul de la perte de la sequence
        perte_seq, _ = erreur_sequence(Y, E_seq)
        somme_erreurs_sequences += perte_seq
        somme_erreurs_caracteres += perte_seq
        nombre_total_caracteres += T

    perte_moyenne_sequence = somme_erreurs_sequences / nb_exemples if nb_exemples > 0 else 0.0
    perte_moyenne_caractere = somme_erreurs_caracteres / nombre_total_caracteres if nombre_total_caracteres > 0 else 0.0

    return perte_moyenne_sequence, perte_moyenne_caractere


if __name__ == "__main__":
    from src.donnees import preparer_donnees
    from src.rnn import initialiser_parametres

    chemin_corpus = os.path.join(os.path.dirname(__file__), "..", "data", "corpus.txt")
    exemples, c2i, i2c, V = preparer_donnees(chemin_corpus)

    # Initialisation aleatoire
    H_dim = 32
    parametres = initialiser_parametres(taille_vocab=V, taille_cachee=H_dim, echelle=0.01)

    # Test sur le 1er exemple
    premier = exemples[0]
    H, Y = propagation_avant(premier["X_one_hot"], parametres)
    perte_seq, perte_seq_moy = erreur_sequence(Y, premier["E_one_hot"])

    # Test global sur le corpus
    perte_glob_seq, perte_glob_car = erreur_globale_corpus(exemples, parametres)

    # Valeur theorique attendue a l'initialisation aleatoire : log(V)
    perte_theorique = log(V)

    print("=== Verification du calcul de la Perte (Entropie Croisee) ===")
    print(f"Taille du vocabulaire V = {V}")
    print(f"Perte theorique initiale attendue = log({V}) = {perte_theorique:.4f}")
    print(f"\n--- Resultats avec poids aleatoires ---")
    print(f"Erreur totale exemple 1 (T={premier['T']}) : {perte_seq:.4f}")
    print(f"Erreur moyenne par caractere exemple 1    : {perte_seq_moy:.4f}")
    print(f"Erreur moyenne par sequence sur le corpus : {perte_glob_seq:.4f}")
    print(f"Erreur moyenne par caractere sur le corpus: {perte_glob_car:.4f}")

    # Verification de coherence mathematique
    ecart = abs(perte_glob_car - perte_theorique)
    print(f"\nEcart avec la valeur theorique : {ecart:.4f}")
    assert ecart < 0.05, f"Attention : la perte initiale ({perte_glob_car:.4f}) s'eloigne trop de log(V) ({perte_theorique:.4f})"

    print("[OK] Test de conformite reussi : La perte initiale est bien tres proche de log(V) !")
