"""
Module : perte.py
Description : Entropie croisee et calcul de la perplexite pour le Tanh-RNN.
Equations du cours :
  - Err(Yt, Et) = - somme_k ( Et[k] * log(Yt[k] + epsilon) )
  - Err(X) = somme_t ( Err(Yt, Et) )
  - Perplexite = exp( Err(X) / T )
Chaque fonction compte moins de 40 lignes.
"""

from math import log, exp
from src.config import EPSILON_LOG
from src.rnn import propagation_avant


def entropie_croisee(Yt, Et_one_hot):
    """
    Concept du cours : Entropie croisee a l'instant t avec stabilite numerique (1e-12).
    Equation : Err(Yt, Et) = - somme_k ( Et[k] * log(Yt[k] + 1e-12) )
    """
    perte_t = 0.0
    for k in range(len(Yt)):
        if Et_one_hot[k] > 0.0:
            perte_t -= Et_one_hot[k] * log(Yt[k] + EPSILON_LOG)
    return perte_t


def erreur_sequence(Y, E_one_hot_sequence):
    """
    Concept du cours : Erreur cumulative sur une sequence complete de longueur T.
    Equation : Err(X) = somme_{t=0}^{T-1} Err(Yt, Et)
    """
    T = len(E_one_hot_sequence)
    perte_totale = 0.0
    for t in range(T):
        perte_totale += entropie_croisee(Y[t], E_one_hot_sequence[t])
    perte_moyenne = perte_totale / T if T > 0 else 0.0
    return perte_totale, perte_moyenne


def calculer_perplexite(perte_moyenne_caractere):
    """
    Concept du cours : Perplexite du modele de langue.
    Equation : Perplexite = exp( perte_moyenne_par_caractere )
    """
    if perte_moyenne_caractere > 50.0:
        perte_moyenne_caractere = 50.0
    return exp(perte_moyenne_caractere)


def erreur_globale_corpus(exemples, parametres):
    """
    Concept du cours : Erreur globale moyenne et perplexite sur le corpus.
    Equation : Err_globale = (1 / N) * somme_i Err(X^(i))
    """
    nb_exemples = len(exemples)
    somme_perte_totale = 0.0
    total_caracteres = 0

    for ex in exemples:
        H, Y = propagation_avant(ex["X_one_hot"], parametres)
        perte_seq, _ = erreur_sequence(Y, ex["E_one_hot"])
        somme_perte_totale += perte_seq
        total_caracteres += ex["T"]

    perte_moyenne_seq = somme_perte_totale / nb_exemples if nb_exemples > 0 else 0.0
    perte_moyenne_car = somme_perte_totale / total_caracteres if total_caracteres > 0 else 0.0
    ppl = calculer_perplexite(perte_moyenne_car)

    return perte_moyenne_seq, perte_moyenne_car, ppl
