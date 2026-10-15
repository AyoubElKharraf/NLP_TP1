"""
Module : perte.py
Description : Calcul de l'entropie croisee, de l'erreur d'une sequence et de la perplexite.
Equations du cours :
  - Err(Yt, Et) = - somme_k Et[k] * log(Yt[k] + 1e-12)
  - Err(X) = somme_t Err(Yt, Et)
  - Err globale = moyenne de Err(X) sur le corpus
  - Perplexite = exp(perte_moyenne_par_caractere)
"""

from math import exp, log
from src.rnn import propagation_avant


def entropie_croisee(Yt, Et_one_hot):
    """
    Equation : Err(Yt, Et) = - somme_{k=0}^{V-1} ( Et[k] * log(Yt[k] + 1e-12) )
    Entropie croisee stable avec epsilon = 1e-12 pour eviter log(0).
    """
    perte = 0.0
    for k in range(len(Yt)):
        if Et_one_hot[k] > 0.0:
            val = Yt[k] if Yt[k] > 1e-12 else 1e-12
            perte -= Et_one_hot[k] * log(val)
    return perte


def erreur_sequence(Y, E_one_hot_sequence):
    """
    Equation : Err(X) = somme_{t=0}^{T-1} Err(Yt, Et)
    Calcule l'erreur totale et moyenne sur une sequence de longueur T.
    """
    T = len(E_one_hot_sequence)
    perte_totale = 0.0
    for t in range(T):
        perte_totale += entropie_croisee(Y[t], E_one_hot_sequence[t])
    perte_moyenne = perte_totale / T if T > 0 else 0.0
    return perte_totale, perte_moyenne


def calculer_perplexite(perte_moyenne_caractere):
    """
    Equation : Perplexite = exp(Err_moyenne_caractere)
    Mesure la qualite du modele de langue (incertitude de prediction par caractere).
    """
    if perte_moyenne_caractere > 50.0:
        return exp(50.0)
    return exp(perte_moyenne_caractere)


def erreur_globale_corpus(exemples, parametres):
    """
    Equation : Err_globale = (1 / N) * somme_{i=1}^N Err(X^(i))
    Calcule la perte moyenne et la perplexite globale sur le corpus.
    """
    nb = len(exemples)
    tot_perte = 0.0
    tot_car = 0
    for ex in exemples:
        _, Y = propagation_avant(ex["X_one_hot"], parametres)
        p_seq, _ = erreur_sequence(Y, ex["E_one_hot"])
        tot_perte += p_seq
        tot_car += ex["T"]

    moy_seq = tot_perte / nb if nb > 0 else 0.0
    moy_car = tot_perte / tot_car if tot_car > 0 else 0.0
    ppl = calculer_perplexite(moy_car)
    return moy_seq, moy_car, ppl
