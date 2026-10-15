"""
Module : entrainement.py
Description : Boucle d'entrainement SGD, suivi de perte, perplexite, courbe ASCII,
et sauvegarde des resultats dans resultats/perte.csv et resultats/exemples.txt.
"""

import time
from math import exp
from src.rnn import initialiser_parametres, propagation_avant
from src.perte import erreur_sequence, calculer_perplexite
from src.bptt import retropropagation_bptt
from src.optimiseur import ecreter_gradients, mettre_a_jour_parametres
from src.vocabulaire import (
    tokeniser,
    construire_vocab,
    encoder_sequence,
    encoder_one_hot_sequence
)
from src.generation import generer_texte
from src.config import (
    DELTA,
    SEUIL_ECRETAGE,
    EPOQUES_MOT,
    EPOQUES_CORPUS,
    TEMPERATURE,
    MAX_LONGUEUR,
    FICHIER_PERTE_CSV,
    FICHIER_EXEMPLES_TXT
)


def entrainer_une_epoque(exemples, parametres, delta=DELTA, seuil=SEUIL_ECRETAGE):
    """
    Execute une epoque d'apprentissage stochastique (SGD) sequence par sequence.
    Retourne la perte moyenne par caractere et la duree en secondes.
    """
    t_debut = time.time()
    perte_totale = 0.0
    nb_caracteres = 0

    for ex in exemples:
        X_seq, E_seq, T = ex["X_one_hot"], ex["E_one_hot"], ex["T"]
        H, Y = propagation_avant(X_seq, parametres)
        p_seq, _ = erreur_sequence(Y, E_seq)
        perte_totale += p_seq
        nb_caracteres += T

        gradients = retropropagation_bptt(X_seq, E_seq, H, Y, parametres)
        ecreter_gradients(gradients, seuil)
        mettre_a_jour_parametres(parametres, gradients, delta)

    duree = time.time() - t_debut
    perte_moyenne = perte_totale / nb_caracteres if nb_caracteres > 0 else 0.0
    return perte_moyenne, duree


def afficher_courbe_perte_ascii(historique, largeur=50, hauteur=10):
    """
    Trace une courbe de decroissance de perte en ASCII dans le terminal.
    Matplotlib etant interdit, cette fonction offre une visualisation immediate.
    """
    if not historique:
        return
    min_p = min(historique)
    max_p = max(historique)
    delta_p = (max_p - min_p) if max_p > min_p else 1.0

    print("\n--- COURBE DE PERTE (ASCII) ---")
    echantillons = []
    pas = max(1, len(historique) // largeur)
    for i in range(0, len(historique), pas):
        echantillons.append(historique[i])

    for h in range(hauteur, -1, -1):
        seuil = min_p + (h / hauteur) * delta_p
        ligne = f"{seuil:5.2f} | "
        for p in echantillons:
            ligne += "*" if abs(p - seuil) <= (delta_p / (2 * hauteur)) or (h == 0 and p <= seuil) else " "
        print(ligne)
    print("      +" + "-" * len(echantillons))
    print(f"       Epoque 1 {' ' * max(0, len(echantillons) - 15)} Epoque {len(historique)}\n")


def sauvegarder_perte_csv(chemin_csv, historique_donnees):
    """Sauvegarde les metriques (epoque, perte, perplexite, temps) au format CSV."""
    with open(chemin_csv, "w", encoding="utf-8") as f:
        f.write("epoque,perte_moyenne,perplexite,temps_s\n")
        for ep, p, ppl, t in historique_donnees:
            f.write(f"{ep},{p:.6f},{ppl:.6f},{t:.4f}\n")


def tester_surapprentissage_mot(nb_epoques=EPOQUES_MOT):
    """
    Validation methodologique : verifie qu'un Tanh-RNN sur-apprend un seul mot.
    La perte doit s'effondrer et le mot etre reproduit sans faute.
    """
    print("\n=== TEST DE SUR-APPRENTISSAGE SUR UN MOT UNIQUE ===")
    mot = "bonjour<fin>"
    tokens = tokeniser(mot)
    c2i, i2c = construire_vocab(tokens)
    V, H_dim = len(c2i), 16

    ind = encoder_sequence(tokens, c2i)
    ex = [{"X_one_hot": encoder_one_hot_sequence(ind[:-1], V),
           "E_one_hot": encoder_one_hot_sequence(ind[1:], V),
           "T": len(ind) - 1}]

    params = initialiser_parametres(V, H_dim, echelle=0.1)
    for ep in range(1, nb_epoques + 1):
        p, _ = entrainer_une_epoque(ex, params, delta=0.1, seuil=5.0)

    _, Y = propagation_avant(ex[0]["X_one_hot"], params)
    pred = "".join([i2c[max(range(V), key=lambda k: Y[t][k])] for t in range(ex[0]["T"])])
    print(f"Cible : '{mot[1:]}' | Predit : '{pred}' | Perte finale : {p:.4f}")
    assert p < 0.2 and pred == mot[1:], "Echec du sur-apprentissage !"
    print("[OK] Sur-apprentissage reussi : perte < 0.2 et texte exact.")


def evaluer_et_sauvegarder_exemples(parametres, c2i, i2c, chemin_sortie=FICHIER_EXEMPLES_TXT):
    """
    Evalue 10 questions (5 presentes dans le corpus et 5 absentes),
    compare le decodage Argmax et avec Temperature 0.8, et sauvegarde le resultat.
    """
    questions_corpus = [
        "bonjour",
        "comment tu t'appelles ?",
        "qui es-tu ?",
        "que fais-tu ?",
        "au revoir"
    ]
    questions_hors_corpus = [
        "quel temps fait-il ?",
        "combien font deux plus deux ?",
        "aimes-tu le chocolat ?",
        "ou se trouve rabat ?",
        "qui a gagne le match ?"
    ]
    toutes_questions = [(q, True) for q in questions_corpus] + [(q, False) for q in questions_hors_corpus]

    lignes = ["=== EVALUATION DE 10 QUESTIONS (5 CORPUS / 5 HORS-CORPUS) ===\n\n"]
    for q, est_corpus in toutes_questions:
        amorce = f"Q: {q}\nR:"
        rep_argmax = generer_texte(amorce, parametres, c2i, i2c, max_caracteres=MAX_LONGUEUR, temperature=0.0).strip()
        rep_temp = generer_texte(amorce, parametres, c2i, i2c, max_caracteres=MAX_LONGUEUR, temperature=TEMPERATURE).strip()
        statut = "OUI (dans le corpus)" if est_corpus else "NON (hors corpus - limite du modele)"

        bloc = f"Question : '{q}'\nPresente dans le corpus : {statut}\n"
        bloc += f"  - Reponse Argmax (temp=0.0) : '{rep_argmax}'\n"
        bloc += f"  - Reponse Tirage (temp={TEMPERATURE}) : '{rep_temp}'\n\n"
        lignes.append(bloc)

    with open(chemin_sortie, "w", encoding="utf-8") as f:
        f.writelines(lignes)
    print(f"[OK] 10 exemples sauvegardes dans : {chemin_sortie}")


def entrainer_sur_corpus(exemples, parametres, nb_epoques=EPOQUES_CORPUS, delta=DELTA, seuil=SEUIL_ECRETAGE):
    """
    Entraine le RNN sur le corpus complet, affiche la perte, la perplexite,
    le temps par epoque, et enregistre l'historique dans perte.csv.
    """
    historique_csv = []
    historique_pertes = []

    print(f"\n--- ENTRAINEMENT DU RNN SUR LE CORPUS ({nb_epoques} EPOQUES) ---")
    for ep in range(1, nb_epoques + 1):
        perte, duree = entrainer_une_epoque(exemples, parametres, delta, seuil)
        ppl = calculer_perplexite(perte)
        historique_pertes.append(perte)
        historique_csv.append((ep, perte, ppl, duree))

        if ep == 1 or ep % 10 == 0 or ep == nb_epoques:
            print(f"Epoque {ep:3d}/{nb_epoques} | Perte : {perte:.4f} | Perplexite : {ppl:6.2f} | Temps : {duree*1000:5.1f} ms")

    sauvegarder_perte_csv(FICHIER_PERTE_CSV, historique_csv)
    afficher_courbe_perte_ascii(historique_pertes)
    return historique_pertes
