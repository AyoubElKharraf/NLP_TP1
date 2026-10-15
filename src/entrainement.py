"""
Module : entrainement.py
Description : Boucle d'entrainement SGD avec suivi temporel, calcul de perplexite,
sauvegarde CSV et trace de la courbe d'apprentissage en ASCII dans le terminal.
Toutes les fonctions comptent strictement moins de 40 lignes.
"""

from math import exp
import time
from src.config import (
    DELTA,
    SEUIL_CLIPPING,
    NB_EPOQUES_SURAPPRENTISSAGE,
    NB_EPOQUES_CORPUS,
    FREQUENCE_PERPLEXITE,
    CHEMIN_PERTE_CSV
)
from src.rnn import initialiser_parametres, propagation_avant, predire_caractere_argmax
from src.perte import erreur_sequence, calculer_perplexite
from src.bptt import retropropagation_bptt
from src.optimiseur import ecreter_gradients, mettre_a_jour_parametres
from src.vocabulaire import (
    tokeniser,
    construire_vocab,
    encoder_sequence,
    encoder_one_hot_sequence
)


def tracer_courbe_ascii(valeurs, hauteur=10, largeur=45):
    """
    Genere un graphique ASCII de l'evolution de la perte pour affichage en terminal.
    """
    v_min, v_max = min(valeurs), max(valeurs)
    delta_v = (v_max - v_min) if v_max > v_min else 1.0
    nb_pts = len(valeurs)
    pts = [valeurs[int(c * (nb_pts - 1) / (largeur - 1))] for c in range(largeur)]

    grille = [[" " for _ in range(largeur)] for _ in range(hauteur)]
    for c in range(largeur):
        row = (hauteur - 1) - int((pts[c] - v_min) / delta_v * (hauteur - 1))
        grille[row][c] = "*"

    out = ["\n--- COURBE DE PERTE EN TERMINAL (ASCII) ---"]
    for r in range(hauteur):
        axe_y = v_max - (r / (hauteur - 1)) * delta_v
        out.append(f"{axe_y:6.3f} | {''.join(grille[r])}")
    out.append("       +" + "-" * largeur)
    out.append(f"       1{' ' * (largeur - 6)}Epoques ({nb_pts})")
    return "\n".join(out)


def pas_entrainement_sequence(ex, parametres, delta=DELTA, seuil=SEUIL_CLIPPING):
    """
    Execute un pas complet sur une sequence : Forward -> Perte -> BPTT -> Clip -> SGD.
    """
    X_seq = ex["X_one_hot"]
    E_seq = ex["E_one_hot"]
    H, Y = propagation_avant(X_seq, parametres)
    perte_seq, _ = erreur_sequence(Y, E_seq)
    gradients = retropropagation_bptt(X_seq, E_seq, H, Y, parametres)
    ecreter_gradients(gradients, seuil)
    mettre_a_jour_parametres(parametres, gradients, delta)
    return perte_seq, ex["T"]


def sauvegarder_historique_csv(historique, chemin=CHEMIN_PERTE_CSV):
    """
    Sauvegarde la perte moyenne et la perplexite par epoque dans un fichier CSV.
    """
    with open(chemin, "w", encoding="utf-8") as f:
        f.write("epoque,perte_moyenne,perplexite\n")
        for ep, perte, ppl in historique:
            f.write(f"{ep},{perte:.6f},{ppl:.6f}\n")


def entrainer_sgd(exemples, parametres, nb_epoques=NB_EPOQUES_CORPUS, delta=DELTA, seuil=SEUIL_CLIPPING):
    """
    Boucle d'entrainement par descente de gradient stochastique sur le corpus.
    Affiche le temps par epoque et la perplexite toutes les 10 epoques.
    """
    historique = []
    print(f"Demarrage entrainement SGD ({nb_epoques} epoques)...")

    for ep in range(1, nb_epoques + 1):
        t0 = time.time()
        perte_totale, nb_cars = 0.0, 0
        for ex in exemples:
            p_seq, t_len = pas_entrainement_sequence(ex, parametres, delta, seuil)
            perte_totale += p_seq
            nb_cars += t_len

        perte_moy = perte_totale / nb_cars
        ppl = calculer_perplexite(perte_moy)
        dt = time.time() - t0
        historique.append((ep, perte_moy, ppl))

        if ep == 1 or ep % FREQUENCE_PERPLEXITE == 0 or ep == nb_epoques:
            print(f"Epoque {ep:3d}/{nb_epoques} | Perte: {perte_moy:.4f} | Perplexite: {ppl:.2f} | Temps: {dt*1000:.1f}ms")

    sauvegarder_historique_csv(historique)
    pertes_seules = [p for _, p, _ in historique]
    print(tracer_courbe_ascii(pertes_seules))
    return historique


def tester_surapprentissage_mot_unique(nb_epoques=NB_EPOQUES_SURAPPRENTISSAGE):
    """
    Validation methodologique : verifie que le Tanh-RNN peut sur-apprendre un mot unique.
    """
    print("\n--- TEST DE SUR-APPRENTISSAGE SUR UN MOT UNIQUE ---")
    mot = "bonjour<fin>"
    tokens = tokeniser(mot)
    c2i, i2c = construire_vocab(tokens)
    V, H_dim = len(c2i), 16

    indices = encoder_sequence(tokens, c2i)
    ex_test = [{
        "X_one_hot": encoder_one_hot_sequence(indices[:-1], V),
        "E_one_hot": encoder_one_hot_sequence(indices[1:], V),
        "T": len(indices) - 1
    }]

    params = initialiser_parametres(taille_vocab=V, taille_cachee=H_dim, borne=0.1)
    for ep in range(1, nb_epoques + 1):
        pas_entrainement_sequence(ex_test[0], params, delta=0.2, seuil=5.0)

    H, Y = propagation_avant(ex_test[0]["X_one_hot"], params)
    perte, _ = erreur_sequence(Y, ex_test[0]["E_one_hot"])
    predit = "".join([predire_caractere_argmax(Y[t], i2c)[0] for t in range(ex_test[0]["T"])])
    attendu = "".join(tokens[1:])

    print(f"Attendu : '{attendu}' | Predit : '{predit}' | Perte finale : {perte:.4f}")
    assert perte < 0.1, f"Echec du sur-apprentissage : perte {perte:.4f} trop elevee"
    print("[OK] Test de sur-apprentissage valide avec succes !")
