"""
Fichier principal : main.py
Description : Point d'entree du projet (orchestration uniquement).
1. Charge les donnees du corpus (paires Q/R)
2. Initialise les poids du Tanh-RNN
3. Lance l'apprentissage par descente de gradient stochastique
4. Ouvre une session interactive de chat en terminal pour interroger le RNN
"""

import os
import sys

from src.donnees import preparer_donnees
from src.rnn import initialiser_parametres
from src.entrainement import entrainer_sgd
from src.generation import generer_texte


def main():
    # 1. Chemins et chargement des donnees
    dossier_base = os.path.dirname(os.path.abspath(__file__))
    chemin_corpus = os.path.join(dossier_base, "data", "corpus.txt")

    print("================================================================")
    print("      TP TALN : RESEAU DE NEURONES RECURRENT (Tanh-RNN)         ")
    print("      Apprentissage de Modeles de Langue & Chatbot             ")
    print("================================================================\n")

    print(f"--> [1/4] Chargement et preparation du corpus ({chemin_corpus})...")
    exemples, c2i, i2c, V = preparer_donnees(chemin_corpus)
    print(f"    * Nombre de sequences : {len(exemples)}")
    print(f"    * Taille du vocabulaire : {V} caracteres uniques\n")

    # 2. Initialisation des parametres
    H_dim = 32
    print(f"--> [2/4] Initialisation des parametres (H={H_dim}, echelle=0.01)...")
    parametres = initialiser_parametres(taille_vocab=V, taille_cachee=H_dim, echelle=0.01)
    print("    * Matrices creees : Wxh (32x34), Whh (32x32), Why (34x32)")
    print("    * Biais crees     : bh (32), by (34)\n")

    # 3. Entrainement SGD
    nb_epoques = 250
    delta = 0.15
    seuil = 5.0
    print(f"--> [3/4] Entrainement par descente de gradient stochastique ({nb_epoques} epoques)...")
    entrainer_sgd(exemples, parametres, nb_epoques=nb_epoques, delta=delta, seuil=seuil, afficher_chaque=50)
    print("    * Entrainement termine avec succes !\n")

    # 4. Demonstration automatique sur les questions du corpus
    print("--> [4/4] Demonstration de generation automatique (Argmax) :")
    questions_corpus = [
        "Bonjour",
        "Comment vas-tu?",
        "Qui es-tu?",
        "Que fais-tu?"
    ]
    for q in questions_corpus:
        amorce = f"Q: {q}\nR:"
        reponse = generer_texte(amorce, parametres, c2i, i2c, max_caracteres=60, temperature=0.0)
        print(f"    Q: {q}")
        print(f"    R: {reponse.strip()}\n")

    # Mode interactif (si un terminal interactif est disponible)
    if sys.stdin.isatty() and "--demo" not in sys.argv:
        print("----------------------------------------------------------------")
        print("  SESSION DE CHAT INTERACTIF")
        print("  Posez vos questions au RNN (tapez 'quitter' pour arreter)")
        print("----------------------------------------------------------------")
        while True:
            try:
                question_utilisateur = input("\nVous > ").strip()
                if not question_utilisateur:
                    continue
                if question_utilisateur.lower() in ["quitter", "exit", "q"]:
                    print("\nAu revoir !")
                    break

                amorce = f"Q: {question_utilisateur}\nR:"
                reponse = generer_texte(amorce, parametres, c2i, i2c, max_caracteres=60, temperature=0.0)
                print(f"RNN  > {reponse.strip()}")
            except (KeyboardInterrupt, EOFError):
                print("\nSession terminee.")
                break


if __name__ == "__main__":
    main()
