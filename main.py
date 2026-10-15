"""
Fichier principal : main.py
Description : Point d'entree orchestrant l'entrainement, les evaluations et le chat.
Aucune logique algorithmique n'est definie ici, seulement des appels aux modules src.
"""

import sys
from src.config import (
    TAILLE_H,
    ECHELLE_POIDS,
    DELTA,
    SEUIL_ECRETAGE,
    EPOQUES_CORPUS,
    TEMPERATURE,
    MAX_LONGUEUR,
    FICHIER_CORPUS,
    FICHIER_EXEMPLES_TXT
)
from src.donnees import preparer_donnees
from src.rnn import initialiser_parametres
from src.entrainement import (
    tester_surapprentissage_mot,
    entrainer_sur_corpus,
    evaluer_et_sauvegarder_exemples
)
from src.generation import generer_texte


def executer_pipeline_apprentissage():
    """Charge les donnees, initialise le Tanh-RNN et lance l'apprentissage."""
    print("================================================================")
    print("      TP TALN : RESEAU DE NEURONES RECURRENT (Tanh-RNN)         ")
    print("      Apprentissage de Modeles de Langue & Chatbot             ")
    print("================================================================\n")

    # 1. Validation de depart par sur-apprentissage sur un mot unique
    tester_surapprentissage_mot()

    # 2. Chargement du corpus (20 paires Q/R en francais)
    print(f"\nChargement du corpus depuis '{FICHIER_CORPUS}'...")
    exemples, c2i, i2c, V = preparer_donnees(FICHIER_CORPUS)
    print(f"Corpus charge : {len(exemples)} paires Q/R, vocabulaire V = {V} caracteres.")

    # 3. Initialisation des parametres
    parametres = initialiser_parametres(taille_vocab=V, taille_cachee=TAILLE_H, echelle=ECHELLE_POIDS)

    # 4. Entrainement sur le corpus complet
    entrainer_sur_corpus(exemples, parametres, nb_epoques=EPOQUES_CORPUS, delta=DELTA, seuil=SEUIL_ECRETAGE)

    # 5. Sauvegarde et evaluation des 10 exemples (5 corpus / 5 hors corpus)
    evaluer_et_sauvegarder_exemples(parametres, c2i, i2c, chemin_sortie=FICHIER_EXEMPLES_TXT)

    return parametres, c2i, i2c


def lancer_session_chat(parametres, c2i, i2c):
    """Boucle interactive de dialogue avec l'utilisateur dans le terminal."""
    print("\n----------------------------------------------------------------")
    print("  SESSION DE CHAT INTERACTIF")
    print("  Posez une question en francais (ou tapez 'quitter' pour sortir)")
    print("----------------------------------------------------------------")
    while True:
        try:
            entree = input("\nVous > ").strip()
            if not entree:
                continue
            if entree.lower() in ["quitter", "exit", "q"]:
                print("Fin de la session.")
                break
            amorce = f"Q: {entree.lower()}\nR:"
            rep = generer_texte(amorce, parametres, c2i, i2c, max_caracteres=MAX_LONGUEUR, temperature=TEMPERATURE)
            print(f"RNN  > {rep.strip()}")
        except (KeyboardInterrupt, EOFError):
            print("\nSortie du chat.")
            break


def main():
    """Point d'entree : execute l'entrainement puis lance le chat."""
    parametres, c2i, i2c = executer_pipeline_apprentissage()

    if "--demo" not in sys.argv:
        lancer_session_chat(parametres, c2i, i2c)


if __name__ == "__main__":
    main()
