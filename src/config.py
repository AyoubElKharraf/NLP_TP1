"""
Module : config.py
Description : Configuration globale et hyperparametres du projet.
Toutes les valeurs modifiables sont centralisees ici conformement au cahier des charges.
"""

import random

# =====================================================================
# 1. Graine aleatoire pour reproductibilite totale
# =====================================================================
GRAINE_ALEATOIRE = 42
random.seed(GRAINE_ALEATOIRE)

# =====================================================================
# 2. Hyperparametres du reseau Tanh-RNN
# =====================================================================
TAILLE_CACHEE = 32          # Dimension de l'etat cache H
BORNE_INIT = 0.1            # Poids initialises uniformement dans [-0.1, 0.1]
DELTA = 0.1                 # Pas de la descente de gradient stochastique (learning rate)
SEUIL_CLIPPING = 5.0        # Seuil d'ecretage du gradient (norme globale)
EPSILON_LOG = 1e-12         # Stabilite numerique pour eviter log(0) dans l'entropie croisee

# =====================================================================
# 3. Parametres d'entrainement
# =====================================================================
NB_EPOQUES_SURAPPRENTISSAGE = 200  # Epoques pour le test sur un mot unique
NB_EPOQUES_CORPUS = 200            # Epoques pour l'entrainement sur le corpus complet
FREQUENCE_PERPLEXITE = 10          # Calculer la perplexite toutes les 10 epoques

# =====================================================================
# 4. Parametres de generation et decodage
# =====================================================================
TEMPERATURE = 0.8                  # Temperature pour l'echantillonnage
LONGUEUR_MAX_GENERATION = 100      # Longueur maximale d'une reponse generee

# =====================================================================
# 5. Chemins des donnees et resultats
# =====================================================================
CHEMIN_CORPUS = "data/corpus.txt"
CHEMIN_PERTE_CSV = "resultats/perte.csv"
CHEMIN_EXEMPLES_TXT = "resultats/exemples.txt"
