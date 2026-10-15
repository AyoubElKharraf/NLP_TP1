"""
Module : config.py
Description : Configuration globale et hyperparametres du projet Tanh-RNN.
Toutes les valeurs modifiables sont centralisees ici pour garantir la reproductibilite.
"""

from random import seed

# Graine aleatoire pour garantir la reproductibilite
GRAINE = 42
seed(GRAINE)

# Architecture du reseau
TAILLE_H = 32              # Dimension de l'etat cache Ht
ECHELLE_POIDS = 0.1        # Initialisation uniforme dans [-0.1, 0.1]

# Optimisation et entrainement
DELTA = 0.1                # Taux d'apprentissage (pas delta)
SEUIL_ECRETAGE = 5.0       # Seuil d'ecretage du gradient (norme globale)
EPOQUES_MOT = 200          # Nombre d'epoques pour le sur-apprentissage sur un mot
EPOQUES_CORPUS = 60        # Nombre d'epoques d'entrainement sur le corpus (ajuste)

# Generation de texte
TEMPERATURE = 0.8          # Temperature pour le tirage probabiliste
MAX_LONGUEUR = 100         # Longueur maximale d'une reponse generee

# Chemins de fichiers
FICHIER_CORPUS = "data/corpus.txt"
FICHIER_PERTE_CSV = "resultats/perte.csv"
FICHIER_EXEMPLES_TXT = "resultats/exemples.txt"
