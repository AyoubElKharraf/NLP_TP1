"""
Script de test unitaire pour maths_utils.py.
Execution : python tests/test_maths.py
Toutes les fonctions comptent strictement moins de 40 lignes.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.maths_utils import (
    tanh_scalaire,
    tanh_vecteur,
    derivee_tanh,
    softmax,
    produit_matrice_vecteur,
    produit_exterieur,
    somme_vecteurs,
    soustraction_vecteurs,
    transposee,
    norme_vecteur,
    norme_globale,
    creer_matrice_aleatoire,
    fixer_graine
)


def tester_tanh():
    """Verifie tanh(0)=0, la symetrie impaire et la derivee."""
    assert abs(tanh_scalaire(0.0)) < 1e-9, "tanh(0) doit valoir 0"
    assert abs(tanh_scalaire(1.5) + tanh_scalaire(-1.5)) < 1e-9, "tanh impaire"
    assert tanh_scalaire(50.0) == 1.0, "saturation haute"
    assert tanh_scalaire(-50.0) == -1.0, "saturation basse"
    assert abs(derivee_tanh(0.0) - 1.0) < 1e-9, "tanh'(0) = 1"
    print("[OK] Tests tanh et derivee valides.")


def tester_softmax():
    """Verifie la normalisation a 1.0 et la robustesse au debordement."""
    v = [1.0, 2.0, 3.0]
    p = softmax(v)
    assert abs(sum(p) - 1.0) < 1e-7, "Somme softmax != 1.0"
    assert p[2] > p[1] > p[0], "Ordre de softmax incoherent"

    grands = [1000.0, 1001.0, 1002.0]
    p_grands = softmax(grands)
    assert abs(sum(p_grands) - 1.0) < 1e-7, "Overflow dans softmax"
    print("[OK] Tests softmax valides.")


def tester_algebre_lineaire():
    """Verifie les produits matriciels, exterieur et la transposee."""
    M = [[1.0, 2.0], [3.0, 4.0]]
    v = [5.0, 6.0]
    assert produit_matrice_vecteur(M, v) == [17.0, 39.0], "Erreur produit MV"

    u, w = [1.0, 2.0], [3.0, 4.0]
    attendu_ext = [[3.0, 4.0], [6.0, 8.0]]
    assert produit_exterieur(u, w) == attendu_ext, "Erreur produit exterieur"

    assert transposee(M) == [[1.0, 3.0], [2.0, 4.0]], "Erreur transposee"
    assert abs(norme_vecteur([3.0, 4.0]) - 5.0) < 1e-9, "Erreur norme vecteur"
    assert abs(norme_globale([M]) - (1 + 4 + 9 + 16)**0.5) < 1e-9, "Erreur norme globale"
    print("[OK] Tests algebre lineaire valides.")


def tester_initialisation():
    """Verifie la generation uniforme des poids."""
    fixer_graine(42)
    M = creer_matrice_aleatoire(3, 4, borne=0.1)
    assert len(M) == 3 and len(M[0]) == 4, "Dimensions incorrectes"
    for row in M:
        for val in row:
            assert -0.1 <= val <= 0.1, f"Valeur hors borne : {val}"
    print("[OK] Tests initialisation valides.")


if __name__ == "__main__":
    print("=== Execution des tests unitaires de maths_utils.py ===")
    tester_tanh()
    tester_softmax()
    tester_algebre_lineaire()
    tester_initialisation()
    print("--> TOUS LES TESTS MATHEMATIQUES ONT REUSSI AVEC SUCCES !")
