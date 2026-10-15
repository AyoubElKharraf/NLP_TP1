"""
Module : test_maths.py
Description : Tests unitaires de maths_utils.py (tanh, softmax stable, algebre lineaire).
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.maths_utils import (
    tanh_scalaire,
    derivee_tanh,
    softmax,
    produit_matrice_vecteur,
    produit_exterieur,
    somme_vecteurs,
    norme_vecteur,
    norme_globale_matrices
)


def tester_tanh():
    """Verifie tanh(0)=0, imparite, saturation et derivee."""
    assert abs(tanh_scalaire(0.0)) < 1e-9, "tanh(0) doit etre 0"
    assert abs(tanh_scalaire(1.0) + tanh_scalaire(-1.0)) < 1e-9, "tanh impaire"
    assert tanh_scalaire(50.0) == 1.0, "saturation haute"
    assert tanh_scalaire(-50.0) == -1.0, "saturation basse"
    assert abs(derivee_tanh(0.0) - 1.0) < 1e-9, "derivee en 0 = 1"
    print("[OK] Test tanh et derivee valide.")


def tester_softmax():
    """Verifie somme a 1, positivité et stabilite aux grands nombres."""
    p = softmax([1.0, 2.0, 3.0])
    assert abs(sum(p) - 1.0) < 1e-7, "Somme softmax = 1"
    assert p[2] > p[1] > p[0], "Ordre conserve"

    # Test de stabilite avec debordement potentiel
    p_grands = softmax([1000.0, 1001.0, 1002.0])
    assert abs(sum(p_grands) - 1.0) < 1e-7, "Softmax stable contre overflow"
    print("[OK] Test softmax valide.")


def tester_algebre_lineaire():
    """Verifie produit matrice-vecteur, produit exterieur et normes."""
    M = [[1.0, 2.0], [3.0, 4.0]]
    v = [5.0, 6.0]
    assert produit_matrice_vecteur(M, v) == [17.0, 39.0], "Produit M*v"
    assert produit_exterieur([1.0, 2.0], [3.0, 4.0]) == [[3.0, 4.0], [6.0, 8.0]], "Produit u*v^T"
    assert somme_vecteurs([1.0, 2.0], [3.0, 4.0]) == [4.0, 6.0], "Somme u+v"
    assert abs(norme_vecteur([3.0, 4.0]) - 5.0) < 1e-9, "Norme L2 vecteur"
    assert abs(norme_globale_matrices([M]) - (1 + 4 + 9 + 16)**0.5) < 1e-9, "Norme Frobenius"
    print("[OK] Test algebre lineaire valide.")


if __name__ == "__main__":
    print("Execution des tests unitaires mathematiques...")
    tester_tanh()
    tester_softmax()
    tester_algebre_lineaire()
    print("--> TOUS LES TESTS MATHEMATIQUES ONT REUSSI !")
