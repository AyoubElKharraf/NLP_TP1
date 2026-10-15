"""
Script de test unitaire pour maths_utils.py.
Peut etre execute directement avec : python tests/test_maths.py
"""

import sys
import os

# Ajout du dossier racine au chemin d'import
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
    norme_globale_matrices,
    creer_matrice_zeros,
    creer_vecteur_zeros,
    creer_matrice_aleatoire,
    fixer_graine
)


def tester_tanh():
    # tanh(0) doit valoir 0
    val_zero = tanh_scalaire(0.0)
    assert abs(val_zero) < 1e-9, f"tanh(0) devrait etre 0, obtenu: {val_zero}"

    # Symetrie impaire : tanh(-x) == -tanh(x)
    val_pos = tanh_scalaire(1.5)
    val_neg = tanh_scalaire(-1.5)
    assert abs(val_pos + val_neg) < 1e-9, "tanh doit etre une fonction impaire"

    # Saturation
    assert tanh_scalaire(100.0) == 1.0, "tanh(100) doit saturer a 1.0"
    assert tanh_scalaire(-100.0) == -1.0, "tanh(-100) doit saturer a -1.0"

    # Derivee : tanh'(0) = 1 - tanh(0)^2 = 1.0
    assert abs(derivee_tanh(0.0) - 1.0) < 1e-9, "derivee_tanh(0) doit etre 1.0"

    print("[OK] Tests tanh et sa derivee reussis.")


def tester_softmax():
    v = [1.0, 2.0, 3.0]
    p = softmax(v)

    # La somme des probabilites doit etre egale a 1.0
    somme = 0.0
    for val in p:
        assert 0.0 <= val <= 1.0, f"Probabilite invalide : {val}"
        somme += val
    assert abs(somme - 1.0) < 1e-7, f"La somme de softmax doit etre 1.0, obtenu: {somme}"

    # L'ordre doit etre respecte (v[2] > v[1] > v[0] => p[2] > p[1] > p[0])
    assert p[2] > p[1] > p[0], "L'ordre des probabilites est incoherent"

    # Test de stabilite numerique contre l'overflow
    grands_nombres = [1000.0, 1001.0, 1002.0]
    p_grands = softmax(grands_nombres)
    somme_grands = 0.0
    for val in p_grands:
        somme_grands += val
    assert abs(somme_grands - 1.0) < 1e-7, "Softmax a echoue sur les grands nombres (overflow)"

    print("[OK] Tests softmax (normalisation et stabilite numerique) reussis.")


def tester_algebre_lineaire():
    # 1. Produit Matrice-Vecteur
    # M (2x2) * v (2) -> [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
    M = [
        [1.0, 2.0],
        [3.0, 4.0]
    ]
    v = [5.0, 6.0]
    res_mv = produit_matrice_vecteur(M, v)
    assert res_mv == [17.0, 39.0], f"Produit matrice-vecteur incorrect : {res_mv}"

    # 2. Produit Exterieur
    # u (2) et v (3) -> Matrice 2x3
    u = [1.0, 2.0]
    w = [3.0, 4.0, 5.0]
    res_ext = produit_exterieur(u, w)
    attendu_ext = [
        [3.0, 4.0, 5.0],
        [6.0, 8.0, 10.0]
    ]
    assert res_ext == attendu_ext, f"Produit exterieur incorrect : {res_ext}"

    # 3. Transposee
    M_T = transposee(M)
    assert M_T == [[1.0, 3.0], [2.0, 4.0]], f"Transposee incorrecte : {M_T}"

    # 4. Norme euclidienne
    norme = norme_vecteur([3.0, 4.0])
    assert abs(norme - 5.0) < 1e-9, f"Norme incorrecte : {norme}"

    # 5. Somme et soustraction
    assert somme_vecteurs([1.0, 2.0], [3.0, 4.0]) == [4.0, 6.0]
    assert soustraction_vecteurs([4.0, 6.0], [1.0, 2.0]) == [3.0, 4.0]

    print("[OK] Tests d'algebre lineaire (produits, transposee, normes) reussis.")


def tester_initialisations():
    fixer_graine(123)
    M = creer_matrice_aleatoire(2, 3, echelle=0.1)
    assert len(M) == 2 and len(M[0]) == 3, "Dimensions matrice aleatoire incorrectes"
    for i in range(2):
        for j in range(3):
            assert -0.1 <= M[i][j] <= 0.1, f"Valeur hors bornes : {M[i][j]}"

    z_mat = creer_matrice_zeros(2, 2)
    assert z_mat == [[0.0, 0.0], [0.0, 0.0]], "Matrice de zeros incorrecte"

    print("[OK] Tests de creation et initialisation reussis.")


if __name__ == "__main__":
    print("Execution des tests unitaires de maths_utils.py...")
    tester_tanh()
    tester_softmax()
    tester_algebre_lineaire()
    tester_initialisations()
    print("--> TOUS LES TESTS MATHEMATIQUES SONT VALIDES avec succes !")
