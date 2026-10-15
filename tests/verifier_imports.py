"""
Script de verification automatique des modules importes dans tout le projet.
Analyse les arbres syntaxiques (AST) de tous les fichiers .py.
"""

import ast
import os


def analyser_imports_fichier(chemin_fichier):
    with open(chemin_fichier, "r", encoding="utf-8") as f:
        arbre = ast.parse(f.read(), filename=chemin_fichier)
    imports = []
    for noeud in ast.walk(arbre):
        if isinstance(noeud, ast.Import):
            for n in noeud.names:
                imports.append(n.name)
        elif isinstance(noeud, ast.ImportFrom):
            if noeud.module:
                imports.append(noeud.module)
    return imports


def scanner_tout_le_projet(racine="."):
    resultats = {}
    for dossier, _, fichiers in os.walk(racine):
        # Ignorer .git ou dossiers caches
        if ".git" in dossier or "__pycache__" in dossier:
            continue
        for fichier in fichiers:
            if fichier.endswith(".py"):
                chemin = os.path.join(dossier, fichier)
                resultats[chemin] = analyser_imports_fichier(chemin)
    return resultats


if __name__ == "__main__":
    scan = scanner_tout_le_projet(".")
    print("=== RAPPORT D'ANALYSE DES IMPORTS DU PROJET ===")
    tous_modules_externes = set()
    for chemin, mods in sorted(scan.items()):
        print(f"\n{chemin} :")
        for m in sorted(set(mods)):
            print(f"  - {m}")
            # Verifier les modules racines
            racine_m = m.split(".")[0]
            if racine_m not in ["src"]:
                tous_modules_externes.add(racine_m)

    print("\n---------------------------------------------")
    print(f"Ensemble des modules racines de la bibliotheque standard utilises :")
    for m in sorted(tous_modules_externes):
        print(f"  * {m}")
