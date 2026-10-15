# TP TALN : Réseau de Neurones Récurrent (Tanh-RNN) from Scratch en Python Pur

> **Master M2I — Cours de Traitement Automatique du Langage Naturel (Pr. A. Mazroui)**  
> **Auteur du projet :** Travaux Pratiques de Master  
> **Contraintes :** 100% Python 3 pur, aucune bibliothèque tierce (`numpy`, `torch`, `sklearn`, `pandas` strictement interdits). Seules fonctions autorisées : `math.exp` et `math.log`.

---

## 1. Vue d'ensemble du Projet

Ce projet implémente, de zéro et sans aucune boîte noire, un **réseau de neurones récurrent de type Tanh (Elman RNN)** au niveau des **caractères**. 

Le modèle est entraîné comme un modèle de langue autoregressif sur un corpus de paires Question / Réponse formatées :
```text
Q: <Question>
R: <Réponse>
<fin>
```
À l'inférence, le réseau reçoit une amorce `Q: ...\nR:` et génère caractère par caractère la réponse la plus adaptée jusqu'à l'émission du jeton terminal `<fin>`.

---

## 2. Structure et Organisation des Fichiers

Le projet suit une architecture modulaire stricte où chaque fichier possède une responsabilité unique :

```text
TP0/
├── data/
│   └── corpus.txt          # Corpus textuel de paires Question / Réponse
├── src/
│   ├── maths_utils.py      # Algèbre linéaire pure : tanh, softmax, produit matrice-vecteur, produit extérieur, normes
│   ├── vocabulaire.py      # Tokenisation, dictionnaire (car <-> idx), encodage One-Hot
│   ├── donnees.py          # Chargement du corpus et construction des séquences supervisées (X, cibles E décalées)
│   ├── rnn.py              # Initialisation des poids, propagation avant (Forward Pass)
│   ├── perte.py            # Entropie croisée, perte d'une séquence et perte globale sur corpus
│   ├── bptt.py             # Rétropropagation à travers le temps (BPTT), accumulation temporelle des gradients
│   ├── optimiseur.py       # Écrêtage global du gradient (Clipping) et mise à jour SGD
│   ├── entrainement.py     # Boucle d'apprentissage stochastique par époques et validation par sur-apprentissage
│   └── generation.py       # Décodage autorégressif pas à pas (Argmax / Échantillonnage avec température)
├── tests/
│   ├── test_maths.py       # Validation unitaire des opérations mathématiques et de la stabilité de softmax
│   └── test_gradient.py    # Gradient Checking : vérification numérique par différences finies centrées
├── main.py                 # Point d'entrée : orchestration, entraînement et session de chat interactif en terminal
└── README.md               # Documentation complète, équations du cours et analyse théorique
```

---

## 3. Notations et Formulations Mathématiques du Cours

### 3.1. Paramètres du Réseau (partagés dans le temps)
- $V$ : taille du vocabulaire discret (nombre de caractères uniques, incluant `<fin>`).
- $H$ : dimension de l'espace caché (par défaut $H = 32$).
- $W_{xh} \in \mathbb{R}^{H \times V}$ : matrice de projection des entrées vers l'état caché.
- $W_{hh} \in \mathbb{R}^{H \times H}$ : matrice de transition récurrente d'état à état.
- $W_{hy} \in \mathbb{R}^{V \times H}$ : matrice de projection de l'état caché vers la couche de sortie.
- $b_h \in \mathbb{R}^H, \; b_y \in \mathbb{R}^V$ : vecteurs de biais (initialisés à zéro).
- $H_0 = \vec{0}$ : état caché initial nul au pas $t = 0$.

---

### 3.2. Propagation Avant (Forward Pass)
Pour chaque pas de temps $t \in [0, T-1]$, avec $X_t \in \{0, 1\}^V$ l'encodage One-Hot du caractère courant :

$$z_{h, t} = W_{hh} \cdot H_{t-1} + W_{xh} \cdot X_t + b_h$$

$$H_t = \tanh(z_{h, t}) = \frac{\exp(z_{h, t}) - \exp(-z_{h, t})}{\exp(z_{h, t}) + \exp(-z_{h, t})}$$

$$z_{y, t} = W_{hy} \cdot H_t + b_y$$

$$Y_t = \text{softmax}(z_{y, t}) \quad \text{avec} \quad Y_t[k] = \frac{\exp(z_{y, t}[k] - \max(z_{y, t}))}{\sum_{j=0}^{V-1} \exp(z_{y, t}[j] - \max(z_{y, t}))}$$

---

### 3.3. Fonction de Perte (Entropie Croisée)
Pour chaque instant $t$, la cible $E_t \in \{0, 1\}^V$ est le caractère suivant décalé d'un cran ($E_t = X_{t+1}$).
La perte ponctuelle est :
$$\text{Err}(Y_t, E_t) = -\sum_{k=0}^{V-1} E_t[k] \cdot \log(Y_t[k])$$

L'erreur cumulative sur une séquence de longueur $T$ :
$$\text{Err}(X) = \sum_{t=0}^{T-1} \text{Err}(Y_t, E_t)$$

L'erreur globale moyenne sur un corpus de $N$ séquences :
$$\text{Err}_{\text{globale}} = \frac{1}{N} \sum_{i=1}^N \text{Err}(X^{(i)})$$

---

### 3.4. Rétropropagation à Travers le Temps (BPTT)
On remonte le temps de $t = T-1$ jusqu'à $t = 0$ :

1. **Gradient de sortie** (combinaison Softmax + Entropie Croisée) :
   $$d z_{y, t} = Y_t - E_t \in \mathbb{R}^V$$

2. **Gradients de la couche de sortie** :
   $$d W_{hy} \leftarrow d W_{hy} + d z_{y, t} \cdot (H_t)^T \quad (\text{produit extérieur})$$
   $$d b_y \leftarrow d b_y + d z_{y, t}$$

3. **Gradient total reçu par l'état caché $H_t$** :
   $$d H_t = (W_{hy})^T \cdot d z_{y, t} + d H_{\text{suivant}}$$
   *(avec $d H_{\text{suivant}} = \vec{0}$ au dernier pas $t = T-1$)*.

4. **Passage de la dérivée de $\tanh$** :
   $$d z_{h, t} = d H_t \odot (1 - H_t^2)$$

5. **Accumulation des gradients récurrents et d'entrée** :
   $$d W_{hh} \leftarrow d W_{hh} + d z_{h, t} \cdot (H_{t-1})^T \quad (\text{accumulation temporelle du récurrent})$$
   $$d W_{xh} \leftarrow d W_{xh} + d z_{h, t} \cdot (X_t)^T$$
   $$d b_h \leftarrow d b_h + d z_{h, t}$$

6. **Transmission vers le pas de temps précédent ($t-1$)** :
   $$d H_{\text{suivant}} = (W_{hh})^T \cdot d z_{h, t}$$

---

### 3.5. Écrêtage (Gradient Clipping) et Optimisation (SGD)
Pour bloquer l'explosion du gradient, on calcule la norme L2 globale $\|grad\| = \sqrt{\sum \theta_{ij}^2}$ sur l'ensemble de tous les gradients.

- **Règle d'écrêtage :**
  $$\text{Si } \|grad\| > \text{seuil} \implies grad \leftarrow \left(\frac{\text{seuil}}{\|grad\|}\right) \cdot grad$$

- **Mise à jour par Descente de Gradient Stochastique (SGD) :**
  $$P(k+1) = P(k) - \delta \cdot grad(\text{Err}(P(k)))$$
  *(avec $\delta = 0.15$ le taux d'apprentissage et $\text{seuil} = 5.0$)*.

---

## 4. Guide d'Exécution et Tests de Validation

### Étape 1 : Validation des fonctions mathématiques
```bash
python tests/test_maths.py
```
*Vérifie $\tanh(0)=0$, la parité impaire, la normalisation à 1 de softmax, la robustesse aux grands nombres et les produits matriciels.*

### Étape 2 : Vérification numérique des gradients (Gradient Checking)
```bash
python tests/test_gradient.py
```
*Compare les gradients BPTT aux différences finies centrées : l'erreur relative maximale obtenue est de $1.08 \times 10^{-6}$, attestant de l'exactitude absolue des dérivées.*

### Étape 3 : Entraînement et test de surapprentissage
```bash
python src/entrainement.py
```
*Valide que le modèle sur-apprend parfaitement une séquence isolée (`Bonjour<fin>`, perte $< 0.02$) puis converge sur le corpus.*

### Étape 4 : Démonstration et Chat Interactif
```bash
# Démonstration automatique sur toutes les questions du corpus :
python main.py --demo

# Session interactive de chat en terminal :
python main.py
```

---

## 5. Analyse Comparative : RNN vs Approche Sac de Mots (Chapitre 8)

Le chapitre 8 du cours aborde les représentations traditionnelles de type **Sac de Mots (*Bag of Words* / BoW)** et modèles n-grammes classiques.

| Critère | Modèle Sac de Mots (BoW) | Réseau Récurrent (Tanh-RNN) |
| :--- | :--- | :--- |
| **Prise en compte de l'ordre** | **Aucune** (invariance par permutation). L'ordre des mots/caractères est totalement perdu. | **Complète** : l'ordre séquentiel dicte directement l'évolution temporelle de $H_t$. |
| **Sensibilité au contexte** | Incapable de distinguer « *le chat mange la souris* » de « *la souris mange le chat* ». | Les mêmes mots dans un ordre différent mènent à des états cachés $H_t$ distincts. |
| **Gestion des négations** | « *bon, pas mauvais* » et « *mauvais, pas bon* » ont les mêmes vecteurs d'occurrences. | La négation inverse dynamiquement la trajectoire latente. |
| **Génération de texte** | Impossible de façon cohérente sans modèle d'ordre. | Génération fluide autoregressive caractère par caractère. |
| **Représentation** | Vecteurs très creux (*sparse*) de grande dimension. | Vecteur dense d'état caché $H_t \in \mathbb{R}^H$ résumant tout l'historique passé. |

---

## 6. Limite Théorique du Tanh-RNN et Motivation du LSTM

### 6.1. Le Problème de Disparition du Gradient (*Vanishing Gradient*)
Dans la boucle BPTT, l'erreur rétropropagée depuis l'instant $t$ jusqu'à un instant lointain $k \ll t$ s'exprime par le produit matriciel en chaîne des Jacobiennes :

$$\frac{\partial H_t}{\partial H_k} = \prod_{j=k+1}^t \frac{\partial H_j}{\partial H_{j-1}} = \prod_{j=k+1}^t \text{diag}\left(1 - H_j^2\right) \cdot W_{hh}^T$$

Ce produit répété $(t - k)$ fois pose deux problèmes majeurs :
1. **Saturation de $\tanh$ :** La dérivée $\tanh'(x) = 1 - \tanh^2(x)$ est strictement bornée dans $[0, 1]$, et atteint $1$ uniquement pour $x = 0$. Dès que le neurone s'active fortement ($|x| > 2$), $1 - H_j^2 \to 0$.
2. **Rayon spectral de $W_{hh}$ :** Si les valeurs propres de $W_{hh}$ ont un module inférieur à $1$, le produit de $(t - k)$ matrices décroît de manière **exponentielle** vers la matrice nulle :
   $$\left\| \frac{\partial H_t}{\partial H_k} \right\| \xrightarrow[t - k \to \infty]{} 0$$

**Conséquence pratique :** Le réseau "oublie" rapidement le début de la phrase dès que la longueur $T$ dépasse 10 à 15 pas temporels. Les premières couches temporelles ne reçoivent plus aucun gradient pour apprendre.

---

### 6.2. La Solution LSTM (*Long Short-Term Memory*)
Pour résoudre ce problème fondamental, Hochreiter & Schmidhuber ont introduit le **LSTM** :
- Remplacement de la simple cellule récurrente par une **cellule à mémoire interne $C_t$** régie par des portes d'entrée ($i_t$), d'oubli ($f_t$) et de sortie ($o_t$).
- **Carrousel d'erreur constant :** La mise à jour de la mémoire s'effectue par une **addition linéaire** :
  $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
- Lors de la rétropropagation, $\frac{\partial C_t}{\partial C_{t-1}} \approx f_t$. Si la porte d'oubli est ouverte ($f_t \approx 1$), le gradient circule à travers le temps **sans aucune atténuation exponentielle**, permettant de capturer des dépendances à très long terme.
