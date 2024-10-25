
# Projet : Apprentissage par Renforcement avec Q-learning et SARSA

Ce projet implémente des algorithmes d'apprentissage par renforcement pour résoudre un problème de navigation sur une grille. Deux méthodes d'apprentissage, **Q-learning** et **SARSA**, sont explorées pour entraîner un agent à trouver le chemin optimal dans un environnement pouvant inclure des pièges et des récompenses différentes.

## Structure des Fichiers

- **`qlearning.ipynb`** : Notebook contenant l'implémentation de l'algorithme Q-learning. Cet algorithme utilise une approche de mise à jour de valeur hors-politique pour apprendre la politique optimale.
- **`SARSA.ipynb`** : Notebook contenant l'implémentation de l'algorithme SARSA, une approche d'apprentissage par renforcement en politique. SARSA met à jour les valeurs d'état-action en suivant la politique actuelle de l'agent.
- **`utils.py`** : Fichier contenant les fonctions utilitaires pour la gestion de la grille, des états, des récompenses et de la politique. Il inclut également des fonctions de visualisation pour les fonctions de valeurs, les transitions et les politiques optimales.

## Fonctionnalités Principales

### 1. Génération des Grilles et des Transitions
   - `dict_transition` : Génère les transitions possibles pour chaque état en fonction des dimensions de la grille et des pièges.
   - `dict_rewards` : Génère les récompenses associées à chaque position sur la grille, y compris les pénalités pour les pièges.

### 2. Q-learning (`qlearning.ipynb`)
   - Initialisation des états et actions.
   - Choix d'action via la méthode epsilon-greedy.
   - Mise à jour de la fonction Q en suivant la formule Q-learning.
   - Affichage des résultats et visualisation de la politique optimale sur la grille.

### 3. SARSA (`SARSA.ipynb`)
   - Mise en œuvre de l'algorithme SARSA pour les mises à jour d'état-action en suivant la politique actuelle.
   - Sélection d'actions en fonction d'une politique epsilon-greedy.
   - Visualisation de la convergence de la fonction de valeur pour chaque état.

### 4. Fonctions Utilitaires et Visualisation (`utils.py`)
   - **Grille et Transitions** : Génération des grilles, gestion des pièges, et configuration des récompenses.
   - **Fonctions de Valeur** : Initialisation des fonctions de valeur et mise à jour pour chaque état.
   - **Choix d'Actions** : Fonctions pour sélectionner des actions en fonction de la politique.
   - **Visualisation** : Affichage des fonctions de valeur sous forme de heatmaps, visualisation des transitions et des politiques optimales.

## Installation et Pré-requis

1. **Python** 3.x
2. Librairies nécessaires (installables via pip) :
   ```bash
   pip install numpy matplotlib seaborn
   ```

## Utilisation

1. **Configurer la Grille et les Paramètres**
   - Définir les dimensions de la grille (`K`), les positions de la case terminale (`T`) et les pièges si nécessaire.
   
2. **Exécuter Q-learning**
   - Ouvrez `qlearning.ipynb` et exécutez chaque cellule pour lancer l'entraînement de l'agent.
   
3. **Exécuter SARSA**
   - Ouvrez `SARSA.ipynb` et exécutez chaque cellule pour entraîner l'agent en utilisant l'algorithme SARSA.

## Résultats

Le projet génère des visualisations des fonctions de valeur pour chaque algorithme, des politiques optimales et des chemins suivis par l'agent. Ces résultats peuvent être utilisés pour comparer les performances de Q-learning et SARSA.

## Contribuer

Les contributions sont les bienvenues. Veuillez créer une *issue* ou soumettre une *pull request* pour des suggestions d'améliorations.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus d’informations.
