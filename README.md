# TARDIS - Prédiction des Retards de Trains SNCF

## À Propos du Projet

TARDIS est un projet Epitech visant à analyser et prédire les retards des trains SNCF en utilisant des techniques de machine learning. Ce projet s'inscrit dans le cadre du cours de données et d'intelligence artificielle, développé en 5 semaines par une équipe de 3 personnes.

### Objectif Principal

Construire un modèle prédictif performant capable d'anticiper le **retard moyen à l'arrivée de tous les trains SNCF** en fonction de diverses caractéristiques (date, service, gares, paramètres opérationnels, etc.), permettant :

- Une meilleure gestion des ressources de transport
- Une meilleure information et planification pour les voyageurs
- L'identification des facteurs clés influençant les retards
- Une compréhension des causes des dysfonctionnements opérationnels

### Méthodologie

Le projet suit un flux complet de machine learning :
1. **Exploration et nettoyage des données** (EDA) - Compréhension des données brutes
2. **Prétraitement et feature engineering** - Préparation des données pour le modèle
3. **Entraînement et évaluation de modèles** - Sélection du meilleur modèle
4. **Déploiement et visualisation** - Dashboard interactif pour les prédictions

## Installation

### Prérequis
- `Python 3.8`+
- `pip`
- `jupyter-notebook` ou `vscode`

### Instructions d'installation

1. **Cloner le dépôt** :
```bash
git clone git@github.com:EpitechPGE1-2025/G-AIA-210-COT-2-1-tardis-18.git
cd G-AIA-210-COT-2-1-tardis-18
```

2. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

## Utilisation

### Lancer le dashboard interactif

```bash
streamlit run tardis_dashboard.py
```

Le dashboard propose une interface conviviale pour explorer les prédictions et visualiser les données.

### Exécuter les notebooks

- **`tardis_eda.ipynb`** : Analyse exploratoire des données (EDA), nettoyage et feature engineering.
- **`tardis_model.ipynb`** : Entraînement, évaluation et sélection du meilleur modèle.

## Structure du Projet

```
├── tardis_eda.ipynb                # Analyse exploratoire et préparation des données
├── tardis_model.ipynb              # Développement et évaluation du modèle ML
├── tardis_dashboard.py             # Application Streamlit interactive
├── cleaned_dataset.csv             # Dataset nettoyé et prêt pour l'entraînement
├── dataset.csv                     # Dataset brut original
├── model.joblib                    # Modèle entraîné sauvegardé
├── requirements.txt                # Dépendances Python
└── README.md                       # Ce fichier
```

## Description Détaillée des Fichiers

### Fichiers de Données

#### **dataset.csv** (14 152 lignes)
Dataset brut original contenant les données complètes sur les retards des trains SNCF. Ce fichier contient :
- **Métadonnées temporelles** : Date, Année, Mois
- **Routes** : Gare de départ, Gare d'arrivée, Service (National/International)
- **Indicateurs de trains** : Nombre de trains programmés, annulés, retardés
- **Mesures de retards** : Retards moyens (départ/arrivée), retards catégorisés (>15min, >30min, >60min)
- **Causes des retards** : Facteurs externes, infrastructure, gestion du trafic, matériel roulant, gestion des gares, gestion des passagers
- **Autres variables** : Durée moyenne des trajets, taux d'annulation
- État : Contient des valeurs manquantes et des incohérences à nettoyer

#### **cleaned_dataset.csv** (11 488 lignes)
Dataset nettoyé et normalisé après prétraitement. Prêt pour l'entraînement du modèle :
- Valeurs aberrantes supprimées
- Valeurs manquantes traitées
- Nouvelles colonnes créées : `Delay_categories` (classification des retards), `is_delayed` (binaire)
- Taux de cancellation calculé
- Encodage des variables catégoriques complété
- Normalisé pour améliorer la performance des modèles

### Notebooks Jupyter

#### **tardis_eda.ipynb** (Exploratory Data Analysis)
Notebook interactif d'analyse exploratoire des données :
- **Exploration initiale** : Structure, types de données, dimensions
- **Analyse statistique** : Distributions, corrélations, statistiques descriptives
- **Visualisations** : Graphiques de distribution, heatmaps de corrélation, tendances temporelles
- **Nettoyage des données** :
  - Identification et traitement des valeurs manquantes
  - Détection et suppression des valeurs aberrantes
  - Gestion des doublons
- **Feature engineering** :
  - Création de nouvelles variables pertinentes
  - Extraction de features temporelles (jour de la semaine, saison, etc.)
  - Encodage des variables catégoriques
- **Résultats** : Génération du fichier `cleaned_dataset.csv`

#### **tardis_model.ipynb** (Model Development)
Notebook d'entraînement et d'évaluation des modèles ML :
- **Chargement et préparation** : Import du dataset nettoyé, split train/test
- **Sélection de modèles** :
  - Régression linéaire
  - Random Forest
  - Gradient Boosting
  - Autres modèles selon les résultats
- **Entraînement** :
  - Tuning des hyperparamètres
  - Cross-validation pour évaluer la robustesse
  - Validation sur ensemble de test
- **Évaluation** :
  - Métriques : MAE (Mean Absolute Error), MSE, RMSE, R²
  - Analyse des résidus
  - Feature importance
  - Courbes d'apprentissage
- **Sélection du meilleur modèle** : Basée sur RMSE et R²
- **Sauvegarde** : Export du modèle entraîné dans `model.joblib`

### Application Dashboard

#### **tardis_dashboard.py**
Application Streamlit interactive pour visualiser et utiliser les prédictions :

Lancer l'application :
```bash
streamlit run tardis_dashboard.py
```

### Fichier de Configuration

#### **requirements.txt**
Liste complète des dépendances Python :
- **pandas** : Manipulation et analyse de données
- **numpy** : Calculs numériques
- **scikit-learn** : Algorithmes de machine learning et prétraitement
- **matplotlib** : Visualisations statiques
- **seaborn** : Visualisations statistiques avancées
- **plotly** : Graphiques interactifs
- **streamlit** : Framework pour le dashboard web
- **jupyter-client, nbconvert, ipykernel** : Support des notebooks Jupyter

## Workflow Complet du Projet

### Étape 1 : Préparation des Données
```bash
jupyter notebook tardis_eda.ipynb
```
- Charger et explorer le dataset brut (`dataset.csv`)
- Nettoyer et transformer les données
- Générer `cleaned_dataset.csv`

### Étape 2 : Entraînement et Évaluation
```bash
jupyter notebook tardis_model.ipynb
```
- Charger le dataset nettoyé
- Entraîner plusieurs modèles
- Évaluer les performances
- Sauvegarder le meilleur modèle dans `model.joblib`

### Étape 3 : Utilisation des Prédictions
```bash
streamlit run tardis_dashboard.py
```
- Accéder à l'interface web interactive
- Effectuer des prédictions sur de nouvelles routes
- Visualiser les résultats et analyses

## Équipe

Ce projet a été réalisé par :

| Nom | Email |
|-----|-------|
| Laurince AGANI | laurince.agani@epitech.eu |
| Frieda CHATIGRE | frieda.chatigre@epitech.eu |
| Régis KOUADOUA | regis.kouadoua@epitech.eu |

## Technologies Utilisées

- **Python 3** - Langage principal
- **Pandas & NumPy** - Manipulation et analyse des données
- **Scikit-learn** - Algorithmes de machine learning et prétraitement
- **Matplotlib & Seaborn** - Visualisations statiques et statistiques
- **Plotly** - Graphiques interactifs
- **Streamlit** - Framework pour le dashboard web interactif
- **Jupyter** - Notebooks interactifs pour l'analyse et le développement
