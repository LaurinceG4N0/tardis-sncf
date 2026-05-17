# TARDIS - Predicting the Unpredictable

Le projet Tardis est un projet Epitech qui consiste à analyser des retards de trains SNCF et prédire via un modèle de machine learning, les retards de ces derniers.

## INSTALLATION

- Dépendances nécessaires :
```bash
pip3 install -r requirements.txt
```

- Clonage du dossier :
```bash
git clone git@github.com:EpitechPGE1-2025/G-AIA-210-COT-2-1-tardis-18.git
```

Ce projet à été réalisé en 5 semaines et en groupe de 3 personnes dont les membres sont:

- Laurince AGANI: laurince.agani@epitech.eu
- Frieda CHATIGRE: frieda.chatigre@epitech.eu
- Régis KOUADOUA: regis.kouadoua@epitech.eu

## USAGE

### Lancer le dashboard

```bash
streamlit run tardis_dashboard.py
```

### Exécuter les notebook

- `tardis_eda.ipynb`: Nettoyage et analyse exploratoire.
- `tardis_model.ipynb`: Entrainement et evaluation des données.

## STRUCTURE DU PROJET

- `tardis_eda.ipynb`: EDA et feature engineering.
- `tardis_model.ipynb`: Modèle ML et sélection.
- `tardis_dashboard.py`: Dashboard streamlit interactif.
- `cleaned_dataset.csv`: Dataset nettoyé.
- `model.joblib`: Modèle sauvegardè.
