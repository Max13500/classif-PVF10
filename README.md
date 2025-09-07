# Classification de défauts de panneaux photovoltaïques

## Contexte

La détection et l’analyse des défauts dans les panneaux photovoltaïques représentent un enjeu critique pour l'optimisation des performances et la maintenance préventive des installations solaires. Dans le contexte énergétique actuel, avec plus d'un million d'installations en France totalisant 24.4 GW de puissance installée fin 2024, la fiabilité et l'efficacité des systèmes photovoltaïques deviennent primordiales.

La thermographie, technique non destructive de capture des variations de température, constitue une approche privilégiée pour cette détection. Les drones équipés de caméras thermiques fournissent une méthode particulièrement efficace, permettant l'inspection rapide et exhaustive de grandes installations photovoltaïques. Sur le plan économique, détecter précocement ces défauts évite des pertes importantes et réduit significativement les coûts d'exploitation et de maintenance.

## Contenu du dépôt

Ce dépôt contient l'ensemble des fichiers (notebooks Jupyter, modules et packages python) correspondant au travail des membres de l'équipe de ce projet, réalisé dans le cadre du cursus **Data Scientist** de la promo _**Novembre 2024 - Continu**_ chez DataScientest.

## Organisation du projet

    ├── LICENSE
    ├── README.md          <- Ce fichier
    ├── data
    │   ├── processed      <- Données pré-traitées après exécution des notebooks
    │   └── raw            <- Les données brutes
    │
    ├── models             <- Modèles entraînés correspondant aux 4 approches évaluées
    │
    ├── notebooks          <- Notebooks Jupyter d'exploration et de travail
    │                         "1.0 - [...].ipynb" -> Notebook d'exploration
    │                         "2.0 - [...].ipynb" -> Notebooks de modélisation/évaluation
    │                         "2.1 - [...].ipynb" -> Notebooks complémentaires
    │
    ├── references         <- Publication scientifique de référence à la base notre travail
    │
    ├── reports            <- Les 2 rapports rendus dans le cadre de ce projet
    │
    ├── requirements.txt   <- Le fichier "requirements.txt" permettant de reproduire l'environnement d'exécution
    │
    └── sep24_cds_pv       <- Code source du projet
        ├── __init__.py    <- Permet d'installer le projet comme un package
        │
        └── streamlit      <- Le Streamlit du projet
            └── app.py     <- Le script principal du site Streamlit
                              Dans ce répertoire, exécuter la commande `streamlit run app.py`

## Récupération du dataset de travail

Le dataset de travail est (à fin août 2025) disponible à l'adresse suivante : https://drive.usercontent.google.com/download?id=1SQq0hETXi8I3Kdq9tDAEVyZgIsRCbOah sous la forme d'un fichier .zip.

Afin d'exécuter les notebooks et de lancer le Streamlit, il est indispensable de dézipper les images brutes du dataset dans le répertoire `data/raw`.

L'arborescence obtenue dans le répertoire `data` doit être la suivante :

```
data
├── processed
└── raw
    └── PVF-10
        ├── PVF_10_110x60
        │   ├── augmented
        │   ├── test
        │   └── train
        ├── PVF_10_112x112
        │   ├── test
        │   └── train
        └── PVF_10_Ori
            ├── test
            └── train
```

## Notebooks

### 1. Phase exploratoire

Le notebook nommé `1.0_Data_Exploration.ipynb` synthétise l'ensemble du travail d'exploration + dataviz réalisé sur le dataset, en amont de la modélisation.

### 2. Phase de modélisation

Le travail de modélisation a été divisé en 4 approches complémentaires.

#### Machine Learning - Features localisés

Les notebooks relatifs à cette approche sont les suivants :
  - `2.0_ML_images_SVM.ipynb` : Optimisation d'un pipeline basé sur le classifieur SVM
  - `2.0_ML_images_RF.ipynb` : Optimisation d'un pipeline basé sur le classifieur RandomForest
  - `2.0_ML_images_DenseKeras.ipynb` : Optimisation d'un pipeline basé sur un réseau de neurones dense

#### Machine Learning - Features non localisés

Le notebook `2.0_ML_stats.ipynb` détaille l'ensemble des travaux d'optimisation de pipelines basés sur les classifieurs **RandomForest**, **DecisionTree**, **XGBoost**, **SVC**, **LightGBM** et **CatBoost**.

2 notebooks complètent ce travail :
  - `2.1_data_augmentation.ipynb` : Travail préparatoire portant sur l'augmentation de données à l'aide de la bibliothèque Albumentations.
  - `2.1_ML_stats_XGBoost.ipynb` :  Evaluation de l'apport de l'oversampling sur le meilleur modèle de cette approche "ML_stats" : **XGBoost**.

#### Deep Learning - CNN _from scratch_

Le notebook `2.0_DL_images_CNN.ipynb` décrit la construction et l'optimisation d'un modèle à base de réseaux de neurones convolutionnels.

#### Deep Learning - Transfer Learning

Les 2 notebooks suivants décrivent l'approche basée sur l'utilisation de modèles CNN pré-entraînés :
  - `2.0_DL_images_MobileNetV2.ipynb` : Transfer Learning avec un modèle pré-entraîné "léger" : **MobileNetV2**
  - `2.0_DL_images_EfficientNetV2.ipynb` : Transfer Learning avec un modèle pré-entraîné plus lourd : **EfficientNetV2B2**

## Rapports

Les 2 rapports suivants ont été rendus au cours du projet :
  - `Rapport_Exploration_PVF10_Rendu1.pdf` : Le rapport d'exploration préliminaire du jeu de données
  - `Classification de Défauts dans les Panneaux Photovoltaïques – Bilan du Projet.pdf` :  le rapport final rendu en fin de projet

## Streamlit

_TODO_

## Docker

_TODO_:
  - Comment utiliser le Dockerfile_dev pour disposer d'un environnement python de dev dans un conteneur Docker
  - Comment utiliser le Dockerfile_streamlit pour lancer le streamlit

## Membres du projet

- [Maxime BENOIT](https://fr.linkedin.com/in/maxime-benoit-92004a329)
- [Sylvain CORDIER](https://fr.linkedin.com/in/sylvain-cordier)
- [Philippe MARECHAL](https://fr.linkedin.com/in/philippe-marechal-74a24a4)


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
