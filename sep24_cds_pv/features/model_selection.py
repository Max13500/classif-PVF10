"""
model_selection subpackage (like sklearn.model_selection)
"""
from collections import namedtuple
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

Splits = namedtuple("Splits", "X_train X_test X_validation y_train y_test y_validation")


def improved_train_test_validation_split(data: pd.DataFrame, 
                                         target_col: str,
                                         train_size: float,
                                         test_size: float, 
                                         validation_size: float, 
                                         stratify: bool = True,
                                         **kwargs):
    """Split the dataset into train, test (and validation) respecting original 'target_col' ratios in each"""

    total_size = train_size + test_size + validation_size
    if abs(total_size - 1.0) > 0.001:
        raise ValueError(f"Sum of (train_size, test_size, validation_size) must be 1.0 (currently {total_size}).")

    features = data.drop(columns=[target_col])
    target = data[target_col]
    
    if stratify:
        stratify = target
    else:
        stratify = None

    if validation_size > 0.:
        X_remain, X_validation, y_remain, y_validation = train_test_split(features, 
                                                                          target, 
                                                                          test_size=validation_size, 
                                                                          stratify=stratify, 
                                                                          **kwargs,
                                                                          )
        new_train_size = train_size / (train_size + test_size)
        if stratify is not None:
            stratify = y_remain
    else:
        X_validation = pd.DataFrame()
        y_validation = pd.Series()
        X_remain = features
        y_remain = target
        new_train_size = train_size
        
    X_train, X_test, y_train, y_test = train_test_split(X_remain, y_remain, train_size=new_train_size, stratify=stratify, **kwargs)

    return Splits(X_train, X_test, X_validation, y_train, y_test, y_validation)


def display_results(y_test, y_pred):

    # Afficher la matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies classes")
    plt.title("Matrice de confusion")
    plt.show()

    # Générer le rapport de classification
    class_report = classification_report(y_test, y_pred)

    # Afficher le rapport
    print("Rapport de classification :\n", class_report)
