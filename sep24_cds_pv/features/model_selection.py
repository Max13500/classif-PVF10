"""
model_selection subpackage (like sklearn.model_selection)
"""
from collections import namedtuple
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

Splits = namedtuple("Splits", "X_train X_test X_validation y_train y_test y_validation")


def improved_train_test_validation_split(data: pd.DataFrame, 
                                         target_col: str,
                                         train_size: float,
                                         test_size: float, 
                                         validation_size: float, 
                                         **kwargs):
    """Split the dataset into train, test (and validation) respecting original 'target_col' ratios in each"""

    total_size = train_size + test_size + validation_size
    if abs(total_size - 1.0) > 0.001:
        raise ValueError(f"Sum of (train_size, test_size, validation_size) must be 1.0 (currently {total_size}).")

    features = data.drop(columns=[target_col])
    target = data[target_col]

    cat_vals = data[target_col].unique()

    splits = {}

    for cat_val in cat_vals:
        selection_mask = data[target_col] == cat_val

        if validation_size > 0.:
            X_remain, X_validation, y_remain, y_validation = train_test_split(features.loc[selection_mask, :], 
                                                                              target[selection_mask], 
                                                                              test_size=validation_size, 
                                                                              **kwargs,
                                                                              )
            new_train_size = train_size / (train_size + test_size)
        else:
            X_validation = pd.DataFrame()
            y_validation = pd.Series()
            X_remain = features.loc[selection_mask, :]
            y_remain = target[selection_mask]
            new_train_size = train_size
            
        X_train, X_test, y_train, y_test = train_test_split(X_remain, y_remain, train_size=new_train_size, **kwargs)

        splits[cat_val] = X_train, X_test, X_validation, y_train, y_test, y_validation

    X_train_full, X_test_full, X_validation_full, y_train_full, y_test_full, y_validation_full = [pd.concat(splits_tuple, axis=0) 
                                                                                                  for splits_tuple in zip(*splits.values())]

    return Splits(X_train_full, X_test_full, X_validation_full, y_train_full, y_test_full, y_validation_full)


def display_results(y_test, y_pred, classes):

    # Afficher la matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", yticklabels=classes, xticklabels=classes)
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies classes")
    plt.title("Matrice de confusion")
    plt.show()

    # Générer le rapport de classification
    class_report = classification_report(y_test, y_pred, target_names=classes)

    # Afficher le rapport
    print("Rapport de classification :\n", class_report)
