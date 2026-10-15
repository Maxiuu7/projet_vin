"""
Prétraitement des données (mêmes étapes que la partie 2.1 du notebook).
On regroupe le code dans des fonctions pour pouvoir le réutiliser
dans l'entraînement et dans les tests unitaires.
"""
import os
import pandas as pd

# Dossier "data" à la racine du projet
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

# Les 12 variables d'entrée, dans l'ordre utilisé par le modèle
VARIABLES = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
             'pH', 'sulphates', 'alcohol', 'type']


def charger_donnees(data_dir=DATA_DIR):
    """Charge les vins rouges et blancs et les fusionne avec une colonne type (rouge = 0, blanc = 1)."""
    rouge = pd.read_csv(os.path.join(data_dir, 'winequality-red.csv'), sep=';')
    blanc = pd.read_csv(os.path.join(data_dir, 'winequality-white.csv'), sep=';')
    rouge['type'] = 0
    blanc['type'] = 1
    return pd.concat([rouge, blanc], ignore_index=True)


def nettoyer(df):
    """Supprime les doublons (un même vin ne doit pas être à la fois dans le train et le test)."""
    return df.drop_duplicates().reset_index(drop=True)


def creer_cible(df, seuil=7):
    """Crée la cible binaire bon_vin (1 si quality >= seuil) et retire quality pour éviter la fuite de la cible."""
    df = df.copy()
    df['bon_vin'] = (df['quality'] >= seuil).astype(int)
    return df.drop(columns=['quality'])


def preparer_donnees(data_dir=DATA_DIR):
    """Enchaîne toutes les étapes et renvoie X (variables) et Y (cible)."""
    df = creer_cible(nettoyer(charger_donnees(data_dir)))
    X = df[VARIABLES]
    Y = df['bon_vin']
    return X, Y
