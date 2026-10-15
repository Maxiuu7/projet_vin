"""
Fonctions pour charger le modèle sauvegardé et prédire la qualité d'un nouveau vin.
Utilisées par les tests.
"""
import os
import joblib
import pandas as pd

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMIN_MODELE = os.path.join(RACINE, 'models', 'modele_vin.joblib')


def charger_modele(chemin=CHEMIN_MODELE):
    """Charge le dictionnaire {scaler, modele, variables} créé par train.py."""
    return joblib.load(chemin)


def predire(vin, artefact):
    """
    vin : dictionnaire {nom de variable: valeur} pour un seul vin.
    Renvoie 1 si le vin est prédit bon, 0 sinon.
    """
    # On remet les variables dans le même ordre que pendant l'entraînement
    X = pd.DataFrame([vin])[artefact['variables']]
    X_norm = artefact['scaler'].transform(X)
    return int(artefact['modele'].predict(X_norm)[0])
