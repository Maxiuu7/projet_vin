"""
Tests unitaires du modèle : il doit se charger, prédire 0 ou 1, et garder une performance minimale.
Lancement : pytest
"""
import os
import pytest
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from preprocessing import preparer_donnees
from predict import charger_modele, predire, CHEMIN_MODELE
from train import entrainer


@pytest.fixture(scope='module')
def artefact():
    # Si le modèle n'existe pas encore, on l'entraîne avant les tests
    if not os.path.exists(CHEMIN_MODELE):
        entrainer()
    return charger_modele()


@pytest.fixture
def vin_exemple():
    # Premier vin rouge du fichier (note réelle = 5, donc pas bon)
    return {'fixed acidity': 7.4, 'volatile acidity': 0.7, 'citric acid': 0.0, 'residual sugar': 1.9,
            'chlorides': 0.076, 'free sulfur dioxide': 11.0, 'total sulfur dioxide': 34.0,
            'density': 0.9978, 'pH': 3.51, 'sulphates': 0.56, 'alcohol': 9.4, 'type': 0}


def test_modele_contient_scaler_et_modele(artefact):
    assert 'scaler' in artefact and 'modele' in artefact
    assert len(artefact['variables']) == 12


def test_prediction_est_0_ou_1(artefact, vin_exemple):
    assert predire(vin_exemple, artefact) in (0, 1)


def test_ordre_des_variables_sans_importance(artefact, vin_exemple):
    # Le même vin avec les clés dans un autre ordre doit donner la même prédiction
    vin_inverse = dict(reversed(list(vin_exemple.items())))
    assert predire(vin_inverse, artefact) == predire(vin_exemple, artefact)


def test_vin_mediocre_predit_pas_bon(artefact, vin_exemple):
    # Ce vin est noté 5/10 par les experts : le modèle doit le classer "pas bon"
    assert predire(vin_exemple, artefact) == 0


def test_performance_minimale(artefact):
    # Garde-fou : le F1 sur le test ne doit pas retomber sous 0.45 (on avait ≈ 0.54)
    X, Y = preparer_donnees()
    _, Xtest, _, ytest = train_test_split(X, Y, test_size=0.25, random_state=1, stratify=Y)
    y_pred = artefact['modele'].predict(artefact['scaler'].transform(Xtest))
    assert f1_score(ytest, y_pred) > 0.45
