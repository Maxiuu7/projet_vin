"""
Tests unitaires du prétraitement : on vérifie que chaque étape de la partie 2.1 fait bien ce qu'elle doit faire.
Lancement : pytest
"""
import pandas as pd
from preprocessing import charger_donnees, nettoyer, creer_cible, preparer_donnees, VARIABLES


def test_chargement_fusionne_rouges_et_blancs():
    df = charger_donnees()
    assert len(df) == 1599 + 4898           # tous les vins des 2 fichiers
    assert set(df['type'].unique()) == {0, 1}  # type bien encodé en 0/1


def test_pas_de_valeur_manquante():
    df = charger_donnees()
    assert df.isnull().sum().sum() == 0


def test_nettoyer_supprime_les_doublons():
    df = nettoyer(charger_donnees())
    assert df.duplicated().sum() == 0
    assert len(df) == 5320


def test_creer_cible_seuil_7():
    # Petit tableau inventé pour vérifier la règle quality >= 7 -> bon vin
    df = pd.DataFrame({'quality': [3, 6, 7, 9], 'alcohol': [9, 10, 12, 13]})
    resultat = creer_cible(df)
    assert list(resultat['bon_vin']) == [0, 0, 1, 1]


def test_quality_retiree_pour_eviter_la_fuite():
    df = creer_cible(nettoyer(charger_donnees()))
    assert 'quality' not in df.columns


def test_preparer_donnees_forme_de_X_et_Y():
    X, Y = preparer_donnees()
    assert list(X.columns) == VARIABLES      # 12 variables, dans le bon ordre
    assert len(X) == len(Y)
    assert set(Y.unique()) == {0, 1}         # cible binaire


def test_proportion_de_bons_vins():
    # On a mesuré environ 19 % de bons vins : si ce chiffre change, une étape a été modifiée
    X, Y = preparer_donnees()
    assert 0.17 < Y.mean() < 0.21
