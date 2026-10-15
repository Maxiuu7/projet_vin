# Classification de la qualité du vin – Projet Machine Learning 1 (Efrei)

Prédire si un vin est **bon (qualité ≥ 7)** ou **pas bon** à partir de ses caractéristiques physico-chimiques.

- **Données** : Wine Quality (rouges + blancs), 5320 vins après suppression des doublons, 19 % de bons vins
- **Modèle final** : SVM RBF (C=100, gamma='scale', class_weight='balanced') → F1 ≈ 0.54, rappel ≈ 0.70

## Structure du projet

```
├── Rapport_technique_ML_qualite_vin.docx   # rapport technique
├── Projet_ML_qualite_vin.ipynb             # notebook complet (2.1 → 2.4)
├── data/                                   # fichiers CSV (rouges et blancs)
├── src/
│   ├── preprocessing.py                    # chargement, doublons, cible binaire
│   ├── train.py                            # entraîne et sauvegarde le modèle final
│   └── predict.py                          # charge le modèle et prédit un vin
├── tests/                                  # tests unitaires (pytest)
├── models/                                 # modèle sauvegardé (créé par train.py)
└── requirements.txt
```

## Lancer les tests unitaires

```bash
pip install -r requirements.txt
python src/train.py   # entraîne le modèle et le sauvegarde dans models/ (facultatif, les tests le font sinon)
pytest                # lance les 12 tests
```

| Fichier | Ce qui est vérifié |
|---|---|
| `test_preprocessing.py` | fusion des 2 fichiers, aucune valeur manquante, doublons supprimés, règle quality ≥ 7, `quality` retirée (pas de fuite de la cible), forme de X et Y, proportion de bons vins ≈ 19 % |
| `test_model.py` | le modèle se charge avec son scaler, la prédiction vaut 0 ou 1, l'ordre des variables n'a pas d'importance, un vin noté 5/10 est prédit « pas bon », F1 sur le test > 0.45 |
