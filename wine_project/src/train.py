"""
Entraîne le modèle final retenu dans le notebook (SVM RBF optimisé + class_weight='balanced')
et le sauvegarde avec son scaler dans models/modele_vin.joblib.

Lancement : python src/train.py
"""
import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preparer_donnees, VARIABLES

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMIN_MODELE = os.path.join(RACINE, 'models', 'modele_vin.joblib')


def entrainer():
    X, Y = preparer_donnees()

    # Même découpage que dans le notebook
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.25, random_state=1, stratify=Y)

    # Le scaler est calculé sur le train uniquement
    scaler = StandardScaler()
    Xtrain_norm = scaler.fit_transform(Xtrain)
    Xtest_norm = scaler.transform(Xtest)

    # Modèle final : hyperparamètres trouvés par Grid Search en 2.4
    modele = SVC(kernel='rbf', C=100, gamma='scale', class_weight='balanced')
    modele.fit(Xtrain_norm, ytrain)

    y_pred = modele.predict(Xtest_norm)
    print(classification_report(ytest, y_pred))
    print('F1 (bon vin) sur le test : {0:.3f}'.format(f1_score(ytest, y_pred)))

    # On sauvegarde le scaler avec le modèle : les nouvelles données doivent être normalisées exactement pareil
    os.makedirs(os.path.dirname(CHEMIN_MODELE), exist_ok=True)
    joblib.dump({'scaler': scaler, 'modele': modele, 'variables': VARIABLES}, CHEMIN_MODELE)
    print('Modèle sauvegardé dans', CHEMIN_MODELE)


if __name__ == '__main__':
    entrainer()
