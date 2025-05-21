import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

data = pd.read_csv('features.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    ada = AdaBoostClassifier(n_estimators=100, random_state=0)
    ada.fit(X_train_scaled, y_train)
    
    y_pred_prob = ada.predict_proba(X_test_scaled)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    auc_scores.append(auc)
    print(f'Fold {fold+1} AUC: {auc:.4f}')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ada_final = AdaBoostClassifier(n_estimators=100, random_state=0)
ada_final.fit(X_scaled, y)

joblib.dump(ada_final, 'adaboost_model.joblib')
joblib.dump(scaler, 'standard_scaler_ada.joblib')

print(f"\nAverage AUC: {np.mean(auc_scores):.4f}")