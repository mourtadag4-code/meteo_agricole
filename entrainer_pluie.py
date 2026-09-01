"""
Entraînement du modèle de prédiction de pluie (J+1), repris et nettoyé du
notebook exploratoire — même feature engineering et mêmes hyperparamètres
déjà trouvés (RandomizedSearchCV), mais :
- pop_lag7 (colonne constante, 0% d'importance) retirée proprement
- le modèle, le scaler ET la liste des features sont sauvegardés
  ENSEMBLE, dans un seul fichier, pour éviter tout risque de
  désynchronisation entre les trois (c'est ce qui causait le bug avec
  features_secheresse.pkl dans le notebook sécheresse).

Lance ce script une fois sur ta machine (même dossier que
dataset_entrainement_niayes.csv) pour régénérer un modele_pluie.joblib
propre et garanti cohérent.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import joblib

FICHIER_DONNEES = "dataset_entrainement_niayes.csv"
FICHIER_MODELE = "modele_pluie.joblib"

SEUIL_PLUIE_MM = 0.2  # pluie significative
COLS_A_DECALER = ["temp_day", "humidity", "pop", "rain_mm", "wind_speed", "uvi"]

# Hyperparamètres déjà trouvés par RandomizedSearchCV dans le notebook
# exploratoire — repris tels quels pour ne pas relancer une recherche
# longue (~AUC 0.852, meilleur seuil 0.23 → F1 0.667 sur le jeu de test).
MEILLEURS_PARAMS = {
    "subsample": 0.7, "n_estimators": 100, "max_depth": 6,
    "learning_rate": 0.01, "colsample_bytree": 0.7,
}
MEILLEUR_SEUIL = 0.23


def construire_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

    for col in COLS_A_DECALER:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)
        df[f"{col}_lag7"] = df[col].shift(7)

    df["temp_roll7"] = df["temp_day"].rolling(7).mean()
    df["humidity_roll7"] = df["humidity"].rolling(7).mean()
    df["pop_roll7"] = df["pop"].rolling(7).mean()
    df["rain_roll7"] = df["rain_mm"].rolling(7).sum()

    return df


# Liste finale des features (pop_lag7 retirée : constante, 0% d'importance)
FEATURES_PLUIE = [
    "temp_day", "humidity", "pressure", "wind_speed", "pop",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "temp_day_lag1", "temp_day_lag2", "temp_day_lag7",
    "humidity_lag1", "humidity_lag2", "humidity_lag7",
    "pop_lag1", "pop_lag2",
    "rain_mm_lag1", "rain_mm_lag2", "rain_mm_lag7",
    "wind_speed_lag1", "wind_speed_lag2", "wind_speed_lag7",
    "temp_roll7", "humidity_roll7", "pop_roll7", "rain_roll7",
]


if __name__ == "__main__":
    print(f"📂 Lecture de {FICHIER_DONNEES} ...")
    df = pd.read_csv(FICHIER_DONNEES)
    df = construire_features(df)

    df["rain_tomorrow_binary"] = (df["rain_mm"].shift(-1) >= SEUIL_PLUIE_MM).astype(int)
    df_clean = df.dropna(subset=FEATURES_PLUIE + ["rain_tomorrow_binary"])
    print(f"✅ {len(df_clean)} lignes conservées après nettoyage "
          f"({df_clean.index[0].date()} → {df_clean.index[-1].date()})")
    print(f"   Proportion de jours avec pluie (≥{SEUIL_PLUIE_MM}mm) : "
          f"{df_clean['rain_tomorrow_binary'].mean():.1%}")

    X = df_clean[FEATURES_PLUIE]
    y = df_clean["rain_tomorrow_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # split chronologique, jamais aléatoire
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scale_pos_weight = (1 - y_train.mean()) / y_train.mean()
    print(f"⚖️  scale_pos_weight : {scale_pos_weight:.2f}")

    print("🌲 Entraînement XGBoost...")
    modele = xgb.XGBClassifier(
        **MEILLEURS_PARAMS, scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=42,
    )
    modele.fit(X_train_scaled, y_train)

    y_proba = modele.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= MEILLEUR_SEUIL).astype(int)

    print(f"\n📈 Évaluation (seuil {MEILLEUR_SEUIL}) :")
    print(classification_report(y_test, y_pred, target_names=["Pas pluie", "Pluie"]))
    print(f"AUC-ROC : {roc_auc_score(y_test, y_proba):.4f}")

    # Modèle, scaler ET liste de features sauvegardés ENSEMBLE — évite le
    # bug de désynchronisation rencontré dans le notebook sécheresse.
    joblib.dump({
        "modele": modele, "scaler": scaler, "features": FEATURES_PLUIE,
        "seuil": MEILLEUR_SEUIL, "cols_a_decaler": COLS_A_DECALER,
    }, FICHIER_MODELE)
    print(f"\n💾 Sauvegardé : {FICHIER_MODELE}")
