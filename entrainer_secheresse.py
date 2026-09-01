"""
Entraînement du modèle de risque sécheresse (pluie cumulée < seuil sur les
7 prochains jours), repris et nettoyé du notebook exploratoire.

Correction apportée : le notebook calculait 'rain_pred' (sortie du modèle
pluie) et l'ajoutait à features_secheresse, MAIS à cause de l'ordre des
cellules, le scaler et le modèle final ont été entraînés sur X_train_scaled
calculé AVANT cet ajout — rain_pred n'a donc jamais réellement influencé
l'entraînement, et le fichier features_secheresse.pkl sauvegardé (29
colonnes) ne correspondait plus aux 28 features réellement apprises par
le modèle. Ce script supprime cette étape inutile et non fonctionnelle,
et sauvegarde modèle/scaler/features ensemble pour éviter toute
désynchronisation future.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import joblib

FICHIER_DONNEES = "dataset_entrainement_niayes.csv"
FICHIER_MODELE = "modele_secheresse.joblib"

SEUIL_SECHERESSE_MM = 2  # mm cumulés sur 7 jours
FENETRE_JOURS = 7
COLS_A_DECALER = ["temp_day", "humidity", "pop", "rain_mm", "wind_speed", "uvi"]

# Hyperparamètres déjà trouvés par RandomizedSearchCV dans le notebook
# exploratoire (AUC 0.941, meilleur seuil 0.62 → F1 0.930 sur le test).
MEILLEURS_PARAMS = {
    "subsample": 0.7, "n_estimators": 300, "max_depth": 8,
    "learning_rate": 0.01, "colsample_bytree": 0.7,
}
MEILLEUR_SEUIL = 0.62

FEATURES_SECHERESSE = [
    "temp_day", "humidity", "pressure", "wind_speed", "pop",
    "month_sin", "month_cos", "doy_sin", "doy_cos",
    "temp_day_lag1", "temp_day_lag2", "temp_day_lag7",
    "humidity_lag1", "humidity_lag2", "humidity_lag7",
    "pop_lag1", "pop_lag2",
    "rain_mm_lag1", "rain_mm_lag2", "rain_mm_lag7",
    "wind_speed_lag1", "wind_speed_lag2", "wind_speed_lag7",
    "temp_roll7", "humidity_roll7", "pop_roll7", "rain_roll7",
]  # pop_lag7 retirée (constante, 0% d'importance — cf. notebook)


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


if __name__ == "__main__":
    print(f"📂 Lecture de {FICHIER_DONNEES} ...")
    df = pd.read_csv(FICHIER_DONNEES)
    df = construire_features(df)

    df["pluie_future_7j"] = df["rain_mm"].rolling(FENETRE_JOURS).sum().shift(-FENETRE_JOURS)
    df["secheresse_binary"] = (df["pluie_future_7j"] < SEUIL_SECHERESSE_MM).astype(int)

    df_clean = df.dropna(subset=FEATURES_SECHERESSE + ["secheresse_binary"])
    print(f"✅ {len(df_clean)} lignes conservées après nettoyage "
          f"({df_clean.index[0].date()} → {df_clean.index[-1].date()})")
    print(f"   Proportion de jours en sécheresse (<{SEUIL_SECHERESSE_MM}mm/7j) : "
          f"{df_clean['secheresse_binary'].mean():.1%}")

    X = df_clean[FEATURES_SECHERESSE]
    y = df_clean["secheresse_binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
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
    print(classification_report(y_test, y_pred, target_names=["Pas sécheresse", "Sécheresse"]))
    print(f"AUC-ROC : {roc_auc_score(y_test, y_proba):.4f}")

    joblib.dump({
        "modele": modele, "scaler": scaler, "features": FEATURES_SECHERESSE,
        "seuil": MEILLEUR_SEUIL, "cols_a_decaler": COLS_A_DECALER,
    }, FICHIER_MODELE)
    print(f"\n💾 Sauvegardé : {FICHIER_MODELE}")
