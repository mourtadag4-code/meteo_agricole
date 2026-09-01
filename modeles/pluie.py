"""
Fonctions de prédiction de pluie (J+1), réutilisables depuis l'app
Streamlit ou tout autre point d'entrée.

⚠️ Contrairement au modèle maladie (un seul jour suffit), celui-ci a
besoin d'AU MOINS 7 jours d'historique consécutifs se terminant au jour
pour lequel on veut prédire "demain" — les features incluent des lags à
7 jours et des moyennes glissantes sur 7 jours.
"""

import numpy as np
import pandas as pd
import joblib

CHEMIN_MODELE = "modele_pluie.joblib"


def charger_modele(chemin=CHEMIN_MODELE):
    return joblib.load(chemin)


def construire_features(df_historique, cols_a_decaler):
    """
    df_historique : DataFrame trié par date croissante, au moins 7 jours,
    se terminant au jour pour lequel on veut prédire "demain".
    """
    df = df_historique.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    df["doy_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

    for col in cols_a_decaler:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)
        df[f"{col}_lag7"] = df[col].shift(7)

    df["temp_roll7"] = df["temp_day"].rolling(7).mean()
    df["humidity_roll7"] = df["humidity"].rolling(7).mean()
    df["pop_roll7"] = df["pop"].rolling(7).mean()
    df["rain_roll7"] = df["rain_mm"].rolling(7).sum()

    return df


def predire_pluie(df_historique, modele_charge=None):
    """
    Retourne (probabilité 0-1, prédiction binaire) pour le jour SUIVANT
    le dernier jour de df_historique.
    df_historique doit contenir au moins 7 jours consécutifs se terminant
    au jour de référence (aujourd'hui), triés par date croissante.
    """
    if modele_charge is None:
        modele_charge = charger_modele()

    modele = modele_charge["modele"]
    scaler = modele_charge["scaler"]
    features = modele_charge["features"]
    seuil = modele_charge["seuil"]
    cols_a_decaler = modele_charge["cols_a_decaler"]

    if len(df_historique) < 8:
        raise ValueError(
            f"Au moins 8 jours d'historique nécessaires (7 pour les lags + "
            f"le jour courant), {len(df_historique)} fournis."
        )

    df_feat = construire_features(df_historique, cols_a_decaler)
    derniere_ligne = df_feat.iloc[[-1]][features]

    if derniere_ligne.isna().any().any():
        raise ValueError(
            "Certaines features sont manquantes pour le dernier jour — "
            "vérifie que l'historique fourni est bien continu (pas de trous)."
        )

    X_scaled = scaler.transform(derniere_ligne)
    proba = modele.predict_proba(X_scaled)[0, 1]
    prediction = int(proba >= seuil)

    return proba, prediction


def interpreter_pluie(proba, prediction):
    if prediction == 1:
        return "🌧️ Pluie probable demain", f"Probabilité estimée : {proba*100:.0f}%. Inutile d'irriguer aujourd'hui si la pluie couvre le besoin."
    else:
        return "☀️ Pas de pluie significative attendue", f"Probabilité estimée : {proba*100:.0f}%."
