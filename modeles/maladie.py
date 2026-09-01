"""
Fonctions de prédiction du risque de maladie fongique, réutilisables
depuis l'app Streamlit ou tout autre point d'entrée (API, script batch).

Isole la logique métier de l'interface, comme convenu — Streamlit importe
ces fonctions, il n'y a pas de duplication de code entre l'app et
l'entraînement.
"""

import numpy as np
import pandas as pd
import joblib

CHEMIN_MODELE = "modele_risque_maladie.joblib"


def point_de_rosee_approx(temp_c, humidite_pct):
    """Formule de Magnus — voir feature_engineering_maladie.py pour le détail."""
    a, b = 17.62, 243.12
    gamma = (a * temp_c) / (b + temp_c) + np.log(humidite_pct / 100.0)
    return (b * gamma) / (a - gamma)


def calculer_features_maladie(meteo_jour):
    """
    Calcule les features nécessaires au modèle à partir d'un dict/Series
    de météo brute pour UN jour (mêmes champs que le dataset d'entraînement
    ou que la réponse /current de meteo_niayes.py, une fois harmonisés).

    meteo_jour attend au minimum : temp_min, temp_max, temp_day,
    temp_night, temp_eve, temp_morn, feels_like_day, pressure, humidity,
    wind_speed, wind_deg, clouds, pop, rain_mm, uvi, ndvi, date
    """
    m = dict(meteo_jour)  # copie défensive, accepte dict ou pandas.Series

    temp_ref = (m["temp_min"] + m["temp_max"]) / 2
    date = pd.to_datetime(m["date"])
    jour_annee = date.dayofyear

    m["interaction_temp_humidite"] = temp_ref * m["humidity"]
    m["saison_sin"] = np.sin(2 * np.pi * jour_annee / 365.25)
    m["saison_cos"] = np.cos(2 * np.pi * jour_annee / 365.25)
    m["point_de_rosee_approx"] = point_de_rosee_approx(temp_ref, m["humidity"])
    m["ecart_rosee_tempmin"] = m["temp_min"] - m["point_de_rosee_approx"]

    return m


def charger_modele(chemin=CHEMIN_MODELE):
    return joblib.load(chemin)


def predire_risque_maladie(meteo_jour, modele_charge=None):
    """
    Retourne la probabilité de risque maladie (0-1) pour un jour donné.
    meteo_jour : dict ou pandas.Series avec les champs météo bruts du jour.
    modele_charge : objet retourné par charger_modele() — le charge lui-même
    si non fourni (pratique pour un usage ponctuel, à éviter en boucle pour
    ne pas relire le fichier à chaque appel).
    """
    if modele_charge is None:
        modele_charge = charger_modele()

    modele, colonnes_features = modele_charge["modele"], modele_charge["features"]

    features = calculer_features_maladie(meteo_jour)
    X = pd.DataFrame([{c: features.get(c, 0) for c in colonnes_features}])

    proba = modele.predict_proba(X)[0, 1]
    return proba


def interpreter_risque(proba):
    """Traduit une probabilité en recommandation textuelle simple pour l'app."""
    if proba >= 0.7:
        return "🔴 Risque élevé", "Conditions favorables aux champignons — envisager un traitement préventif."
    elif proba >= 0.4:
        return "🟠 Risque modéré", "Surveiller l'évolution météo dans les prochains jours."
    else:
        return "🟢 Risque faible", "Conditions actuelles peu favorables au développement fongique."
