"""
Fonctions de prédiction du risque sécheresse (pluie cumulée sur les 7
prochains jours), réutilisables depuis l'app Streamlit.

Même besoin que le modèle pluie : au moins 7 jours d'historique
consécutifs se terminant au jour de référence.
"""

import pandas as pd
import joblib

from modeles.pluie import construire_features  # même feature engineering, réutilisé

CHEMIN_MODELE = "modele_secheresse.joblib"


def charger_modele(chemin=CHEMIN_MODELE):
    return joblib.load(chemin)


def predire_secheresse(df_historique, modele_charge=None):
    """
    Retourne (probabilité 0-1, prédiction binaire) de sécheresse
    (pluie cumulée < seuil) sur les 7 jours suivant le dernier jour de
    df_historique.
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
            f"Au moins 8 jours d'historique nécessaires, {len(df_historique)} fournis."
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


def interpreter_secheresse(proba, prediction):
    if prediction == 1:
        return "🏜️ Risque de sécheresse (7 prochains jours)", f"Probabilité estimée : {proba*100:.0f}%. Planifier l'irrigation en conséquence."
    else:
        return "🌱 Pas de risque de sécheresse notable", f"Probabilité estimée : {proba*100:.0f}%."
