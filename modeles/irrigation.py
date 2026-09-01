"""
Estime le besoin en eau d'irrigation quotidien, selon la méthodologie
standard FAO-56 (Allen et al., 1998) :

    Je prédis le volume d'eau nécessaire aujourd'hui (mm) pour que
    l'agriculteur puisse irriguer la bonne quantité, ni trop ni trop peu.

Contrairement au risque maladie, ceci n'est PAS un modèle entraîné sur un
label incertain — c'est un calcul agronomique physique, reconnu et utilisé
mondialement, qui ne nécessite que des données météo de base (température
min/max, position géographique) déjà disponibles dans notre dataset.

Étapes :
1. ET0 (évapotranspiration de référence, mm/j) — formule de
   Hargreaves-Samani, qui n'exige que temp min/max et le rayonnement
   extraterrestre (calculable à partir de la latitude et du jour de
   l'année, sans capteur).
2. Kc (coefficient cultural, table FAO-56) — ajuste ET0 à la culture et
   son stade de croissance.
3. ETc = ET0 × Kc → besoin en eau réel de la culture (mm/j).
4. Besoin net = ETc − pluie efficace → ce qu'il faut réellement irriguer.
"""

import numpy as np
import pandas as pd

FICHIER_ENTREE = "dataset_entrainement_niayes.csv"
FICHIER_SORTIE = "dataset_besoin_eau_niayes.csv"

LATITUDE = 15.4167  # zone des Niayes, en degrés (coordonnées déjà utilisées pour la météo)

# Coefficients culturaux Kc (FAO-56, Table 12) pour les cultures vedettes
# des Niayes, par stade de croissance. Simplifié à 3 stades (au lieu de 4)
# pour rester praticable sans connaître la date de semis exacte de chaque
# parcelle.
TABLE_KC = {
    "oignon":         {"initial": 0.70, "mi_saison": 1.05, "fin_saison": 0.75},
    "tomate":         {"initial": 0.60, "mi_saison": 1.15, "fin_saison": 0.80},
    "chou":           {"initial": 0.70, "mi_saison": 1.05, "fin_saison": 0.95},
    "carotte":        {"initial": 0.70, "mi_saison": 1.05, "fin_saison": 0.95},
    "pomme_de_terre": {"initial": 0.50, "mi_saison": 1.15, "fin_saison": 0.75},
}

# Fraction de la pluie réellement disponible pour la culture (le reste
# ruisselle ou s'infiltre trop profondément) — approximation courante en
# l'absence de données de sol détaillées.
FRACTION_PLUIE_EFFICACE = 0.8


def rayonnement_extraterrestre(jour_annee, latitude_deg):
    """
    Calcule le rayonnement extraterrestre Ra (MJ/m²/jour), formule
    standard FAO-56 (Annexe 1), à partir du jour de l'année et de la
    latitude — ne nécessite aucune mesure, uniquement la géométrie
    Terre-Soleil.
    """
    phi = np.radians(latitude_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * jour_annee)
    delta = 0.409 * np.sin(2 * np.pi / 365 * jour_annee - 1.39)
    omega_s = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))

    Gsc = 0.0820  # constante solaire, MJ/m²/min
    Ra = (24 * 60 / np.pi) * Gsc * dr * (
        omega_s * np.sin(phi) * np.sin(delta)
        + np.cos(phi) * np.cos(delta) * np.sin(omega_s)
    )
    return Ra


def et0_hargreaves(temp_min, temp_max, jour_annee, latitude_deg=LATITUDE):
    """
    Évapotranspiration de référence ET0 (mm/j), méthode Hargreaves-Samani
    (1985), recommandée par la FAO quand seules les températures sont
    disponibles (pas de mesure de rayonnement solaire, humidité ou vent
    nécessaire pour cette variante simplifiée).
    """
    temp_moy = (temp_min + temp_max) / 2
    Ra = rayonnement_extraterrestre(jour_annee, latitude_deg)
    # Ra est en MJ/m²/j ; le coefficient 0.408 convertit en équivalent mm/j
    et0 = 0.0023 * (Ra * 0.408) * (temp_moy + 17.8) * np.sqrt(np.maximum(temp_max - temp_min, 0))
    return et0


def calculer_besoins_eau(df, culture="oignon", stade="mi_saison"):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    jour_annee = df["date"].dt.dayofyear

    df["et0_mm"] = et0_hargreaves(df["temp_min"], df["temp_max"], jour_annee)

    kc = TABLE_KC[culture][stade]
    df["kc"] = kc
    df["etc_mm"] = df["et0_mm"] * kc

    pluie = df["rain_mm"] if "rain_mm" in df.columns else 0
    df["pluie_efficace_mm"] = pluie * FRACTION_PLUIE_EFFICACE
    df["besoin_irrigation_mm"] = np.maximum(df["etc_mm"] - df["pluie_efficace_mm"], 0)

    return df


if __name__ == "__main__":
    print(f"📂 Lecture de {FICHIER_ENTREE} ...")
    df = pd.read_csv(FICHIER_ENTREE)

    CULTURE = "oignon"  # à changer selon la culture ciblée par l'app
    STADE = "mi_saison"  # simplification : stade fixe faute de date de semis par parcelle

    df_final = calculer_besoins_eau(df, culture=CULTURE, stade=STADE)

    df_final.to_csv(FICHIER_SORTIE, index=False)
    print(f"💾 Sauvegardé : {FICHIER_SORTIE}")

    print(f"\n📊 ET0 (évapotranspiration de référence) — statistiques :")
    print(df_final["et0_mm"].describe())

    print(f"\n📊 Besoin d'irrigation net ({CULTURE}, stade {STADE}) par mois :")
    df_final["mois"] = pd.to_datetime(df_final["date"]).dt.month
    print(df_final.groupby("mois")["besoin_irrigation_mm"].mean().round(2))

    print(f"\nExemple, dernier jour ({df_final['date'].iloc[-1]}) :")
    derniere = df_final.iloc[-1]
    print(f"  ET0 : {derniere['et0_mm']:.2f} mm/j")
    print(f"  ETc ({CULTURE}) : {derniere['etc_mm']:.2f} mm/j")
    print(f"  Pluie efficace : {derniere['pluie_efficace_mm']:.2f} mm")
    print(f"  → Besoin d'irrigation net : {derniere['besoin_irrigation_mm']:.2f} mm")
