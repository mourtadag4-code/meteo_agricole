"""
Plateforme Intelligente Météo & Agricole — Zone des Niayes (Sénégal)

Lancement : streamlit run app.py

Structure :
- Les modèles/calculs vivent dans modeles/ (maladie.py, irrigation.py) et
  sont importés ici, pas réécrits — même logique que celle testée en
  amont pendant l'entraînement.
- Les onglets Pluie / Sécheresse sont des stubs en attendant les scripts
  des 2 premiers modèles.
"""

import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "modeles"))
from meteo_niayes import obtenir_historique_recent, API_KEY, LATITUDE, LONGITUDE
from modeles.maladie import (
    charger_modele as charger_modele_maladie_fn, predire_risque_maladie, interpreter_risque
)
from modeles.irrigation import calculer_besoins_eau, TABLE_KC
from modeles.pluie import charger_modele as charger_modele_pluie_fn, predire_pluie, interpreter_pluie
from modeles.secheresse import charger_modele as charger_modele_secheresse_fn, predire_secheresse, interpreter_secheresse

# --- Configuration de la page ---
st.set_page_config(
    page_title="Météo & Agriculture — Niayes",
    page_icon="🌾",
    layout="wide",
)

FICHIER_DONNEES = "dataset_entrainement_niayes.csv"


@st.cache_data
def charger_donnees():
    df = pd.read_csv(FICHIER_DONNEES, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_resource
def charger_modele_maladie():
    return charger_modele_maladie_fn()


@st.cache_resource
def charger_modele_pluie():
    return charger_modele_pluie_fn()


@st.cache_resource
def charger_modele_secheresse():
    return charger_modele_secheresse_fn()


df_meteo = charger_donnees()
modele_maladie = charger_modele_maladie()
modele_pluie = charger_modele_pluie()
modele_secheresse = charger_modele_secheresse()

# --- En-tête ---
st.title("🌾 Plateforme Intelligente Météo & Agricole")
st.caption("Zone des Niayes, Sénégal — aide à la décision pour l'irrigation et la protection des cultures")

# --- Barre latérale : sélection commune à plusieurs onglets ---
with st.sidebar:
    st.header("⚙️ Paramètres")
    culture = st.selectbox("Culture", options=list(TABLE_KC.keys()), index=0)
    stade = st.selectbox("Stade de croissance", options=["initial", "mi_saison", "fin_saison"], index=1)
    st.divider()

    mode = st.radio("Source des données", ["📅 Historique (démo)", "🔴 Aujourd'hui (en direct)"])

    if mode == "📅 Historique (démo)":
        date_selectionnee = st.date_input(
            "Date à analyser",
            value=df_meteo["date"].max(),
            min_value=df_meteo["date"].min(),
            max_value=df_meteo["date"].max(),
        )
    else:
        st.caption("Appelle l'API OpenWeather pour récupérer les 10 derniers jours réels.")
        if st.button("🔄 Actualiser la météo en direct"):
            st.cache_data.clear()

# --- Récupération des données selon le mode choisi ---
if mode == "📅 Historique (démo)":
    df_historique_recent = df_meteo[df_meteo["date"] <= pd.to_datetime(date_selectionnee)].tail(30)
    ligne_jour = df_meteo[df_meteo["date"] == pd.to_datetime(date_selectionnee)]
    if ligne_jour.empty:
        st.error("Aucune donnée météo pour cette date.")
        st.stop()
    meteo_jour = ligne_jour.iloc[0]
    date_affichee = date_selectionnee
else:
    @st.cache_data(ttl=1800)  # ré-appelle l'API au plus toutes les 30 minutes
    def recuperer_meteo_live():
        return obtenir_historique_recent(API_KEY, LATITUDE, LONGITUDE, jours=10)

    df_live = recuperer_meteo_live()
    if df_live is None or df_live.empty:
        st.error("Impossible de récupérer la météo en direct — vérifie la connexion "
                 "réseau et l'abonnement One Call sur ta clé API.")
        st.stop()

    df_live["date"] = pd.to_datetime(df_live["date"])
    df_historique_recent = df_live
    meteo_jour = df_live.iloc[-1]
    date_affichee = meteo_jour["date"].date()
    st.sidebar.success(f"Météo récupérée en direct — dernier jour : {date_affichee}")

# --- Onglets ---
onglet_apercu, onglet_pluie, onglet_secheresse, onglet_maladie, onglet_eau = st.tabs([
    "📊 Aperçu météo", "🌧️ Pluie", "☀️ Sécheresse", "🍄 Risque maladie", "💧 Besoin en eau"
])

# ==================== APERÇU MÉTÉO ====================
with onglet_apercu:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Température (jour)", f"{meteo_jour['temp_day']:.1f} °C")
    col2.metric("Humidité", f"{meteo_jour['humidity']:.0f} %")
    col3.metric("Pluie", f"{meteo_jour['rain_mm']:.1f} mm")
    col4.metric("Vent", f"{meteo_jour['wind_speed']:.1f} m/s")

    st.subheader(f"Évolution météo récente (jusqu'au {date_affichee})")
    if mode == "📅 Historique (démo)":
        df_recent = df_meteo[df_meteo["date"] <= pd.to_datetime(date_selectionnee)].tail(90)
    else:
        df_recent = df_historique_recent  # seulement les 10 jours récupérés en direct

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_recent["date"], y=df_recent["temp_max"], name="Temp. max", line=dict(color="firebrick")))
    fig.add_trace(go.Scatter(x=df_recent["date"], y=df_recent["temp_min"], name="Temp. min", line=dict(color="royalblue")))
    fig.update_layout(yaxis_title="°C", legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    fig_pluie = px.bar(df_recent, x="date", y="rain_mm", labels={"rain_mm": "Pluie (mm)"})
    st.plotly_chart(fig_pluie, use_container_width=True)

    if "ndvi" in df_recent.columns and df_recent["ndvi"].notna().any():
        fig_ndvi = px.line(df_recent, x="date", y="ndvi", labels={"ndvi": "NDVI (végétation)"})
        st.plotly_chart(fig_ndvi, use_container_width=True)

# ==================== PLUIE ====================
with onglet_pluie:
    if len(df_historique_recent) < 8:
        st.warning("Pas assez d'historique avant cette date (8 jours minimum requis) "
                   "pour calculer les tendances récentes.")
    else:
        try:
            proba_pluie, pred_pluie = predire_pluie(df_historique_recent, modele_charge=modele_pluie)
            niveau_pluie, message_pluie = interpreter_pluie(proba_pluie, pred_pluie)

            col_jauge, col_texte = st.columns([1, 2])
            with col_jauge:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=proba_pluie * 100, number={"suffix": "%"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "royalblue"},
                           "steps": [{"range": [0, 23], "color": "#fff3e0"},
                                     {"range": [23, 100], "color": "#bbdefb"}]},
                    title={"text": "Probabilité de pluie demain"},
                ))
                st.plotly_chart(fig, use_container_width=True)
            with col_texte:
                st.subheader(niveau_pluie)
                st.write(message_pluie)
                st.caption("Seuil de décision optimisé : 23% (au lieu de 50%) — le déséquilibre "
                           "des classes rend ce seuil plus fiable que le seuil par défaut.")
        except ValueError as e:
            st.error(str(e))

# ==================== SÉCHERESSE ====================
with onglet_secheresse:
    if len(df_historique_recent) < 8:
        st.warning("Pas assez d'historique avant cette date (8 jours minimum requis).")
    else:
        try:
            proba_sech, pred_sech = predire_secheresse(df_historique_recent, modele_charge=modele_secheresse)
            niveau_sech, message_sech = interpreter_secheresse(proba_sech, pred_sech)

            col_jauge, col_texte = st.columns([1, 2])
            with col_jauge:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=proba_sech * 100, number={"suffix": "%"},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "peru"},
                           "steps": [{"range": [0, 62], "color": "#e8f5e9"},
                                     {"range": [62, 100], "color": "#ffe0b2"}]},
                    title={"text": "Risque sécheresse (7 prochains jours)"},
                ))
                st.plotly_chart(fig, use_container_width=True)
            with col_texte:
                st.subheader(niveau_sech)
                st.write(message_sech)
                st.caption("Seuil de décision optimisé : 62% (au lieu de 50%).")
        except ValueError as e:
            st.error(str(e))

# ==================== RISQUE MALADIE ====================
with onglet_maladie:
    proba = predire_risque_maladie(meteo_jour, modele_charge=modele_maladie)
    niveau, message = interpreter_risque(proba)

    col_jauge, col_texte = st.columns([1, 2])
    with col_jauge:
        fig_jauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0, 40], "color": "#c8e6c9"},
                    {"range": [40, 70], "color": "#ffe0b2"},
                    {"range": [70, 100], "color": "#ffcdd2"},
                ],
            },
            title={"text": "Probabilité de risque"},
        ))
        st.plotly_chart(fig_jauge, use_container_width=True)

    with col_texte:
        st.subheader(niveau)
        st.write(message)
        st.caption(
            "⚠️ Ce score est basé sur des seuils agronomiques (température × "
            "humidité) et un modèle qui apprend à les reproduire — il ne "
            "s'agit pas d'une prédiction validée sur des cas réels de "
            "maladies observées, faute de données d'incidence disponibles."
        )

    with st.expander("Détail des conditions météo du jour"):
        st.write(f"Température moyenne : {(meteo_jour['temp_min']+meteo_jour['temp_max'])/2:.1f} °C")
        st.write(f"Humidité : {meteo_jour['humidity']:.0f} %")

# ==================== BESOIN EN EAU ====================
with onglet_eau:
    df_eau_jour = calculer_besoins_eau(pd.DataFrame([meteo_jour]), culture=culture, stade=stade)
    ligne = df_eau_jour.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ET0 (référence)", f"{ligne['et0_mm']:.2f} mm/j")
    col2.metric(f"ETc ({culture})", f"{ligne['etc_mm']:.2f} mm/j")
    col3.metric("Pluie efficace", f"{ligne['pluie_efficace_mm']:.2f} mm")
    col4.metric("💧 Besoin net", f"{ligne['besoin_irrigation_mm']:.2f} mm", delta=None)

    if ligne["besoin_irrigation_mm"] < 0.5:
        st.success("Pas d'irrigation nécessaire aujourd'hui — la pluie couvre le besoin.")
    elif ligne["besoin_irrigation_mm"] < 3:
        st.info("Besoin d'irrigation modéré aujourd'hui.")
    else:
        st.warning("Besoin d'irrigation important aujourd'hui.")

    st.caption(
        "Calcul basé sur la méthode FAO-56 (Hargreaves-Samani pour ET0, "
        "coefficient cultural Kc tabulé par la FAO). Le stade de croissance "
        "est choisi manuellement dans la barre latérale, faute de date de "
        "semis par parcelle."
    )

    st.subheader("Besoin d'irrigation, période récente")
    df_recent_eau = calculer_besoins_eau(df_historique_recent, culture=culture, stade=stade)
    fig_eau = px.bar(df_recent_eau, x="date", y="besoin_irrigation_mm",
                      labels={"besoin_irrigation_mm": "Besoin net (mm)"})
    st.plotly_chart(fig_eau, use_container_width=True)
