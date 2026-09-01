import sys
import time
import requests
from datetime import datetime
import json
import os

# --- CONFIGURATION ---
# La clé API n'est plus en dur dans le code : elle est lue depuis une
# variable d'environnement (fichier .env, jamais commité sur GitHub).
# Installe python-dotenv si besoin : pip install python-dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # fonctionne aussi si la variable est déjà définie autrement (ex: CI/CD)

API_KEY = os.environ.get("OPENWEATHER_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Variable d'environnement OPENWEATHER_API_KEY manquante. "
        "Crée un fichier .env à la racine du projet avec : "
        "OPENWEATHER_API_KEY=ta_cle_ici"
    )

LATITUDE = 15.4167
LONGITUDE = -16.6667

# --- FONCTIONS PRINCIPALES ---

def obtenir_meteo_actuelle(api_key, lat, lon):
    """
    Récupère les données météo actuelles via l'API Current Weather (2.5).
    Fonctionne SANS abonnement One Call.

    ⚠️ Non recommandé si tu utilises aussi /timeline/1day (One Call 4.0)
    pour entraîner un modèle prédictif : les deux endpoints ne reposent
    pas forcément sur le même modèle météo sous-jacent, ce qui peut
    introduire une incohérence entre données d'entraînement et données
    d'inférence. Utilise plutôt obtenir_meteo_actuelle_onecall() pour
    rester sur une source unique et cohérente.
    """
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
        "lang": "fr"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("❌ Timeout : Le serveur ne répond pas.", file=sys.stderr)
    except requests.exceptions.HTTPError as http_err:
        print(f"❌ Erreur HTTP : {http_err}", file=sys.stderr)
    except requests.exceptions.RequestException as err:
        print(f"❌ Erreur réseau : {err}", file=sys.stderr)
    return None


def obtenir_meteo_actuelle_onecall(api_key, lat, lon, tentatives=3, timeout=15):
    """
    Récupère la météo actuelle via l'endpoint /current de One Call 4.0.
    Même famille que /timeline/1day (modèle OWHL cohérent) — à privilégier
    pour un pipeline de data science, afin de ne jamais mélanger les
    sources entre entraînement et inférence.

    Nécessite le même abonnement "One Call by Call" que /timeline/1day
    et consomme sur le même quota (1000 appels/jour gratuits).

    Réessaie automatiquement en cas de timeout ou d'erreur réseau
    transitoire (jusqu'à `tentatives` fois), avec un court délai entre
    chaque essai — utile pour une collecte automatisée longue durée où
    un simple aléa réseau ne doit pas faire perdre le point de données.
    """
    url = "https://api.openweathermap.org/data/4.0/onecall/current"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
        "lang": "fr"
    }

    for essai in range(1, tentatives + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"❌ Timeout (tentative {essai}/{tentatives}).", file=sys.stderr)
        except requests.exceptions.HTTPError as http_err:
            print(f"❌ Erreur HTTP : {http_err}", file=sys.stderr)
            try:
                print(f"   Détail API : {http_err.response.text}", file=sys.stderr)
            except Exception:
                pass
            break  # une erreur HTTP (401, 403...) ne se résoudra pas en réessayant
        except requests.exceptions.RequestException as err:
            print(f"❌ Erreur réseau (tentative {essai}/{tentatives}) : {err}", file=sys.stderr)

        if essai < tentatives:
            time.sleep(2 * essai)  # backoff progressif : 2s, puis 4s

    return None


def obtenir_previsions_onecall(api_key, lat, lon):
    """
    Récupère les prévisions journalières via l'API One Call 4.0.
    Endpoint : /timeline/1day (agrégats journaliers, jusqu'à 1,5 an).
    Nécessite un abonnement actif "One Call by Call" sur ton compte
    OpenWeather (séparé de l'accès à /data/2.5/weather).
    """
    url = "https://api.openweathermap.org/data/4.0/onecall/timeline/1day"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
        "lang": "fr"
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("❌ Timeout : Le serveur ne répond pas.", file=sys.stderr)
    except requests.exceptions.HTTPError as http_err:
        print(f"❌ Erreur HTTP : {http_err}", file=sys.stderr)
        # Affiche le corps de la réponse pour comprendre la cause exacte du 401/403
        try:
            print(f"   Détail API : {http_err.response.text}", file=sys.stderr)
        except Exception:
            pass
    except requests.exceptions.RequestException as err:
        print(f"❌ Erreur réseau : {err}", file=sys.stderr)
    return None


def obtenir_historique_recent(api_key, lat, lon, jours=10):
    """
    Récupère les 'jours' derniers jours (incluant aujourd'hui) via
    /timeline/1day, et les aplatit en DataFrame avec les MÊMES colonnes
    que dataset_entrainement_niayes.csv (température, humidité, pluie,
    etc.) — pour que les modèles pluie/sécheresse/maladie/irrigation
    puissent l'utiliser directement, sans changement de format.

    Le jour "aujourd'hui" peut contenir des valeurs estimées/prévues si
    la journée n'est pas terminée (comportement normal de l'API).

    Nécessite pandas (import fait localement pour ne pas alourdir le
    reste du script si cette fonction n'est pas utilisée).
    """
    import pandas as pd
    from datetime import timezone, timedelta

    aujourdhui = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    debut = aujourdhui - timedelta(days=jours - 1)
    start_ts = int(debut.timestamp())

    url = "https://api.openweathermap.org/data/4.0/onecall/timeline/1day"
    params = {
        "lat": lat, "lon": lon, "appid": api_key,
        "units": "metric", "lang": "fr",
        "start": start_ts, "cnt": jours,
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as err:
        print(f"❌ Erreur lors de la récupération de l'historique récent : {err}", file=sys.stderr)
        return None

    lignes = []
    for jour in data.get("data", []):
        temp = jour.get("temp", {})
        feels = jour.get("feels_like", {})
        meteo = jour.get("weather", [{}])[0]
        lignes.append({
            "date": datetime.fromtimestamp(jour["dt"], tz=timezone.utc).strftime("%Y-%m-%d"),
            "temp_min": temp.get("min"), "temp_max": temp.get("max"),
            "temp_day": temp.get("day"), "temp_night": temp.get("night"),
            "temp_eve": temp.get("eve"), "temp_morn": temp.get("morn"),
            "feels_like_day": feels.get("day"),
            "pressure": jour.get("pressure"), "humidity": jour.get("humidity"),
            "wind_speed": jour.get("wind_speed"), "wind_deg": jour.get("wind_deg"),
            "clouds": jour.get("clouds"), "pop": jour.get("pop", 0),
            "rain_mm": jour.get("rain", 0), "uvi": jour.get("uvi"),
            "weather_main": meteo.get("main"), "weather_description": meteo.get("description"),
        })

    if not lignes:
        print("⚠️  Aucune donnée reçue pour l'historique récent.", file=sys.stderr)
        return None

    return pd.DataFrame(lignes).sort_values("date").reset_index(drop=True)


def afficher_meteo_actuelle(data):
    """Affiche les données météo actuelles de manière lisible."""
    if not data:
        return

    print("\n" + "="*50)
    print(f"🌤️  MÉTÉO EN DIRECT - ZONE DES NIAYES")
    print(f"📅 {datetime.now().strftime('%A %d %B %Y à %H:%M')}")
    print("="*50)

    main = data.get('main', {})
    weather = data.get('weather', [{}])[0]
    wind = data.get('wind', {})

    print(f"☁️  Ciel : {weather.get('description', 'N/A').capitalize()}")
    print(f"🌡️ Température : {main.get('temp', 'N/A')}°C")
    print(f"💧 Humidité : {main.get('humidity', 'N/A')}%")
    print(f"💨 Vent : {wind.get('speed', 'N/A')} m/s")
    print(f"📊 Pression : {main.get('pressure', 'N/A')} hPa")
    print("="*50)


def afficher_previsions(data):
    """
    Affiche les prévisions journalières renvoyées par
    /data/4.0/onecall/timeline/1day.

    Structure confirmée de la réponse :
    {
      "lat", "lon", "timezone", "timezone_offset",
      "data": [
        {
          "dt", "sunrise", "sunset", "moonrise", "moonset", "moon_phase",
          "temp": {"day","min","max","night","eve","morn"},
          "feels_like": {...},
          "pressure", "humidity", "wind_speed", "wind_deg",
          "weather": [{"id","main","description","icon"}],
          "clouds", "pop", "rain" (optionnel), "uvi"
        },
        ...
      ],
      "prev", "next"  # URLs de pagination
    }
    """
    if not data:
        return

    jours = data.get('data', [])
    if not jours:
        print("ℹ️  Aucune donnée journalière dans la réponse.")
        return

    print("\n📅 PRÉVISIONS JOURNALIÈRES (One Call 4.0 - timeline/1day)")
    print("-"*50)

    for i, jour in enumerate(jours):
        date_str = datetime.fromtimestamp(jour['dt']).strftime('%A %d %b')
        temp = jour.get('temp', {})
        meteo = jour.get('weather', [{}])[0]
        pluie_mm = jour.get('rain', 0)
        proba_pluie = jour.get('pop', 0) * 100  # pop est une fraction 0-1

        print(f"{i+1}. {date_str} — {meteo.get('description', 'N/A').capitalize()}")
        print(f"   🌡️  {temp.get('min', 'N/A')}°C → {temp.get('max', 'N/A')}°C "
              f"(matin {temp.get('morn', 'N/A')}°C, jour {temp.get('day', 'N/A')}°C, "
              f"soir {temp.get('eve', 'N/A')}°C, nuit {temp.get('night', 'N/A')}°C)")
        print(f"   ☔ Pluie : {pluie_mm} mm | 🎲 Proba : {proba_pluie:.0f}% | "
              f"💧 Humidité : {jour.get('humidity', 'N/A')}% | 💨 Vent : {jour.get('wind_speed', 'N/A')} m/s")
        print()


def sauvegarder_donnees(data, nom_fichier="meteo_niayes.csv"):
    """Sauvegarde les données météo dans un fichier CSV."""
    if not data:
        return

    import pandas as pd

    info = {
        'timestamp': datetime.now().isoformat(),
        'temperature': data.get('main', {}).get('temp'),
        'humidite': data.get('main', {}).get('humidity'),
        'pression': data.get('main', {}).get('pressure'),
        'vent': data.get('wind', {}).get('speed'),
        'description': data.get('weather', [{}])[0].get('description')
    }

    df = pd.DataFrame([info])

    if os.path.exists(nom_fichier):
        df.to_csv(nom_fichier, mode='a', header=False, index=False)
    else:
        df.to_csv(nom_fichier, index=False)

    print(f"💾 Données sauvegardées dans {nom_fichier}")


def sauvegarder_previsions_json(data, nom_fichier="previsions_niayes.json"):
    """Sauvegarde les prévisions brutes en JSON (structure 4.0 non normalisée)."""
    if not data:
        return
    with open(nom_fichier, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Prévisions sauvegardées dans {nom_fichier}")


# --- EXÉCUTION PRINCIPALE ---

if __name__ == "__main__":
    print("🚀 LANCEMENT DU SCRIPT MÉTÉO - PROJET NIAYES")
    print("-"*50)

    # 1. Récupération des données actuelles via /2.5/weather (référence, sans abonnement One Call)
    print("📡 Récupération des données actuelles (2.5)...")
    data_actuelle = obtenir_meteo_actuelle(API_KEY, LATITUDE, LONGITUDE)
    afficher_meteo_actuelle(data_actuelle)

    if data_actuelle:
        sauvegarder_donnees(data_actuelle)

    # 2. Vérification de l'endpoint /current de One Call 4.0
    #    C'est celui qu'utilise collecte_horaire.py — on affiche la réponse
    #    brute ici pour confirmer que la structure "current" est bien celle
    #    attendue avant de lancer une collecte longue durée.
    print("\n📡 Vérification de /current (One Call 4.0)...")
    data_onecall = obtenir_meteo_actuelle_onecall(API_KEY, LATITUDE, LONGITUDE)
    if data_onecall:
        print(json.dumps(data_onecall, indent=2, ensure_ascii=False))
    else:
        print("⚠️  /current indisponible — vérifie l'abonnement 'One Call by Call'.")

    # 3. Tentative de récupération des prévisions (One Call 4.0)
    print("\n📡 Tentative de récupération des prévisions (One Call 4.0)...")
    previsions = obtenir_previsions_onecall(API_KEY, LATITUDE, LONGITUDE)

    if previsions:
        afficher_previsions(previsions)
        sauvegarder_previsions_json(previsions)
    else:
        print("⚠️  Les prévisions ne sont pas disponibles.")
        print("   → Vérifie que l'abonnement 'One Call by Call' est actif pour ta clé API")
        print("     sur https://home.openweathermap.org/subscriptions")

    print("\n✅ Script terminé.")
