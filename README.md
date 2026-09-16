# 🌾 Plateforme d'Aide à la Décision Agricole — Niayes, Sénégal

Plateforme prédictive combinant données météo temps réel et modèles de machine learning pour aider les agriculteurs de la zone des Niayes à anticiper les risques climatiques sur leurs cultures.

Projet initié lors du hackathon **Databeez** (2025), puis retravaillé pour fiabiliser le pipeline de production.

## 🎯 Fonctionnalités

- Récupération de données météo en temps réel via API
- Prédiction de la pluie à J+1 (classification binaire)
- Prédiction du risque de sécheresse sur les 7 prochains jours (classification binaire)
- Interface Streamlit accessible aux agriculteurs

**Démo en ligne :** [meteo-agricole.onrender.com](https://meteo-agricole.onrender.com/)

## 🤖 Modèles

| Modèle | Type | AUC-ROC | F1-score | Seuil de décision |
|---|---|---|---|---|
| Prédiction de pluie (J+1) | XGBoost (classification binaire) | 0.852 | 0.667 | 0.23 |
| Risque de sécheresse (7 jours) | XGBoost (classification binaire) | 0.941 | 0.930 | 0.62 |

### Feature engineering

- **27 variables** : mesures météo brutes (température, humidité, pression, vent, probabilité de précipitation), encodage cyclique de la saisonnalité (sin/cos sur mois et jour de l'année), variables décalées (lags 1, 2, 7 jours) et moyennes/sommes glissantes sur 7 jours
- **Split chronologique** (non aléatoire) pour éviter toute fuite de données temporelle entre entraînement et test
- **Gestion du déséquilibre de classes** via `scale_pos_weight` et optimisation du seuil de décision (recherché spécifiquement plutôt que laissé à 0.5)
- **Hyperparamètres** optimisés par `RandomizedSearchCV`

## 🐛 Rigueur technique : un bug de désynchronisation identifié et corrigé

En nettoyant le notebook exploratoire pour en faire des scripts de production, deux problèmes silencieux ont été identifiés :

1. **`features_secheresse.pkl`** sauvegardait 29 colonnes alors que le modèle final n'en avait réellement appris que 28 — une variable (`rain_pred`, la sortie du modèle pluie) avait été ajoutée aux features *après* que le scaler et le modèle aient déjà été entraînés sur les données mises à l'échelle. Elle n'a donc jamais influencé l'entraînement, mais restait présente dans le fichier de features sauvegardé, créant un risque de plantage silencieux ou de mauvaises prédictions en production.
2. Une variable (`pop_lag7`) s'est révélée constante (0% d'importance) et a été retirée proprement du pipeline final.

**Correction apportée** : les scripts `entrainer_pluie.py` et `entrainer_secheresse.py` sauvegardent désormais le modèle, le scaler et la liste exacte des features **ensemble, dans un seul fichier `.joblib`**, éliminant tout risque futur de désynchronisation entre ces trois éléments.

## 🛠️ Stack technique

`Python` · `XGBoost` · `scikit-learn` · `pandas` · `NumPy` · `Streamlit` · API météo

## 🚀 Reproduire l'entraînement

```bash
pip install -r requirements.txt
python entrainer_pluie.py
python entrainer_secheresse.py
```

## 📁 Structure

```
meteo_agricole/
├── app.py                          # Application Streamlit
├── meteo_niayes.py                 # Récupération des données météo (API)
├── entrainer_pluie.py              # Entraînement du modèle de pluie
├── entrainer_secheresse.py         # Entraînement du modèle de sécheresse
├── dataset_entrainement_niayes.csv # Données d'entraînement
├── modele_pluie.joblib             # Modèle pluie sérialisé (modèle + scaler + features)
├── modele_secheresse.joblib        # Modèle sécheresse sérialisé
└── requirements.txt
```

## 👤 Auteur

Mouhamadoul Mourtadha Gueye — [LinkedIn](https://www.linkedin.com/in/mouhamadoul-mourtadha-gueye/) · [GitHub](https://github.com/mourtadag4-code)
