"""
app_streamlit.py — Prédiction du Risque d'Abandon Scolaire
"""

import streamlit as st
import joblib
import joblib, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Abandon Scolaire — Détection Précoce",
    page_icon="🎓", layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.header{background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);padding:2rem 2.5rem;border-radius:14px;color:white;margin-bottom:1.5rem;}
.header h1{font-size:1.9rem;margin:0;font-weight:700;}
.header p{color:#93b4cc;margin:.4rem 0 0;font-size:.95rem;}

.verdict{border-radius:12px;padding:1.5rem;text-align:center;margin:.8rem 0;}
.verdict-risk{background:#fff0f0;border:2px solid #e74c3c;}
.verdict-safe{background:#f0fff4;border:2px solid #27ae60;}
.verdict .big{font-size:3rem;font-weight:700;margin:.4rem 0;}
.verdict .sub{color:#777;font-size:.9rem;}

/* NOUVEAU BLOC ALERTE */
.alerte-box{
    border-radius:12px;
    padding:1.2rem;
    margin-top:.6rem;
    text-align:center;
    font-weight:600;
}

.alerte-rouge{background:#fdecea;color:#c0392b;border:2px solid #e74c3c;}
.alerte-orange{background:#fff4e6;color:#e67e22;border:2px solid #f39c12;}
.alerte-bleu{background:#eef6ff;color:#1565c0;border:2px solid #42a5f5;}
.alerte-vert{background:#edf7ed;color:#2e7d32;border:2px solid #66bb6a;}

.stButton>button{background:linear-gradient(135deg,#203a43,#2c5364);color:white;border:none;border-radius:10px;padding:.65rem 1.5rem;font-size:1rem;font-weight:600;width:100%;}
footer,#MainMenu{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    m = joblib.load(os.path.join(base,"best_model.pkl"))
    s = joblib.load(os.path.join(base,"scaler.pkl"))
    return m, s

model, scaler = load_model()

FEATURES = ['average_grade','absenteeism_rate','study_time_hours','global_score','study_efficiency']

def build_X(grade, absence, study):
    gs = (grade + study * 2) / 2
    se = grade * study
    return pd.DataFrame([[grade, absence, study, gs, se]], columns=FEATURES)

def calculer_alerte(grade, absence, study, proba):
    c1 = grade  < 10
    c2 = absence > 0.30
    c3 = study   < 1.0
    if c1 and c2 and c3: return "ALERTE_PRECOCE"
    if c1 and c2:         return "PRIORITE"
    if c1 or c2 or c3 or proba >= 0.35: return "SURVEILLANCE"
    return "OK"

def predire(grade, absence, study):
    X  = build_X(grade, absence, study)
    Xs = scaler.transform(X)
    p  = int(model.predict(Xs)[0])
    pr = float(model.predict_proba(Xs)[0][1])
    return p, pr, calculer_alerte(grade, absence, study, pr)

def actions(alerte):
    if alerte == "ALERTE_PRECOCE":
        return ["Convoquer l'étudiant et sa famille dans les 24h",
                "Ouvrir un dossier de suivi renforcé",
                "Attribuer un tuteur académique dédié",
                "Contacter le service d'orientation",
                "Bilan hebdomadaire obligatoire"]
    if alerte == "PRIORITE":
        return ["Entretien individuel sous 48h",
                "Soutien scolaire ciblé",
                "Identifier les causes des absences",
                "Suivi mensuel de la progression"]
    if alerte == "SURVEILLANCE":
        return ["Surveiller sur les 2 prochaines semaines",
                "Discussion avec l'étudiant",
                "Encourager le soutien académique"]
    return ["Maintenir le suivi standard","Valoriser les bons résultats"]

st.markdown("""
<div class="header">
  <h1>Prédiction du Risque d'Abandon Scolaire</h1>
  <p>Outil d’aide à la détection précoce du risque d’abandon scolaire</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Analyse", "À propos"])

# ================= ANALYSE =================
with tab1:
    with st.sidebar:
        st.markdown("## Profil de l'étudiant")
        grade   = st.slider("Moyenne générale (/20)", 0.0, 20.0, 12.0, 0.1)
        absence = st.slider("Taux d'absentéisme", 0.0, 0.5, 0.15, 0.01)
        study   = st.slider("Heures d'étude / jour", 0.0, 5.0, 2.0, 0.1)

    pred, proba, alerte = predire(grade, absence, study)

    col_g, col_d = st.columns([1.05, 1], gap="large")

    with col_g:
        if pred == 1:
            cls,col,emo,txt = "verdict-risk","#e74c3c","🔴","RISQUE D'ABANDON"
        else:
            cls,col,emo,txt = "verdict-safe","#27ae60","🟢","PAS DE RISQUE"

        st.markdown(f"""
<div class="verdict {cls}">
  <div style="font-size:1.6rem;font-weight:700">{emo} {txt}</div>
  <div class="big" style="color:{col}">{proba:.0%}</div>
  <div class="sub">probabilité estimée</div>
</div>""", unsafe_allow_html=True)

        # ===== NOUVEAU NIVEAU D'ALERTE =====
        st.markdown("### Niveau d'alerte")

        if alerte == "ALERTE_PRECOCE":
            texte = "Alerte précoce - intervention immédiate"
            css = "alerte-rouge"
        elif alerte == "PRIORITE":
            texte = "Priorité - intervention rapide"
            css = "alerte-orange"
        elif alerte == "SURVEILLANCE":
            texte = "Surveillance recommandée"
            css = "alerte-bleu"
        else:
            texte = "Situation stable"
            css = "alerte-vert"

        st.markdown(f"""
<div class="alerte-box {css}">
{texte}<br>
<small>{proba:.0%} de risque</small>
</div>
""", unsafe_allow_html=True)

        st.markdown("### Plan d'action")
        for a in actions(alerte):
            st.write("-", a)

    with col_d:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba*100,
            title={"text":"Probabilité (%)"},
            gauge={"axis":{"range":[0,100]}}
        ))
        st.plotly_chart(fig, use_container_width=True)

# ================= A PROPOS =================
with tab2:
    st.markdown("### À propos du projet")
    st.markdown("""
Ce projet s’inscrit dans un mini-projet en intelligence artificielle portant sur la prédiction du risque d’abandon scolaire.

### Objectif
L’objectif est d’identifier les élèves susceptibles d’abandonner leurs études afin de permettre une intervention rapide et adaptée.

### Problème traité
Il s’agit d’un problème de classification :
- 1 → élève à risque d’abandon  
- 0 → élève sans risque  

L’idée est simple : utiliser les données scolaires pour anticiper les situations à risque.

### Données utilisées
Chaque élève est décrit à partir de plusieurs informations, notamment :
- sa moyenne générale  
- son taux d’absentéisme  
- son temps d’étude  

Ces éléments permettent d’évaluer son niveau d’engagement scolaire.

### Démarche suivie
Le projet a été réalisé en plusieurs étapes :
- préparation et nettoyage des données  
- analyse des facteurs influençant l’abandon  
- création de nouvelles variables (score global, efficacité)  
- entraînement de plusieurs modèles de machine learning  
- sélection du modèle le plus performant  
- déploiement sous forme d’application interactive  

### Fonctionnement de l’application
L’utilisateur renseigne les informations d’un élève.

L’application :
- estime la probabilité d’abandon  
- attribue un niveau d’alerte  
- propose des actions à mettre en place  

### Outils utilisés
- Python  
- Pandas  
- Scikit-learn  
- Streamlit  

Auteur : Emmanuel DJEDJE
""")

st.markdown("---")
st.markdown("<center style='color:gray;font-size:0.8rem'>Projet académique</center>", unsafe_allow_html=True)
