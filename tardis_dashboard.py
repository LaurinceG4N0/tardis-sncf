#!/usr/bin/env python3
##
## EPITECH PROJECT, 2026
## tardis
## File description:
## tardis_dashboard
##

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# CONFIGURATION
st.set_page_config(
    page_title="TARDIS - Retards",
    page_icon="train",
    layout="wide",
)

# STYLE GLOBAL - compatible dark mode
st.markdown(
    """
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 10px;
    padding: 16px;
    background-color: rgba(255, 255, 255, 0.05);
}

[data-testid="stMetricLabel"] {
    font-size: 0.85rem;
    opacity: 0.75;
}

[data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-weight: 700;
}

h1, h2, h3 {
    font-weight: 600;
}

div[data-testid="stButton"] > button {
    width: 100%;
}
</style>
""",
    unsafe_allow_html=True,
)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e0e0e0",
    margin=dict(l=10, r=10, t=30, b=10),
)


def apply_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)"
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)"
    )
    return fig


# CHARGEMENT DES DONNÉES
@st.cache_data
def charger_donnees():
    if not os.path.exists("cleaned_dataset.csv"):
        st.warning("Dataset introuvable - utilisation des données de démonstration.")
        return generer_donnees_demo()
    return pd.read_csv("cleaned_dataset.csv", sep=";", encoding="utf-8")


def generer_donnees_demo():
    rng = np.random.default_rng(42)
    n = 500

    gares = [
        "PARIS MONTPARNASSE",
        "BORDEAUX ST JEAN",
        "LYON PART DIEU",
        "MARSEILLE ST CHARLES",
        "NANTES",
        "TOULOUSE MATABIAU",
    ]

    departs = rng.choice(gares, n)
    arrivees = [rng.choice([g for g in gares if g != d]) for d in departs]

    return pd.DataFrame(
        {
            "Year": rng.choice([2020, 2021, 2022, 2023], n),
            "Month": rng.integers(1, 13, n),
            "Service": rng.choice(["National", "TGV inOui"], n),
            "Departure station": departs,
            "Arrival station": arrivees,
            "Number of scheduled trains": rng.integers(50, 300, n),
            "Number of cancelled trains": rng.integers(0, 15, n),
            "Average delay of all trains at arrival": np.abs(rng.normal(5, 7, n)),
            "Average delay of all trains at departure": np.abs(rng.normal(4, 5, n)),
            "Number of trains delayed > 15min": rng.integers(0, 50, n),
            "Pct delay due to infrastructure": rng.uniform(0, 40, n),
            "Pct delay due to traffic management": rng.uniform(0, 30, n),
            "Pct delay due to rolling stock": rng.uniform(0, 25, n),
            "Pct delay due to external causes": rng.uniform(0, 20, n),
            "Pct delay due to passenger handling (crowding, disabled persons, connections)": rng.uniform(
                0, 15, n
            ),
            "Pct delay due to station management and equipment reuse": rng.uniform(
                0, 15, n
            ),
            "Average journey time": rng.uniform(1.0, 5.0, n),
            "Delay categories": rng.choice(["Faible", "Modéré", "Fort"], n),
        }
    )


@st.cache_resource
def charger_modele():
    if os.path.exists("model.joblib"):
        return joblib.load("model.joblib")
    return None


def engineer_features(input_df: pd.DataFrame) -> pd.DataFrame:
    """Applique le même feature engineering que dans le notebook (7 features).

    Temporelles     : month_sin, month_cos, season, is_peak_month
    Opérationnelles : cancellation_rate, pct_heavily_delayed, stress_score
    """
    df = input_df.copy()
    month_col = df["Month"]

    # Temporelles
    df["month_sin"] = np.sin(2 * np.pi * month_col / 12)
    df["month_cos"] = np.cos(2 * np.pi * month_col / 12)
    df["season"] = pd.cut(
        month_col,
        bins=[0, 3, 6, 9, 12],
        labels=[1, 2, 3, 4],
        include_lowest=True,
    ).astype(float)
    df["is_peak_month"] = month_col.isin([7, 8, 12, 1]).astype(int)

    # Opérationnelles
    n_prog = df["Number of scheduled trains"].clip(lower=1)

    df["cancellation_rate"] = df["Number of cancelled trains"] / n_prog

    if "Number of trains delayed > 15min" in df.columns:
        df["pct_heavily_delayed"] = df["Number of trains delayed > 15min"] / n_prog
    else:
        df["pct_heavily_delayed"] = 0.0

    infra_col = "Pct delay due to infrastructure"
    trafic_col = "Pct delay due to traffic management"
    if infra_col in df.columns and trafic_col in df.columns:
        df["stress_score"] = df[infra_col] + df[trafic_col]
    else:
        df["stress_score"] = 0.0

    return df


# CHARGEMENT EFFECTIF
df = charger_donnees()
model = charger_modele()

COL_RETARD = "Average delay of all trains at arrival"
COL_DEPART = "Departure station"
COL_ARRIVEE = "Arrival station"
COL_PROGRAMME = "Number of scheduled trains"
COL_ANNULE = "Number of cancelled trains"

# SIDEBAR
st.sidebar.title("TARDIS")

annees = sorted(df["Year"].dropna().unique()) if "Year" in df.columns else []
services = sorted(df["Service"].dropna().unique()) if "Service" in df.columns else []

annees_sel = st.sidebar.multiselect("Années", annees, default=annees)
services_sel = st.sidebar.multiselect("Services", services, default=services)
mois_sel = st.sidebar.slider("Mois", 1, 12, (1, 12))

page = st.sidebar.radio(
    "Navigation",
    ["Vue générale", "Analyse des retards", "Gares et routes", "Prédiction"],
)

if model:
    st.sidebar.success("Modèle chargé")
else:
    st.sidebar.info("Aucun modèle détecté")

# FILTRES
df_filtre = df.copy()
if annees_sel:
    df_filtre = df_filtre[df_filtre["Year"].isin(annees_sel)]
if services_sel:
    df_filtre = df_filtre[df_filtre["Service"].isin(services_sel)]
df_filtre = df_filtre[
    (df_filtre["Month"] >= mois_sel[0]) & (df_filtre["Month"] <= mois_sel[1])
]

# PAGE : VUE GÉNÉRALE
if page == "Vue générale":
    st.title("TARDIS - Retards SNCF")
    st.caption(f"{len(df_filtre):,} lignes sélectionnées sur {len(df):,}")

    col1, col2, col3, col4 = st.columns(4)
    retard_moyen = df_filtre[COL_RETARD].mean()
    ponctualite = (df_filtre[COL_RETARD] < 5).mean() * 100
    total_trains = df_filtre[COL_PROGRAMME].sum()
    taux_annul = df_filtre[COL_ANNULE].sum() / max(total_trains, 1) * 100

    col1.metric("Retard moyen", f"{retard_moyen:.1f} min")
    col2.metric("Ponctualité", f"{ponctualite:.1f}%")
    col3.metric("Trains programmés", f"{total_trains:,.0f}")
    col4.metric("Taux d'annulation", f"{taux_annul:.2f}%")

    st.divider()

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Retard moyen par année")
        par_annee = df_filtre.groupby("Year")[COL_RETARD].mean().reset_index()
        fig = px.bar(
            par_annee,
            x="Year",
            y=COL_RETARD,
            color=COL_RETARD,
            color_continuous_scale="Blues",
        )
        fig.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Année",
            yaxis_title="Retard moyen (min)",
        )
        st.plotly_chart(apply_theme(fig), width="stretch")

    with col_g2:
        st.subheader("Causes des retards")
        causes = {
            "Infrastructure": "Pct delay due to infrastructure",
            "Trafic": "Pct delay due to traffic management",
            "Matériel": "Pct delay due to rolling stock",
            "Externe": "Pct delay due to external causes",
            "Passagers": "Pct delay due to passenger handling (crowding, disabled persons, connections)",
            "Gare": "Pct delay due to station management and equipment reuse",
        }
        valeurs = {
            label: df_filtre[col].mean()
            for label, col in causes.items()
            if col in df_filtre.columns
        }
        fig2 = px.pie(
            names=list(valeurs.keys()), values=list(valeurs.values()), hole=0.45
        )
        st.plotly_chart(apply_theme(fig2), width="stretch")

    st.subheader("Évolution mensuelle")
    par_mois = df_filtre.groupby(["Year", "Month"])[COL_RETARD].mean().reset_index()
    par_mois["Periode"] = (
        par_mois["Year"].astype(str) + "-" + par_mois["Month"].astype(str).str.zfill(2)
    )
    fig3 = px.line(par_mois, x="Periode", y=COL_RETARD, color="Year", markers=True)
    fig3.update_layout(xaxis_title="Période", yaxis_title="Retard moyen (min)")
    st.plotly_chart(apply_theme(fig3), width="stretch")

# PAGE : ANALYSE DES RETARDS
elif page == "Analyse des retards":
    st.title("Analyse des retards")

    seuil = st.slider("Seuil critique (minutes)", 5, 30, 15)

    fig = px.histogram(df_filtre, x=COL_RETARD, nbins=40)
    fig.add_vline(
        x=df_filtre[COL_RETARD].mean(),
        line_dash="dash",
        annotation_text="Moyenne",
        annotation_position="top right",
    )
    fig.add_vline(
        x=seuil,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Seuil {seuil} min",
        annotation_position="top left",
    )
    st.plotly_chart(apply_theme(fig), width="stretch")

    st.divider()

    col_groupby = st.selectbox("Comparer par", ["Service", "Year", "Month"])
    fig_box = px.box(df_filtre, x=col_groupby, y=COL_RETARD, color=col_groupby)
    st.plotly_chart(apply_theme(fig_box), width="stretch")

    st.divider()

    col_dep = "Average delay of all trains at departure"
    if col_dep in df_filtre.columns:
        st.subheader("Départ vs arrivée")
        sample = df_filtre.sample(min(800, len(df_filtre)))

        # Trendline manuelle
        fig_scatter = px.scatter(
            sample, x=col_dep, y=COL_RETARD, color="Service", opacity=0.6
        )

        for service_name, grp in sample.groupby("Service"):
            if len(grp) >= 2:
                m, b = np.polyfit(grp[col_dep].fillna(0), grp[COL_RETARD].fillna(0), 1)
                x_line = np.linspace(grp[col_dep].min(), grp[col_dep].max(), 50)
                fig_scatter.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=m * x_line + b,
                        mode="lines",
                        name=f"{service_name} (tendance)",
                        line=dict(dash="dash", width=1.5),
                    )
                )

        st.plotly_chart(apply_theme(fig_scatter), width="stretch")
    else:
        st.info("Colonne de retard au départ indisponible.")

# PAGE : GARES ET ROUTES
elif page == "Gares et routes":
    st.title("Gares et routes")

    n_gares = st.slider("Nombre de gares", 5, 30, 15)

    par_gare = (
        df_filtre.groupby(COL_DEPART)[COL_RETARD]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Retard moyen", "count": "Observations"})
        .query("Observations >= 3")
        .sort_values("Retard moyen", ascending=False)
        .head(n_gares)
    )

    fig_gares = px.bar(
        par_gare,
        x="Retard moyen",
        y=COL_DEPART,
        orientation="h",
        color="Retard moyen",
        color_continuous_scale="Blues",
    )
    fig_gares.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(apply_theme(fig_gares), width="stretch")

    st.divider()

    df_filtre = df_filtre.copy()
    df_filtre["route"] = df_filtre[COL_DEPART] + " -> " + df_filtre[COL_ARRIVEE]
    routes = sorted(df_filtre["route"].unique())

    col1, col2 = st.columns(2)
    with col1:
        route1 = st.selectbox("Route A", routes)
    with col2:
        route2 = st.selectbox("Route B", routes, index=min(1, len(routes) - 1))

    def stats_route(route):
        sub = df_filtre[df_filtre["route"] == route]
        return {
            "Observations": len(sub),
            "Retard moyen": round(sub[COL_RETARD].mean(), 2),
            "Retard médian": round(sub[COL_RETARD].median(), 2),
            "Retard max": round(sub[COL_RETARD].max(), 2),
        }

    comp = pd.DataFrame({route1: stats_route(route1), route2: stats_route(route2)}).T
    st.dataframe(comp, width="stretch")

    st.divider()
    st.subheader("Export")
    st.dataframe(par_gare, width="stretch")
    csv = par_gare.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger le CSV", csv, file_name="gares_retards.csv", mime="text/csv"
    )

# PAGE : PRÉDICTION
elif page == "Prédiction":
    st.title("Prédiction")

    if model is None:
        st.error("model.joblib introuvable")
        st.stop()

    gares = sorted(
        set(df[COL_DEPART].dropna().unique()) | set(df[COL_ARRIVEE].dropna().unique())
    )
    services = sorted(df["Service"].dropna().unique())
    delay_categories = (
        sorted(df["Delay categories"].dropna().unique())
        if "Delay categories" in df.columns
        else ["Faible", "Modéré", "Fort"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        gare_dep = st.selectbox("Gare de départ", gares)
        gare_arr = st.selectbox("Gare d'arrivée", [g for g in gares if g != gare_dep])
        service = st.selectbox("Service", services)

    with col2:
        annee = st.selectbox("Année", sorted(df["Year"].dropna().unique()))
        mois = st.selectbox("Mois", list(range(1, 13)))
        n_prog = st.number_input("Trains programmés", 1, 500, 120)

    with col3:
        n_annul = st.number_input("Trains annulés", 0, 100, 3)
        retard_dep = st.slider("Retard au départ (min)", 0.0, 30.0, 4.0, 0.5)
        n_15min = st.number_input("Retards > 15 min", 0, 300, 15)

    col4, col5 = st.columns(2)
    with col4:
        duree_trajet = st.slider("Durée moyenne du trajet (h)", 0.5, 6.0, 3.0, 0.5)
    with col5:
        delay_cat = st.selectbox("Catégorie de retard", delay_categories)

    st.subheader("Causes")
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        pct_infra = st.slider("Infrastructure", 0, 100, 20)
        pct_traf = st.slider("Trafic", 0, 100, 25)
    with col_c2:
        pct_mat = st.slider("Matériel", 0, 100, 20)
        pct_ext = st.slider("Externe", 0, 100, 10)
    with col_c3:
        pct_pax = st.slider("Passagers", 0, 100, 10)
        pct_gare = st.slider("Gare", 0, 100, 15)

    if st.button("Lancer la prédiction", type="primary"):
        input_df = pd.DataFrame(
            [
                {
                    "Service": service,
                    "Departure station": gare_dep,
                    "Arrival station": gare_arr,
                    "Average journey time": duree_trajet,
                    "Number of scheduled trains": n_prog,
                    "Number of cancelled trains": n_annul,
                    "Average delay of all trains at departure": retard_dep,
                    "Number of trains delayed > 15min": n_15min,
                    "Pct delay due to external causes": pct_ext,
                    "Pct delay due to infrastructure": pct_infra,
                    "Pct delay due to traffic management": pct_traf,
                    "Pct delay due to rolling stock": pct_mat,
                    "Pct delay due to station management and equipment reuse": pct_gare,
                    "Pct delay due to passenger handling (crowding, disabled persons, connections)": pct_pax,
                    "Year": annee,
                    "Month": mois,
                    "Delay categories": delay_cat,
                }
            ]
        )

        input_df = engineer_features(input_df)

        try:
            prediction = float(model.predict(input_df)[0])
            prediction = max(0.0, prediction)

            st.metric("Retard prédit", f"{prediction:.1f} min")

            if prediction < 3:
                st.success("Service considéré ponctuel")
            elif prediction < 8:
                st.warning("Retard faible")
            elif prediction < 15:
                st.warning("Retard modéré")
            else:
                st.error("Retard critique")

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")
