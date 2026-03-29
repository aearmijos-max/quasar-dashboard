import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Quasar Research",
    page_icon="📊",
    layout="wide"
)

# =========================
# ESTILO
# =========================
st.markdown("""
<style>
    .main {background-color: #f7f7f7;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    h1, h2, h3 {color: #b30000;}
    .kpi-card {
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid #e5e5e5;
        box-shadow: 0 4px 14px rgba(0,0,0,0.05);
    }
    .small-note {
        color: #666;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# RUTAS
# =========================
DATA_DIR = Path("data")
PATH_RANKING = DATA_DIR / "bi_ranking.csv"
PATH_COMPANIA = DATA_DIR / "bi_compania.csv"
PATH_SEGMENTO = DATA_DIR / "bi_segmento.csv"
PATH_CIIU = DATA_DIR / "bi_ciiu.csv"

# =========================
# FUNCIONES
# =========================
def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where((b == 0) | pd.isna(b), np.nan, a / b)

def to_numeric_if_exists(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(show_spinner=True)
def load_data():
    if not PATH_RANKING.exists():
        raise FileNotFoundError(f"No existe {PATH_RANKING}")
    if not PATH_COMPANIA.exists():
        raise FileNotFoundError(f"No existe {PATH_COMPANIA}")
    if not PATH_SEGMENTO.exists():
        raise FileNotFoundError(f"No existe {PATH_SEGMENTO}")
    if not PATH_CIIU.exists():
        raise FileNotFoundError(f"No existe {PATH_CIIU}")

    # -------- ranking --------
    df = pd.read_csv(PATH_RANKING, low_memory=False)

    numeric_cols = [
        "anio","expediente","posicion_general","cia_imvalores","id_estado_financiero",
        "ingresos_ventas","activos","patrimonio","utilidad_an_imp","impuesto_renta",
        "n_empleados","ingresos_totales","utilidad_ejercicio","utilidad_neta",
        "cod_segmento","liquidez_corriente","prueba_acida","end_activo","end_patrimonial",
        "margen_bruto","margen_operacional","rent_neta_ventas","roe","roa",
        "deuda_total","deuda_total_c_plazo","total_gastos"
    ]
    df = to_numeric_if_exists(df, numeric_cols)

    for c in ["ciiu_n1", "ciiu_n6"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    # -------- compania --------
    comp = pd.read_csv(PATH_COMPANIA, low_memory=False)
    comp.columns = comp.columns.str.strip().str.lower()
    comp = comp.rename(columns={"provinvia": "provincia", "provinvcia": "provincia"})
    if "expediente" in comp.columns:
        comp["expediente"] = pd.to_numeric(comp["expediente"], errors="coerce")

    comp_keep = [c for c in ["expediente", "ruc", "nombre", "tipo", "provincia"] if c in comp.columns]
    comp = comp[comp_keep].drop_duplicates(subset=["expediente"])

    # -------- segmento --------
    seg = pd.read_csv(PATH_SEGMENTO, low_memory=False)
    seg.columns = seg.columns.str.strip().str.lower()
    if "id_segmento" in seg.columns:
        seg["id_segmento"] = pd.to_numeric(seg["id_segmento"], errors="coerce")

    seg_desc_col = "segmento" if "segmento" in seg.columns else seg.columns[-1]
    seg = seg[[c for c in ["id_segmento", seg_desc_col] if c in seg.columns]].copy()
    seg = seg.rename(columns={seg_desc_col: "descripcion_segmento"})

    # -------- ciiu --------
    ciiu = pd.read_csv(PATH_CIIU, low_memory=False)
    ciiu.columns = ciiu.columns.str.strip().str.lower()

    if "ciiu" not in ciiu.columns or "descripcion" not in ciiu.columns:
        raise ValueError(
            f"bi_ciiu.csv debe tener columnas 'ciiu' y 'descripcion'. Tiene: {ciiu.columns.tolist()}"
        )

    ciiu["ciiu"] = ciiu["ciiu"].astype("string").str.strip()
    ciiu["descripcion"] = ciiu["descripcion"].astype("string").str.strip()
    ciiu = ciiu[["ciiu", "descripcion"]].drop_duplicates(subset=["ciiu"])

    # -------- merges --------
    df = df.merge(comp, on="expediente", how="left")

    if "cod_segmento" in df.columns and "id_segmento" in seg.columns:
        df = df.merge(seg, left_on="cod_segmento", right_on="id_segmento", how="left")

    if "ciiu_n1" in df.columns:
        tmp = ciiu.rename(columns={"ciiu": "ciiu_n1_key", "descripcion": "descripcion_ciiu_general"})
        df = df.merge(tmp, left_on="ciiu_n1", right_on="ciiu_n1_key", how="left").drop(columns=["ciiu_n1_key"], errors="ignore")

    if "ciiu_n6" in df.columns:
        tmp = ciiu.rename(columns={"ciiu": "ciiu_n6_key", "descripcion": "descripcion_ciiu"})
        df = df.merge(tmp, left_on="ciiu_n6", right_on="ciiu_n6_key", how="left").drop(columns=["ciiu_n6_key"], errors="ignore")

    if "descripcion_ciiu" in df.columns and "descripcion_ciiu_general" in df.columns:
        df["descripcion_ciiu"] = df["descripcion_ciiu"].fillna(df["descripcion_ciiu_general"])

    # -------- derivados --------
    if {"ingresos_ventas", "n_empleados"}.issubset(df.columns):
        df["ingresos_por_empleado"] = safe_div(df["ingresos_ventas"], df["n_empleados"])

    if {"deuda_total", "activos"}.issubset(df.columns):
        df["ratio_deuda_total"] = safe_div(df["deuda_total"], df["activos"])

    # tamaño empresa por cuantiles
    if "ingresos_ventas" in df.columns:
        positive = df["ingresos_ventas"].dropna()
        positive = positive[positive > 0]
        if len(positive) > 10:
            q1, q2, q3 = positive.quantile([0.25, 0.50, 0.75]).values
            def size_label(x):
                if pd.isna(x) or x <= 0: return "Sin dato"
                if x <= q1: return "Micro"
                if x <= q2: return "Pequeña"
                if x <= q3: return "Mediana"
                return "Grande"
            df["company_size"] = df["ingresos_ventas"].apply(size_label)
        else:
            df["company_size"] = "Sin dato"
    else:
        df["company_size"] = "Sin dato"

    # score quasar simplificado
    score_cols_pos = [c for c in ["roe", "roa", "liquidez_corriente", "prueba_acida", "margen_operacional"] if c in df.columns]
    score_cols_neg = [c for c in ["end_activo", "end_patrimonial", "ratio_deuda_total"] if c in df.columns]

    parts = []
    for c in score_cols_pos:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            s = s.clip(s.quantile(0.01), s.quantile(0.99))
            scaled = 100 * (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else pd.Series(50, index=s.index)
            parts.append(scaled.fillna(50))

    for c in score_cols_neg:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            s = s.clip(s.quantile(0.01), s.quantile(0.99))
            scaled = 100 * (s - s.min()) / (s.max() - s.min()) if s.max() != s.min() else pd.Series(50, index=s.index)
            parts.append((100 - scaled).fillna(50))

    if parts:
        df["SCORE_QUASAR"] = pd.concat(parts, axis=1).mean(axis=1)
    else:
        df["SCORE_QUASAR"] = 50

    def riesgo(score):
        if pd.isna(score): return "Sin dato"
        if score >= 75: return "Verde"
        if score >= 55: return "Amarillo"
        if score >= 35: return "Naranja"
        return "Rojo"

    df["riesgo_quasar"] = df["SCORE_QUASAR"].apply(riesgo)

    # limpieza texto
    for c in ["nombre","ruc","tipo","provincia","descripcion_segmento","descripcion_ciiu","descripcion_ciiu_general","company_size"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("Sin dato").str.strip()

    return df


def compute_yoy(grouped_df, col):
    out = grouped_df[["anio", col]].copy()
    out[f"{col}_yoy"] = out[col].pct_change() * 100
    return out

# =========================
# CARGA
# =========================
try:
    df = load_data()
except Exception as e:
    st.error(f"Error cargando datos: {e}")
    st.stop()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Quasar Research")
st.sidebar.caption("Filtros principales")

years = sorted([int(x) for x in df["anio"].dropna().unique().tolist()]) if "anio" in df.columns else []
year_options = ["Todos"] + [str(y) for y in years]

selected_year = st.sidebar.selectbox("Año", year_options, index=len(year_options)-1 if len(year_options) > 1 else 0)

provincias = ["Todas"] + sorted(df["provincia"].dropna().astype(str).unique().tolist()) if "provincia" in df.columns else ["Todas"]
selected_prov = st.sidebar.selectbox("Provincia", provincias)

segmentos = ["Todos"] + sorted(df["descripcion_segmento"].dropna().astype(str).unique().tolist()) if "descripcion_segmento" in df.columns else ["Todos"]
selected_seg = st.sidebar.selectbox("Segmento", segmentos)

tipos = ["Todos"] + sorted(df["tipo"].dropna().astype(str).unique().tolist()) if "tipo" in df.columns else ["Todos"]
selected_tipo = st.sidebar.selectbox("Tipo de empresa", tipos)

ciiu_general = ["Todos"] + sorted(df["descripcion_ciiu_general"].dropna().astype(str).unique().tolist()) if "descripcion_ciiu_general" in df.columns else ["Todos"]
selected_ciiu_general = st.sidebar.selectbox("CIIU general", ciiu_general)

search_actividad = st.sidebar.text_input("Buscar actividad CIIU")
search_empresa = st.sidebar.text_input("Buscar empresa / RUC")

sizes = ["Todos"] + sorted(df["company_size"].dropna().astype(str).unique().tolist()) if "company_size" in df.columns else ["Todos"]
selected_size = st.sidebar.selectbox("Tamaño empresa", sizes)

compare_companies = st.sidebar.multiselect(
    "Comparar empresas (hasta 15)",
    options=sorted(df["nombre"].dropna().astype(str).unique().tolist())[:5000],
    max_selections=15
)

# =========================
# FILTRADO
# =========================
df_f = df.copy()

if selected_year != "Todos":
    df_f = df_f[df_f["anio"] == int(selected_year)]

if selected_prov != "Todas":
    df_f = df_f[df_f["provincia"] == selected_prov]

if selected_seg != "Todos":
    df_f = df_f[df_f["descripcion_segmento"] == selected_seg]

if selected_tipo != "Todos":
    df_f = df_f[df_f["tipo"] == selected_tipo]

if selected_ciiu_general != "Todos":
    df_f = df_f[df_f["descripcion_ciiu_general"] == selected_ciiu_general]

if search_actividad:
    df_f = df_f[df_f["descripcion_ciiu"].str.contains(search_actividad, case=False, na=False)]

if search_empresa:
    df_f = df_f[
        df_f["nombre"].str.contains(search_empresa, case=False, na=False) |
        df_f["ruc"].str.contains(search_empresa, case=False, na=False)
    ]

if selected_size != "Todos":
    df_f = df_f[df_f["company_size"] == selected_size]

# =========================
# HEADER
# =========================
st.title("📊 Quasar Research")
st.caption("Inteligencia financiera empresarial • Valores monetarios en miles de USD")

# =========================
# KPIs
# =========================
def money_k(x):
    if pd.isna(x):
        return "—"
    return f"{x/1000:,.0f}"

col1, col2, col3, col4 = st.columns(4)

empresas = df_f["expediente"].nunique() if "expediente" in df_f.columns else 0
ing_ventas = df_f["ingresos_ventas"].sum() if "ingresos_ventas" in df_f.columns else np.nan
ing_totales = df_f["ingresos_totales"].sum() if "ingresos_totales" in df_f.columns else np.nan
util_neta = df_f["utilidad_neta"].sum() if "utilidad_neta" in df_f.columns else np.nan
score_avg = df_f["SCORE_QUASAR"].mean() if "SCORE_QUASAR" in df_f.columns else np.nan

with col1:
    st.markdown('<div class="kpi-card"><div class="small-note">Empresas analizadas</div><h2>{}</h2></div>'.format(f"{empresas:,}"), unsafe_allow_html=True)
with col2:
    st.markdown('<div class="kpi-card"><div class="small-note">Ingresos ventas (miles USD)</div><h2>{}</h2></div>'.format(money_k(ing_ventas)), unsafe_allow_html=True)
with col3:
    st.markdown('<div class="kpi-card"><div class="small-note">Ingresos totales (miles USD)</div><h2>{}</h2></div>'.format(money_k(ing_totales)), unsafe_allow_html=True)
with col4:
    st.markdown('<div class="kpi-card"><div class="small-note">Utilidad neta (miles USD)</div><h2>{}</h2><div class="small-note">Score promedio: {:.2f}</div></div>'.format(money_k(util_neta), score_avg if pd.notna(score_avg) else 0), unsafe_allow_html=True)

st.markdown("")

# =========================
# GRÁFICOS 1
# =========================
c1, c2 = st.columns(2)

with c1:
    st.subheader("Evolución temporal del mercado")
    if {"anio","ingresos_ventas","ingresos_totales","utilidad_neta"}.issubset(df_f.columns):
        evo = df_f.groupby("anio")[["ingresos_ventas","ingresos_totales","utilidad_neta"]].sum().reset_index().sort_values("anio")
        for c in ["ingresos_ventas","ingresos_totales","utilidad_neta"]:
            evo[c] = evo[c] / 1000
        fig = px.line(
            evo,
            x="anio",
            y=["ingresos_ventas","ingresos_totales","utilidad_neta"],
            markers=True,
            color_discrete_sequence=["#b30000", "#d90429", "#7f1d1d"]
        )
        fig.update_layout(
            legend_title_text="Indicador",
            xaxis_title="Año",
            yaxis_title="Miles USD",
            margin=dict(l=20, r=20, t=20, b=20),
            height=430
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay columnas suficientes para este gráfico.")

with c2:
    st.subheader("Variaciones anuales (%)")
    if {"anio","ingresos_ventas","ingresos_totales","utilidad_neta"}.issubset(df_f.columns):
        evo = df_f.groupby("anio")[["ingresos_ventas","ingresos_totales","utilidad_neta"]].sum().reset_index().sort_values("anio")
        yoy_ventas = compute_yoy(evo, "ingresos_ventas")[["anio","ingresos_ventas_yoy"]]
        yoy_tot = compute_yoy(evo, "ingresos_totales")[["anio","ingresos_totales_yoy"]]
        yoy_util = compute_yoy(evo, "utilidad_neta")[["anio","utilidad_neta_yoy"]]
        yoy = yoy_ventas.merge(yoy_tot, on="anio", how="outer").merge(yoy_util, on="anio", how="outer")
        fig = px.line(
            yoy,
            x="anio",
            y=["ingresos_ventas_yoy","ingresos_totales_yoy","utilidad_neta_yoy"],
            markers=True,
            color_discrete_sequence=["#b30000", "#d90429", "#7f1d1d"]
        )
        fig.update_layout(
            legend_title_text="Variación",
            xaxis_title="Año",
            yaxis_title="% variación anual",
            margin=dict(l=20, r=20, t=20, b=20),
            height=430
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay columnas suficientes para este gráfico.")

# =========================
# GRÁFICOS 2
# =========================
c3, c4 = st.columns(2)

with c3:
    st.subheader("Rentabilidad: ROE vs ROA")
    if {"roe","roa","activos","nombre","provincia","riesgo_quasar"}.issubset(df_f.columns):
        dsc = df_f.dropna(subset=["roe","roa"]).copy()
        if len(dsc) > 0:
            med_roe = dsc["roe"].median()
            med_roa = dsc["roa"].median()

            color_map = {
                "Verde": "#16a34a",
                "Amarillo": "#eab308",
                "Naranja": "#f97316",
                "Rojo": "#dc2626",
                "Sin dato": "#6b7280"
            }

            fig = px.scatter(
                dsc.sample(min(len(dsc), 2500)),
                x="roe",
                y="roa",
                color="riesgo_quasar",
                size="activos" if "activos" in dsc.columns else None,
                hover_data=["nombre","provincia","utilidad_neta"],
                color_discrete_map=color_map
            )
            fig.add_vline(x=med_roe, line_dash="dash", line_color="#b30000")
            fig.add_hline(y=med_roa, line_dash="dash", line_color="#b30000")
            fig.update_layout(
                xaxis_title="ROE",
                yaxis_title="ROA",
                margin=dict(l=20, r=20, t=20, b=20),
                height=430
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay datos para ROE/ROA.")
    else:
        st.info("No hay columnas suficientes para este gráfico.")

with c4:
    st.subheader("Comparativo entre empresas")
    if compare_companies:
        dcmp = df_f[df_f["nombre"].isin(compare_companies)].copy()
    else:
        if "ingresos_ventas" in df_f.columns:
            top_names = (
                df_f.groupby("nombre")["ingresos_ventas"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
                .index.tolist()
            )
            dcmp = df_f[df_f["nombre"].isin(top_names)].copy()
        else:
            dcmp = df_f.copy()

    if {"nombre","ingresos_ventas","utilidad_neta","activos"}.issubset(dcmp.columns) and len(dcmp) > 0:
        grp = dcmp.groupby("nombre")[["ingresos_ventas","utilidad_neta","activos"]].sum().reset_index()
        for c in ["ingresos_ventas","utilidad_neta","activos"]:
            grp[c] = grp[c] / 1000
            grp[c] = grp[c].round(0)
        fig = px.bar(
            grp,
            x="nombre",
            y=["ingresos_ventas","utilidad_neta","activos"],
            barmode="group",
            color_discrete_sequence=["#b30000", "#d90429", "#7f1d1d"]
        )
        fig.update_layout(
            xaxis_title="Empresas",
            yaxis_title="Miles USD",
            margin=dict(l=20, r=20, t=20, b=80),
            height=430
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Selecciona empresas para comparar.")

# =========================
# GRÁFICOS 3
# =========================
c5, c6 = st.columns(2)

with c5:
    st.subheader("Top segmentos")
    if {"descripcion_segmento","ingresos_ventas"}.issubset(df_f.columns):
        top_seg = (
            df_f.groupby("descripcion_segmento")["ingresos_ventas"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_seg["ingresos_ventas"] = top_seg["ingresos_ventas"] / 1000
        fig = px.bar(
            top_seg,
            x="descripcion_segmento",
            y="ingresos_ventas",
            color_discrete_sequence=["#b30000"]
        )
        fig.update_layout(
            xaxis_title="Segmento",
            yaxis_title="Miles USD",
            margin=dict(l=20, r=20, t=20, b=80),
            height=430
        )
        st.plotly_chart(fig, use_container_width=True)

with c6:
    st.subheader("Top sectores económicos")
    if {"descripcion_ciiu","ingresos_ventas"}.issubset(df_f.columns):
        top_sec = (
            df_f.groupby("descripcion_ciiu")["ingresos_ventas"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        top_sec["short_label"] = top_sec["descripcion_ciiu"].str.split().str[:2].str.join(" ")
        top_sec["ingresos_ventas"] = top_sec["ingresos_ventas"] / 1000
        fig = px.bar(
            top_sec,
            x="short_label",
            y="ingresos_ventas",
            hover_data=["descripcion_ciiu"],
            color_discrete_sequence=["#b30000"]
        )
        fig.update_layout(
            xaxis_title="Sector económico",
            yaxis_title="Miles USD",
            margin=dict(l=20, r=20, t=20, b=60),
            height=430
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================
# TOP COMPAÑÍAS
# =========================
st.subheader("Top compañías")
if {"nombre","ingresos_ventas"}.issubset(df_f.columns):
    top_comp = (
        df_f.groupby("nombre")["ingresos_ventas"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    top_comp["ingresos_ventas"] = top_comp["ingresos_ventas"] / 1000
    fig = px.bar(
        top_comp.sort_values("ingresos_ventas", ascending=True),
        x="ingresos_ventas",
        y="nombre",
        orientation="h",
        color_discrete_sequence=["#b30000"]
    )
    fig.update_layout(
        xaxis_title="Miles USD",
        yaxis_title="",
        margin=dict(l=20, r=20, t=20, b=20),
        height=520
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# TABLA
# =========================
st.subheader("Ranking ampliado de empresas")
table_cols = [
    "anio","nombre","ruc","provincia","tipo","descripcion_segmento",
    "descripcion_ciiu_general","descripcion_ciiu","company_size","riesgo_quasar",
    "SCORE_QUASAR","ingresos_ventas","ingresos_totales","activos","patrimonio",
    "utilidad_an_imp","utilidad_ejercicio","utilidad_neta","total_gastos",
    "roe","roa","liquidez_corriente","prueba_acida","end_activo","end_patrimonial"
]
table_cols = [c for c in table_cols if c in df_f.columns]

show_df = df_f[table_cols].copy()
for c in ["ingresos_ventas","ingresos_totales","activos","patrimonio","utilidad_an_imp","utilidad_ejercicio","utilidad_neta","total_gastos"]:
    if c in show_df.columns:
        show_df[c] = (show_df[c] / 1000).round(0)

st.dataframe(show_df.head(4000), use_container_width=True)

# =========================
# GLOSARIO
# =========================
with st.expander("Glosario financiero"):
    glossary = pd.DataFrame({
        "Indicador": [
            "SCORE QUASAR", "ROE", "ROA", "Liquidez corriente", "Prueba ácida",
            "Variación anual", "Ingresos ventas", "Ingresos totales", "Utilidad neta"
        ],
        "Definición": [
            "Índice 0-100 que resume rentabilidad, liquidez y riesgo financiero.",
            "Rentabilidad sobre el patrimonio.",
            "Rentabilidad sobre activos.",
            "Capacidad para cubrir obligaciones de corto plazo.",
            "Liquidez estricta sin depender tanto de inventarios.",
            "Cambio porcentual frente al año anterior.",
            "Ventas o ingresos operativos principales del negocio.",
            "Total de ingresos recibidos en el año fiscal.",
            "Ganancia final después de costos, gastos e impuestos."
        ]
    })
    st.dataframe(glossary, use_container_width=True)
