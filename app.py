import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, joblib, copy
warnings.filterwarnings('ignore')

from sklearn.preprocessing  import LabelEncoder, StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, roc_auc_score, confusion_matrix, roc_curve)
from imblearn.over_sampling  import SMOTE

st.set_page_config(page_title="Multiple Disease Prediction System", 
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Sora', sans-serif;
    background-color: #080d1a !important;
    color: #e2e8f4;
}

section[data-testid="stSidebar"] {
    background: #0c1222 !important;
    border-right: 1px solid #1a2540;
}
section[data-testid="stSidebar"] > div { padding-top: 1rem; }

/* Hide default streamlit chrome */
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem 2.5rem !important; max-width: 100% !important; }

/* Sidebar nav radio */
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] > div {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
div[data-testid="stRadio"] > div > label {
    display: flex !important;
    align-items: center;
    padding: 10px 14px;
    border-radius: 8px;
    cursor: pointer;
    color: #8899bb !important;
    font-size: 0.88rem;
    font-weight: 500;
    transition: all 0.15s;
    border: 1px solid transparent;
}
div[data-testid="stRadio"] > div > label:hover {
    background: #111d35;
    color: #c8d8f0 !important;
}
div[data-testid="stRadio"] > div > label[data-checked="true"],
div[data-testid="stRadio"] > div > label:has(input:checked) {
    background: #0e2048;
    border-color: #2563eb;
    color: #60a5fa !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0c1222;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #1a2540;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    font-size: 0.83rem;
    font-weight: 500;
    color: #5a7099;
    padding: 7px 18px;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: #162040 !important;
    color: #60a5fa !important;
    border: 1px solid #1e3a6e !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #0c1630;
    border: 1px solid #1a2d55;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
}
div[data-testid="stMetricLabel"] p { font-size: 0.72rem !important; color: #5a7099 !important; text-transform: uppercase; letter-spacing: 0.08em; }
div[data-testid="stMetricValue"] { font-size: 1.7rem !important; color: #e2e8f4 !important; font-weight: 700 !important; }
div[data-testid="stMetricDelta"] { font-size: 0.75rem !important; color: #4a90d9 !important; }

/* Sliders & inputs dark */
.stSlider > div > div > div { background: #1a2d55 !important; }
.stSlider [data-testid="stThumbValue"] { background: #2563eb; color: white; }
div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div { background: #0c1630 !important; border-color: #1a2d55 !important; }
div[data-testid="stSelectbox"] label,
.stSlider label { color: #8899bb !important; font-size: 0.82rem !important; }

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 600;
    font-size: 0.9rem;
    padding: 0.7rem 1rem;
    letter-spacing: 0.02em;
    transition: all 0.2s;
    box-shadow: 0 4px 15px rgba(37,99,235,0.35);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8);
    box-shadow: 0 6px 20px rgba(37,99,235,0.5);
    transform: translateY(-1px);
}

/* Dataframe */
.stDataFrame { border: 1px solid #1a2d55 !important; border-radius: 10px; overflow: hidden; }

/* Divider */
hr { border-color: #1a2540 !important; margin: 1.5rem 0; }

/* Custom classes */
.stat-card {
    background: linear-gradient(145deg, #0d1a36, #0a1428);
    border: 1px solid #1a2d55;
    border-radius: 14px;
    padding: 1.4rem;
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: #2563eb55; }

.stat-label {
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #4a6080;
    margin-bottom: 0.25rem;
}
.stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.1rem;
}
.stat-sub {
    font-size: 0.73rem;
    color: #4a6080;
}

.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #c8d8f0;
    letter-spacing: 0.01em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a2540;
    margin-left: 0.5rem;
}

.page-header {
    margin-bottom: 1.8rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #1a2540;
}
.page-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f0f4ff;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
}
.page-sub {
    font-size: 0.88rem;
    color: #4a6888;
    font-weight: 400;
}

.result-card-high {
    background: linear-gradient(145deg, #1f0a0a, #2d0f0f);
    border: 1px solid #7f1d1d;
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.result-card-low {
    background: linear-gradient(145deg, #071a0f, #0a2016);
    border: 1px solid #14532d;
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.result-pct {
    font-size: 3.5rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
    margin: 0.5rem 0;
}
.result-label-text {
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    opacity: 0.65;
    margin-top: 0.3rem;
}

.tag {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.info-strip {
    background: #0c1630;
    border: 1px solid #1a2d55;
    border-left: 3px solid #2563eb;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    font-size: 0.8rem;
    color: #6888aa;
    margin-bottom: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# Chart theme
FIG_BG = '#080d1a'
AX_BG  = '#0c1222'
GRID_C = '#1a2540'
TEXT_C = '#8899bb'

def style_ax(ax, grid=True):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TEXT_C, labelsize=8)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.title.set_color('#c8d8f0')
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    if grid:
        ax.yaxis.grid(True, color=GRID_C, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

def style_fig(fig):
    fig.patch.set_facecolor(FIG_BG)
    for ax in fig.get_axes():
        style_ax(ax)


@st.cache_data(show_spinner=False)
def build_preprocessed_data():
    df_dia = pd.read_csv('diabetes.csv')
    df_d   = df_dia.copy()
    df_d[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = \
        df_d[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0, np.nan)
    X_d_raw = df_d.drop('Outcome', axis=1);  y_d = df_d['Outcome']
    imp_d   = SimpleImputer(strategy='median')
    X_d     = pd.DataFrame(imp_d.fit_transform(X_d_raw), columns=X_d_raw.columns)
    sc_d    = StandardScaler()
    X_d_sc  = pd.DataFrame(sc_d.fit_transform(X_d), columns=X_d.columns)

    df_hrt = pd.read_csv('heart_disease_uci.csv')
    df_h   = df_hrt.copy()
    df_h['target'] = (df_h['num'] > 0).astype(int)
    df_h   = df_h.drop(columns=['id','dataset','num'], errors='ignore')
    for col in ['fbs','exang']:
        df_h[col] = df_h[col].map({True:1,False:0,'TRUE':1,'FALSE':0}).astype(float)
    le = LabelEncoder()
    for col in ['sex','cp','restecg','slope','thal']:
        df_h[col] = le.fit_transform(df_h[col].astype(str))
    X_h_raw = df_h.drop('target', axis=1);  y_h = df_h['target']
    imp_h   = SimpleImputer(strategy='median')
    X_h     = pd.DataFrame(imp_h.fit_transform(X_h_raw), columns=X_h_raw.columns)
    sc_h    = StandardScaler()
    X_h_sc  = pd.DataFrame(sc_h.fit_transform(X_h), columns=X_h.columns)

    df_kid = pd.read_csv('kidney_disease.csv')
    df_k   = df_kid.copy()
    df_k   = df_k.drop(columns=['id'], errors='ignore')
    for col in df_k.select_dtypes(include='object').columns:
        df_k[col] = df_k[col].str.strip()
    df_k['target_k'] = (df_k['classification'] == 'ckd').astype(int)
    df_k   = df_k.drop(columns=['classification'])
    bmap   = {'yes':1,'no':0,'normal':1,'abnormal':0,'present':1,'notpresent':0,'good':1,'poor':0}
    for col in ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane']:
        df_k[col] = df_k[col].map(bmap)
    for col in ['pcv','wc','rc']:
        df_k[col] = pd.to_numeric(df_k[col], errors='coerce')
    X_k_raw = df_k.drop('target_k', axis=1);  y_k = df_k['target_k']
    imp_k   = SimpleImputer(strategy='median')
    X_k     = pd.DataFrame(imp_k.fit_transform(X_k_raw), columns=X_k_raw.columns)
    sc_k    = StandardScaler()
    X_k_sc  = pd.DataFrame(sc_k.fit_transform(X_k), columns=X_k.columns)

    return {
        'diabetes': {'Xsc': X_d_sc, 'y': y_d, 'feat': list(X_d.columns),
                     'imp': imp_d, 'sc': sc_d, 'raw': df_dia},
        'heart':    {'Xsc': X_h_sc, 'y': y_h, 'feat': list(X_h.columns),
                     'imp': imp_h, 'sc': sc_h, 'raw': df_hrt},
        'kidney':   {'Xsc': X_k_sc, 'y': y_k, 'feat': list(X_k.columns),
                     'imp': imp_k, 'sc': sc_k, 'raw': df_kid},
    }


@st.cache_resource(show_spinner=False)
def load_models():
    return {
        'diabetes': joblib.load('diabetes_model.pkl'),
        'heart':    joblib.load('heart_model.pkl'),
        'kidney':   joblib.load('kidney_model.pkl'),
    }


@st.cache_resource(show_spinner=False)
def run_evaluation(_prep):
    sm   = SMOTE(random_state=42)
    BASE = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'SVM':                 SVC(probability=True, random_state=42),
    }
    gbs = load_models()
    out = {}
    for key, cfg in _prep.items():
        X, y = cfg['Xsc'].values, cfg['y'].values
        Xtr_r, Xte, ytr_r, yte = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
        Xtr, ytr = sm.fit_resample(Xtr_r, ytr_r)
        res = {}
        for mname, m in BASE.items():
            m = copy.deepcopy(m); m.fit(Xtr, ytr)
            yp = m.predict(Xte); yprob = m.predict_proba(Xte)[:,1]
            fpr, tpr, _ = roc_curve(yte, yprob)
            res[mname] = dict(model=m,
                acc=float(accuracy_score(yte,yp)),
                prec=float(precision_score(yte,yp,zero_division=0)),
                rec=float(recall_score(yte,yp,zero_division=0)),
                f1=float(f1_score(yte,yp,zero_division=0)),
                auc=float(roc_auc_score(yte,yprob)),
                cm=confusion_matrix(yte,yp), fpr=fpr, tpr=tpr)
        gb = gbs[key]
        yp = gb.predict(Xte); yprob = gb.predict_proba(Xte)[:,1]
        fpr, tpr, _ = roc_curve(yte, yprob)
        res['Gradient Boosting'] = dict(model=gb,
            acc=float(accuracy_score(yte,yp)),
            prec=float(precision_score(yte,yp,zero_division=0)),
            rec=float(recall_score(yte,yp,zero_division=0)),
            f1=float(f1_score(yte,yp,zero_division=0)),
            auc=float(roc_auc_score(yte,yprob)),
            cm=confusion_matrix(yte,yp), fpr=fpr, tpr=tpr)
        out[key] = (res, cfg['feat'])
    return out


with st.spinner(""):
    prep   = build_preprocessed_data()
    models = load_models()
    evals  = run_evaluation(prep)

ALL    = ['Logistic Regression', 'Random Forest', 'SVM', 'Gradient Boosting']
BEST   = 'Gradient Boosting'
COLORS = {'diabetes':'#3b82f6', 'heart':'#f43f5e', 'kidney':'#10b981'}
ICONS  = {'diabetes':'🩸', 'heart':'🫀', 'kidney':'🫘'}

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0.5rem 1.5rem;'>
         <div style='font-size:1.15rem;font-weight:800;color:#e2e8f4;letter-spacing:-0.01em;'>Multiple Disease Prediction System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.65rem;font-weight:600;color:#2a4060;text-transform:uppercase;letter-spacing:0.12em;padding:0 0.3rem 0.5rem;'>Navigation</div>", unsafe_allow_html=True)

    page = st.radio("Navigation",
                    ["Overview", "Data Explorer", "Model Performance", "Predict"],
                    label_visibility="collapsed")

  

    


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <div class='page-header'>
        <div class='page-title'>Overview</div>
        <div class='page-sub'>Performance summary across all three disease prediction models</div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    best_auc = max(evals['diabetes'][0][BEST]['auc'],
                   evals['heart'][0][BEST]['auc'],
                   evals['kidney'][0][BEST]['auc'])
    c1.metric("Diseases Covered",  "3",             "Diabetes · Heart · Kidney")
    c2.metric("Total Records",     "2,088",          "768 + 920 + 400")
    c3.metric("Best ROC-AUC",      f"{best_auc:.3f}", "Gradient Boosting")
    c4.metric("Models Compared",   "4",              "LR · RF · SVM · GB")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Disease Modules</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    dinfo = [
        ('diabetes','Diabetes',      '768 records','8 features', 'Pima Indian Women'),
        ('heart',   'Heart Disease', '920 records','13 features','UCI Heart Disease'),
        ('kidney',  'Kidney Disease','400 records','24 features','UCI CKD Dataset'),
    ]
    for col, (key, label, recs, feats, src) in zip([c1,c2,c3], dinfo):
        e   = evals[key][0][BEST]
        clr = COLORS[key]
        ico = ICONS[key]
        with col:
            st.markdown(f"""
            <div class='stat-card' style='border-top:2px solid {clr};'>
                <div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:1rem;'>
                    <span style='font-size:1.3rem;'>{ico}</span>
                    <div>
                        <div style='font-weight:700;color:#dde8ff;font-size:0.95rem;'>{label}</div>
                        <div style='font-size:0.7rem;color:#3a5878;'>{src} · {recs} · {feats}</div>
                    </div>
                </div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;'>
                    <div style='background:#080d1a;border-radius:8px;padding:0.7rem;text-align:center;'>
                        <div class='stat-label'>Accuracy</div>
                        <div class='stat-value' style='color:{clr};'>{e["acc"]:.1%}</div>
                    </div>
                    <div style='background:#080d1a;border-radius:8px;padding:0.7rem;text-align:center;'>
                        <div class='stat-label'>ROC-AUC</div>
                        <div class='stat-value' style='color:{clr};'>{e["auc"]:.3f}</div>
                    </div>
                    <div style='background:#080d1a;border-radius:8px;padding:0.7rem;text-align:center;'>
                        <div class='stat-label'>F1 Score</div>
                        <div class='stat-value' style='color:{clr};'>{e["f1"]:.3f}</div>
                    </div>
                    <div style='background:#080d1a;border-radius:8px;padding:0.7rem;text-align:center;'>
                        <div class='stat-label'>Recall</div>
                        <div class='stat-value' style='color:{clr};'>{e["rec"]:.3f}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>AUC Comparison — All Models</div>", unsafe_allow_html=True)

    short = ['LR','RF','SVM','GB']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (key, label, _, _, _) in zip(axes, dinfo):
        clr  = COLORS[key]
        aucs = [evals[key][0][m]['auc'] for m in ALL]
        bc   = [clr if m==BEST else '#1a2d55' for m in ALL]
        ec   = [clr if m==BEST else '#263a60' for m in ALL]
        bars = ax.bar(short, aucs, color=bc, edgecolor=ec, linewidth=1, width=0.55)
        for bar, val in zip(bars, aucs):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.008,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8,
                    color=clr if val==max(aucs) else TEXT_C, fontweight='600')
        ax.set_ylim(0.55, 1.1)
        ax.set_title(f'{ICONS[key]} {label}', fontsize=10, fontweight='700', pad=10)
        ax.set_ylabel('AUC', fontsize=8)
        style_ax(ax)
    style_fig(fig)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.markdown("""
    <div class='page-header'>
        <div class='page-title'>Data Explorer</div>
        <div class='page-sub'>Raw datasets, feature distributions, and correlation analysis</div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🩸  Diabetes", "🫀  Heart Disease", "🫘  Kidney Disease"])

    def explorer(df, target_col, color, str_target=False):
        c1, c2, c3, c4 = st.columns(4)
        try:
            pos = (df[target_col].str.strip()=='ckd').mean()*100 if str_target \
                  else (df[target_col]>0).mean()*100
            pos_str = f"{pos:.1f}%"
        except Exception:
            pos_str = "—"
        c1.metric("Rows",    df.shape[0])
        c2.metric("Features",df.shape[1]-1)
        c3.metric("Missing", f"{df.isnull().mean().mean()*100:.1f}%")
        c4.metric("Positive Class", pos_str)

        st.markdown("<div class='section-title' style='margin-top:1.2rem;'>Sample Data</div>", unsafe_allow_html=True)
        st.dataframe(df.head(8), use_container_width=True, height=220)

        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in [target_col,'id']]
        if not num_cols: return

        st.markdown("<div class='section-title' style='margin-top:1.2rem;'>Feature Distributions</div>", unsafe_allow_html=True)
        ncols = 4
        nrows = max(1,(len(num_cols)+ncols-1)//ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(13, nrows*2.8))
        flat = np.array(axes).flatten()
        for i, col in enumerate(num_cols):
            vals = df[col].dropna().values
            flat[i].hist(vals, bins=22, color=color+'88', edgecolor=color, linewidth=0.6)
            flat[i].set_title(col, fontsize=8.5, pad=5, color='#c8d8f0')
            style_ax(flat[i], grid=False)
        for j in range(len(num_cols), len(flat)):
            flat[j].set_visible(False)
        style_fig(fig)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig); plt.close(fig)

        st.markdown("<div class='section-title' style='margin-top:1.2rem;'>Correlation Matrix</div>", unsafe_allow_html=True)
        num_df = df.select_dtypes(include=[np.number]).drop(columns=['id'], errors='ignore')
        if num_df.shape[1] > 1:
            fig2, ax2 = plt.subplots(figsize=(10,5.5))
            mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
            cmap = sns.diverging_palette(220, 20, as_cmap=True)
            sns.heatmap(num_df.corr(), mask=mask, ax=ax2, cmap=cmap, center=0,
                        annot=True, fmt='.2f', annot_kws={'size':7,'color':'#c8d8f0'},
                        linewidths=0.4, linecolor='#080d1a', cbar_kws={'shrink':0.7})
            ax2.tick_params(colors=TEXT_C, labelsize=7.5)
            ax2.set_facecolor(AX_BG)
            style_fig(fig2)
            plt.tight_layout()
            st.pyplot(fig2); plt.close(fig2)

    with tab1: explorer(prep['diabetes']['raw'], 'Outcome',        '#3b82f6')
    with tab2: explorer(prep['heart']['raw'],    'num',            '#f43f5e')
    with tab3: explorer(prep['kidney']['raw'],   'classification', '#10b981', str_target=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown("""
    <div class='page-header'>
        <div class='page-title'>Model Performance</div>
        <div class='page-sub'>Metrics, ROC curves, confusion matrices, and feature importance</div>
    </div>""", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🩸  Diabetes", "🫀  Heart Disease", "🫘  Kidney Disease"])

    def perf(key, label, color):
        res, feat_names = evals[key]

        st.markdown("<div class='section-title'>Metrics — All Models</div>", unsafe_allow_html=True)
        rows = [{'Model':m, 'Accuracy':f"{res[m]['acc']:.4f}",
                 'Precision':f"{res[m]['prec']:.4f}", 'Recall':f"{res[m]['rec']:.4f}",
                 'F1':f"{res[m]['f1']:.4f}", 'AUC':f"{res[m]['auc']:.4f}"}
                for m in ALL]
        st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

        left, right = st.columns(2)

        with left:
            st.markdown("<div class='section-title' style='margin-top:1rem;'>ROC Curves</div>", unsafe_allow_html=True)
            rc = ['#3b82f6','#10b981','#f59e0b','#f43f5e']
            fig, ax = plt.subplots(figsize=(5.5,4.5))
            for m, c in zip(ALL, rc):
                lw = 2.2 if m==BEST else 1.4
                ax.plot(res[m]['fpr'], res[m]['tpr'], color=c, lw=lw,
                        label=f"{m}  {res[m]['auc']:.3f}",
                        alpha=1.0 if m==BEST else 0.7)
            ax.plot([0,1],[0,1],'--',color='#2a3a55',lw=1)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{label} — ROC Curves', fontsize=10, fontweight='600')
            ax.legend(fontsize=7.5, framealpha=0.15, labelcolor='white',
                      facecolor='#0c1222', edgecolor='#1a2540')
            style_ax(ax)
            style_fig(fig)
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

        with right:
            st.markdown("<div class='section-title' style='margin-top:1rem;'>Confusion Matrix — GB</div>", unsafe_allow_html=True)
            cm = res[BEST]['cm']
            fig2, ax2 = plt.subplots(figsize=(4.5,3.8))
            cmap = sns.light_palette(color, as_cmap=True, n_colors=8)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax2, cmap=cmap,
                        xticklabels=['Negative','Positive'],
                        yticklabels=['Negative','Positive'],
                        linewidths=2, linecolor='#080d1a',
                        annot_kws={'size':16,'weight':'bold','color':'#0a1428'})
            ax2.set_xlabel('Predicted', color=TEXT_C)
            ax2.set_ylabel('Actual',    color=TEXT_C)
            ax2.tick_params(colors=TEXT_C)
            ax2.set_facecolor(AX_BG)
            style_fig(fig2)
            plt.tight_layout()
            st.pyplot(fig2); plt.close(fig2)

        st.markdown("<div class='section-title' style='margin-top:1rem;'>Feature Importance — Gradient Boosting</div>", unsafe_allow_html=True)
        gb = res[BEST]['model']
        if hasattr(gb,'feature_importances_'):
            imps = gb.feature_importances_
            idx  = np.argsort(imps)[-12:]
            fig3, ax3 = plt.subplots(figsize=(9,3.5))
            bars = ax3.barh([feat_names[i] for i in idx], imps[idx],
                            color=color+'99', edgecolor=color, linewidth=0.8, height=0.6)
            for bar, val in zip(bars, imps[idx]):
                ax3.text(val+0.001, bar.get_y()+bar.get_height()/2,
                         f'{val:.3f}', va='center', fontsize=7.5, color=TEXT_C)
            ax3.set_xlabel('Importance Score')
            ax3.set_title('Top Features', fontsize=10, fontweight='600')
            style_ax(ax3)
            style_fig(fig3)
            plt.tight_layout()
            st.pyplot(fig3); plt.close(fig3)

    with tab1: perf('diabetes','Diabetes',     '#3b82f6')
    with tab2: perf('heart',  'Heart Disease', '#f43f5e')
    with tab3: perf('kidney', 'Kidney Disease','#10b981')


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.markdown("""
    <div class='page-header'>
        <div class='page-title'>Live Prediction</div>
        <div class='page-sub'>Enter patient values to get an instant AI risk assessment</div>
    </div>""", unsafe_allow_html=True)

    def show_result(pred, prob, pos_lbl, neg_lbl):
        if int(pred) == 1:
            st.markdown(f"""
            <div class='result-card-high'>
                <div style='font-size:0.75rem;font-weight:600;color:#991b1b;text-transform:uppercase;letter-spacing:0.1em;'>⚠ High Risk Detected</div>
                <div class='result-pct' style='color:#f87171;'>{prob:.0%}</div>
                <div class='result-label-text' style='color:#f87171;'>{pos_lbl}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-card-low'>
                <div style='font-size:0.75rem;font-weight:600;color:#065f46;text-transform:uppercase;letter-spacing:0.1em;'>✓ Low Risk</div>
                <div class='result-pct' style='color:#34d399;'>{1-prob:.0%}</div>
                <div class='result-label-text' style='color:#34d399;'>{neg_lbl}</div>
            </div>""", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(4, 0.85))
        clr = '#f43f5e' if int(pred)==1 else '#10b981'
        ax.barh([''],  [prob],      color=clr,     height=0.55, alpha=0.9)
        ax.barh([''],  [1-prob],    color='#1a2d55',height=0.55, left=prob)
        ax.axvline(0.5, color='#3a5070', lw=1.5, linestyle='--', alpha=0.8)
        ax.set_xlim(0,1)
        ax.set_xticks([0,.25,.5,.75,1])
        ax.set_xticklabels(['0%','25%','50%','75%','100%'], fontsize=7, color=TEXT_C)
        ax.set_yticks([])
        style_ax(ax, grid=False)
        style_fig(fig)
        plt.tight_layout(pad=0.2)
        st.pyplot(fig); plt.close(fig)

    def run_predict(key, raw_values):
        row  = np.array([raw_values], dtype=float)
        row  = prep[key]['imp'].transform(row)
        row  = prep[key]['sc'].transform(row)
        prob = float(models[key].predict_proba(row)[0][1])
        pred = int(models[key].predict(row)[0])
        return pred, prob

    tab1, tab2, tab3 = st.tabs(["🩸  Diabetes", "🫀  Heart Disease", "🫘  Kidney Disease"])

    with tab1:
        left, right = st.columns([3, 2])
        with left:
            st.markdown("<div class='section-title'>Patient Values</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                preg    = st.slider("Pregnancies",            0,  20,   2,        key='d_preg')
                glucose = st.slider("Glucose (mg/dL)",       50, 250, 120,        key='d_gluc')
                bp      = st.slider("Blood Pressure (mmHg)", 30, 130,  72,        key='d_bp')
                skin    = st.slider("Skin Thickness (mm)",    0, 100,  20,        key='d_skin')
            with c2:
                insulin = st.slider("Insulin (µU/mL)",        0, 900,  80,        key='d_ins')
                bmi     = st.slider("BMI",                  10.0,70.0,28.0,  0.1, key='d_bmi')
                dpf     = st.slider("Diabetes Pedigree Fn", 0.0, 3.0,  0.5, 0.01, key='d_dpf')
                age     = st.slider("Age",                  10,  100,  35,        key='d_age')
            btn1 = st.button("Run Prediction", type="primary", key="btn_d")
        with right:
            st.markdown("<div class='section-title'>Result</div>", unsafe_allow_html=True)
            if btn1:
                raw = [preg,
                       glucose if glucose>0 else np.nan,
                       bp      if bp>0      else np.nan,
                       skin    if skin>0    else np.nan,
                       insulin if insulin>0 else np.nan,
                       bmi     if bmi>0     else np.nan,
                       dpf, age]
                pred, prob = run_predict('diabetes', raw)
                show_result(pred, prob,
                            "estimated probability of diabetes",
                            "estimated probability of no diabetes")
            else:
                st.markdown("<div style='height:160px;border:1px dashed #1a2d55;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#2a4060;font-size:0.85rem;'>Awaiting input →</div>", unsafe_allow_html=True)

    with tab2:
        left, right = st.columns([3, 2])
        with left:
            st.markdown("<div class='section-title'>Patient Values</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                h_age  = st.slider("Age",                       20, 90, 52,        key='h_age')
                h_sex  = st.selectbox("Sex",                    ['Male','Female'],  key='h_sex')
                h_cp   = st.selectbox("Chest Pain Type",
                                      ['typical angina','atypical angina',
                                       'non-anginal','asymptomatic'],              key='h_cp')
                h_rbp  = st.slider("Resting Blood Pressure",   80, 220, 130,       key='h_rbp')
                h_chol = st.slider("Cholesterol (mg/dL)",     100, 600, 240,       key='h_chol')
                h_fbs  = st.selectbox("Fasting Blood Sugar >120", ['No','Yes'],    key='h_fbs')
                h_ecg  = st.selectbox("Resting ECG",
                                      ['normal','lv hypertrophy',
                                       'st-t abnormality'],                        key='h_ecg')
            with c2:
                h_thalch = st.slider("Max Heart Rate",         60, 220, 150,       key='h_thalch')
                h_exang  = st.selectbox("Exercise Angina",     ['No','Yes'],        key='h_exang')
                h_op     = st.slider("ST Depression",          0.0, 7.0, 1.0, 0.1, key='h_op')
                h_slope  = st.selectbox("ST Slope",
                                        ['upsloping','flat','downsloping'],         key='h_slope')
                h_ca     = st.slider("Major Vessels (0-3)",    0, 3, 0,            key='h_ca')
                h_thal   = st.selectbox("Thal",
                                        ['normal','fixed defect',
                                         'reversable defect'],                     key='h_thal')
            btn2 = st.button("Run Prediction", type="primary", key="btn_h")
        with right:
            st.markdown("<div class='section-title'>Result</div>", unsafe_allow_html=True)
            if btn2:
                raw = [h_age,
                       {'Male':1,'Female':0}[h_sex],
                       {'typical angina':3,'atypical angina':0,'non-anginal':2,'asymptomatic':1}[h_cp],
                       h_rbp, h_chol, 1 if h_fbs=='Yes' else 0,
                       {'normal':1,'lv hypertrophy':0,'st-t abnormality':2}[h_ecg],
                       h_thalch, 1 if h_exang=='Yes' else 0, h_op,
                       {'upsloping':2,'flat':1,'downsloping':0}[h_slope],
                       h_ca, {'normal':2,'fixed defect':0,'reversable defect':1}[h_thal]]
                pred, prob = run_predict('heart', raw)
                show_result(pred, prob,
                            "estimated probability of heart disease",
                            "estimated probability of no heart disease")
            else:
                st.markdown("<div style='height:160px;border:1px dashed #1a2d55;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#2a4060;font-size:0.85rem;'>Awaiting input →</div>", unsafe_allow_html=True)

    with tab3:
        left, right = st.columns([3, 2])
        with left:
            st.markdown("<div class='section-title'>Patient Values</div>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                k_age = st.slider("Age",                    1,100,50,          key='k_age')
                k_bp  = st.slider("Blood Pressure",        50,180,80,          key='k_bp')
                k_sg  = st.selectbox("Specific Gravity",   [1.005,1.010,1.015,1.020,1.025], key='k_sg')
                k_al  = st.slider("Albumin (0-5)",          0,  5, 1,          key='k_al')
                k_su  = st.slider("Sugar (0-5)",            0,  5, 0,          key='k_su')
                k_rbc = st.selectbox("Red Blood Cells",    ['Normal','Abnormal'],       key='k_rbc')
                k_pc  = st.selectbox("Pus Cell",           ['Normal','Abnormal'],       key='k_pc')
                k_pcc = st.selectbox("Pus Cell Clumps",   ['Not Present','Present'],   key='k_pcc')
            with c2:
                k_ba   = st.selectbox("Bacteria",          ['Not Present','Present'],   key='k_ba')
                k_bgr  = st.slider("Blood Glucose",       50,500,120,          key='k_bgr')
                k_bu   = st.slider("Blood Urea (mg/dL)",   1,300, 36,          key='k_bu')
                k_sc   = st.slider("Serum Creatinine",   0.1,20.0,1.2,0.1,    key='k_sc')
                k_sod  = st.slider("Sodium (mEq/L)",    100,165,140,           key='k_sod')
                k_pot  = st.slider("Potassium (mEq/L)", 2.0,10.0,4.5,0.1,     key='k_pot')
                k_hemo = st.slider("Haemoglobin (g/dL)",3.0,18.0,14.0,0.1,    key='k_hemo')
            with c3:
                k_pcv = st.slider("Packed Cell Volume",  10, 55,42,            key='k_pcv')
                k_wc  = st.slider("WBC Count",         2000,20000,8000,         key='k_wc')
                k_rc  = st.slider("RBC Count",          1.0, 8.0,5.0,0.1,     key='k_rc')
                k_htn = st.selectbox("Hypertension",   ['No','Yes'],            key='k_htn')
                k_dm  = st.selectbox("Diabetes",       ['No','Yes'],            key='k_dm')
                k_cad = st.selectbox("Coronary Artery Disease",['No','Yes'],    key='k_cad')
                k_app = st.selectbox("Appetite",       ['Good','Poor'],         key='k_app')
                k_pe  = st.selectbox("Pedal Edema",    ['No','Yes'],            key='k_pe')
                k_ane = st.selectbox("Anemia",         ['No','Yes'],            key='k_ane')
            btn3 = st.button("Run Prediction", type="primary", key="btn_k")
        with right:
            st.markdown("<div class='section-title'>Result</div>", unsafe_allow_html=True)
            if btn3:
                yn  = lambda v: 1 if v=='Yes'     else 0
                na  = lambda v: 1 if v=='Normal'  else 0
                pre = lambda v: 1 if v=='Present' else 0
                raw = [k_age,k_bp,k_sg,k_al,k_su,
                       na(k_rbc),na(k_pc),pre(k_pcc),pre(k_ba),
                       k_bgr,k_bu,k_sc,k_sod,k_pot,k_hemo,
                       k_pcv,k_wc,k_rc,
                       yn(k_htn),yn(k_dm),yn(k_cad),
                       1 if k_app=='Good' else 0,yn(k_pe),yn(k_ane)]
                pred, prob = run_predict('kidney', raw)
                show_result(pred, prob,
                            "estimated probability of kidney disease",
                            "estimated probability of no kidney disease")
            else:
                st.markdown("<div style='height:160px;border:1px dashed #1a2d55;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#2a4060;font-size:0.85rem;'>Awaiting input →</div>", unsafe_allow_html=True)

st.markdown("""
<div style='margin-top:3rem;padding-top:1rem;border-top:1px solid #1a2540;
            text-align:center;font-size:0.7rem;color:#1e3050;'>
   
</div>""", unsafe_allow_html=True)