import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Logistic Regression · Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* Dark background */
  .stApp {
    background: #0d0f14;
    color: #e8eaf0;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #13151c !important;
    border-right: 1px solid #1e2230;
  }

  /* Headers */
  h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.5px;
  }

  /* Main title */
  .main-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    line-height: 1.2;
    margin-bottom: 4px;
  }
  .main-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #5c6278;
    margin-bottom: 2rem;
  }

  /* Accent line */
  .accent-bar {
    width: 48px;
    height: 3px;
    background: linear-gradient(90deg, #4f8ef7, #a78bfa);
    border-radius: 2px;
    margin-bottom: 1.2rem;
  }

  /* Metric cards */
  .metric-card {
    background: #181b24;
    border: 1px solid #1e2230;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
  }
  .metric-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #5c6278;
    margin-bottom: 6px;
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.7rem;
    font-weight: 700;
    color: #e8eaf0;
  }

  /* Result card */
  .result-admitted {
    background: linear-gradient(135deg, #0f2d1a, #132b1a);
    border: 1px solid #1a5c30;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
  }
  .result-rejected {
    background: linear-gradient(135deg, #2d0f0f, #2b1313);
    border: 1px solid #5c1a1a;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
  }
  .result-label {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 8px;
  }
  .result-admitted .result-label { color: #4ade80; }
  .result-rejected .result-label { color: #f87171; }
  .result-prob {
    font-size: 0.85rem;
    color: #8890a8;
    margin-top: 6px;
  }

  /* Probability bar */
  .prob-bar-container {
    background: #1e2230;
    border-radius: 8px;
    height: 8px;
    overflow: hidden;
    margin: 10px 0;
  }

  /* Input sliders custom */
  .stSlider label {
    font-size: 0.82rem !important;
    color: #8890a8 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }
  .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #4f8ef7 !important;
  }

  /* Section separators */
  .section-head {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #4f8ef7;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
  }

  /* Divider */
  hr { border-color: #1e2230 !important; }

  /* Number input */
  input[type="number"] {
    background: #181b24 !important;
    border: 1px solid #1e2230 !important;
    color: #e8eaf0 !important;
    border-radius: 8px !important;
  }

  /* Hide Streamlit branding */
  #MainMenu, footer { visibility: hidden; }

  /* Button */
  .stButton > button {
    background: linear-gradient(135deg, #4f8ef7, #7c6af5);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    padding: 0.6rem 1.4rem;
    width: 100%;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ─── Helper Functions ────────────────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)

def map_feature(X1, X2, degree=6):
    """Polynomial feature mapping for dataset 2."""
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    out = []
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))
    return np.stack(out, axis=1)

def load_model():
    path = "model.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-head">⚙ Configuration</div>', unsafe_allow_html=True)

    dataset_choice = st.selectbox(
        "Jeu de données",
        ["Dataset 1 — Admissions universitaires", "Dataset 2 — Contrôle qualité microchips"],
        help="Choisissez le modèle correspondant à votre jeu de données"
    )

    st.markdown("---")
    st.markdown('<div class="section-head">📂 Modèle</div>', unsafe_allow_html=True)

    model_file = st.file_uploader(
        "Charger model.pkl (optionnel)",
        type=["pkl"],
        help="Chargez votre fichier model.pkl sauvegardé depuis le notebook"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#3a3f52; line-height:1.7;'>
    <b style='color:#5c6278;'>Régression Logistique</b><br>
    Modèle entraîné sur données d'apprentissage automatique.<br><br>
    <span style='color:#4f8ef7;'>Dataset 1</span> — 2 features linéaires<br>
    <span style='color:#a78bfa;'>Dataset 2</span> — 27 features polynomiales
    </div>
    """, unsafe_allow_html=True)

# ─── Load or Build Model ─────────────────────────────────────────────────────
model = None
if model_file is not None:
    model = joblib.load(model_file)

is_dataset1 = "Dataset 1" in dataset_choice

# Default fallback weights (approximations)
if model is None:
    if is_dataset1:
        w_default = np.array([0.18, 0.17])
        b_default = -25.16
    else:
        w_default = np.array([
            0.34, 0.17, 0.36, -0.36, 0.30, -0.23,
            -0.55, -0.29, -0.17, -0.51, -0.14, -0.07,
            0.27, 0.25, 0.05, -0.46, -0.09, -0.07,
            -0.42, -0.24, -0.01, -0.30, 0.01, 0.05,
            -0.27, -0.05, -0.25
        ])
        b_default = 1.27
    w = w_default
    b = b_default
else:
    w = model["w"].flatten()
    b = float(model["b"])

# ─── Main Layout ────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 Classifier</div>', unsafe_allow_html=True)
st.markdown(f'<div class="main-subtitle">{dataset_choice}</div>', unsafe_allow_html=True)
st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

col_inputs, col_result = st.columns([1, 1], gap="large")

# ─── Input Panel ─────────────────────────────────────────────────────────────
with col_inputs:
    st.markdown('<div class="section-head">📥 Entrées</div>', unsafe_allow_html=True)

    if is_dataset1:
        score1 = st.slider("Score Examen 1", 0.0, 100.0, 65.0, 0.5,
                           help="Note obtenue à l'examen 1")
        score2 = st.slider("Score Examen 2", 0.0, 100.0, 65.0, 0.5,
                           help="Note obtenue à l'examen 2")

        X_input = np.array([score1, score2])
        prob = predict_proba(X_input, w, b)

    else:
        test1 = st.slider("Test Microchip 1", -1.5, 1.5, 0.05, 0.01,
                          help="Résultat du test microchip 1")
        test2 = st.slider("Test Microchip 2", -1.5, 1.5, 0.70, 0.01,
                          help="Résultat du test microchip 2")

        X_mapped = map_feature(np.array([test1]), np.array([test2]))
        X_input_raw = np.array([test1, test2])
        prob = predict_proba(X_mapped[0], w, b)

    prediction = 1 if prob >= 0.5 else 0

    st.markdown("---")
    st.markdown('<div class="section-head">📊 Paramètres du modèle</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Dimension w</div>
          <div class="metric-value">{len(w)}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Biais b</div>
          <div class="metric-value">{b:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        src = "Fichier" if model_file else "Défaut"
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Source</div>
          <div class="metric-value" style="font-size:1rem;">{src}</div>
        </div>""", unsafe_allow_html=True)

# ─── Result Panel ────────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-head">🎯 Résultat</div>', unsafe_allow_html=True)

    if is_dataset1:
        admitted_label = "✅ Admis"
        rejected_label = "❌ Refusé"
    else:
        admitted_label = "✅ Accepté"
        rejected_label = "❌ Rejeté"

    if prediction == 1:
        st.markdown(f"""
        <div class="result-admitted">
          <div style="font-size:2.5rem;">✅</div>
          <div class="result-label">{admitted_label}</div>
          <div class="result-prob">Probabilité : {prob:.1%}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-rejected">
          <div style="font-size:2.5rem;">❌</div>
          <div class="result-label">{rejected_label}</div>
          <div class="result-prob">Probabilité : {prob:.1%}</div>
        </div>""", unsafe_allow_html=True)

    # Probability gauge
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-head">📈 Probabilité</div>', unsafe_allow_html=True)

    bar_color = "#4ade80" if prediction == 1 else "#f87171"
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; font-size:0.75rem; color:#5c6278; margin-bottom:4px;">
      <span>0%</span><span>50%</span><span>100%</span>
    </div>
    <div class="prob-bar-container">
      <div style="width:{prob*100:.1f}%; height:100%; background:linear-gradient(90deg, {bar_color}99, {bar_color}); border-radius:8px; transition:width 0.4s ease;"></div>
    </div>
    <div style="text-align:center; font-family:'Space Mono',monospace; font-size:1.5rem; font-weight:700; color:{bar_color}; margin-top:8px;">
      {prob:.1%}
    </div>
    """, unsafe_allow_html=True)

    # Sigmoid value display
    z_val = np.dot(w, X_input if is_dataset1 else X_mapped[0]) + b
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#181b24; border:1px solid #1e2230; border-radius:10px; padding:1rem; font-family:'Space Mono',monospace; font-size:0.8rem; color:#5c6278;">
      <span style="color:#4f8ef7;">z</span> = {z_val:.4f}<br>
      <span style="color:#a78bfa;">σ(z)</span> = {prob:.6f}
    </div>
    """, unsafe_allow_html=True)

# ─── Decision Boundary Plot ──────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-head">🗺 Frontière de Décision</div>', unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(8, 4.5))
fig.patch.set_facecolor('#0d0f14')
ax.set_facecolor('#13151c')

if is_dataset1:
    x_min, x_max = 30, 100
    y_min, y_max = 30, 100

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = sigmoid(np.dot(grid, w) + b).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1],
                colors=['#2d0f1a', '#0f2d1a'], alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0.5],
               colors=['#4f8ef7'], linewidths=2)

    # Mark current point
    color_pt = '#4ade80' if prediction == 1 else '#f87171'
    ax.scatter([score1], [score2], color=color_pt, s=180,
               zorder=5, edgecolors='white', linewidths=1.5, label="Votre point")

    ax.set_xlabel("Score Examen 1", color='#5c6278', fontsize=9)
    ax.set_ylabel("Score Examen 2", color='#5c6278', fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

else:
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_flat = np.c_[xx.ravel(), yy.ravel()]
    grid_mapped = map_feature(grid_flat[:, 0], grid_flat[:, 1])
    Z = sigmoid(np.dot(grid_mapped, w) + b).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1],
                colors=['#2d0f1a', '#0f2d1a'], alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0.5],
               colors=['#a78bfa'], linewidths=2)

    color_pt = '#4ade80' if prediction == 1 else '#f87171'
    ax.scatter([test1], [test2], color=color_pt, s=180,
               zorder=5, edgecolors='white', linewidths=1.5, label="Votre point")

    ax.set_xlabel("Test Microchip 1", color='#5c6278', fontsize=9)
    ax.set_ylabel("Test Microchip 2", color='#5c6278', fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

ax.tick_params(colors='#3a3f52', labelsize=8)
for spine in ax.spines.values():
    spine.set_edgecolor('#1e2230')

patch_yes = mpatches.Patch(color='#4ade80', label='Positif (1)', alpha=0.7)
patch_no  = mpatches.Patch(color='#f87171', label='Négatif (0)', alpha=0.7)
ax.legend(handles=[patch_yes, patch_no], facecolor='#181b24',
          edgecolor='#1e2230', labelcolor='#8890a8', fontsize=8)

ax.set_title("Frontière de décision (σ = 0.5)", color='#5c6278',
             fontsize=9, pad=10, fontfamily='monospace')

st.pyplot(fig)
plt.close()