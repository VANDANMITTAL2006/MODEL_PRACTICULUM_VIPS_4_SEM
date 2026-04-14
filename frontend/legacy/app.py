"""
Streamlit Frontend â€” AI Personalized Learning System
Pages: Dashboard | Predict | Recommendations | Update Quiz | Profile
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import random

# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AI Learning System",
    page_icon="ðŸŽ“",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â”€â”€ Custom CSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
}
[data-testid="stSidebar"] * {
    color: #e0e0ff !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 1.05rem;
    padding: 6px 0;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(100, 100, 255, 0.25);
    border-radius: 16px;
    padding: 22px 26px;
    margin: 6px 0;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(80,80,255,0.25);
}
.metric-card h3 { color: #a0a0d0; font-weight: 500; margin: 0 0 6px; font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; }
.metric-card .value { font-size: 2.2rem; font-weight: 700; color: #ffffff; }
.metric-card .sub { font-size: 0.8rem; color: #7070a0; margin-top: 4px; }

/* Badge */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.badge-success { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.badge-warning { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.badge-danger  { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid rgba(239,68,68,0.3);  }
.badge-info    { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }

/* Topic pill */
.topic-pill {
    display: inline-block;
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(168,85,247,0.2));
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 999px;
    padding: 6px 18px;
    margin: 5px 4px;
    color: #c4b5fd;
    font-size: 0.88rem;
    font-weight: 500;
}

/* Section header */
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e0e0ff;
    margin: 20px 0 12px;
    border-left: 4px solid #6366f1;
    padding-left: 12px;
}

/* Score colors */
.score-fast    { color: #34d399; }
.score-low     { color: #fbbf24; }
.score-struggle{ color: #ef4444; }

/* Streak */
.streak-box {
    background: linear-gradient(135deg, #f59e0b22, #ef444422);
    border: 1px solid #f59e0b44;
    border-radius: 12px;
    padding: 14px 20px;
    text-align: center;
    color: #fcd34d;
    font-weight: 700;
    font-size: 1.1rem;
}

/* Title */
h1.page-title {
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0;
}

/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   DARK THEME â€” Main content area
   Matches sidebar: #0f0c29 â†’ #302b63 â†’ #24243e
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

/* Main app background */
.stApp {
    background: linear-gradient(160deg, #0f0c29 0%, #1a1640 40%, #24243e 100%) !important;
}

/* Main block container */
[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
}
[data-testid="block-container"] {
    background: transparent !important;
}

/* All text */
.stApp, .stApp p, .stApp span, .stApp label,
.stApp li, .stApp div, .stApp h1, .stApp h2,
.stApp h3, .stApp h4, .stApp h5, .stApp h6 {
    color: #d0d0f0;
}
.stApp .stCaption, .stApp small { color: #8080b0 !important; }

/* Markdown text */
.stMarkdown { color: #c8c8ee; }

/* st.metric boxes */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1a2e, #16213e) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"]  { color: #a0a0d0 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"]  { color: #ffffff !important; }
[data-testid="stMetricDelta"]  { color: #818cf8 !important; }

/* Forms / containers */
[data-testid="stForm"] {
    background: rgba(15, 12, 41, 0.6) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

/* Inputs & text areas */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
textarea {
    background: rgba(26, 26, 62, 0.9) !important;
    color: #e0e0ff !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    border-radius: 8px !important;
}
.stTextInput > div > div > input:focus,
textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.25) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: rgba(26, 26, 62, 0.9) !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    border-radius: 8px !important;
    color: #e0e0ff !important;
}
.stSelectbox svg { fill: #818cf8 !important; }

/* Dropdown options */
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] li {
    background: #1e1e3a !important;
    color: #e0e0ff !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"] li:hover {
    background: rgba(99,102,241,0.25) !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div > div {
    background: #4338ca !important;
}
[data-testid="stSlider"] > div > div {
    color: #c4b5fd !important;
}

/* Radio buttons */
.stRadio > div { gap: 8px; }
.stRadio label { color: #c0c0e0 !important; }

/* Buttons */
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(79,70,229,0.4) !important;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(79,70,229,0.55) !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: rgba(20, 18, 50, 0.7) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #c4b5fd !important;
}

/* Dataframe / table */
[data-testid="stDataFrame"] {
    background: rgba(15,12,41,0.8) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
}

/* Info / success / error / warning boxes */
[data-testid="stAlert"] {
    border-radius: 10px !important;
}
.stAlert[data-baseweb="notification"] { background: rgba(20,18,50,0.8) !important; }

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    border-radius: 999px !important;
}
[data-testid="stProgress"] > div {
    background: rgba(99,102,241,0.15) !important;
    border-radius: 999px !important;
}

/* Horizontal rule */
hr {
    border-color: rgba(100,100,255,0.18) !important;
}

/* Top bar / header */
header[data-testid="stHeader"] {
    background: rgba(15,12,41,0.95) !important;
    border-bottom: 1px solid rgba(99,102,241,0.2) !important;
}

/* Deploy button area */
[data-testid="stToolbar"] {
    background: rgba(15,12,41,0.95) !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #818cf8 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0f0c29; }
::-webkit-scrollbar-thumb { background: #4338ca; border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }

</style>
""", unsafe_allow_html=True)

# â”€â”€ Session state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if "student_id" not in st.session_state:
    st.session_state.student_id = "S0001"
if "streak" not in st.session_state:
    st.session_state.streak = 1
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "last_recommend" not in st.session_state:
    st.session_state.last_recommend = None
if "quiz_history" not in st.session_state:
    st.session_state.quiz_history = []
if "difficulty_level" not in st.session_state:
    st.session_state.difficulty_level = "Medium"

SUBJECTS = [
    "Algebra", "Geometry", "Calculus", "Statistics", "Physics",
    "Chemistry", "Biology", "History", "Literature", "Computer Science"
]

LEARNING_STYLES = ["Visual", "Auditory", "Kinesthetic", "Reading/Writing"]
SUPPORT_LEVELS = ["Low", "Medium", "High"]
STRESS_LEVELS  = ["Low", "Medium", "High"]

# â”€â”€ API calls â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=4)
        return r.json()
    except Exception:
        return None

def api_predict(payload):
    try:
        r = requests.post(f"{API_BASE}/predict-performance", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "âš ï¸ Cannot connect to backend. Is it running? (`uvicorn api.main:app --reload`)"
    except Exception as e:
        return None, str(e)

def api_recommend(payload):
    try:
        r = requests.post(f"{API_BASE}/recommend-content", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "âš ï¸ Cannot connect to backend."
    except Exception as e:
        return None, str(e)

def api_profile(student_id):
    try:
        r = requests.get(f"{API_BASE}/student-profile/{student_id}", timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "âš ï¸ Cannot connect to backend."
    except Exception as e:
        return None, str(e)

def api_update(payload):
    try:
        r = requests.post(f"{API_BASE}/update-after-quiz", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "âš ï¸ Cannot connect to backend."
    except Exception as e:
        return None, str(e)

# â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def badge_for_type(learning_type):
    if learning_type == "Fast Learner":
        return '<span class="badge badge-success">âš¡ Fast Learner</span>'
    elif learning_type == "Struggling Learner":
        return '<span class="badge badge-danger">âš ï¸ Struggling Learner</span>'
    else:
        return '<span class="badge badge-warning">ðŸ”„ Low Engagement</span>'

def score_color(score):
    if score >= 75:  return "#34d399"
    elif score >= 55: return "#fbbf24"
    else:             return "#ef4444"

def gauge_chart(value, title="Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={"reference": 65},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#666"},
            "bar": {"color": score_color(value)},
            "bgcolor": "#1e1e3a",
            "steps": [
                {"range": [0, 55],   "color": "rgba(239,68,68,0.12)"},
                {"range": [55, 75],  "color": "rgba(251,191,36,0.12)"},
                {"range": [75, 100], "color": "rgba(52,211,153,0.12)"},
            ],
            "threshold": {
                "line": {"color": "#818cf8", "width": 3},
                "thickness": 0.85,
                "value": 75,
            },
        },
        title={"text": title, "font": {"color": "#a0a0d0", "size": 15}},
        number={"font": {"color": "#ffffff", "size": 38}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=240,
        margin=dict(t=40, b=10, l=20, r=20),
    )
    return fig

def line_chart(history, title="Score History"):
    df = pd.DataFrame({"Quiz": [f"Q{i+1}" for i in range(len(history))], "Score": history})
    fig = px.line(df, x="Quiz", y="Score", title=title, markers=True)
    fig.update_traces(line_color="#818cf8", marker_color="#c084fc", marker_size=9)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,12,41,0.7)",
        font_color="#b0b0d0",
        height=260,
        margin=dict(t=40, b=20, l=10, r=10),
        xaxis=dict(gridcolor="rgba(100,100,255,0.1)"),
        yaxis=dict(gridcolor="rgba(100,100,255,0.1)", range=[0, 105]),
        title_font_color="#c4b5fd",
    )
    return fig

def bar_chart(labels, values, title=""):
    colors = [score_color(v) for v in values]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=[f"{v:.1f}" for v in values], textposition="outside"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,12,41,0.7)",
        font_color="#b0b0d0",
        height=240,
        margin=dict(t=30, b=20, l=10, r=10),
        xaxis=dict(gridcolor="rgba(100,100,255,0.1)"),
        yaxis=dict(gridcolor="rgba(100,100,255,0.1)", range=[0, 110]),
        title=title, title_font_color="#c4b5fd",
    )
    return fig

# â”€â”€ Sidebar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.markdown("## ðŸŽ“ AI Learning System")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["ðŸ  Dashboard", "ðŸ”® Predict", "ðŸ“š Recommendations", "âœï¸ Update Quiz", "ðŸ‘¤ Profile"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Student ID**")
    student_id_input = st.text_input("", value=st.session_state.student_id, label_visibility="collapsed")
    if student_id_input != st.session_state.student_id:
        st.session_state.student_id = student_id_input
        st.session_state.last_prediction = None
        st.session_state.last_recommend = None

    st.markdown("---")
    st.markdown(f"""<div class="streak-box">ðŸ”¥ Learning Streak<br>{st.session_state.streak} day(s)</div>""", unsafe_allow_html=True)
    st.markdown("---")

    health = api_health()
    if health and health.get("status") == "healthy":
        st.markdown('<span class="badge badge-success">â— Backend Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-danger">â— Backend Offline</span>', unsafe_allow_html=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  PAGE: DASHBOARD
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
if page == "ðŸ  Dashboard":
    st.markdown('<h1 class="page-title">Dashboard Overview</h1>', unsafe_allow_html=True)
    st.caption(f"Student ID: **{st.session_state.student_id}** Â· Adaptive Learning Engine v1.0")
    st.markdown("---")

    pred = st.session_state.last_prediction

    if pred:
        score  = pred["predicted_score"]
        ltype  = pred["learning_type"]
        wareas = pred["weak_areas"]
        eng    = pred.get("engagement_score", 0)
        cons   = pred.get("consistency_score", 0)

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.plotly_chart(gauge_chart(score, "Predicted Score"), use_container_width=True)
        with col2:
            st.markdown('<div class="metric-card"><h3>Learning Type</h3>', unsafe_allow_html=True)
            st.markdown(badge_for_type(ltype), unsafe_allow_html=True)
            st.markdown(f'<div class="sub">{pred.get("explanation", "")}</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="metric-card"><h3>Engagement Score</h3>', unsafe_allow_html=True)
            st.progress(int(eng))
            st.markdown(f'<div class="value">{eng:.1f}<span style="font-size:1rem;color:#7070a0">/100</span></div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h3>Consistency Score</h3>', unsafe_allow_html=True)
            st.progress(int(min(cons, 100)))
            st.markdown(f'<div class="value">{cons:.1f}<span style="font-size:1rem;color:#7070a0">/100</span></div></div>', unsafe_allow_html=True)

            st.markdown('<div class="metric-card"><h3>Weak Areas</h3>', unsafe_allow_html=True)
            for w in wareas:
                st.markdown(f'<span class="badge badge-danger">âš ï¸ {w}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            hist = st.session_state.quiz_history
            if hist:
                scores = [h["score"] for h in hist]
                st.plotly_chart(line_chart(scores, "Quiz Score History"), use_container_width=True)
            else:
                st.info("ðŸ“ˆ No quiz history yet. Go to **Update Quiz** to add entries.")

        with col_b:
            labels = ["Predicted", "Engagement", "Consistency"]
            values = [score, eng, cons]
            st.plotly_chart(bar_chart(labels, values, "Key Metrics"), use_container_width=True)

        if pred.get("recommended_topics"):
            st.markdown('<div class="section-header">ðŸ“š Quick Topic Recommendations</div>', unsafe_allow_html=True)
            pills = " ".join([f'<span class="topic-pill">ðŸ“Œ {t}</span>' for t in pred["recommended_topics"]])
            st.markdown(pills, unsafe_allow_html=True)
    else:
        st.info("ðŸ‘ˆ  Go to **Predict** to analyse a student and populate the dashboard.")
        # Show a teaser
        with st.expander("ðŸŽ¯ What this system does", expanded=True):
            st.markdown("""
| Feature | Description |
|---------|-------------|
| ðŸ”® **Performance Prediction** | XGBoost model predicts final score |
| ðŸ§  **Student Segmentation** | KMeans clusters learners into 3 types |
| ðŸ“š **Hybrid Recommendations** | Content-based + Collaborative filtering |
| ðŸ”„ **Real-time Adaptation** | Updates after every quiz attempt |
| ðŸ“Š **Rich Analytics** | Gauge charts, line graphs, metric cards |
""")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  PAGE: PREDICT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "ðŸ”® Predict":
    st.markdown('<h1 class="page-title">Predict Performance</h1>', unsafe_allow_html=True)
    st.caption("Enter student details to get an AI-powered performance prediction.")
    st.markdown("---")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("##### ðŸ“‹ Basic Info")
            age           = st.slider("Age", 14, 25, 18)
            gender        = st.selectbox("Gender", ["Male", "Female", "Other"])
            learning_style= st.selectbox("Learning Style", LEARNING_STYLES)
            parental_supp = st.selectbox("Parental Support", SUPPORT_LEVELS, index=1)
            stress_level  = st.selectbox("Stress Level", STRESS_LEVELS, index=1)
        with col2:
            st.markdown("##### ðŸ“Š Academic Stats")
            attendance       = st.slider("Attendance (%)", 20, 100, 75)
            assignment_score = st.slider("Assignment Score", 10, 100, 70)
            quiz_score       = st.slider("Quiz Score", 5, 100, 65)
            previous_score   = st.slider("Previous Score", 10, 100, 62)
        with col3:
            st.markdown("##### ðŸŽ¯ Engagement Info")
            time_spent   = st.slider("Study Time (hrs/day)", 0.5, 15.0, 5.0, step=0.5)
            attempts     = st.slider("Attempts per Topic", 1, 15, 3)
            subject_weak = st.selectbox("Weak Subject", SUBJECTS)
            internet     = st.radio("Internet Access", [1, 0], format_func=lambda x: "Yes" if x else "No")
            extra        = st.radio("Extracurricular", [1, 0], format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("ðŸš€ Predict Performance", use_container_width=True)

    if submitted:
        payload = {
            "student_id": st.session_state.student_id,
            "age": age, "gender": gender, "learning_style": learning_style,
            "attendance": attendance, "assignment_score": assignment_score,
            "quiz_score": quiz_score, "time_spent_hours": time_spent,
            "attempts": attempts, "previous_score": previous_score,
            "internet_access": internet, "parental_support": parental_supp,
            "extracurricular": extra, "stress_level": stress_level,
            "subject_weakness": subject_weak,
        }
        with st.spinner("ðŸ¤– Running AI prediction â€¦"):
            result, err = api_predict(payload)

        if err:
            st.error(err)
        elif result:
            st.session_state.last_prediction = result
            st.success("âœ… Prediction complete!")
            st.markdown("---")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(gauge_chart(result["predicted_score"], "Predicted Score"), use_container_width=True)
            with c2:
                st.markdown(f"""
<div class="metric-card">
  <h3>Learning Type</h3>
  {badge_for_type(result['learning_type'])}
  <div class="sub" style="margin-top:10px">{result['explanation']}</div>
</div>
""", unsafe_allow_html=True)
                st.markdown(f"""
<div class="metric-card">
  <h3>Engagement / Consistency</h3>
  <div style="color:#818cf8;font-size:1.1rem">âš¡ {result['engagement_score']:.1f} &nbsp;|&nbsp; ðŸ“ {result['consistency_score']:.1f}</div>
</div>
""", unsafe_allow_html=True)
            with c3:
                st.markdown('<div class="metric-card"><h3>Weak Areas</h3>', unsafe_allow_html=True)
                for w in result["weak_areas"]:
                    st.markdown(f'<span class="badge badge-danger">âš ï¸ {w}</span><br>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-header">ðŸ“š Recommended Topics</div>', unsafe_allow_html=True)
            pills = " ".join([f'<span class="topic-pill">ðŸ“Œ {t}</span>' for t in result["recommended_topics"]])
            st.markdown(pills, unsafe_allow_html=True)

        # JSON view
        if result:
            with st.expander("ðŸ” Full API Response"):
                st.json(result)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  PAGE: RECOMMENDATIONS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "ðŸ“š Recommendations":
    st.markdown('<h1 class="page-title">Content Recommendations</h1>', unsafe_allow_html=True)
    st.caption("Hybrid content-based + collaborative filtering.")
    st.markdown("---")

    with st.form("recommend_form"):
        c1, c2 = st.columns(2)
        with c1:
            r_quiz    = st.slider("Quiz Score", 5, 100, 55)
            r_subject = st.selectbox("Weak Subject", SUBJECTS)
            r_topics  = st.slider("Topics to recommend", 3, 10, 5)
        with c2:
            r_engagement = st.slider("Engagement Score", 0.0, 100.0, 45.0)
            r_consistency= st.slider("Consistency Score", 0.0, 100.0, 55.0)
            r_attempts   = st.slider("Attempts", 1, 15, 4)
        go_rec = st.form_submit_button("ðŸŽ¯ Get Recommendations", use_container_width=True)

    if go_rec:
        payload = {
            "student_id": st.session_state.student_id,
            "quiz_score": r_quiz,
            "subject_weakness": r_subject,
            "engagement_score": r_engagement,
            "consistency_score": r_consistency,
            "attempts": r_attempts,
            "num_topics": r_topics,
        }
        with st.spinner("ðŸ” Fetching personalized recommendations â€¦"):
            result, err = api_recommend(payload)

        if err:
            st.error(err)
        elif result:
            st.session_state.last_recommend = result
            st.success("âœ… Recommendations ready!")
            st.markdown("---")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="section-header">ðŸ“˜ Content-Based Topics</div>', unsafe_allow_html=True)
                for i, t in enumerate(result.get("content_based_topics", []), 1):
                    st.markdown(f"**{i}.** {t}")
            with c2:
                st.markdown('<div class="section-header">ðŸ‘¥ Collaborative Topics</div>', unsafe_allow_html=True)
                for i, t in enumerate(result.get("collaborative_topics", []), 1):
                    st.markdown(f"**{i}.** {t}")

            st.markdown("---")
            st.markdown('<div class="section-header">â­ Top Hybrid Picks</div>', unsafe_allow_html=True)
            pills = " ".join([f'<span class="topic-pill">ðŸ† {t}</span>' for t in result.get("recommended_topics", [])])
            st.markdown(pills, unsafe_allow_html=True)

            with st.expander("ðŸ” Full API Response"):
                st.json(result)

    # Show last result if exists
    elif st.session_state.last_recommend:
        st.info("Showing cached recommendations. Fill the form and click **Get Recommendations** to refresh.")
        pills = " ".join([f'<span class="topic-pill">ðŸ“Œ {t}</span>' for t in st.session_state.last_recommend.get("recommended_topics", [])])
        st.markdown(pills, unsafe_allow_html=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  PAGE: UPDATE QUIZ
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "âœï¸ Update Quiz":
    st.markdown('<h1 class="page-title">Update After Quiz</h1>', unsafe_allow_html=True)
    st.caption("Record a new quiz result â€” the AI will recalibrate your profile in real time.")
    st.markdown("---")

    with st.form("quiz_update_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            u_subject    = st.selectbox("Subject", SUBJECTS)
            u_quiz_score = st.slider("New Quiz Score", 0, 100, 60)
        with c2:
            u_time    = st.slider("Time Spent (hrs)", 0.5, 10.0, 2.0, step=0.5)
            u_attempts= st.slider("Attempts", 1, 15, 2)
        with c3:
            st.markdown("##### â„¹ï¸ Difficulty Adjustment")
            st.session_state.difficulty_level = st.selectbox(
                "Difficulty Level", ["Easy", "Medium", "Hard"],
                index=["Easy","Medium","Hard"].index(st.session_state.difficulty_level)
            )
            st.caption("Difficulty modifies how the system weighs your score.")

        update_btn = st.form_submit_button("ðŸ”„ Update Profile", use_container_width=True)

    if update_btn:
        # Adjust for difficulty
        diff_map = {"Easy": 0.85, "Medium": 1.0, "Hard": 1.15}
        adj_score = min(u_quiz_score * diff_map[st.session_state.difficulty_level], 100)

        payload = {
            "student_id": st.session_state.student_id,
            "subject": u_subject,
            "new_quiz_score": round(adj_score, 1),
            "time_spent_hours": u_time,
            "attempts": u_attempts,
        }

        with st.spinner("âš¡ Adapting learning profile â€¦"):
            result, err = api_update(payload)
            time.sleep(0.5)

        if err:
            st.error(err)
        elif result:
            # Update streak
            st.session_state.streak += 1
            # Log history
            st.session_state.quiz_history.append({
                "subject": u_subject, "score": result["new_quiz_score"],
                "predicted": result["new_predicted_score"]
            })
            st.success(result["message"])
            st.markdown("---")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(gauge_chart(result["new_predicted_score"], "Updated Score"), use_container_width=True)
            with c2:
                st.markdown(f"""
<div class="metric-card">
  <h3>Updated Learning Type</h3>
  {badge_for_type(result["learning_type"])}      
</div>
<div class="metric-card">
  <h3>Difficulty Applied</h3>
  <div class="value" style="font-size:1.4rem">{st.session_state.difficulty_level}</div>
  <div class="sub">Adjusted Score: {result["new_quiz_score"]}</div>
</div>
""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
<div class="metric-card">
  <h3>Streak ðŸ”¥</h3>
  <div class="value">{st.session_state.streak} day(s)</div>
</div>
<div class="metric-card">
  <h3>Engagement / Consistency</h3>
  <div style="color:#818cf8">âš¡ {result['engagement_score']:.1f} &nbsp;|&nbsp; ðŸ“ {result['consistency_score']:.1f}</div>
</div>
""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown('<div class="section-header">ðŸ“š Updated Recommendations</div>', unsafe_allow_html=True)
            pills = " ".join([f'<span class="topic-pill">ðŸ“Œ {t}</span>' for t in result.get("updated_recommendations", [])])
            st.markdown(pills, unsafe_allow_html=True)

            with st.expander("ðŸ” Full API Response"):
                st.json(result)

    # Quiz history table
    if st.session_state.quiz_history:
        st.markdown("---")
        st.markdown('<div class="section-header">ðŸ“œ Quiz History</div>', unsafe_allow_html=True)
        hist_df = pd.DataFrame(st.session_state.quiz_history)
        hist_df.index = hist_df.index + 1
        st.dataframe(hist_df.style.background_gradient(subset=["score","predicted"], cmap="RdYlGn"), use_container_width=True)

        if len(hist_df) > 1:
            st.plotly_chart(line_chart(hist_df["predicted"].tolist(), "Predicted Score Trend"), use_container_width=True)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
#  PAGE: PROFILE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
elif page == "ðŸ‘¤ Profile":
    st.markdown('<h1 class="page-title">Student Profile</h1>', unsafe_allow_html=True)
    st.caption("Full academic analytics for the selected student.")
    st.markdown("---")

    c1, c2 = st.columns([3, 1])
    with c1:
        pid = st.text_input("Enter Student ID", value=st.session_state.student_id)
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_btn = st.button("ðŸ“¥ Load Profile", use_container_width=True)

    if fetch_btn or pid:
        with st.spinner(f"Loading profile for **{pid}** â€¦"):
            profile, err = api_profile(pid)

        if err:
            st.error(err)
        elif profile:
            st.success(f"âœ… Loaded profile: **{profile['student_id']}**")
            st.markdown("---")

            # Header row
            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.metric("Predicted Score", f"{profile.get('predicted_score', 0):.1f}")
            hc2.metric("Quiz Score",       f"{profile.get('quiz_score', 0):.1f}")
            hc3.metric("Attendance",        f"{profile.get('attendance', 0):.1f}%")
            hc4.metric("Assignment",        f"{profile.get('assignment_score', 0):.1f}")

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
<div class="metric-card">
  <h3>Student Details</h3>
  <table style="color:#c0c0e0;width:100%;font-size:0.9rem">
    <tr><td>ðŸ†” ID</td><td><b>{profile.get('student_id')}</b></td></tr>
    <tr><td>ðŸ“… Age</td><td>{profile.get('age')}</td></tr>
    <tr><td>âš§ Gender</td><td>{profile.get('gender')}</td></tr>
    <tr><td>ðŸ“– Learning Style</td><td>{profile.get('learning_style')}</td></tr>
    <tr><td>ðŸ’ª Strength</td><td>{profile.get('subject_strength')}</td></tr>
    <tr><td>âš ï¸ Weakness</td><td>{profile.get('subject_weakness')}</td></tr>
    <tr><td>ðŸŒ Internet</td><td>{"Yes" if profile.get('internet_access') else "No"}</td></tr>
    <tr><td>ðŸ‘¨â€ðŸ‘©â€ðŸ‘§ Parental Support</td><td>{profile.get('parental_support')}</td></tr>
    <tr><td>ðŸ˜“ Stress Level</td><td>{profile.get('stress_level')}</td></tr>
    <tr><td>ðŸ”¥ Streak</td><td>{profile.get('streak', 1)} day(s)</td></tr>
  </table>
</div>
""", unsafe_allow_html=True)

            with c2:
                st.plotly_chart(gauge_chart(profile.get("predicted_score", 0), "Predicted Score"), use_container_width=True)
                st.markdown(f"""
<div class="metric-card">
  <h3>Cluster / Segment</h3>
  {badge_for_type(profile.get('cluster_label', 'Unknown'))}
</div>
""", unsafe_allow_html=True)

            st.markdown("---")
            c3, c4 = st.columns(2)
            with c3:
                history = profile.get("score_history", [])
                if history:
                    st.plotly_chart(line_chart(history, "Score Progression"), use_container_width=True)

            with c4:
                eng  = profile.get('engagement_score', 0)
                cons = profile.get('consistency_score', 0)
                quiz = profile.get('quiz_score', 0)
                st.plotly_chart(bar_chart(["Engagement", "Consistency", "Quiz"], [eng, cons, quiz], "Score Breakdown"), use_container_width=True)

            if profile.get("weak_areas"):
                st.markdown('<div class="section-header">âš ï¸ Weak Areas</div>', unsafe_allow_html=True)
                for w in profile["weak_areas"]:
                    st.markdown(f'<span class="badge badge-danger">âš ï¸ {w}</span>', unsafe_allow_html=True)

            with st.expander("ðŸ” Full Profile Data"):
                st.json(profile)

