from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

BASE = Path(__file__).resolve().parents[1]
DESKTOP = Path(r"C:\Users\SRISHTI MITTAL\OneDrive\Desktop")
OUTPUT = DESKTOP / "AI_Personalized_Learning_System_Complete_Guide.pdf"

MODEL_ROWS = [
    ["stacking", 4.696359433786332, 3.845160797949065, 0.7023669361617088, 5.806422153257836, 1.4223375693106313],
    ["extra_trees", 4.75448963762522, 3.9573799999999837, 0.6949533012976377, 6.029415502311759, 1.7130887301587834],
    ["hgb", 4.95266260106762, 4.0251298246711915, 0.6689938945375291, 6.03947415276482, 1.5506800917809662],
    ["rf", 4.985039427074941, 4.0747321495146105, 0.6646520050930649, 6.199363512667706, 1.8351765426006488],
    ["gbr", 5.291720255145205, 4.238086823922466, 0.6221214210138072, 6.4109237418190235, 1.6939648078063763],
    ["xgb", 5.313448508733565, 4.260848731994629, 0.619011846815025, 6.460933042256562, 1.5699449455954813],
]

LEGACY = {"mae": 3.6926900739683246, "rmse": 4.5400386617919715, "r2": 0.7218509241486418, "train_rows": 400, "test_rows": 100}
RECO_GLOBAL = {"hit": 1.0, "precision": 0.424, "recall": 0.7066666666666667, "ndcg": 0.6214187731580791, "map": 0.6264833333333334, "coverage": 1.0, "diversity": 1.0, "novelty": 0.8544, "cold_start_hit": 0.0, "struggling_hit": 1.0}
RECO_SEGMENTS = {
    "regular": {"hit": 1.0, "precision": 0.4398936170212766, "recall": 0.7331560283687943, "ndcg": 0.6378696494413455, "map": 0.625354609929078, "diversity": 1.0, "novelty": 0.8882978723404256},
    "struggling": {"hit": 1.0, "precision": 0.3758064516129032, "recall": 0.6263440860215053, "ndcg": 0.5715354708797881, "map": 0.6299059139784946, "diversity": 1.0, "novelty": 0.7516129032258064},
}
ONLINE = {"p50": 7.85, "p95": 69.29, "p99": 69.29, "cache": 0.0, "ctr": 0.06349206349206349, "completion": 0.0}

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=1, fontSize=24, leading=28, textColor=colors.HexColor("#111827")))
styles.add(ParagraphStyle(name="SubCenter", parent=styles["BodyText"], alignment=1, fontSize=11, leading=15, textColor=colors.HexColor("#374151")))
styles.add(ParagraphStyle(name="Section", parent=styles["Heading1"], fontSize=16, leading=19, spaceBefore=8, spaceAfter=6, textColor=colors.HexColor("#111827")))
styles.add(ParagraphStyle(name="SubSection", parent=styles["Heading2"], fontSize=12.5, leading=15, spaceBefore=6, spaceAfter=4, textColor=colors.HexColor("#1f2937")))
styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=9.5, leading=13.2, textColor=colors.HexColor("#111827")))
styles.add(ParagraphStyle(name="MonoSmall", parent=styles["Code"], fontName="Courier", fontSize=8.3, leading=10.5, textColor=colors.HexColor("#111827")))

story = []


def fmt(value):
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def bullets(items, font_size=9.5):
    text = "<br/>".join([f"• {item}" for item in items])
    story.append(Paragraph(text, ParagraphStyle("Bullets", parent=styles["BodySmall"], leftIndent=12, fontSize=font_size, leading=font_size + 3)))


def table(rows, widths=None):
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f9fafb"), colors.white]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)


def page_num(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def maybe_image(path: Path, width=17.0 * cm, height=10.0 * cm):
    if path.exists():
        story.append(Image(str(path), width=width, height=height))
        story.append(Spacer(1, 0.15 * cm))


doc = SimpleDocTemplate(
    str(OUTPUT),
    pagesize=A4,
    rightMargin=1.6 * cm,
    leftMargin=1.6 * cm,
    topMargin=1.5 * cm,
    bottomMargin=1.4 * cm,
    title="AI Personalized Learning System - Complete Guide",
)

story.extend([
    Spacer(1, 1.0 * cm),
    Paragraph("AI Personalized Learning System", styles["TitleCenter"]),
    Spacer(1, 0.2 * cm),
    Paragraph("Complete beginner-friendly project guide with architecture, API workflow, ML pipeline, metrics, frontend behavior, and run instructions.", styles["SubCenter"]),
    Spacer(1, 0.5 * cm),
])

story.append(Paragraph("What this project does", styles["Section"]))
story.append(Paragraph(
    "This system predicts student performance, classifies learning risk, recommends topics, records user feedback, and adapts future recommendations from interaction data. The backend is a FastAPI service, the machine learning layer trains and serves score prediction models plus a recommender, and the frontend is a React dashboard that guides a learner from onboarding to personalized study actions.",
    styles["BodySmall"],
))
bullets([
    "Primary goal: turn a student's profile into a predicted score plus actionable topic recommendations.",
    "Secondary goal: keep learning recommendations updated after clicks, feedback, and quiz completion.",
    "Operational goal: support offline training, online serving, and lightweight observability in one repository.",
])
story.append(Spacer(1, 0.25 * cm))
story.append(Paragraph("Repository shape", styles["Section"]))
story.append(Preformatted(
"""project/
├── api/          FastAPI backend, schemas, services, cache, observability
├── ml/           training, inference, monitoring, recommender, artifacts
├── data/         raw CSV data and event logs
├── frontend/     React app, dashboard, onboarding, learning flow
├── scripts/      reporting and evaluation helpers
├── config/       runtime configuration templates
└── tests/       API, ML, recommender tests""",
    styles["MonoSmall"],
))

story.append(PageBreak())
story.append(Paragraph("Technology Stack", styles["Section"]))
table([
    ["Layer", "Main tools", "Why it matters"],
    ["Backend API", "FastAPI, Uvicorn, Pydantic", "Validates requests and exposes prediction/recommendation endpoints."],
    ["ML / data", "pandas, numpy, scikit-learn, joblib", "Loads data, engineers features, trains models, persists artifacts."],
    ["Modeling", "RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, XGBoost, stacking", "Compares candidate regressors and chooses the best score predictor."],
    ["Recommendation", "Retrieval, hybrid reranking, adaptive ranking", "Finds, scores, and reranks learning topics."],
    ["Frontend", "React 18, Vite, Zustand, React Router, Tailwind CSS, Framer Motion, Axios", "Provides onboarding, dashboard, and study flow UI."],
    ["Observability", "event logging, online metrics, cache, A/B bucket assignment", "Tracks latency, click-through, cache hits, and request behavior."],
], [2.7 * cm, 5.8 * cm, 7.0 * cm])
story.append(Spacer(1, 0.25 * cm))
story.append(Paragraph("Core data assets", styles["Section"]))
bullets([
    "Student dataset: data/raw/Student_Performance.csv.",
    "Event log: data/events/events.csv plus deduplication keys for feedback ingestion.",
    "Model artifacts: ml/artifacts/ contains pickle files, metrics JSON/CSV, training summaries, and charts.",
])
story.append(Paragraph("How the whole system works", styles["Section"]))
story.append(Preformatted(
"""1. User fills the onboarding form.
2. Frontend sends the profile to POST /analyze-user.
3. Backend validates the schema, builds features, predicts a score, and assigns a risk level.
4. Backend returns prediction + recommendation topics.
5. Dashboard renders the performance card and learning cards.
6. User opens a topic, sends feedback, or completes a sprint.
7. The store refreshes prediction and recommendations from the updated signals.""",
    styles["MonoSmall"],
))

story.append(PageBreak())
story.append(Paragraph("Backend API", styles["Section"]))
story.append(Paragraph(
    "The backend is centered on FastAPI. It loads trained artifacts from ml/artifacts, configures permissive CORS, and exposes a small set of endpoints for health, prediction, recommendation, profile lookup, feedback, and quiz updates.",
    styles["BodySmall"],
))
table([
    ["Method", "Path", "Purpose"],
    ["GET", "/", "Simple status message and version."],
    ["GET", "/health", "Checks model loading and schema version."],
    ["POST", "/analyze-user", "Unified onboarding analysis: prediction plus recommendations."],
    ["POST", "/predict-performance", "Returns predicted score, risk, confidence, weak areas, and explanation."],
    ["POST", "/recommend-content", "Builds ranked recommendation topics with topic-level prediction insight."],
    ["POST", "/feedback-event", "Ingests feedback and updates the event/feature loop."],
    ["GET", "/student-profile/{student_id}", "Returns a student snapshot from the dataset."],
    ["POST", "/update-after-quiz", "Updates a learner after a new quiz attempt."],
], [2.0 * cm, 4.2 * cm, 9.3 * cm])
story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph("Prediction request schema", styles["SubSection"]))
bullets([
    "The stricter prediction request includes age, gender, learning style, attendance, assignment score, quiz score, time spent, attempts, previous score, internet access, parental support, extracurricular, stress level, subject weakness, engagement score, and consistency score.",
    "Schema versioning is enforced with SCHEMA_VERSION = 2.0.",
    "The backend clamps and validates values to keep inputs in realistic ranges.",
])
story.append(Paragraph("Backend decision logic", styles["SubSection"]))
bullets([
    "Risk banding: score < 50 means high risk, 50-69 means medium risk, 70+ means low risk.",
    "Confidence estimation prefers ensemble spread when the model supports tree estimators; otherwise it falls back to a heuristic based on predicted score distance and behavioral inputs.",
    "Topic-level insights adjust base predictions using the learner's weak subject and topic keywords such as fundamentals, advanced, or basics.",
])
story.append(Paragraph("Current serving behavior", styles["SubSection"]))
story.append(Paragraph(
    "The unified /analyze-user flow is the onboarding path used by the React app. It returns a prediction object and a small list of recommendations. The more detailed /predict-performance and /recommend-content endpoints exist for richer intelligence and are used by the newer dashboard components and backend orchestration.",
    styles["BodySmall"],
))

story.append(PageBreak())
story.append(Paragraph("Machine Learning Pipeline", styles["Section"]))
story.append(Paragraph(
    "The training pipeline builds a regression model for final score prediction, evaluates multiple regressors, stores the best model, and also builds a 3-cluster student segmentation layer. Feature engineering is shared between training and serving so that the model sees the same structure in both places.",
    styles["BodySmall"],
))
bullets([
    "Feature engineering adds engagement, consistency, learning efficiency, interaction terms, and behavioral embedding-style proxies.",
    "Categorical columns are encoded with LabelEncoder and numeric columns are standardized with StandardScaler.",
    "The production feature set stored in preprocessing_metadata.json has 29 feature columns after engineering and encoding.",
    "A KMeans model with 3 clusters maps learners to Fast Learner, Low Engagement, and Struggling Learner segments.",
])
story.append(Paragraph("Model comparison", styles["SubSection"]))
comp = [["Model", "RMSE", "MAE", "R2", "MAPE", "Calibration error"]]
for row in MODEL_ROWS:
    comp.append([row[0], fmt(row[1]), fmt(row[2]), fmt(row[3]), fmt(row[4]), fmt(row[5])])
table(comp, [2.3 * cm, 2.5 * cm, 2.5 * cm, 2.2 * cm, 2.6 * cm, 3.0 * cm])
story.append(Spacer(1, 0.15 * cm))
story.append(Paragraph("Best model choice", styles["SubSection"]))
bullets([
    "The stacking model is the strongest by RMSE in the saved comparison table.",
    "Random Forest is the closest single-model baseline.",
    "XGBoost is competitive but not the final winner in this saved comparison run.",
])
story.append(Paragraph("Legacy clean-input model", styles["SubSection"]))
table([
    ["Metric", "Value"],
    ["MAE", fmt(LEGACY["mae"])],
    ["RMSE", fmt(LEGACY["rmse"])],
    ["R2", fmt(LEGACY["r2"])],
    ["Training rows", str(LEGACY["train_rows"])],
    ["Test rows", str(LEGACY["test_rows"])],
], [4.0 * cm, 8.5 * cm])
story.append(Spacer(1, 0.15 * cm))
story.append(Paragraph("Prediction charts", styles["SubSection"]))
maybe_image(BASE / "ml" / "artifacts" / "prediction_metrics_grid.png", width=17.0 * cm, height=10.3 * cm)
maybe_image(BASE / "ml" / "artifacts" / "prediction_delta_vs_previous.png", width=17.0 * cm, height=8.2 * cm)

story.append(PageBreak())
story.append(Paragraph("Recommendation Engine", styles["Section"]))
story.append(Paragraph(
    "The recommender uses a hybrid structure. It combines content-based topic selection, collaborative retrieval, adaptive ranking, and a diversity/novelty rerank. The backend enriches every recommended topic with a predicted score, risk level, and human-readable reason.",
    styles["BodySmall"],
))
bullets([
    "Topic library spans subjects like Algebra, Geometry, Statistics, Physics, Chemistry, Biology, History, Literature, and Computer Science.",
    "Hybrid ranking mixes retrieval candidates with user features and item features.",
    "Fallbacks exist for hybrid, segment-popular, and global-popular recommendations if the main path fails.",
])
table([
    ["Metric", "Global value"],
    ["Hit@5", fmt(RECO_GLOBAL["hit"])],
    ["Precision@5", fmt(RECO_GLOBAL["precision"])],
    ["Recall@5", fmt(RECO_GLOBAL["recall"])],
    ["NDCG@5", fmt(RECO_GLOBAL["ndcg"])],
    ["MAP@5", fmt(RECO_GLOBAL["map"])],
    ["Coverage", fmt(RECO_GLOBAL["coverage"])],
    ["Diversity", fmt(RECO_GLOBAL["diversity"])],
    ["Novelty", fmt(RECO_GLOBAL["novelty"])],
    ["Cold-start Hit@5", fmt(RECO_GLOBAL["cold_start_hit"])],
    ["Struggling Hit@5", fmt(RECO_GLOBAL["struggling_hit"])],
], [4.6 * cm, 7.9 * cm])
story.append(Spacer(1, 0.15 * cm))
story.append(Paragraph("Segment metrics", styles["SubSection"]))
seg_rows = [["Segment", "Hit@5", "Precision@5", "Recall@5", "NDCG@5", "MAP@5", "Diversity", "Novelty"]]
for name, metrics in RECO_SEGMENTS.items():
    seg_rows.append([name, fmt(metrics["hit"]), fmt(metrics["precision"]), fmt(metrics["recall"]), fmt(metrics["ndcg"]), fmt(metrics["map"]), fmt(metrics["diversity"]), fmt(metrics["novelty"])])
table(seg_rows, [2.2 * cm, 1.7 * cm, 2.1 * cm, 2.0 * cm, 2.0 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm])
story.append(Spacer(1, 0.15 * cm))
story.append(Paragraph("Recommendation charts", styles["SubSection"]))
maybe_image(BASE / "ml" / "artifacts" / "recommendation_metrics_bar.png", width=17.0 * cm, height=9.2 * cm)

story.append(PageBreak())
story.append(Paragraph("Frontend Workflow", styles["Section"]))
story.append(Paragraph(
    "The frontend is a React single-page app. It starts at onboarding, saves the user profile in a Zustand store, navigates to the dashboard after analysis, and then lets the learner open topic cards to run study sessions and send feedback.",
    styles["BodySmall"],
))
table([
    ["File / area", "Role"],
    ["src/App.jsx", "Routes users between onboarding, dashboard, and learning pages."],
    ["src/pages/DashboardPage.jsx", "Shows the prediction card and recommended topics."],
    ["src/pages/LearningPage.jsx", "Shows a topic sprint with feedback and completion controls."],
    ["src/store/useLearningStore.js", "Holds prediction, recommendations, topic state, and refresh actions."],
    ["src/services/recommendationApi.js", "Calls /analyze-user and validates the API response shape."],
    ["src/components/PerformanceCard.jsx", "Visualizes predicted score, risk, and refresh action."],
    ["src/components/RecommendationSection.jsx", "Renders recommendation cards in sections."],
], [4.6 * cm, 8.0 * cm])
story.append(Spacer(1, 0.15 * cm))
story.append(Paragraph("User journey", styles["SubSection"]))
bullets([
    "Onboarding collects the learner's basic profile and study context.",
    "The dashboard shows predicted score, risk level, and recommendation list.",
    "Opening a topic sends the learner into a timed learning sprint.",
    "Feedback buttons and completion events update the store and can trigger a fresh analysis.",
])
story.append(Paragraph("Routes", styles["SubSection"]))
bullets([
    "/onboarding - entry flow for new users.",
    "/dashboard - main analysis and recommendation view.",
    "/learn/:itemId - topic study view for a selected recommendation.",
])
story.append(Paragraph("Frontend packages", styles["SubSection"]))
bullets([
    "react and react-dom for the UI.",
    "react-router-dom for navigation.",
    "zustand for persistent state.",
    "axios for API calls.",
    "tailwindcss and framer-motion for visual styling and motion.",
])

story.append(PageBreak())
story.append(Paragraph("Runtime and Operations", styles["Section"]))
story.append(Paragraph(
    "The system includes basic runtime support for caching, A/B assignment, event ingestion, and online metrics. These features keep the serving layer practical instead of purely demo-only.",
    styles["BodySmall"],
))
table([
    ["Component", "Observed behavior"],
    ["Cache", "Recommendation responses are cached for 180 seconds."],
    ["A/B testing", "Users are assigned to an experiment bucket with a 50/50 split."],
    ["Online metrics", "Latency, cache hits, CTR, and completion proxies are tracked in ml/artifacts/online_metrics.json."],
    ["Feedback loop", "Feedback events feed the feature store and adaptive recommender."],
    ["Observability", "Structured logging and timed spans are used around key endpoints."],
], [3.0 * cm, 9.6 * cm])
story.append(Spacer(1, 0.15 * cm))
story.append(Paragraph("Current online metrics snapshot", styles["SubSection"]))
bullets([
    f"Latency p50: {ONLINE['p50']} ms",
    f"Latency p95: {ONLINE['p95']} ms",
    f"Latency p99: {ONLINE['p99']} ms",
    f"Cache hit rate: {ONLINE['cache']}",
    f"CTR: {ONLINE['ctr']:.4f}",
    f"Completion-at-K proxy: {ONLINE['completion']}",
])
story.append(Paragraph("Project status notes", styles["SubSection"]))
bullets([
    "The docs mark the system as working/production-style, but monitoring and production database wiring are still listed as future work in the README checklist.",
    "Unified endpoint test notes mention that predictions can cluster in a narrow score band, so recommendation ranking carries much of the user-facing differentiation.",
    "The saved online metrics show a small CTR and no cache hits yet, which is consistent with a fresh or lightly exercised deployment.",
])

story.append(PageBreak())
story.append(Paragraph("How to run it locally", styles["Section"]))
bullets([
    "Backend: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000",
    "Frontend: cd frontend && npm install && npm run dev",
    "Tests: pytest -q, python test_pipeline.py, python test_e2e.py",
    "Optional legacy UI: streamlit run frontend/legacy/app.py",
])
story.append(Paragraph("Typical development flow", styles["SubSection"]))
story.append(Preformatted(
"""1. Train or refresh models with the training scripts.
2. Start the backend API.
3. Start the React frontend.
4. Open the onboarding page.
5. Submit a student profile.
6. Review the prediction card and recommendation cards.
7. Use the learning page to provide feedback and update the model snapshot.""",
    styles["MonoSmall"],
))
story.append(Paragraph("Key file locations", styles["SubSection"]))
bullets([
    "Backend entry point: api/main.py",
    "Schema contracts: api/schemas/contracts.py",
    "Frontend store: frontend/src/store/useLearningStore.js",
    "Recommendation API client: frontend/src/services/recommendationApi.js",
    "Training pipeline: ml/training/train_model.py",
    "Recommendation evaluation: ml/training/evaluate_recommender.py",
    "Saved metrics: ml/artifacts/",
])
story.append(Paragraph("Glossary for beginners", styles["SubSection"]))
bullets([
    "MAE: average absolute prediction error.",
    "RMSE: prediction error with extra penalty for large mistakes.",
    "R2: how much variance the model explains; closer to 1 is better.",
    "Hit@K: whether at least one relevant topic appears in the top K recommendations.",
    "Precision@K: how many of the top K items are relevant.",
    "Recall@K: how much of the relevant set appears in the top K items.",
    "NDCG and MAP: ranking-quality metrics that reward better ordering of useful items.",
    "Risk level: a coarse label used to make model output easier to understand.",
])
story.append(Spacer(1, 0.2 * cm))
story.append(Paragraph("Bottom line", styles["Section"]))
story.append(Paragraph(
    "This is not a single-model demo. It is an integrated learning platform with a score predictor, a topic recommender, a feedback loop, and a React dashboard that turns model output into an interactive study experience.",
    styles["BodySmall"],
))

doc.build(story, onFirstPage=page_num, onLaterPages=page_num)
print(f"Created: {OUTPUT}")
