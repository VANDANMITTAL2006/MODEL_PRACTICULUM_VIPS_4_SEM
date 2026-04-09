# AI Personalized Learning System

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python data/generate_dataset.py
```

### 3. Train Models
```bash
python models/train_model.py
```

### 4. Start Backend (Terminal 1)
```bash
uvicorn backend.main:app --reload
```

### 5. Start Frontend (Terminal 2)
```bash
streamlit run frontend/app.py
```

---

## Project Structure
```
project/
├── data/
│   ├── generate_dataset.py    ← synthetic dataset generator
│   ├── data_pipeline.py       ← preprocessing, feature engineering
│   └── Student_Performance.csv
├── models/
│   ├── train_model.py         ← XGBoost + RandomForest + KMeans training
│   ├── model.pkl              ← XGBoost (primary)
│   ├── rf_model.pkl           ← Random Forest
│   ├── scaler.pkl             ← StandardScaler
│   ├── encoders.pkl           ← LabelEncoders
│   ├── kmeans.pkl             ← clustering model
│   └── cluster_mapping.pkl    ← cluster → label mapping
├── backend/
│   ├── main.py                ← FastAPI app (6 endpoints)
│   └── recommender.py         ← hybrid recommendation engine
├── frontend/
│   └── app.py                 ← Streamlit UI (5 pages)
├── requirements.txt
└── setup_and_run.bat          ← Windows one-click setup
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root info |
| GET | `/health` | Health check + model status |
| POST | `/predict-performance` | Predict student's final score |
| POST | `/recommend-content` | Hybrid content recommendations |
| GET | `/student-profile/{id}` | Full student analytics |
| POST | `/update-after-quiz` | Real-time adaptive update |

---

## ML Models

| Model | Role |
|-------|------|
| XGBoost Regressor | Primary performance predictor |
| Random Forest | Secondary / ensemble comparison |
| KMeans (k=3) | Student segmentation |
| KNN | Collaborative filtering |

## Student Segments
- ⚡ **Fast Learner** — high engagement + high score
- 🔄 **Low Engagement** — medium score, low study time
- ⚠️ **Struggling Learner** — low score, high attempts
