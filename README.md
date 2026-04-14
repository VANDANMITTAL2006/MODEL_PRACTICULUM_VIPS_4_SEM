# AI Personalized Learning Platform

Production-style ML system with clean boundaries between serving, ML lifecycle, data assets, and frontend applications.

## Architecture

- api/: FastAPI serving layer (core infra, schemas, service adapters)
- ml/: model lifecycle (training, inference, monitoring, recommender, artifacts)
- data/: raw inputs and event logs
- frontend/: React app (primary) + legacy Streamlit app isolated under frontend/legacy/
- scripts/: operational entry points (setup, reports, evaluations)
- config/: runtime configuration templates
- tests/: domain-oriented tests (api, ml, recommender)

## Project Structure

```text
project/
├── api/
│   ├── core/
│   ├── routes/
│   ├── schemas/
│   ├── services/
│   └── main.py
├── ml/
│   ├── data/
│   ├── training/
│   ├── inference/
│   ├── monitoring/
│   ├── recommender/
│   └── artifacts/
├── data/
│   ├── raw/
│   └── events/
├── frontend/
│   ├── src/
│   └── legacy/
│       └── app.py
├── scripts/
├── config/
└── tests/
    ├── api/
    ├── ml/
    └── recommender/
```

## Quick Start

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Generate data and train

```bash
python ml/data/generate_dataset.py
python ml/training/train_model.py
```

3. Evaluate/tune/retrain recommender

```bash
python ml/training/evaluate_recommender.py
python ml/training/tune_recommender.py
python ml/training/retraining_orchestrator.py
```

4. Start API

```bash
uvicorn api.main:app --reload
```

5. Start React frontend

```bash
cd frontend
npm install
npm run dev
```

Optional legacy UI

```bash
streamlit run frontend/legacy/app.py
```

## Tests

```bash
pytest -q
```

## Data and Artifact Policy

- Input dataset: data/raw/Student_Performance.csv
- Event logs: data/events/events.csv
- Model binaries/metrics: ml/artifacts/
- Heavy generated outputs are excluded by .gitignore
