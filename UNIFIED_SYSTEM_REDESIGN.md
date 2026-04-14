# Unified AI Learning System - System Redesign

## 🎯 Overview

**New Architecture:** `USER INPUT → PREDICTION → RECOMMENDATION → UI DISPLAY`

All prediction and recommendation logic is now unified in a **single endpoint** that understands both the user's ability and their learning needs in one pass.

---

## 🚀 The New Unified Endpoint

### `POST /analyze-user`

**Input:**
```json
{
  "schema_version": "2.0",
  "student_id": "S0001",
  "attendance": 75,
  "assignment_score": 70,
  "quiz_score": 62,
  "time_spent_hours": 5,
  "attempts": 3,
  "subject_weakness": "Algebra"
}
```

**Output:**
```json
{
  "prediction": {
    "predicted_score": 68.5,
    "risk_level": "medium",
    "confidence": 0.82
  },
  "recommendations": [
    {
      "topic": "Algebra Fundamentals",
      "predicted_score": 58.2,
      "risk_level": "high",
      "reason": "Predicted low performance in Algebra Fundamentals; prioritize this topic to prevent learning gaps.",
      "difficulty": "easy"
    },
    {
      "topic": "Linear Equations",
      "predicted_score": 68.5,
      "risk_level": "medium",
      "reason": "Moderate predicted performance in Linear Equations; a focused practice burst can improve mastery.",
      "difficulty": "medium"
    },
    {
      "topic": "Statistics Basics",
      "predicted_score": 75.1,
      "risk_level": "low",
      "reason": "Strong predicted performance in Statistics Basics; use this as momentum-building practice.",
      "difficulty": "hard"
    }
  ],
  "student_id": "S0001",
  "model_version": "legacy",
  "latency_ms": 45.2
}
```

---

## 📊 System Flow

```
┌─────────────────────────────────────────════════┐
│           User Onboarding Form                   │
│  (quiz_score, engagement, consistency, subject) │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  POST /analyze-user  │
        └──────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
    ┌─────────────┐   ┌──────────────────┐
    │ PREDICTION  │   │  RECOMMENDATION  │
    │             │   │   GENERATION     │
    │ • Score     │   │                  │
    │ • Risk      │   │ • Per-topic      │
    │ • Confidence│   │   predictions    │
    │             │   │ • Risk-aware     │
    │             │   │   reranking      │
    └─────────────┘   └──────────────────┘
         │                   │
         └─────────┬─────────┘
                   ▼
        ┌──────────────────────┐
        │   Unified Response   │
        │                      │
        │ {prediction, recs}   │
        └──────────────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │   Frontend State Update   │
        │                          │
        │ • prediction snapshot    │
        │ • risk buckets (H/M/L)   │
        │ • recommendations list   │
        └──────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌──────────┐ ┌──────────┐  ┌────────────┐
│Performance│ │Risk Viz.|  │ Recommend. │
│  Card     │ │Buckets  │  │   Cards   │
└──────────┘ └──────────┘  └────────────┘
```

---

## 🔧 Implementation Details

### Backend: `/analyze-user` Logic

```python
@app.post("/analyze-user")
def analyze_user(req: PredictRequest):
    # 1. PREDICTION PHASE
    X = make_feature_vector(req_dict, feature_cols)
    predicted_score = model.predict(X)[0]        # e.g., 68.5
    risk_level = risk_level_from_score(predicted_score)  # "medium"
    confidence = estimate_prediction_confidence(...)     # 0.82
    
    # 2. FEATURE PREP
    student_features = {
        "quiz_score": req.quiz_score,
        "engagement_score": engagement_score,
        "consistency_score": consistency_score,
        "attempts": req.attempts,
    }
    
    # 3. RECOMMENDATION GENERATION
    # Pass BOTH prediction + user features to recommender
    recommendations = hybrid_recommend(
        subject_weakness=req.subject_weakness,
        quiz_score=req.quiz_score,
        student_features=student_features,
        df=df,                          # Historical data
        num_topics=5
    )
    
    # 4. ENRICH EACH RECOMMENDATION WITH PREDICTIONS
    enriched = []
    for topic in recommendations:
        topic_pred_score, topic_risk, reason = topic_prediction_insight(
            predicted_score, topic, req.subject_weakness
        )
        enriched.append({
            "topic": topic,
            "predicted_score": topic_pred_score,
            "risk_level": topic_risk,
            "reason": reason,
            "difficulty": map_difficulty(topic_pred_score)
        })
    
    # 5. RETURN UNIFIED RESPONSE
    return {
        "prediction": {
            "predicted_score": predicted_score,
            "risk_level": risk_level,
            "confidence": confidence,
        },
        "recommendations": enriched
    }
```

### Frontend: Unified Flow

**Before (Separate Calls):**
```javascript
// Call 1: Get prediction
const predictionData = await predictPerformance(userFeatures);

// Call 2: Get recommendations (without prediction insight)
const recommendationData = await getRecommendations(userFeatures);

// Merge in UI (hacky)
```

**After (Unified Call):**
```javascript
// Single call = prediction + recommendations from one analysis
const analysisResult = await analyzeUser(userFeatures);

// State update combines both instantly
const nextState = {
  prediction: analysisResult.prediction,
  recommendations: analysisResult.recommendations,
  riskBuckets: buildRiskBuckets(analysisResult.recommendations),
};
```

---

## 📱 UI Display

### Dashboard Layout (After User Submits Onboarding)

```
┌─────────────────────────────────────────────────────────┐
│  SECTION 1: PREDICTION + RISK VISUALIZATION            │
│                                                         │
│  ┌──────────────────────┐  ┌─────────────────────────┐ │
│  │ Predicted Score: 68.5 │  │  HIGH-RISK TOPICS     │ │
│  │ Risk: MEDIUM 🟡       │  │  • Algebra Adv.       │ │
│  │ Confidence: 82%       │  │  • Geometry           │ │
│  │                       │  │                       │ │
│  │ "Based on your...     │  │  MEDIUM-RISK TOPICS   │ │
│  │ you'll score around ... │  │  • Linear Equations   │ │
│  │                       │  │                       │ │
│  │ [Refresh Insight]     │  │  STRONG AREAS         │ │
│  └──────────────────────┘  │  • Statistics         │ │
│                            │  • Calculus           │ │
│                            └─────────────────────────┘ │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  SECTION 2: PERSONALIZED RECOMMENDATIONS              │
│                                                         │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────┐   │
│  │ Algebra Fund.  │ │Linear Equations│ │Statistics  │   │
│  │                │ │                │ │            │   │
│  │ Score: 58.2    │ │ Score: 68.5    │ │ Score: 75.1│   │
│  │ Risk: 🔴 HIGH  │ │ Risk: 🟡 MED   │ │ Risk: 🟢 LOW│  │
│  │                │ │                │ │            │   │
│  │ "Predicted low │ │ "Moderate..."  │ │ "Strong..."│   │
│  │ performance..."│ │                │ │            │   │
│  │                │ │                │ │            │   │
│  │ [Start] [Save] │ │ [Start] [Save] │ │[Start][Save]│   │
│  └────────────────┘ └────────────────┘ └────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 Real-Time Updates

**Workflow after user action:**

```
User Action: Click "Complete" on Algebra lesson
         │
         ▼
   POST /feedback-event
         │
         ▼ (background)
   FeedbackLoop processes event
         │
         ▼
   Adaptive recommender updates
         │
         ▼
   Frontend silently calls:
   POST /analyze-user (with same user features)
         │
         ▼
   New prediction + ranked recommendations
         │
         ▼
   Dashboard refreshes silently
         │
         └─► User sees updated performance card
         └─► User sees reranked recommendations
```

Example: If user completes "Algebra Fundamentals" (high-risk):
- Predicted score might increase 2–3 points
- Risk level might drop from "high" to "medium"
- Recommendations reorder to keep high-risk topics prioritized

---

## 💪 Key Advantages

### 1. **Coherent System Narrative**
   - Single unified response tells a story: "Here's your score, here's your risk, here's what to learn"
   - No fragmented data; everything is contextual

### 2. **Efficient API**
   - One call instead of two parallelized calls
   - Lower latency (combines computation)
   - Simpler client-side state management

### 3. **Per-Topic Predictions**
   - Each recommendation includes topic-level predicted score + risk
   - User understands why each topic is suggested
   - "You'll likely score 58 on this, but 75 on that"

### 4. **Risk-Aware Ranking**
   - Recommendations auto-sort by learning need, not just engagement
   - High-risk topics bubble up
   - Self-directed learning feels smarter

### 5. **Production Feel**
   - Feels like intelligent tutoring system (ITS)
   - User perceives AI understanding their ability + needs
   - Not just "popular topics" but "topics for you"

---

## 📦 API Endpoints (Full)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/analyze-user` | POST | **NEW** Unified prediction + recommendation |
| `/predict-performance` | POST | Legacy (still works) |
| `/recommend-content` | POST | Legacy (still works) |
| `/feedback-event` | POST | Post user interaction |
| `/update-after-quiz` | POST | Quiz score update |
| `/student-profile/{id}` | GET | Get student profile |
| `/health` | GET | Health check |

---

## 🧪 Testing

### Test the Unified Endpoint

```bash
curl -X POST http://localhost:8000/analyze-user \
  -H "Content-Type: application/json" \
  -d '{
    "schema_version": "2.0",
    "student_id": "S0001",
    "attendance": 75,
    "assignment_score": 70,
    "quiz_score": 62,
    "time_spent_hours": 5,
    "attempts": 3,
    "subject_weakness": "Algebra"
  }'
```

### Interactive API Docs

**Swagger UI:** http://localhost:8000/docs

---

## 🔗 Code Files Updated

- ✅ [api/main.py](./api/main.py) — New `/analyze-user` endpoint + enrichment logic
- ✅ [frontend/src/services/recommendationApi.js](./frontend/src/services/recommendationApi.js) — `analyzeUser()` service call
- ✅ [frontend/src/store/useLearningStore.js](./frontend/src/store/useLearningStore.js) — Unified state fetch + real-time refresh
- ✅ [frontend/src/pages/DashboardPage.jsx](./frontend/src/pages/DashboardPage.jsx) — PerformanceCard + risk viz
- ✅ [frontend/src/components/RecommendationCard.jsx](./frontend/src/components/RecommendationCard.jsx) — Risk badges + predicted scores
- ✅ [frontend/src/utils/topicMeta.js](./frontend/src/utils/topicMeta.js) — Enhanced topic card mapping

---

## 🎓 Mental Model

Think of the system as a **smart tutor**:

1. **Listen:** "Tell me about yourself" (onboarding form)
2. **Understand:** "Based on your profile, I predict you'll score X" (prediction)
3. **Analyze:** "Your weak points are Y; your strengths are Z" (risk buckets)
4. **Recommend:** "Here's what you should learn, in priority order" (ranked recommendations)
5. **Adapt:** "After each lesson, I'll recalibrate and update your learning path" (real-time refresh)

This is how modern AI learning platforms (Duolingo, YouTube Learning) work.

---

## 📊 System is Now Live

- ✅ Backend: http://localhost:8000/docs
- ✅ Frontend: http://localhost:5173
- ✅ Unified endpoint: POST /analyze-user
- ✅ Real-time updates: After each user action

**Next steps:**
1. Go to http://localhost:5173
2. Fill onboarding form
3. Watch dashboard render prediction + risk-aware recommendations
4. Click interactions and see predictions/recommendations refresh silently
