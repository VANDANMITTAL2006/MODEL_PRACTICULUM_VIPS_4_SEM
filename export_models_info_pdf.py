from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem


def build_pdf(output_path: str) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="AI Personalized Learning System - Models and Algorithms",
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    body_style.leading = 14

    small_style = ParagraphStyle(
        "SmallBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=13,
    )

    story = []

    story.append(Paragraph("AI Personalized Learning System", title_style))
    story.append(Paragraph("Models and Algorithms Used", heading_style))
    story.append(Spacer(1, 0.3 * cm))

    story.append(Paragraph(
        "This report explains which machine learning models and algorithms are used in the project and how each one is applied in the workflow.",
        body_style,
    ))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("1) Data Preparation Pipeline", heading_style))
    story.append(Paragraph(
        "Before training any model, the dataset is cleaned and transformed:",
        body_style,
    ))
    prep_points = [
        "Missing numeric values are filled using mean values.",
        "Missing categorical values are filled using mode values.",
        "Engineered features are created: engagement_score, consistency_score, and learning_efficiency.",
        "Categorical columns are encoded using LabelEncoder.",
        "Numeric features are normalized using StandardScaler.",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(p, small_style)) for p in prep_points],
            bulletType="bullet",
            leftIndent=14,
        )
    )
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("2) Prediction Models", heading_style))

    story.append(Paragraph("A. XGBoost Regressor (Primary Model)", body_style))
    xgb_points = [
        "Algorithm type: Gradient Boosted Decision Trees for regression.",
        "Use: Predicts a student final_score from processed features.",
        "How used: Trained with GridSearchCV for hyperparameter tuning and selected as the main deployed predictor.",
        "Runtime: Loaded as model.pkl and used by the /predict-performance and /update-after-quiz backend flows.",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(p, small_style)) for p in xgb_points],
            bulletType="bullet",
            leftIndent=14,
        )
    )

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("B. Random Forest Regressor (Secondary/Baseline)", body_style))
    rf_points = [
        "Algorithm type: Bagging ensemble of decision trees.",
        "Use: Baseline comparator during model training.",
        "How used: Trained and evaluated with RMSE, R2, and cross-validation.",
        "Runtime: Saved as rf_model.pkl but not currently used for live API prediction by default.",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(p, small_style)) for p in rf_points],
            bulletType="bullet",
            leftIndent=14,
        )
    )
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("3) Segmentation Model", heading_style))
    story.append(Paragraph("KMeans Clustering", body_style))
    km_points = [
        "Algorithm type: Unsupervised clustering (k = 3).",
        "Use: Groups students by behavioral/learning patterns.",
        "Input features: engagement_score, consistency_score, learning_efficiency.",
        "Cluster labels are mapped to learner categories: Fast Learner, Low Engagement, Struggling Learner.",
        "Artifacts: kmeans.pkl and cluster_mapping.pkl.",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(p, small_style)) for p in km_points],
            bulletType="bullet",
            leftIndent=14,
        )
    )
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("4) Recommendation Algorithms", heading_style))

    story.append(Paragraph("A. Content-Based Recommendation", body_style))
    cb_points = [
        "Uses student weak subject and quiz score threshold rules.",
        "If quiz score is low, recommends foundational topics first.",
        "Topic catalog is stored in TOPIC_LIBRARY.",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(p, small_style)) for p in cb_points],
            bulletType="bullet",
            leftIndent=14,
        )
    )

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("B. Collaborative Recommendation (KNN Similarity)", body_style))
    knn_points = [
        "Algorithm type: K-Nearest Neighbors similarity search (NearestNeighbors).",
        "Distance metric: Euclidean distance.",
        "Features used: quiz_score, engagement_score, consistency_score, attempts.",
        "Finds similar students and suggests topics from their strong subjects.",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(p, small_style)) for p in knn_points],
            bulletType="bullet",
            leftIndent=14,
        )
    )

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("C. Hybrid Recommendation", body_style))
    hybrid_points = [
        "Combines content-based and collaborative outputs.",
        "Deduplicates merged topic list.",
        "Returns final recommended_topics list through the /recommend-content API.",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(p, small_style)) for p in hybrid_points],
            bulletType="bullet",
            leftIndent=14,
        )
    )
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("5) Model Evaluation and Selection", heading_style))
    eval_points = [
        "Metrics used: RMSE and R2 score.",
        "Cross-validation: 5-fold CV used for stability checks.",
        "Hyperparameter optimization: GridSearchCV for XGBoost.",
        "Selected deployment model: tuned XGBoost model.",
    ]
    story.append(
        ListFlowable(
            [ListItem(Paragraph(p, small_style)) for p in eval_points],
            bulletType="bullet",
            leftIndent=14,
        )
    )

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Generated on: April 9, 2026", small_style))

    doc.build(story)


if __name__ == "__main__":
    output_file = r"C:\Users\SRISHTI MITTAL\OneDrive\Desktop\models_and_algorithms_workflow.pdf"
    build_pdf(output_file)
    print(f"PDF created at: {output_file}")
