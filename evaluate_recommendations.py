"""
Evaluation of Recommendation System Accuracy
=============================================
Measures the accuracy of:
1. Content-Based Filtering
2. Collaborative Filtering (KNN-based)
3. Hybrid Recommendation

Metrics Used:
- Precision@K: What fraction of recommended items are relevant
- Recall@K: What fraction of relevant items were recommended
- Hit Rate@K: How often the system recommends at least one relevant item
- Coverage: What fraction of items can be recommended
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from backend.recommender import (
    content_based_recommend,
    collaborative_recommend,
    hybrid_recommend,
    TOPIC_LIBRARY
)
from data.data_pipeline import load_data

# ────────────────────────────────────────────────────────────
# Evaluation Helper Functions
# ────────────────────────────────────────────────────────────

def get_all_topics():
    """Return flat list of all topics in the library."""
    return [topic for topics in TOPIC_LIBRARY.values() for topic in topics]


def get_subject_topics(subject):
    """Return all topics for a given subject."""
    return TOPIC_LIBRARY.get(subject, [])


def generate_ground_truth(df):
    """
    Generate ground truth 'relevant' topics for each student.
    A topic is considered relevant if:
    - It belongs to their weak subject area
    - OR their quiz_score in that subject is below threshold
    """
    ground_truth = {}

    for idx, row in df.iterrows():
        student_id = row.get('student_id', f'S{idx}')
        weak_subject = row.get('subject_weakness', 'Algebra')
        quiz_score = row.get('quiz_score', 50)

        # Relevant topics = topics from weak subject
        relevant_topics = set(get_subject_topics(weak_subject))

        # If quiz_score is low, also include foundational topics
        if quiz_score < 60:
            relevant_topics.update(get_subject_topics(weak_subject)[:3])

        ground_truth[student_id] = relevant_topics

    return ground_truth


def precision_at_k(recommended, relevant, k):
    """Calculate Precision@K."""
    if not relevant:
        return 0.0
    recommended_k = set(recommended[:k])
    relevant_count = len(recommended_k & relevant)
    return relevant_count / min(k, len(recommended)) if recommended else 0.0


def recall_at_k(recommended, relevant, k):
    """Calculate Recall@K."""
    if not relevant:
        return 0.0
    recommended_k = set(recommended[:k])
    relevant_count = len(recommended_k & relevant)
    return relevant_count / len(relevant)


def hit_rate_at_k(recommended, relevant, k):
    """Calculate Hit Rate@K (1 if at least one hit, 0 otherwise)."""
    if not relevant:
        return 0.0
    recommended_k = set(recommended[:k])
    return 1.0 if len(recommended_k & relevant) > 0 else 0.0


def ndcg_at_k(recommended, relevant, k):
    """Calculate Normalized Discounted Cumulative Gain@K."""
    if not relevant:
        return 0.0

    dcg = 0.0
    idcg = 0.0

    # DCG: Sum of relevance / log2(position + 1)
    for i, topic in enumerate(recommended[:k]):
        if topic in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0

    # IDCG: Ideal DCG (all relevant items at top)
    for i in range(min(len(relevant), k)):
        idcg += 1.0 / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


# ────────────────────────────────────────────────────────────
# Evaluation Functions for Each Method
# ────────────────────────────────────────────────────────────

def evaluate_content_based(df, ground_truth, k_values=[3, 5, 10]):
    """Evaluate Content-Based Filtering."""
    print("\n" + "="*60)
    print("  CONTENT-BASED FILTERING EVALUATION")
    print("="*60)

    results = {k: {'precision': [], 'recall': [], 'hit_rate': [], 'ndcg': []} for k in k_values}

    for idx, row in df.iterrows():
        student_id = row.get('student_id', f'S{idx}')
        weak_subject = row.get('subject_weakness', 'Algebra')
        quiz_score = row.get('quiz_score', 50)

        # Get recommendations
        rec_result = content_based_recommend(weak_subject, quiz_score, num_topics=max(k_values))
        recommended = rec_result['topics']
        relevant = ground_truth.get(student_id, set())

        # Calculate metrics
        for k in k_values:
            results[k]['precision'].append(precision_at_k(recommended, relevant, k))
            results[k]['recall'].append(recall_at_k(recommended, relevant, k))
            results[k]['hit_rate'].append(hit_rate_at_k(recommended, relevant, k))
            results[k]['ndcg'].append(ndcg_at_k(recommended, relevant, k))

    # Aggregate results
    summary = {}
    for k in k_values:
        summary[k] = {
            'precision': np.mean(results[k]['precision']),
            'recall': np.mean(results[k]['recall']),
            'hit_rate': np.mean(results[k]['hit_rate']),
            'ndcg': np.mean(results[k]['ndcg']),
        }

        print(f"\n  @K={k}:")
        print(f"    Precision: {summary[k]['precision']:.4f}")
        print(f"    Recall:    {summary[k]['recall']:.4f}")
        print(f"    Hit Rate:  {summary[k]['hit_rate']:.4f}")
        print(f"    NDCG:      {summary[k]['ndcg']:.4f}")

    return summary


def evaluate_collaborative(df, ground_truth, k_values=[3, 5, 10], test_size=0.2):
    """Evaluate Collaborative Filtering using train/test split."""
    print("\n" + "="*60)
    print("  COLLABORATIVE FILTERING (KNN) EVALUATION")
    print("="*60)

    # Split into train (for KNN database) and test (for evaluation)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    print(f"\n  Train size: {len(train_df)}, Test size: {len(test_df)}")

    results = {k: {'precision': [], 'recall': [], 'hit_rate': [], 'ndcg': []} for k in k_values}

    for idx, row in test_df.iterrows():
        student_id = row.get('student_id', f'S{idx}')
        quiz_score = row.get('quiz_score', 50)

        student_features = {
            'quiz_score': quiz_score,
            'engagement_score': row.get('engagement_score', 50),
            'consistency_score': row.get('consistency_score', 60),
            'attempts': row.get('attempts', 3),
        }

        # Get recommendations using train set as reference
        rec_result = collaborative_recommend(student_features, train_df, num_topics=max(k_values))
        recommended = rec_result['topics']
        relevant = ground_truth.get(student_id, set())

        # Calculate metrics
        for k in k_values:
            results[k]['precision'].append(precision_at_k(recommended, relevant, k))
            results[k]['recall'].append(recall_at_k(recommended, relevant, k))
            results[k]['hit_rate'].append(hit_rate_at_k(recommended, relevant, k))
            results[k]['ndcg'].append(ndcg_at_k(recommended, relevant, k))

    # Aggregate results
    summary = {}
    for k in k_values:
        summary[k] = {
            'precision': np.mean(results[k]['precision']) if results[k]['precision'] else 0,
            'recall': np.mean(results[k]['recall']) if results[k]['recall'] else 0,
            'hit_rate': np.mean(results[k]['hit_rate']) if results[k]['hit_rate'] else 0,
            'ndcg': np.mean(results[k]['ndcg']) if results[k]['ndcg'] else 0,
        }

        print(f"\n  @K={k}:")
        print(f"    Precision: {summary[k]['precision']:.4f}")
        print(f"    Recall:    {summary[k]['recall']:.4f}")
        print(f"    Hit Rate:  {summary[k]['hit_rate']:.4f}")
        print(f"    NDCG:      {summary[k]['ndcg']:.4f}")

    return summary


def evaluate_hybrid(df, ground_truth, k_values=[3, 5, 10]):
    """Evaluate Hybrid Recommendation System."""
    print("\n" + "="*60)
    print("  HYBRID RECOMMENDATION EVALUATION")
    print("="*60)

    results = {k: {'precision': [], 'recall': [], 'hit_rate': [], 'ndcg': []} for k in k_values}

    for idx, row in df.iterrows():
        student_id = row.get('student_id', f'S{idx}')
        weak_subject = row.get('subject_weakness', 'Algebra')
        quiz_score = row.get('quiz_score', 50)

        student_features = {
            'quiz_score': quiz_score,
            'engagement_score': row.get('engagement_score', 50),
            'consistency_score': row.get('consistency_score', 60),
            'attempts': row.get('attempts', 3),
        }

        # Get hybrid recommendations
        rec_result = hybrid_recommend(weak_subject, quiz_score, student_features, df, num_topics=max(k_values))
        recommended = rec_result['recommended_topics']
        relevant = ground_truth.get(student_id, set())

        # Calculate metrics
        for k in k_values:
            results[k]['precision'].append(precision_at_k(recommended, relevant, k))
            results[k]['recall'].append(recall_at_k(recommended, relevant, k))
            results[k]['hit_rate'].append(hit_rate_at_k(recommended, relevant, k))
            results[k]['ndcg'].append(ndcg_at_k(recommended, relevant, k))

    # Aggregate results
    summary = {}
    for k in k_values:
        summary[k] = {
            'precision': np.mean(results[k]['precision']),
            'recall': np.mean(results[k]['recall']),
            'hit_rate': np.mean(results[k]['hit_rate']),
            'ndcg': np.mean(results[k]['ndcg']),
        }

        print(f"\n  @K={k}:")
        print(f"    Precision: {summary[k]['precision']:.4f}")
        print(f"    Recall:    {summary[k]['recall']:.4f}")
        print(f"    Hit Rate:  {summary[k]['hit_rate']:.4f}")
        print(f"    NDCG:      {summary[k]['ndcg']:.4f}")

    return summary


def compare_methods(content_results, collab_results, hybrid_results, k_values=[3, 5, 10]):
    """Compare all methods side by side."""
    print("\n" + "="*60)
    print("  METHOD COMPARISON SUMMARY")
    print("="*60)

    print("\n  Precision@K Comparison:")
    print(f"  {'K':<5} {'Content-Based':<15} {'Collaborative':<15} {'Hybrid':<15}")
    print(f"  {'-'*50}")
    for k in k_values:
        c = content_results.get(k, {}).get('precision', 0)
        col = collab_results.get(k, {}).get('precision', 0)
        h = hybrid_results.get(k, {}).get('precision', 0)
        print(f"  {k:<5} {c:<15.4f} {col:<15.4f} {h:<15.4f}")

    print("\n  Recall@K Comparison:")
    print(f"  {'K':<5} {'Content-Based':<15} {'Collaborative':<15} {'Hybrid':<15}")
    print(f"  {'-'*50}")
    for k in k_values:
        c = content_results.get(k, {}).get('recall', 0)
        col = collab_results.get(k, {}).get('recall', 0)
        h = hybrid_results.get(k, {}).get('recall', 0)
        print(f"  {k:<5} {c:<15.4f} {col:<15.4f} {h:<15.4f}")

    print("\n  Hit Rate@K Comparison:")
    print(f"  {'K':<5} {'Content-Based':<15} {'Collaborative':<15} {'Hybrid':<15}")
    print(f"  {'-'*50}")
    for k in k_values:
        c = content_results.get(k, {}).get('hit_rate', 0)
        col = collab_results.get(k, {}).get('hit_rate', 0)
        h = hybrid_results.get(k, {}).get('hit_rate', 0)
        print(f"  {k:<5} {c:<15.4f} {col:<15.4f} {h:<15.4f}")

    print("\n  NDCG@K Comparison:")
    print(f"  {'K':<5} {'Content-Based':<15} {'Collaborative':<15} {'Hybrid':<15}")
    print(f"  {'-'*50}")
    for k in k_values:
        c = content_results.get(k, {}).get('ndcg', 0)
        col = collab_results.get(k, {}).get('ndcg', 0)
        h = hybrid_results.get(k, {}).get('ndcg', 0)
        print(f"  {k:<5} {c:<15.4f} {col:<15.4f} {h:<15.4f}")


def calculate_coverage(df, k=10):
    """Calculate catalog coverage - what fraction of topics can be recommended."""
    all_topics = set(get_all_topics())
    recommended_topics = set()

    for idx, row in df.sample(min(100, len(df))).iterrows():
        weak_subject = row.get('subject_weakness', 'Algebra')
        quiz_score = row.get('quiz_score', 50)

        rec_result = hybrid_recommend(weak_subject, quiz_score, {
            'quiz_score': quiz_score,
            'engagement_score': 50,
            'consistency_score': 60,
            'attempts': 3,
        }, df, num_topics=k)

        recommended_topics.update(rec_result['recommended_topics'])

    coverage = len(recommended_topics) / len(all_topics)
    print(f"\n  Catalog Coverage@{k}: {coverage:.4f} ({len(recommended_topics)}/{len(all_topics)} topics)")
    return coverage


# ────────────────────────────────────────────────────────────
# Main Evaluation
# ────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  AI PERSONALIZED LEARNING - RECOMMENDATION EVALUATION")
    print("="*60)

    # Load data
    print("\nLoading dataset...")
    try:
        df = load_data()
        print(f"  Loaded {len(df)} students")
    except FileNotFoundError:
        print("  Dataset not found. Generating synthetic data...")
        from data.generate_dataset import generate_dataset
        df = generate_dataset()
        print(f"  Generated {len(df)} students")

    # Generate ground truth
    print("\nGenerating ground truth...")
    ground_truth = generate_ground_truth(df)
    avg_relevant = np.mean([len(gt) for gt in ground_truth.values()])
    print(f"  Average relevant topics per student: {avg_relevant:.2f}")

    # Evaluate each method
    k_values = [3, 5, 10]

    content_results = evaluate_content_based(df, ground_truth, k_values)
    collab_results = evaluate_collaborative(df, ground_truth, k_values)
    hybrid_results = evaluate_hybrid(df, ground_truth, k_values)

    # Compare methods
    compare_methods(content_results, collab_results, hybrid_results, k_values)

    # Coverage
    calculate_coverage(df)

    # Summary
    print("\n" + "="*60)
    print("  EVALUATION SUMMARY")
    print("="*60)

    # Find best method for each metric
    best_precision = max(
        ('Content-Based', content_results.get(5, {}).get('precision', 0)),
        ('Collaborative', collab_results.get(5, {}).get('precision', 0)),
        ('Hybrid', hybrid_results.get(5, {}).get('precision', 0)),
        key=lambda x: x[1]
    )

    best_recall = max(
        ('Content-Based', content_results.get(5, {}).get('recall', 0)),
        ('Collaborative', collab_results.get(5, {}).get('recall', 0)),
        ('Hybrid', hybrid_results.get(5, {}).get('recall', 0)),
        key=lambda x: x[1]
    )

    best_hit_rate = max(
        ('Content-Based', content_results.get(5, {}).get('hit_rate', 0)),
        ('Collaborative', collab_results.get(5, {}).get('hit_rate', 0)),
        ('Hybrid', hybrid_results.get(5, {}).get('hit_rate', 0)),
        key=lambda x: x[1]
    )

    print(f"\n  Best Precision@5: {best_precision[0]} ({best_precision[1]:.4f})")
    print(f"  Best Recall@5:    {best_recall[0]} ({best_recall[1]:.4f})")
    print(f"  Best Hit Rate@5:  {best_hit_rate[0]} ({best_hit_rate[1]:.4f})")

    print("\n" + "="*60)
    print("  Evaluation Complete!")
    print("="*60)

    return {
        'content_based': content_results,
        'collaborative': collab_results,
        'hybrid': hybrid_results,
    }


if __name__ == "__main__":
    results = main()
