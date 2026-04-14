#!/usr/bin/env python3
"""
Simulates the exact workflow a user would experience:
1. Fill onboarding form
2. Submit to /analyze-user endpoint
3. Receive prediction + recommendations
4. Display dashboard data
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def simulate_user_workflow():
    """Simulate complete user journey."""
    
    print("\n" + "="*70)
    print("🎓 USER WORKFLOW SIMULATION")
    print("="*70)
    
    # Step 1: Onboarding Form Input
    print("\n[Step 1] User fills onboarding form at /onboarding")
    print("-" * 70)
    
    form_data = {
        "userId": "student_demo_001",
        "skillLevel": "Intermediate",
        "subjectWeakness": "Calculus",
        "quizScore": 72,
        "engagementScore": 68,
        "consistencyScore": 75,
        "attempts": 4,
    }
    
    print("Form Data Entered:")
    for key, value in form_data.items():
        print(f"  • {key}: {value}")
    
    # Step 2: Frontend Maps to API Payload
    print("\n[Step 2] Frontend service (toPredictPayload) maps form → API payload")
    print("-" * 70)
    
    api_payload = {
        "schema_version": "2.0",
        "student_id": form_data["userId"],
        "age": 20,
        "gender": "Male",
        "learning_style": "Visual",
        "attendance": form_data["consistencyScore"],  # Derived
        "assignment_score": (form_data["consistencyScore"] + form_data["quizScore"]) / 2,  # Derived
        "quiz_score": form_data["quizScore"],
        "time_spent_hours": form_data["engagementScore"] / 10,  # Derived
        "attempts": form_data["attempts"],
        "previous_score": form_data["quizScore"] * 0.8 + form_data["consistencyScore"] * 0.2,  # Derived
        "internet_access": 1,
        "parental_support": "Medium",
        "extracurricular": 0,
        "stress_level": "Medium",
        "engagement_score": form_data["engagementScore"],
        "consistency_score": form_data["consistencyScore"],
        "subject_weakness": form_data["subjectWeakness"],
    }
    
    print(f"API Payload Generated (18 fields):")
    print(f"  • Required: attendance={api_payload['attendance']:.1f}, assignment={api_payload['assignment_score']:.1f}, quiz={api_payload['quiz_score']}")
    print(f"  • Optional: engagement={api_payload['engagement_score']}, consistency={api_payload['consistency_score']}")
    print(f"  • Subject weakness: {api_payload['subject_weakness']}")
    
    # Step 3: Backend Prediction
    print("\n[Step 3] Backend processes request: /analyze-user")
    print("-" * 70)
    
    try:
        start = time.perf_counter()
        response = requests.post(
            f"{BASE_URL}/analyze-user",
            json=api_payload,
            timeout=10
        )
        elapsed = (time.perf_counter() - start) * 1000
        
        if response.status_code != 200:
            print(f"❌ ERROR {response.status_code}: {response.text}")
            return False
        
        result = response.json()
        
        print(f"✅ Backend Response (HTTP 200) - {elapsed:.0f}ms")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    
    # Step 4: Display Prediction
    print("\n[Step 4] Frontend displays prediction on dashboard")
    print("-" * 70)
    
    prediction = result.get("prediction", {})
    score = prediction.get("predicted_score", 0)
    risk = prediction.get("risk_level", "unknown")
    confidence = prediction.get("confidence", 0)
    
    print(f"📊 Performance Card:")
    print(f"  Predicted Score: {score:.1f}/100")
    print(f"  Risk Level: {risk.upper()}")
    print(f"  Confidence: {confidence:.1%}")
    
    # Step 5: Display Recommendations
    print("\n[Step 5] Frontend displays recommendations")
    print("-" * 70)
    
    recommendations = result.get("recommendations", [])
    print(f"📚 Topics Recommended (showing first 5 of {len(recommendations)}):\n")
    
    risk_buckets = {"high": [], "medium": [], "low": []}
    for i, rec in enumerate(recommendations[:5], 1):
        topic = rec.get("topic", "Unknown")
        pred_score = rec.get("predicted_score", 0)
        rec_risk = rec.get("risk_level", "unknown")
        reason = rec.get("reason", "")
        
        risk_buckets[rec_risk].append(topic)
        
        print(f"  {i}. {topic}")
        print(f"     └─ Predicted Score: {pred_score:.1f} | Risk: {rec_risk}")
    
    print(f"\n🎯 Risk Visualization:")
    print(f"  High-risk topics: {', '.join(risk_buckets['high']) if risk_buckets['high'] else 'None'}")
    print(f"  Medium-risk topics: {', '.join(risk_buckets['medium']) if risk_buckets['medium'] else 'None'}")
    print(f"  Strong areas: {', '.join(risk_buckets['low']) if risk_buckets['low'] else 'None'}")
    
    # Step 6: Verify Data Quality
    print("\n[Step 6] Verify data quality ✓")
    print("-" * 70)
    
    checks = [
        ("Prediction score in [0, 100]", 0 <= score <= 100),
        ("Risk level is {low, medium, high}", risk in ["low", "medium", "high"]),
        ("Confidence in [0, 1]", 0 <= confidence <= 1),
        ("Has recommendations", len(recommendations) > 0),
        ("Response time < 3 seconds", elapsed < 3000),
        ("Backend processed features", score > 0),  # Not 0.0
    ]
    
    all_pass = True
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"  [{status}] {check_name}")
        if not result:
            all_pass = False
    
    # Summary
    print("\n" + "="*70)
    print("📋 WORKFLOW SUMMARY")
    print("="*70)
    print(f"✅ Form Input → API Mapping → Backend Processing → Dashboard Display: COMPLETE")
    print(f"✅ Prediction: {score:.1f} ({risk}, confidence {confidence:.0%})")
    print(f"✅ Recommendations: {len(recommendations)} topics generated")
    print(f"✅ All validation checks: {'PASS' if all_pass else 'FAIL'}")
    
    return all_pass

if __name__ == "__main__":
    try:
        success = simulate_user_workflow()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        exit(1)
