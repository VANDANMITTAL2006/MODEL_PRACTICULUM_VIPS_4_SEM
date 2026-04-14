#!/usr/bin/env python3
"""
End-to-end test of the unified prediction + recommendation system.
Tests multiple scenarios to ensure the pipeline works correctly.
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

TEST_CASES = [
    {
        "name": "High performer with strong engagement",
        "payload": {
            "student_id": "e2e_001",
            "age": 20,
            "gender": "Female",
            "learning_style": "Visual",
            "attendance": 0.95,
            "assignment_score": 92,
            "quiz_score": 90,
            "time_spent_hours": 15,
            "attempts": 8,
            "previous_score": 88,
            "internet_access": True,
            "parental_support": "High",
            "extracurricular": True,
            "stress_level": "Low",
            "engagement_score": 85,
            "consistency_score": 93,
            "subject_weakness": "Statistics",
            "schema_version": "2.0"
        }
    },
    {
        "name": "Moderate performer with variable engagement",
        "payload": {
            "student_id": "e2e_002",
            "age": 22,
            "gender": "Male",
            "learning_style": "Kinesthetic",
            "attendance": 0.72,
            "assignment_score": 68,
            "quiz_score": 71,
            "time_spent_hours": 8,
            "attempts": 4,
            "previous_score": 65,
            "internet_access": True,
            "parental_support": "Medium",
            "extracurricular": False,
            "stress_level": "Medium",
            "engagement_score": 60,
            "consistency_score": 70,
            "subject_weakness": "Calculus",
            "schema_version": "2.0"
        }
    },
    {
        "name": "Struggling student needing support",
        "payload": {
            "student_id": "e2e_003",
            "age": 19,
            "gender": "Non-binary",
            "learning_style": "Auditory",
            "attendance": 0.55,
            "assignment_score": 45,
            "quiz_score": 48,
            "time_spent_hours": 4,
            "attempts": 2,
            "previous_score": 42,
            "internet_access": False,
            "parental_support": "Low",
            "extracurricular": False,
            "stress_level": "High",
            "engagement_score": 25,
            "consistency_score": 50,
            "subject_weakness": "Physics",
            "schema_version": "2.0"
        }
    }
]

def test_scenario(test_case):
    """Test a single scenario."""
    print(f"\n{'='*70}")
    print(f"📋 Scenario: {test_case['name']}")
    print(f"{'='*70}")
    
    payload = test_case['payload']
    try:
        start = time.perf_counter()
        response = requests.post(f"{BASE_URL}/analyze-user", json=payload, timeout=10)
        elapsed = (time.perf_counter() - start) * 1000
        
        if response.status_code != 200:
            print(f"❌ Status {response.status_code}: {response.text}")
            return False
        
        data = response.json()
        pred = data.get("prediction", {})
        recs = data.get("recommendations", [])
        
        # Validate prediction
        score = pred.get("predicted_score", 0)
        risk = pred.get("risk_level", "unknown")
        confidence = pred.get("confidence", 0)
        
        print(f"✅ Score: {score:.1f}/100 | Risk: {risk} | Confidence: {confidence:.1%} | Response time: {elapsed:.0f}ms")
        
        # Validate recommendations
        print(f"📚 Recommendations ({len(recs)}):")
        for i, rec in enumerate(recs[:3], 1):
            topic = rec.get("topic", "N/A")
            pred_score = rec.get("predicted_score", 0)
            risk_level = rec.get("risk_level", "N/A")
            print(f"   {i}. {topic} (pred: {pred_score:.1f}, risk: {risk_level})")
        
        # Basic validation
        checks = [
            ("Score in range [0,100]", 0 <= score <= 100),
            ("Risk level valid", risk in ["low", "medium", "high"]),
            ("Confidence in range [0,1]", 0 <= confidence <= 1),
            ("Has recommendations", len(recs) > 0),
            ("Response time < 5s", elapsed < 5000),
        ]
        
        all_pass = all(check[1] for check in checks)
        for check_name, result in checks:
            print(f"   {'✓' if result else '✗'} {check_name}")
        
        return all_pass
        
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Backend not running")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run all test cases."""
    print("\n" + "="*70)
    print("🚀 END-TO-END PREDICTION SYSTEM TEST")
    print("="*70)
    
    results = []
    for test_case in TEST_CASES:
        passed = test_scenario(test_case)
        results.append((test_case['name'], passed))
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 TEST SUMMARY")
    print(f"{'='*70}")
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    print(f"Passed: {passed_count}/{total_count}")
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    return passed_count == total_count

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
