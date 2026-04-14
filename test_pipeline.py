#!/usr/bin/env python3
"""Quick test of the unified prediction pipeline."""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_analyze_user():
    """Test the /analyze-user endpoint with complete payload."""
    
    payload = {
        "student_id": "test_001",
        "age": 20,
        "gender": "Male",
        "learning_style": "Visual",
        "attendance": 0.85,
        "assignment_score": 78,
        "quiz_score": 82,
        "time_spent_hours": 12.5,
        "attempts": 5,
        "previous_score": 75,
        "internet_access": True,
        "parental_support": "Medium",
        "extracurricular": False,
        "stress_level": "Low",
        "engagement_score": 75,
        "consistency_score": 80,
        "subject_weakness": "Algebra",
        "schema_version": "2.0"
    }
    
    print("=" * 60)
    print("Testing /analyze-user Endpoint")
    print("=" * 60)
    print(f"\n📤 Request Payload ({len(payload)} fields):")
    for k, v in payload.items():
        print(f"   {k}: {v}")
    
    try:
        response = requests.post(f"{BASE_URL}/analyze-user", json=payload, timeout=10)
        
        print(f"\n📥 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✅ SUCCESS! Response:")
            print(json.dumps(data, indent=2))
            
            # Analyze prediction
            pred = data.get("prediction", {})
            print("\n📊 Prediction Analysis:")
            print(f"   Predicted Score: {pred.get('predicted_score', 'N/A')} / 100")
            print(f"   Risk Level: {pred.get('risk_level', 'N/A')}")
            print(f"   Confidence: {pred.get('confidence', 0.0):.1%}")
            print(f"   Explanation: {pred.get('explanation', 'N/A')}")
            
            # Check recommendations
            recs = data.get("recommendations", [])
            print(f"\n📋 Recommendations ({len(recs)} items):")
            for i, rec in enumerate(recs, 1):
                print(f"   {i}. {rec.get('title', 'N/A')} (risk: {rec.get('risk_level', 'N/A')})")
            
            return True
        else:
            print(f"\n❌ ERROR Response:")
            print(f"   {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n❌ CONNECTION ERROR: Backend not running on http://localhost:8000")
        print("   Start it with: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_analyze_user()
    exit(0 if success else 1)
