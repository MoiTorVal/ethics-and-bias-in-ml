#!/usr/bin/env python3
"""
Quick test of the loan prediction functionality
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predict_loan import quick_train_model, predict_single_applicant

def test_prediction():
    """Test the prediction with sample data"""
    
    print("🧪 Testing Loan Approval Prediction")
    print("="*40)
    
    # Train the model
    print("Training model...")
    model, encoders, scaler = quick_train_model()
    
    # Test data for a sample applicant
    test_cases = [
        {
            'name': 'High-Quality Applicant',
            'data': {
                'gender': 'Male',
                'race': 'White', 
                'education': 'Bachelor',
                'age': 35,
                'income': 75000,
                'credit_score': 720,
                'loan_amount': 300000,
                'employment_years': 8,
                'debt_to_income': 25,
                'property_value': 400000,
                'down_payment_pct': 20
            }
        },
        {
            'name': 'High-Risk Applicant',
            'data': {
                'gender': 'Female',
                'race': 'Black',
                'education': 'High School',
                'age': 25,
                'income': 35000,
                'credit_score': 580,
                'loan_amount': 200000,
                'employment_years': 2,
                'debt_to_income': 45,
                'property_value': 220000,
                'down_payment_pct': 5
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print('='*50)
        
        data = test_case['data']
        
        # Calculate loan-to-value ratio
        loan_to_value = (data['loan_amount'] / data['property_value']) * 100
        
        try:
            # Encode categorical variables
            gender_encoded = encoders['gender'].transform([data['gender']])[0]
            race_encoded = encoders['race'].transform([data['race']])[0] 
            education_encoded = encoders['education'].transform([data['education']])[0]
            
            # Create feature vector
            import numpy as np
            features = np.array([[
                data['age'], data['income'], data['credit_score'], 
                data['loan_amount'], data['employment_years'],
                data['debt_to_income'], loan_to_value, data['down_payment_pct'],
                gender_encoded, race_encoded, education_encoded
            ]])
            
            # Make prediction
            if scaler is not None:
                features = scaler.transform(features)
                
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0, 1]
            
            # Display results
            print(f"\nApplicant: {data['gender']} {data['race']} with {data['education']}")
            print(f"Age: {data['age']}, Income: ${data['income']:,}")
            print(f"Credit Score: {data['credit_score']}")
            print(f"Loan: ${data['loan_amount']:,} on ${data['property_value']:,} property")
            print(f"LTV: {loan_to_value:.1f}%, DTI: {data['debt_to_income']:.1f}%")
            
            status = "✅ APPROVED" if prediction == 1 else "❌ REJECTED"
            print(f"\nPREDICTION: {status}")
            print(f"Probability: {probability:.1%}")
            
            # Risk assessment
            if probability >= 0.8:
                risk = "🟢 Very Low Risk"
            elif probability >= 0.6:
                risk = "🟡 Low Risk"
            elif probability >= 0.4:
                risk = "🟠 Medium Risk"
            else:
                risk = "🔴 High Risk"
            
            print(f"Risk Level: {risk}")
            
        except Exception as e:
            print(f"Error in prediction: {e}")
    
    print(f"\n{'='*50}")
    print("✅ TESTING COMPLETED")
    print("="*50)
    print("To run interactive predictions:")
    print("  python predict_loan.py")
    print("or")
    print("  python main.py")

if __name__ == "__main__":
    test_prediction()
