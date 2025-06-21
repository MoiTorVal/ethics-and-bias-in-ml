#!/usr/bin/env python3
"""
Standalone Loan Approval Prediction Tool
Run this script to quickly predict loan approval for individual applicants.
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add the current directory to Python path to import from main.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import generate_loan_data, prepare_data_for_modeling, split_data, train_models
except ImportError as e:
    print(f"Error importing from main.py: {e}")
    print("Please make sure main.py is in the same directory.")
    sys.exit(1)

def quick_train_model():
    """Quickly train a model for prediction"""
    print("Training loan approval model...")
    
    # Generate data and train model
    loan_data = generate_loan_data(n_samples=3000)  # Smaller dataset for speed
    X, y, encoders, model_data = prepare_data_for_modeling(loan_data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    model_results, trained_models, scaler = train_models(X_train, y_train, X_val, y_val)
    
    # Get best model
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['f1'])
    best_model_info = trained_models[best_model_name]
    best_model = best_model_info['model']
    model_scaler = best_model_info['scaler']
    
    print(f"Model trained successfully! Best model: {best_model_name}")
    
    return best_model, encoders, model_scaler

def predict_single_applicant(model, encoders, scaler=None):
    """Predict loan approval for a single applicant"""
    
    print("\n" + "="*50)
    print("LOAN APPROVAL PREDICTOR")
    print("="*50)
    print("Enter applicant information:")
    
    try:
        # Collect input with validation
        gender = input("\n1. Gender (Male/Female): ").strip().title()
        while gender not in ['Male', 'Female']:
            gender = input("   Please enter 'Male' or 'Female': ").strip().title()
        
        race = input("2. Race (White/Black/Hispanic/Asian/Other): ").strip().title()
        while race not in ['White', 'Black', 'Hispanic', 'Asian', 'Other']:
            race = input("   Please enter White/Black/Hispanic/Asian/Other: ").strip().title()
        
        education = input("3. Education (High School/Some College/Bachelor/Graduate): ").strip().title()
        while education not in ['High School', 'Some College', 'Bachelor', 'Graduate']:
            education = input("   Please enter High School/Some College/Bachelor/Graduate: ").strip().title()
        
        age = int(input("4. Age: "))
        income = float(input("5. Annual Income ($): "))
        credit_score = int(input("6. Credit Score (300-850): "))
        loan_amount = float(input("7. Loan Amount ($): "))
        employment_years = float(input("8. Years of Employment: "))
        debt_to_income = float(input("9. Debt-to-Income Ratio (%): "))
        property_value = float(input("10. Property Value ($): "))
        down_payment_pct = float(input("11. Down Payment (% of property value): "))
        
        # Calculate loan-to-value ratio (handle zero property value)
        if property_value > 0:
            loan_to_value = (loan_amount / property_value) * 100
        else:
            # If property value is 0, assume it's a personal loan (no collateral)
            # Set loan-to-value to 100% to indicate high risk
            loan_to_value = 100.0
            property_value = loan_amount  # Set property value equal to loan for calculations
        
        # Encode categorical variables
        gender_encoded = encoders['gender'].transform([gender])[0]
        race_encoded = encoders['race'].transform([race])[0] 
        education_encoded = encoders['education'].transform([education])[0]
        
        # Create feature vector
        features = np.array([[
            age, income, credit_score, loan_amount, employment_years,
            debt_to_income, loan_to_value, down_payment_pct,
            gender_encoded, race_encoded, education_encoded
        ]])
        
        # Make prediction
        if scaler is not None:
            features = scaler.transform(features)
            
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0, 1]
        
        # Display results
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        
        status = "✅ LIKELY TO BE APPROVED" if prediction == 1 else "❌ LIKELY TO BE REJECTED"
        print(f"\nDecision: {status}")
        print(f"Approval Probability: {probability:.1%}")
        
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
        
        # Quick recommendations
        print(f"\nKey Factors:")
        print(f"  • Credit Score: {credit_score} {'✅' if credit_score >= 650 else '⚠️'}")
        print(f"  • Debt-to-Income: {debt_to_income:.1f}% {'✅' if debt_to_income <= 30 else '⚠️'}")
        print(f"  • Loan-to-Value: {loan_to_value:.1f}% {'✅' if loan_to_value <= 80 else '⚠️'}")
        print(f"  • Income: ${income:,.0f} {'✅' if income >= 40000 else '⚠️'}")
        
        if prediction == 0:
            print(f"\n💡 To improve chances:")
            if credit_score < 650:
                print(f"   • Improve credit score to 650+")
            if debt_to_income > 30:
                print(f"   • Reduce debt-to-income ratio to under 30%")
            if loan_to_value > 80:
                print(f"   • Increase down payment")
                
    except (ValueError, KeyboardInterrupt):
        print("\nInvalid input or cancelled.")
        return

def main():
    """Main function"""
    print("🏦 Loan Approval Prediction Tool")
    print("=" * 40)
    
    # Train model
    model, encoders, scaler = quick_train_model()
    
    while True:
        try:
            predict_single_applicant(model, encoders, scaler)
            
            again = input("\nPredict for another applicant? (y/n): ").strip().lower()
            if again not in ['y', 'yes']:
                break
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
    
    print("\nThank you for using the Loan Approval Predictor!")

if __name__ == "__main__":
    main()
