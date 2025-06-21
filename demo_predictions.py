#!/usr/bin/env python3
"""
Demo script showing different loan approval scenarios
"""

import subprocess
import sys
import os

def run_prediction_example(name, inputs):
    """Run a prediction example with given inputs"""
    print(f"\n{'='*60}")
    print(f"EXAMPLE: {name}")
    print('='*60)
    
    # Convert inputs to string format for echo command
    input_string = "\\n".join(inputs) + "\\nn"  # Add 'n' to not continue
    
    cmd = f'cd /Users/shmoi/Documents/ethics-and-bias-in-ml && echo -e "{input_string}" | /Users/shmoi/Documents/ethics-and-bias-in-ml/.venv/bin/python predict_loan.py'
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        # Only print the prediction results part
        output_lines = result.stdout.split('\n')
        
        # Find the start of applicant input and results
        start_printing = False
        for line in output_lines:
            if "LOAN APPROVAL PREDICTOR" in line:
                start_printing = True
            elif "Thank you for using" in line:
                break
            
            if start_printing:
                print(line)
                
    except Exception as e:
        print(f"Error running example: {e}")

def main():
    """Run several example predictions"""
    
    print("🏦 LOAN APPROVAL PREDICTION EXAMPLES")
    print("="*60)
    print("This demo shows how the loan approval predictor works with different scenarios.")
    
    # Example 1: High-quality applicant
    run_prediction_example(
        "High-Quality Applicant (Should be approved)",
        [
            "Male",           # Gender
            "White",          # Race  
            "Bachelor",       # Education
            "35",             # Age
            "85000",          # Income
            "750",            # Credit Score
            "250000",         # Loan Amount
            "10",             # Employment Years
            "20",             # Debt-to-Income Ratio
            "350000",         # Property Value
            "25"              # Down Payment %
        ]
    )
    
    # Example 2: Risky applicant
    run_prediction_example(
        "High-Risk Applicant (Likely rejected)",
        [
            "Female",         # Gender
            "Black",          # Race
            "High School",    # Education  
            "25",             # Age
            "35000",          # Income
            "580",            # Credit Score
            "200000",         # Loan Amount
            "2",              # Employment Years
            "45",             # Debt-to-Income Ratio
            "220000",         # Property Value
            "5"               # Down Payment %
        ]
    )
    
    # Example 3: Borderline case
    run_prediction_example(
        "Borderline Applicant (Could go either way)",
        [
            "Male",           # Gender
            "Hispanic",       # Race
            "Some College",   # Education
            "42",             # Age
            "55000",          # Income
            "650",            # Credit Score
            "180000",         # Loan Amount
            "6",              # Employment Years
            "32",             # Debt-to-Income Ratio
            "250000",         # Property Value
            "15"              # Down Payment %
        ]
    )
    
    print(f"\n{'='*60}")
    print("EXAMPLES COMPLETED")
    print('='*60)
    print("To run your own predictions, use:")
    print("  python predict_loan.py")
    print("or")
    print("  python main.py")
    print("and choose option 1 when prompted.")

if __name__ == "__main__":
    main()
