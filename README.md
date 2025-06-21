# Loan Approval Prediction Tool - Usage Guide

## 🏦 Interactive Loan Approval Prediction

You now have **two ways** to predict loan approvals for individual applicants:

### Method 1: Full Analysis + Interactive Mode

```bash
python main.py
```

This will:

1. Run the complete loan dataset analysis
2. Train machine learning models with 60/20/20 train/validation/test split
3. Show bias analysis results
4. Launch interactive prediction mode

When prompted with:

```
Options:
1. Predict loan approval for new applicant
2. Exit

Enter your choice (1 or 2):
```

Choose **1** to enter applicant details.

### Method 2: Quick Prediction Tool

```bash
python predict_loan.py
```

This is a faster option that:

1. Quickly trains a model (smaller dataset for speed)
2. Goes straight to interactive prediction

## 📝 Input Information Required

When predicting for a new applicant, you'll be asked for:

### Personal Information:

- **Gender**: Male or Female
- **Race**: White, Black, Hispanic, Asian, or Other
- **Education**: High School, Some College, Bachelor, or Graduate
- **Age**: 18-80 years

### Financial Information:

- **Annual Income**: In dollars (e.g., 75000)
- **Credit Score**: 300-850 range
- **Requested Loan Amount**: In dollars
- **Years of Employment**: Number of years
- **Debt-to-Income Ratio**: Percentage (0-100)
- **Property Value**: In dollars (enter 0 for personal loans without collateral)
- **Down Payment Percentage**: Percentage of property value (enter 0 if no down payment)

## 📊 Example Scenarios

### High-Quality Applicant (Likely Approved):

- Male, White, Bachelor's degree, Age 35
- Income: $85,000, Credit Score: 750
- Loan: $250,000, Property: $350,000
- Employment: 10 years, DTI: 20%, Down Payment: 25%

### High-Risk Applicant (Likely Rejected):

- Female, Black, High School, Age 25
- Income: $35,000, Credit Score: 580
- Loan: $200,000, Property: $220,000
- Employment: 2 years, DTI: 45%, Down Payment: 5%

### Borderline Case:

- Male, Hispanic, Some College, Age 42
- Income: $55,000, Credit Score: 650
- Loan: $180,000, Property: $250,000
- Employment: 6 years, DTI: 32%, Down Payment: 15%

### Personal Loan (No Collateral):

- Female, Asian, Bachelor's degree, Age 29
- Income: $45,000, Credit Score: 720
- Loan: $25,000, Property: $0 (personal loan)
- Employment: 5 years, DTI: 15%, Down Payment: 0%

## ⚠️ Important Notes

### The tool demonstrates bias patterns found in real lending:

- **Racial disparities**: White applicants typically get higher approval rates
- **Gender differences**: Male applicants may have slight advantages
- **Education bias**: Higher education correlates with approvals

### The model provides:

- ✅/❌ **Approval Decision**
- **Probability percentage** (0-100%)
- **Risk Level** (🟢 Low to 🔴 High)
- **Key factor analysis**
- **Improvement recommendations**

## 🔄 Running Multiple Predictions

Both tools allow you to make multiple predictions in a session. After each prediction, you'll be asked if you want to predict for another applicant.

## 📁 Files Generated

The analysis creates:

- `loan_approval_data.csv` - Original synthetic dataset (5,000 records)
- `model_predictions.csv` - Test set with model predictions

## 🎯 Try It Now!

1. Open terminal in the project directory
2. Run `python predict_loan.py` for quick predictions
3. Or run `python main.py` for full analysis + predictions
4. Follow the prompts to enter applicant information
5. Get instant loan approval predictions with explanations!
