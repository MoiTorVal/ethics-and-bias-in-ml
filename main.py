import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_loan_data(n_samples=5000):
    """Generate synthetic loan approval dataset"""
    
    # Demographics
    genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
    races = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
                           n_samples, p=[0.6, 0.15, 0.15, 0.08, 0.02])
    ages = np.random.normal(40, 12, n_samples).clip(18, 80).astype(int)
    
    # Economic factors
    income = np.random.lognormal(10.5, 0.8, n_samples).clip(20000, 500000)
    credit_scores = np.random.normal(650, 120, n_samples).clip(300, 850).astype(int)
    loan_amounts = np.random.lognormal(11, 0.6, n_samples).clip(5000, 800000)
    employment_years = np.random.exponential(5, n_samples).clip(0, 40)
    debt_to_income = np.random.beta(2, 5, n_samples) * 50  # 0-50%
    
    # Education levels
    education = np.random.choice(['High School', 'Some College', 'Bachelor', 'Graduate'], 
                               n_samples, p=[0.3, 0.25, 0.3, 0.15])
    
    # Property information
    property_values = loan_amounts * np.random.uniform(1.2, 2.0, n_samples)
    down_payment_pct = np.random.beta(2, 3, n_samples) * 30  # 0-30%
    
    # Create DataFrame
    data = pd.DataFrame({
        'gender': genders,
        'race': races,
        'age': ages,
        'income': income,
        'credit_score': credit_scores,
        'loan_amount': loan_amounts,
        'employment_years': employment_years,
        'debt_to_income_ratio': debt_to_income,
        'education': education,
        'property_value': property_values,
        'down_payment_pct': down_payment_pct
    })
    
    # Calculate loan-to-value ratio
    data['loan_to_value'] = (data['loan_amount'] / data['property_value']) * 100
    
    # Generate approval decisions with some bias patterns
    approval_prob = (
        (data['credit_score'] - 300) / 550 * 0.4 +  # Credit score influence
        (data['income'] / 100000) * 0.2 +  # Income influence
        (40 - data['debt_to_income_ratio']) / 40 * 0.2 +  # DTI influence
        (data['employment_years'] / 20) * 0.1 +  # Employment history
        (30 - data['loan_to_value']) / 30 * 0.1  # LTV influence
    )
    
    # Add bias factors (this simulates real-world discrimination)
    bias_adjustment = np.where(data['race'] == 'White', 0.05, 0)
    bias_adjustment += np.where(data['gender'] == 'Male', 0.02, 0)
    bias_adjustment += np.where(data['education'].isin(['Bachelor', 'Graduate']), 0.03, 0)
    
    approval_prob = (approval_prob + bias_adjustment).clip(0, 1)
    
    # Generate binary approval decisions
    data['approved'] = np.random.binomial(1, approval_prob, n_samples)
    
    return data

def analyze_loan_data(data):
    """Analyze loan approval patterns and potential bias"""
    
    print("=== LOAN APPROVAL DATASET ANALYSIS ===\n")
    
    # Basic statistics
    print("Dataset Overview:")
    print(f"Total applications: {len(data):,}")
    print(f"Approved applications: {data['approved'].sum():,} ({data['approved'].mean():.1%})")
    print(f"Rejected applications: {(~data['approved'].astype(bool)).sum():,} ({(1-data['approved'].mean()):.1%})")
    print("\n" + "="*50 + "\n")
    
    # Demographic breakdown
    print("APPROVAL RATES BY DEMOGRAPHICS:")
    print("\nBy Gender:")
    gender_stats = data.groupby('gender')['approved'].agg(['count', 'sum', 'mean']).round(3)
    gender_stats.columns = ['Applications', 'Approved', 'Approval_Rate']
    print(gender_stats)
    
    print("\nBy Race:")
    race_stats = data.groupby('race')['approved'].agg(['count', 'sum', 'mean']).round(3)
    race_stats.columns = ['Applications', 'Approved', 'Approval_Rate']
    print(race_stats.sort_values('Approval_Rate', ascending=False))
    
    print("\nBy Education:")
    edu_stats = data.groupby('education')['approved'].agg(['count', 'sum', 'mean']).round(3)
    edu_stats.columns = ['Applications', 'Approved', 'Approval_Rate']
    print(edu_stats.sort_values('Approval_Rate', ascending=False))
    
    print("\n" + "="*50 + "\n")
    
    # Financial characteristics
    print("FINANCIAL CHARACTERISTICS BY APPROVAL STATUS:")
    financial_comparison = data.groupby('approved')[['income', 'credit_score', 'loan_amount', 
                                                   'debt_to_income_ratio', 'loan_to_value']].mean().round(2)
    financial_comparison.index = ['Rejected', 'Approved']
    print(financial_comparison)
    
    print("\n" + "="*50 + "\n")
    
    # Age analysis
    print("APPROVAL RATES BY AGE GROUPS:")
    data['age_group'] = pd.cut(data['age'], bins=[0, 25, 35, 45, 55, 100], 
                              labels=['18-25', '26-35', '36-45', '46-55', '56+'])
    age_stats = data.groupby('age_group')['approved'].agg(['count', 'mean']).round(3)
    age_stats.columns = ['Applications', 'Approval_Rate']
    print(age_stats)
    
    return data

def create_visualizations(data):
    """Create visualizations for loan approval analysis"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Loan Approval Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Approval rates by race
    race_approval = data.groupby('race')['approved'].mean().sort_values(ascending=True)
    axes[0,0].barh(race_approval.index, race_approval.values, color='skyblue')
    axes[0,0].set_title('Approval Rate by Race')
    axes[0,0].set_xlabel('Approval Rate')
    for i, v in enumerate(race_approval.values):
        axes[0,0].text(v + 0.01, i, f'{v:.1%}', va='center')
    
    # 2. Approval rates by gender
    gender_approval = data.groupby('gender')['approved'].mean()
    axes[0,1].bar(gender_approval.index, gender_approval.values, color=['lightcoral', 'lightblue'])
    axes[0,1].set_title('Approval Rate by Gender')
    axes[0,1].set_ylabel('Approval Rate')
    for i, v in enumerate(gender_approval.values):
        axes[0,1].text(i, v + 0.01, f'{v:.1%}', ha='center')
    
    # 3. Credit score distribution by approval status
    approved_scores = data[data['approved'] == 1]['credit_score']
    rejected_scores = data[data['approved'] == 0]['credit_score']
    axes[0,2].hist([rejected_scores, approved_scores], bins=30, alpha=0.7, 
                   label=['Rejected', 'Approved'], color=['red', 'green'])
    axes[0,2].set_title('Credit Score Distribution')
    axes[0,2].set_xlabel('Credit Score')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].legend()
    
    # 4. Income vs Loan Amount (colored by approval)
    approved_mask = data['approved'] == 1
    axes[1,0].scatter(data[~approved_mask]['income'], data[~approved_mask]['loan_amount'], 
                     alpha=0.5, c='red', label='Rejected', s=10)
    axes[1,0].scatter(data[approved_mask]['income'], data[approved_mask]['loan_amount'], 
                     alpha=0.5, c='green', label='Approved', s=10)
    axes[1,0].set_title('Income vs Loan Amount')
    axes[1,0].set_xlabel('Annual Income ($)')
    axes[1,0].set_ylabel('Loan Amount ($)')
    axes[1,0].legend()
    axes[1,0].ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    # 5. Debt-to-Income ratio by approval
    axes[1,1].boxplot([data[data['approved'] == 0]['debt_to_income_ratio'],
                       data[data['approved'] == 1]['debt_to_income_ratio']], 
                      labels=['Rejected', 'Approved'])
    axes[1,1].set_title('Debt-to-Income Ratio by Approval Status')
    axes[1,1].set_ylabel('Debt-to-Income Ratio (%)')
    
    # 6. Approval rate by education level
    edu_approval = data.groupby('education')['approved'].mean().sort_values(ascending=False)
    axes[1,2].bar(range(len(edu_approval)), edu_approval.values, color='orange')
    axes[1,2].set_title('Approval Rate by Education Level')
    axes[1,2].set_xticks(range(len(edu_approval)))
    axes[1,2].set_xticklabels(edu_approval.index, rotation=45)
    axes[1,2].set_ylabel('Approval Rate')
    for i, v in enumerate(edu_approval.values):
        axes[1,2].text(i, v + 0.01, f'{v:.1%}', ha='center')
    
    plt.tight_layout()
    plt.show()

def bias_analysis(data):
    """Perform detailed bias analysis"""
    
    print("BIAS ANALYSIS REPORT:")
    print("="*50)
    
    # Statistical analysis of approval rates
    from scipy.stats import chi2_contingency
    
    # Gender bias test
    gender_crosstab = pd.crosstab(data['gender'], data['approved'])
    chi2_gender, p_gender, _, _ = chi2_contingency(gender_crosstab)
    print(f"\nGender Bias Analysis:")
    print(f"Chi-square test p-value: {p_gender:.6f}")
    if p_gender < 0.05:
        print("⚠️  Statistically significant difference in approval rates by gender")
    else:
        print("✅ No statistically significant gender bias detected")
    
    # Race bias test
    race_crosstab = pd.crosstab(data['race'], data['approved'])
    chi2_race, p_race, _, _ = chi2_contingency(race_crosstab)
    print(f"\nRace Bias Analysis:")
    print(f"Chi-square test p-value: {p_race:.6f}")
    if p_race < 0.05:
        print("⚠️  Statistically significant difference in approval rates by race")
    else:
        print("✅ No statistically significant racial bias detected")
    
    # Intersectional analysis
    print(f"\nIntersectional Analysis (Race & Gender):")
    intersectional = data.groupby(['race', 'gender'])['approved'].agg(['count', 'mean']).round(3)
    intersectional.columns = ['Applications', 'Approval_Rate']
    print(intersectional[intersectional['Applications'] >= 50])  # Only show groups with sufficient data
    
    # Controlled comparison (similar credit profiles)
    print(f"\nControlled Comparison (Credit Score 650-750, Income $50k-$100k):")
    controlled_sample = data[
        (data['credit_score'].between(650, 750)) & 
        (data['income'].between(50000, 100000))
    ]
    
    if len(controlled_sample) > 100:
        controlled_race = controlled_sample.groupby('race')['approved'].mean().sort_values(ascending=False)
        print("Approval rates in controlled sample:")
        for race, rate in controlled_race.items():
            print(f"  {race}: {rate:.1%}")
    
    return data

def prepare_data_for_modeling(data):
    """Prepare data for machine learning modeling"""
    
    # Create a copy for modeling
    model_data = data.copy()
    
    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    
    le_gender = LabelEncoder()
    le_race = LabelEncoder()
    le_education = LabelEncoder()
    
    model_data['gender_encoded'] = le_gender.fit_transform(model_data['gender'])
    model_data['race_encoded'] = le_race.fit_transform(model_data['race'])
    model_data['education_encoded'] = le_education.fit_transform(model_data['education'])
    
    # Select features for modeling
    feature_columns = [
        'age', 'income', 'credit_score', 'loan_amount', 'employment_years',
        'debt_to_income_ratio', 'loan_to_value', 'down_payment_pct',
        'gender_encoded', 'race_encoded', 'education_encoded'
    ]
    
    X = model_data[feature_columns]
    y = model_data['approved']
    
    # Store encoders for later use
    encoders = {
        'gender': le_gender,
        'race': le_race,
        'education': le_education,
        'feature_names': feature_columns
    }
    
    return X, y, encoders, model_data

def split_data(X, y, random_state=42):
    """Split data into train (60%), validation (20%), and test (20%) sets"""
    
    print("Splitting data into train/validation/test sets...")
    print("- Training set: 60%")
    print("- Validation set: 20%") 
    print("- Test set: 20%")
    
    # First split: 80% train+val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Second split: 60% train (75% of temp), 20% val (25% of temp)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=random_state, stratify=y_temp
    )
    
    print(f"\nDataset split sizes:")
    print(f"- Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"- Validation set: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"- Test set: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_models(X_train, y_train, X_val, y_val):
    """Train multiple models and compare performance"""
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    
    print("\n" + "="*60)
    print("TRAINING MACHINE LEARNING MODELS")
    print("="*60)
    
    # Scale features for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Train and evaluate models
    model_results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for Logistic Regression and SVM
        if name in ['Logistic Regression', 'SVM']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_proba)
        
        model_results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        trained_models[name] = {
            'model': model,
            'scaler': scaler if name in ['Logistic Regression', 'SVM'] else None
        }
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  AUC-ROC: {auc:.3f}")
    
    return model_results, trained_models, scaler

def evaluate_best_model(model_results, trained_models, X_test, y_test, encoders):
    """Evaluate the best performing model on test set"""
    
    print("\n" + "="*60)
    print("MODEL COMPARISON AND FINAL EVALUATION")
    print("="*60)
    
    # Find best model based on F1-score
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['f1'])
    best_model_info = trained_models[best_model_name]
    best_model = best_model_info['model']
    scaler = best_model_info['scaler']
    
    print(f"\nBest performing model: {best_model_name}")
    print(f"Validation F1-Score: {model_results[best_model_name]['f1']:.3f}")
    
    # Test set evaluation
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        y_test_pred = best_model.predict(X_test_scaled)
        y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_test_pred = best_model.predict(X_test)
        y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Final test metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"\nFinal Test Set Performance:")
    print(f"  Accuracy: {test_accuracy:.3f}")
    print(f"  Precision: {test_precision:.3f}")
    print(f"  Recall: {test_recall:.3f}")
    print(f"  F1-Score: {test_f1:.3f}")
    print(f"  AUC-ROC: {test_auc:.3f}")
    
    # Feature importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        print(f"\nTop 5 Most Important Features ({best_model_name}):")
        feature_importance = pd.DataFrame({
            'feature': encoders['feature_names'],
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head().iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    # Confusion Matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"              Predicted")
    print(f"              No    Yes")
    print(f"Actual No   {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Yes  {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    return best_model, best_model_name, {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'auc': test_auc
    }

def analyze_model_bias(best_model, best_model_name, X_test, y_test, model_data, encoders, scaler=None):
    """Analyze potential bias in the trained model"""
    
    print("\n" + "="*60)
    print("MODEL BIAS ANALYSIS")
    print("="*60)
    
    # Get test set predictions
    if best_model_name in ['Logistic Regression', 'SVM'] and scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        test_predictions = best_model.predict(X_test_scaled)
        test_probabilities = best_model.predict_proba(X_test_scaled)[:, 1]
    else:
        test_predictions = best_model.predict(X_test)
        test_probabilities = best_model.predict_proba(X_test)[:, 1]
    
    # Create test results dataframe
    test_indices = X_test.index
    test_results = model_data.loc[test_indices].copy()
    test_results['predicted'] = test_predictions
    test_results['prediction_probability'] = test_probabilities
    
    print("Approval Rate Comparison (Actual vs Predicted):")
    print("\nBy Race:")
    race_comparison = test_results.groupby('race').agg({
        'approved': 'mean',
        'predicted': 'mean'
    }).round(3)
    race_comparison.columns = ['Actual_Rate', 'Predicted_Rate']
    race_comparison['Difference'] = (race_comparison['Predicted_Rate'] - race_comparison['Actual_Rate']).round(3)
    print(race_comparison)
    
    print("\nBy Gender:")
    gender_comparison = test_results.groupby('gender').agg({
        'approved': 'mean',
        'predicted': 'mean'
    }).round(3)
    gender_comparison.columns = ['Actual_Rate', 'Predicted_Rate']
    gender_comparison['Difference'] = (gender_comparison['Predicted_Rate'] - gender_comparison['Actual_Rate']).round(3)
    print(gender_comparison)
    
    # Fairness metrics
    print(f"\nFairness Metrics Analysis:")
    
    # Equal Opportunity (True Positive Rate parity)
    for group_col in ['race', 'gender']:
        print(f"\nEqual Opportunity Analysis by {group_col.title()}:")
        for group in test_results[group_col].unique():
            group_data = test_results[test_results[group_col] == group]
            if len(group_data) > 10:  # Only analyze groups with sufficient data
                # True Positive Rate (Sensitivity/Recall)
                tpr = ((group_data['approved'] == 1) & (group_data['predicted'] == 1)).sum() / (group_data['approved'] == 1).sum()
                print(f"  {group}: True Positive Rate = {tpr:.3f}")
    
    return test_results

def predict_loan_approval(best_model, encoders, scaler=None):
    """Interactive function to predict loan approval for a new applicant"""
    
    print("\n" + "="*60)
    print("LOAN APPROVAL PREDICTION FOR NEW APPLICANT")
    print("="*60)
    print("Please enter the following information:")
    
    try:
        # Collect user input
        print("\n1. Personal Information:")
        gender = input("Gender (Male/Female): ").strip().title()
        while gender not in ['Male', 'Female']:
            print("Please enter 'Male' or 'Female'")
            gender = input("Gender (Male/Female): ").strip().title()
        
        race = input("Race (White/Black/Hispanic/Asian/Other): ").strip().title()
        while race not in ['White', 'Black', 'Hispanic', 'Asian', 'Other']:
            print("Please enter one of: White, Black, Hispanic, Asian, Other")
            race = input("Race (White/Black/Hispanic/Asian/Other): ").strip().title()
        
        education = input("Education (High School/Some College/Bachelor/Graduate): ").strip().title()
        while education not in ['High School', 'Some College', 'Bachelor', 'Graduate']:
            print("Please enter one of: High School, Some College, Bachelor, Graduate")
            education = input("Education (High School/Some College/Bachelor/Graduate): ").strip().title()
        
        age = int(input("Age: "))
        while age < 18 or age > 80:
            print("Please enter an age between 18 and 80")
            age = int(input("Age: "))
        
        print("\n2. Financial Information:")
        income = float(input("Annual Income ($): "))
        while income < 0:
            print("Please enter a positive income")
            income = float(input("Annual Income ($): "))
        
        credit_score = int(input("Credit Score (300-850): "))
        while credit_score < 300 or credit_score > 850:
            print("Please enter a credit score between 300 and 850")
            credit_score = int(input("Credit Score (300-850): "))
        
        loan_amount = float(input("Requested Loan Amount ($): "))
        while loan_amount <= 0:
            print("Please enter a positive loan amount")
            loan_amount = float(input("Requested Loan Amount ($): "))
        
        employment_years = float(input("Years of Employment: "))
        while employment_years < 0:
            print("Please enter a non-negative number of years")
            employment_years = float(input("Years of Employment: "))
        
        debt_to_income_ratio = float(input("Debt-to-Income Ratio (%): "))
        while debt_to_income_ratio < 0 or debt_to_income_ratio > 100:
            print("Please enter a percentage between 0 and 100")
            debt_to_income_ratio = float(input("Debt-to-Income Ratio (%): "))
        
        property_value = float(input("Property Value ($): "))
        while property_value <= 0:
            print("Please enter a positive property value")
            property_value = float(input("Property Value ($): "))
        
        down_payment_pct = float(input("Down Payment Percentage (%): "))
        while down_payment_pct < 0 or down_payment_pct > 100:
            print("Please enter a percentage between 0 and 100")
            down_payment_pct = float(input("Down Payment Percentage (%): "))
        
    except ValueError:
        print("Invalid input. Please enter numeric values where required.")
        return
    except KeyboardInterrupt:
        print("\nPrediction cancelled.")
        return
    
    # Calculate derived features (handle zero property value)
    if property_value > 0:
        loan_to_value = (loan_amount / property_value) * 100
    else:
        # If property value is 0, assume it's a personal loan (no collateral)
        # Set loan-to-value to 100% to indicate high risk
        loan_to_value = 100.0
        property_value = loan_amount  # Set property value equal to loan for calculations
    
    # Create feature vector
    try:
        # Encode categorical variables
        gender_encoded = encoders['gender'].transform([gender])[0]
        race_encoded = encoders['race'].transform([race])[0]
        education_encoded = encoders['education'].transform([education])[0]
        
        # Create feature array in the same order as training
        features = np.array([[
            age, income, credit_score, loan_amount, employment_years,
            debt_to_income_ratio, loan_to_value, down_payment_pct,
            gender_encoded, race_encoded, education_encoded
        ]])
        
        # Make prediction
        if scaler is not None:
            features_scaled = scaler.transform(features)
            prediction = best_model.predict(features_scaled)[0]
            probability = best_model.predict_proba(features_scaled)[0, 1]
        else:
            prediction = best_model.predict(features)[0]
            probability = best_model.predict_proba(features)[0, 1]
        
        # Display results
        print("\n" + "="*60)
        print("LOAN APPROVAL PREDICTION RESULTS")
        print("="*60)
        
        print(f"\nApplicant Profile:")
        print(f"  Name: {gender} {race} applicant")
        print(f"  Age: {age}")
        print(f"  Education: {education}")
        print(f"  Annual Income: ${income:,.2f}")
        print(f"  Credit Score: {credit_score}")
        print(f"  Requested Loan: ${loan_amount:,.2f}")
        print(f"  Property Value: ${property_value:,.2f}")
        print(f"  Loan-to-Value Ratio: {loan_to_value:.1f}%")
        print(f"  Debt-to-Income Ratio: {debt_to_income_ratio:.1f}%")
        print(f"  Employment Years: {employment_years:.1f}")
        print(f"  Down Payment: {down_payment_pct:.1f}%")
        
        print(f"\nPrediction Results:")
        approval_status = "APPROVED" if prediction == 1 else "REJECTED"
        print(f"  Decision: {approval_status}")
        print(f"  Approval Probability: {probability:.1%}")
        
        # Risk assessment
        if probability >= 0.7:
            risk_level = "Low Risk"
            risk_color = "🟢"
        elif probability >= 0.5:
            risk_level = "Medium Risk"
            risk_color = "🟡"
        elif probability >= 0.3:
            risk_level = "High Risk"
            risk_color = "🟠"
        else:
            risk_level = "Very High Risk"
            risk_color = "🔴"
        
        print(f"  Risk Assessment: {risk_color} {risk_level}")
        
        # Provide recommendations
        print(f"\nRecommendations:")
        if prediction == 0:  # Rejected
            print("  To improve approval chances:")
            if credit_score < 650:
                print(f"    • Improve credit score (currently {credit_score}, target 650+)")
            if debt_to_income_ratio > 30:
                print(f"    • Reduce debt-to-income ratio (currently {debt_to_income_ratio:.1f}%, target <30%)")
            if loan_to_value > 80:
                print(f"    • Increase down payment to reduce loan-to-value ratio (currently {loan_to_value:.1f}%)")
            if income < 50000:
                print(f"    • Consider increasing income or co-applicant")
        else:  # Approved
            print("  Congratulations! Your application is likely to be approved.")
            if probability < 0.8:
                print("  Consider:")
                if down_payment_pct < 20:
                    print(f"    • Increasing down payment for better terms")
                if credit_score < 750:
                    print(f"    • A higher credit score could qualify for better interest rates")
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        print("Please ensure all inputs are valid.")

def interactive_mode(best_model, encoders, scaler=None):
    """Run interactive prediction mode"""
    
    print("\n" + "="*60)
    print("INTERACTIVE LOAN APPROVAL PREDICTION")
    print("="*60)
    
    while True:
        try:
            print("\nOptions:")
            print("1. Predict loan approval for new applicant")
            print("2. Exit")
            
            choice = input("\nEnter your choice (1 or 2): ").strip()
            
            if choice == '1':
                predict_loan_approval(best_model, encoders, scaler)
                
                # Ask if user wants to try another prediction
                another = input("\nWould you like to make another prediction? (y/n): ").strip().lower()
                if another not in ['y', 'yes']:
                    break
                    
            elif choice == '2':
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

def main():
    """Main function to run the loan approval analysis"""
    
    # Generate synthetic loan data
    print("Generating synthetic loan approval dataset...")
    loan_data = generate_loan_data(n_samples=5000)
    
    # Perform basic analysis
    loan_data = analyze_loan_data(loan_data)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(loan_data)
    
    # Perform bias analysis
    bias_analysis(loan_data)
    
    # Prepare data for modeling
    X, y, encoders, loan_data_prepared = prepare_data_for_modeling(loan_data)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Train models
    model_results, trained_models, scaler = train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate best model
    best_model, best_model_name, test_metrics = evaluate_best_model(model_results, trained_models, X_test, y_test, encoders)
    
    # Create detailed evaluation plots and reports
    print("\nGenerating detailed evaluation plots...")
    final_best_model_name, y_test_pred, y_test_proba = create_detailed_evaluation_plots(
        model_results, trained_models, X_test, y_test, encoders
    )
    
    # Print comprehensive evaluation report
    confusion_matrix_result = print_detailed_evaluation_report(
        model_results, final_best_model_name, y_test, y_test_pred, y_test_proba
    )
    
    # Analyze model bias
    test_results = analyze_model_bias(best_model, best_model_name, X_test, y_test, loan_data_prepared, encoders, scaler)
    
    # Save data for further analysis
    loan_data.to_csv('loan_approval_data.csv', index=False)
    test_results.to_csv('model_predictions.csv', index=False)
    
    print(f"\n✅ Original dataset saved as 'loan_approval_data.csv'")
    print(f"✅ Model predictions saved as 'model_predictions.csv'")
    print(f"Dataset contains {len(loan_data):,} loan applications")
    print(f"Best model: {best_model_name} (Test F1-Score: {test_metrics['f1']:.3f})")
    
    return loan_data, best_model, test_results

# Function to create detailed evaluation plots and reports
def create_detailed_evaluation_plots(model_results, trained_models, X_test, y_test, encoders):
    """
    Create comprehensive evaluation plots including ROC curves, confusion matrices,
    and detailed metrics for all models.
    
    Returns:
        tuple: (best_model_name, y_test_pred, y_test_proba) for the best performing model
    """
    print("\n" + "="*80)
    print("DETAILED MODEL EVALUATION WITH VISUALIZATIONS")
    print("="*80)
    
    # Find the best model
    best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['accuracy'])
    best_model = trained_models[best_model_name]['model']
    
    # Create subplots for visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Model Evaluation Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Model Accuracy Comparison
    model_names = list(model_results.keys())
    accuracies = [model_results[name]['accuracy'] for name in model_names]
    
    axes[0, 0].bar(range(len(model_names)), accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_xticks(range(len(model_names)))
    axes[0, 0].set_xticklabels(model_names, rotation=45)
    axes[0, 0].set_ylim(0, 1)
    
    # Add accuracy values on bars
    for i, acc in enumerate(accuracies):
        axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    # 2. ROC Curves for all models
    axes[0, 1].set_title('ROC Curves - All Models')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    roc_data = {}
    
    for i, (name, model_info) in enumerate(trained_models.items()):
        model = model_info['model']
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
        
        color = colors[i % len(colors)]
        axes[0, 1].plot(fpr, tpr, color=color, lw=2, 
                       label=f'{name} (AUC = {roc_auc:.3f})')
    
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].legend(loc='lower right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confusion Matrix for Best Model
    best_y_pred = best_model.predict(X_test)
    best_y_proba = best_model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, best_y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
    axes[0, 2].set_title(f'Confusion Matrix - {best_model_name}')
    axes[0, 2].set_xlabel('Predicted')
    axes[0, 2].set_ylabel('Actual')
    
    # 4. Precision, Recall, F1-Score Comparison
    metrics = ['Precision', 'Recall', 'F1-Score']
    x_pos = np.arange(len(model_names))
    
    precision_scores = [model_results[name]['precision'] for name in model_names]
    recall_scores = [model_results[name]['recall'] for name in model_names]
    f1_scores = [model_results[name]['f1'] for name in model_names]
    
    width = 0.25
    axes[1, 0].bar(x_pos - width, precision_scores, width, label='Precision', alpha=0.8)
    axes[1, 0].bar(x_pos, recall_scores, width, label='Recall', alpha=0.8)
    axes[1, 0].bar(x_pos + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision, Recall, and F1-Score Comparison')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(model_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1)
    
    # 5. Feature Importance (for Random Forest)
    if best_model_name == 'Random Forest' and hasattr(best_model, 'feature_importances_'):
        feature_names = X_test.columns
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Take top 10 features
        top_features = min(10, len(feature_names))
        axes[1, 1].bar(range(top_features), importances[indices[:top_features]])
        axes[1, 1].set_title(f'Top {top_features} Feature Importances - {best_model_name}')
        axes[1, 1].set_xlabel('Features')
        axes[1, 1].set_ylabel('Importance')
        axes[1, 1].set_xticks(range(top_features))
        axes[1, 1].set_xticklabels([feature_names[i] for i in indices[:top_features]], rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'Feature Importance\nNot Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Feature Importance')
    
    # 6. Prediction Probability Distribution
    axes[1, 2].hist(best_y_proba[y_test == 0], alpha=0.7, label='Rejected', bins=20, color='red')
    axes[1, 2].hist(best_y_proba[y_test == 1], alpha=0.7, label='Approved', bins=20, color='green')
    axes[1, 2].set_xlabel('Prediction Probability')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title(f'Prediction Probability Distribution - {best_model_name}')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()
    
    return best_model_name, best_y_pred, best_y_proba

def print_detailed_evaluation_report(model_results, best_model_name, y_test, y_test_pred, y_test_proba):
    """
    Print comprehensive evaluation report with business insights.
    """
    print("\n" + "="*80)
    print("DETAILED EVALUATION REPORT")
    print("="*80)
    
    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    
    print(f"\nBEST MODEL: {best_model_name}")
    print("-" * 40)
    
    # Detailed metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    
    print(f"\nCONFUSION MATRIX BREAKDOWN:")
    print("-" * 40)
    print(f"True Positives (Correctly Approved):  {tp:4d}")
    print(f"True Negatives (Correctly Rejected):  {tn:4d}")
    print(f"False Positives (Wrongly Approved):   {fp:4d}")
    print(f"False Negatives (Wrongly Rejected):   {fn:4d}")
    
    # Business Impact Analysis
    print(f"\nBUSINESS IMPACT ANALYSIS:")
    print("-" * 40)
    
    total_predictions = len(y_test)
    approval_rate = (tp + fp) / total_predictions
    rejection_rate = (tn + fn) / total_predictions
    
    print(f"Total Loan Applications Evaluated: {total_predictions}")
    print(f"Model Approval Rate: {approval_rate:.2%}")
    print(f"Model Rejection Rate: {rejection_rate:.2%}")
    
    # Risk Assessment
    if fp > 0:
        default_risk = fp / (tp + fp)
        print(f"Estimated Default Risk: {default_risk:.2%} (based on false positives)")
    
    if fn > 0:
        missed_opportunity = fn / (tn + fn)
        print(f"Missed Business Opportunity: {missed_opportunity:.2%} (based on false negatives)")
    
    # Model Reliability
    print(f"\nMODEL RELIABILITY:")
    print("-" * 40)
    
    if precision >= 0.8:
        print("✓ HIGH PRECISION: Model is reliable for approvals")
    elif precision >= 0.6:
        print("⚠ MODERATE PRECISION: Review approval decisions carefully")
    else:
        print("✗ LOW PRECISION: High risk of approving bad loans")
    
    if recall >= 0.8:
        print("✓ HIGH RECALL: Model captures most good loan candidates")
    elif recall >= 0.6:
        print("⚠ MODERATE RECALL: Missing some good loan opportunities")
    else:
        print("✗ LOW RECALL: Missing many good loan opportunities")
    
    print(f"\nRECOMMENDations:")
    print("-" * 40)
    
    if precision < 0.7:
        print("• Consider raising approval threshold to reduce false positives")
    if recall < 0.7:
        print("• Consider lowering approval threshold to capture more good loans")
    if roc_auc < 0.8:
        print("• Model performance could be improved with more features or data")
    if fp > fn:
        print("• Focus on reducing false positives to minimize default risk")
    elif fn > fp:
        print("• Focus on reducing false negatives to maximize business opportunities")
    
    return {
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'metrics': {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc},
        'business_insights': {'approval_rate': approval_rate, 'rejection_rate': rejection_rate}
    }

if __name__ == "__main__":
    # Run the analysis
    loan_data, best_model, test_results = main()
    
    # Display sample data
    print("\nSample of the original dataset:")
    print("="*60)
    sample_columns = ['gender', 'race', 'age', 'income', 'credit_score', 
                     'loan_amount', 'debt_to_income_ratio', 'approved']
    print(loan_data[sample_columns].head(10).to_string(index=False))
    
    print("\nSample of model predictions on test set:")
    print("="*60)
    prediction_columns = ['gender', 'race', 'credit_score', 'income', 'approved', 'predicted', 'prediction_probability']
    print(test_results[prediction_columns].head(10).to_string(index=False, float_format='%.3f'))
    
    # Get the encoders and scaler from the analysis
    X, y, encoders, model_data = prepare_data_for_modeling(loan_data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    model_results, trained_models, scaler = train_models(X_train, y_train, X_val, y_val)
    
    # Get the best model info
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['f1'])
    best_model_info = trained_models[best_model_name]
    model_scaler = best_model_info['scaler']
    
    # Start interactive mode
    interactive_mode(best_model, encoders, model_scaler)