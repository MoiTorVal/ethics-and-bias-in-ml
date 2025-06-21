import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
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
    
    # Save data for further analysis
    loan_data.to_csv('loan_approval_data.csv', index=False)
    print(f"\n✅ Dataset saved as 'loan_approval_data.csv'")
    print(f"Dataset contains {len(loan_data):,} loan applications")
    
    return loan_data

if __name__ == "__main__":
    # Run the analysis
    loan_data = main()
    
    # Display sample data
    print("\nSample of the dataset:")
    print("="*50)
    sample_columns = ['gender', 'race', 'age', 'income', 'credit_score', 
                     'loan_amount', 'debt_to_income_ratio', 'approved']
    print(loan_data[sample_columns].head(10).to_string(index=False))