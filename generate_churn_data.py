import pandas as pd
import numpy as np

def generate_churn_data(num_samples=5000):
    np.random.seed(42)

    # 1. Generate Features
    # IDs
    user_ids = range(1, num_samples + 1)
    
    # Age: Normal distribution around 35, clipped to [18, 80]
    ages = np.random.normal(35, 10, num_samples).astype(int)
    ages = np.clip(ages, 18, 80)
    
    # Gender: 0 (Male), 1 (Female) - roughly 50/50
    genders = np.random.randint(0, 2, num_samples)
    
    # Purchase Count (Last 3 months): Poisson distribution
    # Loyal customers buy more.
    purchase_counts = np.random.poisson(3, num_samples)
    
    # Days Since Last Login: Exponential-ish distribution
    # Many people login recently, fewer people haven't logged in for a long time.
    days_since_login = np.random.exponential(30, num_samples).astype(int)
    days_since_login = np.clip(days_since_login, 0, 365)
    
    # Average Transaction Amount: Lognormal
    avg_transaction = np.random.lognormal(10, 0.5, num_samples) # Around 22000
    
    # Claim Count: Poisson, usually low
    claim_counts = np.random.poisson(0.5, num_samples)
    
    # 2. Determine Churn Probability (Imposing correlations)
    # Goal: 
    # - High days_since_login -> High Churn
    # - High claim_count -> High Churn
    # - High purchase_count -> Low Churn
    # - High avg_transaction -> Low Churn
    
    # Normalize features for calculation roughly to 0-1 scale impact
    norm_login = days_since_login / 365.0
    norm_claims = claim_counts / 5.0
    norm_purchase = purchase_counts / 10.0
    norm_transaction = avg_transaction / 100000.0
    
    # Probability Formula (Logits)
    # Base churn risk - Higher base to ensure some churn
    logits = -1.5 
    
    # Penalties (Risk factors)
    logits += 3.5 * norm_login # Strong factor: If long time no login, high chance
    logits += 2.0 * norm_claims 
    
    # Benefits (Retention factors)
    # Reduced impact of purchase to allow more churns even if they bought something
    logits -= 1.5 * norm_purchase 
    logits -= 0.5 * norm_transaction
    
    # Random Noise
    logits += np.random.normal(0, 1.0, num_samples) # More noise
    
    # Sigmoid to probability
    probs = 1 / (1 + np.exp(-logits))
    
    # 3. Assign Labels based on probability
    # If prob > 0.5, Churn (1). 
    churn_labels = (probs > 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'User_ID': user_ids,
        'Age': ages,
        'Gender': genders,
        'Purchase_Count_3M': purchase_counts,
        'Days_Since_Last_Login': days_since_login,
        'Avg_Transaction_Amt': avg_transaction.astype(int),
        'Claim_Count': claim_counts,
        'Churn': churn_labels
    })
    
    return df

if __name__ == "__main__":
    print("Generating data...")
    df = generate_churn_data(10000)
    
    # Check Imbalance
    churn_rate = df['Churn'].mean()
    print(f"Data Generated: {len(df)} samples")
    print(f"Churn Rate: {churn_rate:.2%}")
    print(df['Churn'].value_counts())
    
    file_path = 'shopping_mall_churn.csv'
    df.to_csv(file_path, index=False)
    print(f"Saved to {file_path}")
