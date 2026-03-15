import pandas as pd
import numpy as np
import random
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_ecommerce_data(num_records=5000):
    print(f"Generating synthetic e-commerce dataset with {num_records} records...")
    
    # 1. Basic Customer Info
    customer_ids = [f"C{str(i).zfill(5)}" for i in range(1, num_records + 1)]
    genders = np.random.choice(["Male", "Female", "Other"], size=num_records, p=[0.48, 0.48, 0.04])
    ages = np.random.normal(loc=35, scale=12, size=num_records).astype(int)
    ages = np.clip(ages, 18, 75)
    
    # 2. Platform Usage & Engagement
    tenure_months = np.random.randint(1, 60, size=num_records)
    device_type = np.random.choice(["Mobile App", "Desktop", "Tablet"], size=num_records, p=[0.7, 0.25, 0.05])
    app_time_mins = np.random.normal(loc=120, scale=45, size=num_records)
    app_time_mins = np.clip(app_time_mins, 10, 500).round(1)
    
    # 3. RFM Features (Recency, Frequency, Monetary)
    recency_days = np.random.exponential(scale=60, size=num_records).astype(int) # Increased scale for more recency
    recency_days = np.clip(recency_days, 1, 365)
    
    purchase_frequency = np.random.poisson(lam=3, size=num_records) + 1 # Min 1 purchase
    
    # Base Monetary Value on frequency
    monetary_value = purchase_frequency * np.random.uniform(50, 500, size=num_records).round(2)
    
    # 4. Customer Experience
    satisfaction_score = np.random.choice([1, 2, 3, 4, 5], size=num_records, p=[0.2, 0.2, 0.2, 0.2, 0.2]) # Equal probability
    support_tickets = np.random.poisson(lam=2, size=num_records) # More support tickets
    
    for i in range(num_records):
        if support_tickets[i] > 3:
            satisfaction_score[i] = max(1, satisfaction_score[i] - 1)
            
    # 5. Define Churn Logic (Target Variable)
    churn_probability = np.zeros(num_records)
    
    churn_probability += (recency_days / 365) * 0.45  # Recency risk
    churn_probability += (1 - (satisfaction_score / 5)) * 0.35 # Satisfaction risk
    churn_probability += (support_tickets / 10) * 0.25 # Support risk
    churn_probability -= (tenure_months / 60) * 0.15 # Tenure loyalty
    
    churn_probability += np.random.normal(0, 0.1, size=num_records)
    
    # Threshold for churn to achieve ~25% churn rate
    churn_status = (churn_probability > 0.25).astype(int)
    
    df = pd.DataFrame({
        "CustomerID": customer_ids,
        "Age": ages,
        "Gender": genders,
        "Tenure_Months": tenure_months,
        "Device_Type": device_type,
        "App_Usage_Time_Mins": app_time_mins,
        "Recency_Days": recency_days,
        "Purchase_Frequency": purchase_frequency,
        "Total_Spending": monetary_value,
        "Satisfaction_Score": satisfaction_score,
        "Support_Tickets": support_tickets,
        "Churn": churn_status
    })
    
    missing_idx = np.random.choice(df.index, size=int(num_records * 0.02), replace=False)
    df.loc[missing_idx, 'Age'] = np.nan
    
    output_path = os.path.join(os.path.dirname(__file__), "ecommerce_churn_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path}")
    print(f"Total Churn Rate: {(df['Churn'].mean() * 100):.2f}%")
    
if __name__ == "__main__":
    generate_ecommerce_data()
