# ==============================================================================
# E-commerce Customer Churn Prediction & Analysis Pipeline
# Author: Fresher Data Analyst Profile
# Description: This script performs End-to-End data analysis, preprocessing, 
#              and trains a Random Forest model to predict Customer Churn.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_data(file_path):
    print("Loading Dataset...")
    try:
        df = pd.read_csv(file_path)
        print(f"Data Loaded Successfully! Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please run dataset_generator.py first.")
        return None

def perform_eda(df, output_dir):
    print("\nStarting Exploratory Data Analysis (EDA)...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Churn Distribution
    plt.figure()
    sns.countplot(x='Churn', data=df, palette='Set2')
    plt.title('Customer Churn Distribution')
    plt.savefig(os.path.join(output_dir, "churn_distribution.png"))
    plt.close()
    
    # 2. Gender vs Churn
    plt.figure()
    sns.countplot(x='Gender', hue='Churn', data=df, palette='Set1')
    plt.title('Churn by Gender')
    plt.savefig(os.path.join(output_dir, "churn_by_gender.png"))
    plt.close()
    
    # 3. Numeric Features Correlation Heatmap
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numeric Features')
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()
    
    # 4. Satisfaction Score vs Churn
    plt.figure()
    sns.countplot(x='Satisfaction_Score', hue='Churn', data=df, palette='viridis')
    plt.title('Churn vs Satisfaction Score')
    plt.savefig(os.path.join(output_dir, "churn_vs_satisfaction.png"))
    plt.close()
    
    print(f"EDA visualisations saved to '{output_dir}/'")

def train_churn_model(df):
    print("\nPreparing Data for Modeling...")
    # Drop identifier columns
    df = df.drop(columns=['CustomerID'])
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Identify numeric and categorical columns
    numeric_features = ['Age', 'Tenure_Months', 'App_Usage_Time_Mins', 'Recency_Days', 
                        'Purchase_Frequency', 'Total_Spending', 'Satisfaction_Score', 'Support_Tickets']
    categorical_features = ['Gender', 'Device_Type']
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Append classifier to preprocessing pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Training shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    print("\nTraining Random Forest Model...")
    pipeline.fit(X_train, y_train)
    
    print("\nEvaluating Model...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print("\n--- MODEL PERFORMANCE ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature Importance Extraction
    model = pipeline.named_steps['classifier']
    categorical_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = categorical_encoder.get_feature_names_out(categorical_features)
    
    all_feature_names = numeric_features + list(cat_feature_names)
    importances = model.feature_importances_
    
    feature_imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(10), palette='magma')
    plt.title('Top 10 Feature Importances in Churn Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "visualisations", "feature_importance.png"))
    plt.close()
    print("Feature importance plot saved.")
    
    return pipeline

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "ecommerce_churn_data.csv")
    viz_dir = os.path.join(current_dir, "visualisations")
    
    # 1. Load Data
    df = load_data(data_path)
    
    if df is not None:
        # 2. Perform EDA
        perform_eda(df, viz_dir)
        
        # 3. Train Model
        model_pipeline = train_churn_model(df)
        
        # 4. Save Model for deployment/portfolio
        model_path = os.path.join(current_dir, "random_forest_churn_model.pkl")
        joblib.dump(model_pipeline, model_path)
        print(f"\nModel successfully saved to {model_path}")
        print("\n--- END OF PIPELINE ---")
