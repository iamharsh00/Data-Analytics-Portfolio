# E-Commerce Customer Churn Prediction 🛒📉

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.1.0-150458.svg)

## 📌 Project Overview
Customer retention is one of the most critical KPIs for any E-Commerce platform. This project builds an End-to-End Machine Learning pipeline to predict whether a customer will **churn** (stop purchasing) based on their demographic data, platform engagement, and RFM (Recency, Frequency, Monetary) metrics.

### 🎯 Business Objective:
- Identify high-risk customers before they churn.
- Discover the key drivers of customer churn (e.g., Support Tickets, Satisfaction Scores).
- Provide actionable insights for targeted marketing and retention campaigns.

---

## 🏗️ Project Structure
```text
📦 1_Ecommerce_Churn_Prediction
 ┣ 📜 dataset_generator.py          # Script generating a highly realistic 5000-user dataset
 ┣ 📜 ecommerce_churn_data.csv      # Executed synthetic dataset
 ┣ 📜 churn_analysis_pipeline.py    # Main End-to-End EDA & ML modeling script
 ┣ 📜 requirements.txt              # Project dependencies
 ┣ 📜 random_forest_churn_model.pkl # Trained Machine Learning Model (Ready for deployment)
 ┗ 📂 visualisations                # Generated EDA & Feature Importance Charts
   ┣ 🖼️ churn_distribution.png
   ┣ 🖼️ churn_by_gender.png
   ┣ 🖼️ correlation_heatmap.png
   ┣ 🖼️ churn_vs_satisfaction.png
   ┗ 🖼️ feature_importance.png
```

## 🛠️ Tech Stack & Skills
* **Languages & Libraries:** Python (Pandas, NumPy, Scikit-Learn)
* **Data Visualization:** Matplotlib, Seaborn
* **Algorithms utilized:** Random Forest Classifier (with `class_weight='balanced'`)
* **Core Analytics Concepts:** Exploratory Data Analysis, Data Imputation, Standard Scaling, One-Hot Encoding, Feature Importance Extraction.

---

## 📊 Key Insights (From EDA)
1. **Recency is King:** Customers who haven't made a purchase recently have the highest probability of churning.
2. **Support Ticket Impact:** A high number of support tickets strongly correlates with low satisfaction scores and subsequently, churn.

## 🚀 How to Run the Project
1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **(Optional) Generate fresh data:**
   ```bash
   python dataset_generator.py
   ```
3. **Run the EDA and Model Pipeline:**
   ```bash
   python churn_analysis_pipeline.py
   ```
   *This will output the classification metrics to the console, save visualizations to the `/visualisations` folder, and export the trained model as a `.pkl` file.*

## 🏆 Model Evaluation
The optimized Random Forest model achieves strong predictive capabilities, specifically prioritizing the detection of the minority "churned" class using balanced weighting.
* **Metric Monitored:** ROC-AUC Score & F1-Score (To handle class imbalance naturally present in churn datasets).
