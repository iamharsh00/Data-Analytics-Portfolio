# Food Delivery Recommender & Sentiment Analysis 🍔📊

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.26.0-FF4B4B.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)

## 📌 Project Overview
This project simulates the environment of massive food delivery aggregators like Swiggy, Zomato, or UberEats. It takes restaurant datasets and user reviews, processes the Natural Language parameters, and applies an AI Recommender Engine.

### 🎯 Business Objective:
- Help users discover new restaurants based on preferences (Cuisine, Price point).
- Evaluate customer feedback computationally (NLP Sentiment evaluation conceptualization).
- Understand restaurant market distributions.

---

## 🏗️ Project Structure
```text
📦 2_Food_Recommender_System
 ┣ 📜 restaurant_data_generator.py  # Script generating synthetic restaurants & review datasets
 ┣ 📜 restaurants.csv               # Synthetic Restaurants Metadata
 ┣ 📜 reviews.csv                   # Synthetic Customer Reviews
 ┣ 📜 app.py                        # Streamlit Frontend Web App with ML engine
 ┣ 📜 requirements.txt              # Project dependencies
 ┗ 📜 README.md                     # Documentation
```

## 🛠️ Tech Stack & Skills
* **Languages & Analytics:** Python (Pandas, NumPy)
* **Web Framework:** Streamlit (For rapid analytical dashboard deployment)
* **Machine Learning / NLP:** Scikit-Learn (TF-IDF Vectorizer, Cosine Similarity)

## 🚀 How to Run the App
1. **Ensure dependencies are installed:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch the Streamlit interface:**
   ```bash
   streamlit run app.py
   ```
   *This will open the highly interactive web dashboard in your default browser.*

## 🏆 Key Features & AI Logic
1. **Content-Based Recommender:** Converts textual features (Cuisines and Cost tags) into numerical TF-IDF matrices, then computes the `Cosine Similarity` between restaurants to surface the 5 most statistically similar options.
2. **Review Aggregation & Dashboards:** Clean, interactive UI metrics that quickly display market summaries without needing complex BI software.

### 🧠 The Math Behind the Magic:
The system calculates the angle between multi-dimensional restaurant vectors:
$$\text{Similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
High similarity scores (closer to 1.0) indicate that two restaurants share high-overlap in cuisine and pricing profiles, ensuring relevant recommendations.

### 🏢 Industry Application:
This architecture mirrors production systems used to solve the **"Cold Start" problem** in recommendation engines, where collaborative filtering (user-item) is unavailable for new users or items.
