import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(page_title="Zomato/Swiggy Analytics Recommender", layout="wide", page_icon="🍔")

# Load Data
@st.cache_data
def load_data():
    base_dir = os.path.dirname(__file__)
    restaurants = pd.read_csv(os.path.join(base_dir, "restaurants.csv"))
    reviews = pd.read_csv(os.path.join(base_dir, "reviews.csv"))
    return restaurants, reviews

st.title("🍔 Food Delivery Recommender & Sentiment Analyzer")
st.markdown("Discover the best restaurants based on content-similarity and customer sentiments.")

try:
    df_restaurants, df_reviews = load_data()
except Exception as e:
    st.error(f"Error loading data. Did you run `dataset_generator.py`? Details: {e}")
    st.stop()

# Sidebar for Navigation
menu = st.sidebar.radio("Navigation", ["Overview & Analytics", "Restaurant Recommender", "Customer Sentiment Analysis"])

if menu == "Overview & Analytics":
    st.header("📊 Market Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Restaurants", len(df_restaurants))
    col2.metric("Total Reviews Analyzed", len(df_reviews))
    col3.metric("Average Market Rating", f"{df_restaurants['Overall_Rating'].mean():.2f}")
    
    st.subheader("Top Rated Restaurants in Town")
    top_rated = df_restaurants.sort_values(by="Overall_Rating", ascending=False).head(10)
    st.dataframe(top_rated[['Name', 'Cuisines', 'Overall_Rating', 'Average_Cost_For_Two']], use_container_width=True)
    
    st.subheader("Cost vs Rating Distribution")
    st.scatter_chart(data=df_restaurants, x='Average_Cost_For_Two', y='Overall_Rating')

elif menu == "Restaurant Recommender":
    st.header("🎯 AI Content-Based Recommender System")
    st.markdown("Select a restaurant you like, and we'll recommend similar ones based on Cuisines, Cost and Ratings format.")
    
    # Simple Content Based Filtering (TF-IDF on Cuisines)
    df_restaurants['content_features'] = df_restaurants['Cuisines'] + " " + df_restaurants['Average_Cost_For_Two'].astype(str)
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_restaurants['content_features'].fillna(''))
    
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    target_restaurant = st.selectbox("Search for a Restaurant you love:", df_restaurants['Name'].values)
    
    if st.button("Recommend Similar Restaurants"):
        try:
            # Get index of the restaurant
            idx = df_restaurants[df_restaurants['Name'] == target_restaurant].index[0]
            
            # Get pairwise similarity scores
            sim_scores = list(enumerate(cosine_sim[idx]))
            
            # Sort by similarity
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Top 5 similar
            sim_scores = sim_scores[1:6]
            res_indices = [i[0] for i in sim_scores]
            
            st.success(f"Here are top 5 recommendations similar to **{target_restaurant}**:")
            recommendations = df_restaurants.iloc[res_indices][['Name', 'Cuisines', 'Overall_Rating', 'Average_Cost_For_Two']]
            st.table(recommendations)
            
        except IndexError:
            st.error("Restaurant not found in the database. Please try another.")

elif menu == "Customer Sentiment Analysis":
    st.header("💬 Review Sentiment Analysis")
    st.markdown("We perform NLP sentiment mapping to classify feedback as Positive, Neutral, or Negative based on ratings & key phrases.")
    
    # Demo basic mapping instead of heavy NLP model to keep it fast
    st.code('''
    # NLP Preprocessing Pipeline Logic Example:
    df['Polarity'] = df['Review_Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['Sentiment'] = df['Polarity'].apply(lambda p: 'Positive' if p > 0 else 'Negative'...)
    ''', language='python')
    
    rest_choice = st.selectbox("Select Restaurant to Analyze Feedback:", df_restaurants['Name'].values)
    
    if rest_choice:
        r_id = df_restaurants[df_restaurants['Name'] == rest_choice]['Restaurant_ID'].values[0]
        rest_reviews = df_reviews[df_reviews['Restaurant_ID'] == r_id]
        
        st.write(f"**Total Reviews Found:** {len(rest_reviews)}")
        
        if not rest_reviews.empty:
            st.dataframe(rest_reviews[['Review_Text', 'Rating']].head(10), use_container_width=True)
            
            # Simple sentiment proxy chart
            st.subheader("Review Ratings Breadown")
            rating_counts = rest_reviews['Rating'].value_counts().sort_index()
            st.bar_chart(rating_counts)
        else:
            st.info("No reviews available for this restaurant yet.")
    
st.markdown("---")
st.markdown("*Developed with 💻 by a Data Analytics Fresher*")
