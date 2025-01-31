import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load dataset
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data[['name', 'positive_ratings', 'negative_ratings', 'average_playtime', 'genres', 'categories', 'price', 'owners', 'platforms', 'steamspy_tags', 'short_description']]

file_path = "fix_steam_data.csv"
data = load_data(file_path)

# Feature Engineering
data1 = data[['name', 'positive_ratings', 'negative_ratings', 'average_playtime', 'genres', 'categories', 'owners', 'platforms']]
data1 = data1.dropna()
data1['owners'] = data1['owners'].astype(str).str.replace(',', '').str.replace('-', ' ').str.split().str[0].fillna(0).astype(int)
data1 = data1[(data1['positive_ratings'] + data1['negative_ratings'] >= 100) & (data1['average_playtime'] > 0) & (data1['owners'] > 0)]

# Compute satisfaction score
data1['satisfaction_score'] = (
    (data1['positive_ratings'] / (data1['positive_ratings'] + data1['negative_ratings'])) * 0.6 +
    (data1['average_playtime'] / data1['average_playtime'].max()) * 0.4
) * np.log1p(data1['owners'])

# Precompute content-based features
@st.cache_data
def compute_tfidf_matrix(text_features):
    tfidf = TfidfVectorizer(stop_words='english')
    return tfidf.fit_transform(text_features)

@st.cache_data
def compute_cosine_similarity(_matrix):
    return cosine_similarity(_matrix)

# Content-Based Filtering
data1['text_features'] = data1['genres'] + ' ' + data1['categories'] + ' ' + data1['platforms']
tfidf_matrix = compute_tfidf_matrix(data1['text_features'])
cosine_sim_cbf = compute_cosine_similarity(tfidf_matrix)

# Function for content-based filtering
def content_based_filtering(data1, game_name, top_n=5):
    indices = pd.Series(data1.index, index=data1['name']).drop_duplicates()
    if game_name not in indices:
        return None
    
    idx = indices[game_name]
    sim_scores = list(enumerate(cosine_sim_cbf[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    game_indices = [i[0] for i in sim_scores]
    return data1.iloc[game_indices][['name', 'genres', 'categories', 'platforms']]

# Precompute count vectorizer features
data['combined_features'] = data['name'] + ' ' + data['genres'] + ' ' + data['categories'] + ' ' + data['steamspy_tags'] + ' ' + data['short_description']
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(data['combined_features'])
cosine_sim_count = compute_cosine_similarity(count_matrix)

# Function for count vectorizer recommendations
def get_game_recommendations(game_name, top_n=5):
    try:
        idx = data.index[data['name'] == game_name].tolist()[0]
    except IndexError:
        return None
    
    sim_scores = list(enumerate(cosine_sim_count[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    game_indices = [i[0] for i in sim_scores]
    return data[['name', 'genres', 'categories']].iloc[game_indices]

# Streamlit UI
st.title("Game Recommender System 🎮")
st.write("Masukkan nama game untuk mendapatkan rekomendasi berdasarkan Content-Based Filtering dan Count Vectorizer.")

game_input = st.text_input("Masukkan nama game: ")

if game_input:
    st.subheader("Rekomendasi Berdasarkan Content-Based Filtering")
    cbf_results = content_based_filtering(data1, game_input)
    if cbf_results is not None:
        st.dataframe(cbf_results)
    else:
        st.write(f"Game '{game_input}' tidak ditemukan dalam dataset.")

    st.subheader("Rekomendasi Berdasarkan Count Vectorizer")
    count_results = get_game_recommendations(game_input)
    if count_results is not None:
        st.dataframe(count_results)
    else:
        st.write(f"Game '{game_input}' tidak ditemukan dalam dataset.")
