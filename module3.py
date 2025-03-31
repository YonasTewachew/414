import pandas as pd
import numpy as np
import re
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

def extract_genres(genres_str):
    try:
        genres_list = ast.literal_eval(genres_str)
        genre_names = [genre['name'].lower() for genre in genres_list]
        return " ".join(genre_names)
    except Exception as e:
        return ""

df = pd.read_csv("tmdb_5000_movies.csv")
df['clean_overview'] = df['overview'].apply(clean_text)
df['clean_genres'] = df['genres'].apply(extract_genres)
df['combined_text'] = df['clean_overview'] + " " + df['clean_genres']

tfidf = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['combined_text'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):
    try:
        idx = indices[title]
    except KeyError:
        raise KeyError(f"Movie '{title}' not found in the dataset. Please check the title or update the query.")
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[movie_indices][['title']]
    recommendations = recommendations.copy()
    recommendations['similarity'] = [i[1] for i in sim_scores]
    return recommendations

query_films = ["Pan's Labyrinth", "Eternal Sunshine of the Spotless Mind", "Ex Machina"]

for film in query_films:
    print(f"\nTop 10 films similar to '{film}':")
    try:
        recs = get_recommendations(film)
        print(recs.to_string(index=False))
    except KeyError as e:
        print(e)
