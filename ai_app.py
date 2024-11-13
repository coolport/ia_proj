import pandas as pd
import os
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

test = os.getcwd()
# print(test)
# Load CSV, only handle 5k rows for safety/testing
newmil = pd.read_csv("/home/aidan/millionsongs_improved/Music Info.csv")
newmil_sample = newmil.head(5000)
# print(newmil.head(1))  # print format for reference

# PRE-PROCESSING
newmil_sample.dropna(subset=["tags"], inplace=True)
newmil_sample["tags"] = newmil_sample["tags"].str.lower()

# VECTORIZATION
# TfidfVectorizer to convert genre tags into a TFIDF matrix
tfidf_vectorizer = TfidfVectorizer(analyzer="word", stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(newmil_sample["tags"])

# CALCULATE SIMILARITIES
similarity_matrix = cosine_similarity(tfidf_matrix)


# RECOMMENDATION FUNCTION
def recommend_song(song_name):
    # Get the index of the song that matches the title
    idx = newmil_sample[newmil_sample["name"] == song_name].index[0]

    # Get the pairwise similarity scores of all songs with that song
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # Sort the songs based on similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar songs (excluding the first one which is the song itself)
    sim_scores = sim_scores[1:11]  # Get top 10 excluding itself

    # Get the song indices
    song_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar songs
    return newmil_sample["name"].iloc[song_indices]


recommendations = recommend_song("Rape Me")
print(recommendations)
