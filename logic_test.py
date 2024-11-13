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
# newmil_sample["name"] = newmil_sample["name"].str.lower()

# VECTORIZATION
# TfidfVectorizer to convert genre tags into a TFIDF matrix

tfidf_vectorizer = TfidfVectorizer(analyzer="word", stop_words="english")
tfidf_matrix = tfidf_vectorizer.fit_transform(newmil_sample["tags"])

# CALCULATE SIMILARITIES

similarity_matrix = cosine_similarity(tfidf_matrix)

# RECOMMENDATION FUNCTION


def recommend_song(song_name):
    # handle cases
    song_name = song_name.lower()
    newmil_sample["lower_name"] = newmil_sample["name"].str.lower()

    # Check if the song exists in the dataset
    if song_name in newmil_sample["lower_name"].values:
        # Get the index of the matching song
        idx = newmil_sample[newmil_sample["lower_name"] == song_name].index[0]

        # Compute similarity scores
        sim_scores = list(enumerate(similarity_matrix[idx]))

        # Sort the scores and exclude the input song itself
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

        # Get the song indices
        song_indices = [i[0] for i in sim_scores]

        # Return the top 10 most similar songs
        return newmil_sample["name"].iloc[song_indices]
    else:
        return None


while True:
    song_name = input("Enter song title: ").strip()
    recommendations = recommend_song(song_name)

    if recommendations is not None:
        print("Recommended songs:")
        print(recommendations)
        break
    else:
        print("Song not found. Please try again.")