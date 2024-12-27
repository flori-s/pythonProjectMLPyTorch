# data_loader.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to load data from CSV files
def load_data(cv_path, vacature_path, labels_path):
    # Check if the files exist
    if not os.path.exists(cv_path):
        raise FileNotFoundError(f"File not found: {cv_path}")
    if not os.path.exists(vacature_path):
        raise FileNotFoundError(f"File not found: {vacature_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"File not found: {labels_path}")

    # Read the CSV files into DataFrames
    cv_df = pd.read_csv(cv_path)
    vacature_df = pd.read_csv(vacature_path)
    labels_df = pd.read_csv(labels_path)
    return cv_df, vacature_df, labels_df


# Function to prepare features for the model
def prepare_features(cv_df, vacature_df, labels_df):
    # Combine relevant text fields for vectorization
    cv_df['combined_text'] = cv_df['Functie'] + " " + cv_df['Vaardigheden'] + " " + cv_df['Opleiding']
    vacature_df['combined_text'] = vacature_df['Functietitel'] + " " + vacature_df['Vaardigheden'] + " " + vacature_df['Opleiding']

    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer()
    all_text = pd.concat([cv_df['combined_text'], vacature_df['combined_text']])
    tfidf.fit(all_text)

    # Transform the combined text fields into TF-IDF matrices
    cv_tfidf = tfidf.transform(cv_df['combined_text']).toarray()
    vacature_tfidf = tfidf.transform(vacature_df['combined_text']).toarray()

    # Convert cv_id and vacature_id to integers
    labels_df['cv_id'] = labels_df['cv_id'].astype(int)
    labels_df['vacature_id'] = labels_df['vacature_id'].astype(int)

    # Create feature combinations based on TF-IDF vectors
    features = []
    for _, row in labels_df.iterrows():
        cv_id = int(row['cv_id']) - 1  # Adjust index to 0-based
        vacature_id = int(row['vacature_id']) - 1  # Adjust index to 0-based
        combined_features = cv_tfidf[cv_id] + vacature_tfidf[vacature_id]
        features.append(combined_features)

    # Create DataFrame for features and Series for labels
    X = pd.DataFrame(features)
    y = labels_df['similarity_score'].values
    return X, y