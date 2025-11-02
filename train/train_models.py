import pandas as pd
import pickle
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# -------------------------------
# 1. Load cleaned datasets
# -------------------------------
ratings_file = "clean_ratings.csv"
movies_file = "clean_movies.csv"

ratings = pd.read_csv(ratings_file)
movies = pd.read_csv(movies_file)

print(f"Loaded cleaned ratings: {ratings.shape}")
print(f"Loaded cleaned movies: {movies.shape}")

# -------------------------------
# 2. Prepare Surprise Dataset
# -------------------------------
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# -------------------------------
# 3. Collaborative Filtering (SVD)
# -------------------------------
print("🔹 Training SVD (Collaborative Filtering)...")
svd_model = SVD(n_factors=100, random_state=42, verbose=True)
svd_model.fit(trainset)

# Evaluate SVD
predictions = svd_model.test(testset)
print("✅ SVD Evaluation Metrics:")
rmse = accuracy.rmse(predictions, verbose=True)
mae = accuracy.mae(predictions, verbose=True)

# Save SVD model
with open("svd_model.pkl", "wb") as f:
    pickle.dump(svd_model, f)

# -------------------------------
# 4. Content-Based Filtering
# -------------------------------
print("🔹 Training Content-Based Model (TF-IDF + NearestNeighbors)...")

# Combine text features
movies["content"] = movies["title"] + " " + movies["genres"]

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(movies["content"])

# Nearest Neighbors
nn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
nn_model.fit(tfidf_matrix)

# Save Content-based models
with open("tfidf_model.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("nn_model.pkl", "wb") as f:
    pickle.dump(nn_model, f)

with open("movies_meta.pkl", "wb") as f:
    pickle.dump(movies[["movieId", "title"]], f)

print("✅ Content-Based Model Trained and Saved!")

# -------------------------------
# 5. Save Metrics
# -------------------------------
metrics = {
    "SVD_RMSE": rmse,
    "SVD_MAE": mae,
    "n_users": ratings["userId"].nunique(),
    "n_movies": ratings["movieId"].nunique(),
    "n_ratings": len(ratings)
}

with open("model_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

print("\n🎉 All models trained, evaluated, and saved successfully!")
print("Models saved: svd_model.pkl, tfidf_model.pkl, nn_model.pkl, movies_meta.pkl")
print("Metrics saved: model_metrics.pkl")
