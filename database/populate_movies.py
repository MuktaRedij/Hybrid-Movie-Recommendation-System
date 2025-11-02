# populate_movies_id_title.py
import pandas as pd
from sqlalchemy import create_engine, text

# ---------------- CONFIG ----------------
CSV_PATH = "D:\\Projects\\Movie-Recommendation-System-main\\data\\tmdb_5000_movies.csv"  # your CSV path
DB_URL = "mysql+pymysql://root@localhost:3306/movie_recommender"
engine = create_engine(DB_URL)

# ---------------- READ CSV ----------------
df = pd.read_csv(CSV_PATH, usecols=["movie_id", "title"])  # only load movie_id & title
print(f"Loaded {len(df)} movies from CSV")

# ---------------- CREATE TABLE IF NOT EXISTS ----------------
with engine.begin() as conn:
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS movies (
        movie_id INT PRIMARY KEY,
        title VARCHAR(255)
    )
    """))

# ---------------- INSERT / UPDATE ----------------
with engine.begin() as conn:
    for _, row in df.iterrows():
        conn.execute(text("""
            INSERT INTO movies (movie_id, title)
            VALUES (:movie_id, :title)
            ON DUPLICATE KEY UPDATE
                title=:title
        """), {
            "movie_id": int(row["movie_id"]),
            "title": row["title"]
        })

print("✅ All movies inserted/updated successfully")
