# 🎬 Hybrid Movie Recommendation System

An **AI-based movie recommendation system** that combines **Content-Based Filtering (CBF)** and **Collaborative Filtering (CF)** using machine learning techniques.  
The system delivers personalized movie suggestions by analyzing user preferences, movie metadata, and social interactions.  
It uses **XAMPP** for database management (MySQL) and **Streamlit** for an interactive web-based interface.

---

## 🚀 Features

- 🔍 **Hybrid Recommendation Engine:** Combines content-based and collaborative models for improved accuracy.  
- 🧠 **Content-Based Filtering:** Uses TF-IDF vectorization and cosine similarity on movie metadata.  
- 🤝 **Collaborative Filtering:** Implements Singular Value Decomposition (SVD) on user–movie ratings.  
- 🗄️ **Database Integration:** MySQL database hosted locally using **XAMPP**.  
- 🎨 **Interactive Web App:** Built with Streamlit for a clean and responsive UI.  
- 👥 **Social Recommendation System:**  
  - Follow other users.  
  - View movies your friends have **searched** or **watched (rated)**.  
  - Get inspiration from friends’ activities.  
- ⚡ **Efficient Performance:** Generates recommendations in under 3 seconds.

---

## 🧰 Tech Stack

| Category | Technology |
|-----------|-------------|
| Programming Language | Python |
| Libraries | Scikit-Learn, Pandas, NumPy, Surprise, SQLAlchemy |
| Web Framework | Streamlit |
| Database | MySQL (via XAMPP) |
| Algorithms | TF-IDF, Cosine Similarity, SVD |
| Tools | VS Code, Git, Jupyter Notebook, XAMPP |

---

## 🎞️ Dataset Information

This project uses a combination of **MovieLens** and **TMDb (The Movie Database)** datasets to train and evaluate the hybrid recommendation model.

---

### 🎬 1️⃣ MovieLens Dataset
**Source:** [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/)  

- Contains user ratings for movies on a scale of 0.5 to 5.0  
- Files used:
  - `movies.csv` – Movie IDs, titles, and genres  
  - `ratings.csv` – User–movie rating matrix  
  - `tags.csv` – User-assigned tags  
  - `links.csv` – Mapping between MovieLens and TMDb/IMDb IDs  

**Purpose:**  
Used to train the **Collaborative Filtering** model based on **Singular Value Decomposition (SVD)** for learning user preferences and predicting ratings.

---

### 🎞️ 2️⃣ TMDb 5000 Movie Dataset
**Source:** [https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)  

- Provides detailed movie metadata such as:
  - Overview (plot summary)
  - Cast and crew details
  - Genres, keywords, production companies, and popularity  
- Files used:
  - `tmdb_5000_movies.csv`
  - `tmdb_5000_credits.csv`

**Purpose:**  
Used for **Content-Based Filtering**, where metadata features like genres, keywords, and overview are vectorized using **TF-IDF** and compared using **cosine similarity**.

---

### 🧹 Data Preprocessing Steps
- Removed duplicates, null values, and irrelevant columns  
- Cleaned text data (lowercasing, punctuation removal, lemmatization)  
- Merged MovieLens and TMDb datasets using common IDs  
- Computed **TF-IDF matrix** for movie descriptions  
- Calculated the **cosine similarity matrix** and saved it for fast lookup

---

### 🧠 Model Artifacts

| File | Purpose |
|------|----------|
| **movie_data.pkl** | Stores the **Content-Based Filtering model data** — includes TF-IDF vectors and cosine similarity matrix |
| **svd_model.pkl** | Stores the **Collaborative Filtering model** trained using SVD |
| **clean_movies.csv** | Cleaned and merged movie metadata |
| **clean_ratings.csv** | Filtered user–movie ratings used for training |

---

## 🔑 TMDb API Configuration

The **content-based model** is trained using static TMDb datasets (`tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`),  
while the **TMDb API** is used at runtime to fetch **movie posters** and **real-time metadata** for display in the web app.

---

### 🧭 Steps to Get an API Key

1. Go to [https://www.themoviedb.org/](https://www.themoviedb.org/)  
2. Create a free account and verify your email.  
3. Navigate to **Settings → API → Request an API Key**.  
4. Once approved, you’ll receive an **API key** (a string like `abcd1234efgh5678`).

---

### ⚙️ Configure Your API Key in `app.py`

1. Open your project folder and locate the file **`app.py`**.  
2. Scroll to the **“POSTER FETCH”** section — you will find a line like this:

   ```python
   api_key = "YOUR_TMDB_API_KEY"

### ⚙️ Configure Your API Key

1. In your project folder, create a new file named:
1. Open your project folder and locate the file **`app.py`**.  
2. Scroll to the **“POSTER FETCH”** section — you will find a line like this:

   ```python
   api_key = "YOUR_TMDB_API_KEY"
   ```
3. Replace "YOUR_TMDB_API_KEY" with your actual TMDb API key.

## ⚙️ Installation and Setup
### 1️⃣ Clone the repository
```bash
git clone https://github.com/<YOUR_USERNAME>/Hybrid-Movie-Recommendation-System.git
cd Hybrid-Movie-Recommendation-System
```
### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # On Windows
# OR
source venv/bin/activate   # On macOS/Linux
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🗄️ XAMPP and MySQL Setup

### 1️⃣ Install XAMPP  
Download and install **XAMPP** from [https://www.apachefriends.org](https://www.apachefriends.org),  
then start the **Apache** and **MySQL** modules from the XAMPP Control Panel.

---

### 2️⃣ Open phpMyAdmin  
Go to [http://localhost/phpmyadmin](http://localhost/phpmyadmin)

---

### 3️⃣ Create a Database  
Create a new database named: movie_recommender

---

### 4️⃣ Import SQL File  
Import the SQL schema file: 
movie_recommender.sql

(found in your project directory)

---

### 5️⃣ Configure Database Connection  
Update your database credentials in your Python or Streamlit app  
(usually in `app.py` or a configuration file):

```python
host = "localhost"
user = "root"
password = ""
database = "movie_recommender"
```

---

### 6️⃣ Populate the Database
Run the following command to insert data into your MySQL tables:
```bash
python populate_movies.py
```
### ▶️ Run the Streamlit App
After setting up the database and virtual environment, start your application with:
```bash
streamlit run app.py
```
Then open the displayed local URL in your browser, for example:
```arduino
http://localhost:8501
```
---
## 🧠 How It Works

### 🎬 Content-Based Filtering (CBF)
- Uses **TF-IDF vectorization** on movie metadata (overview, genres, keywords, etc.).  
- Computes **cosine similarity** to find movies similar to the user’s choice.

---

### 🤝 Collaborative Filtering (CF)
- Uses **Singular Value Decomposition (SVD)** to analyze the user–movie rating matrix.  
- Predicts **unknown ratings** based on similar users.

---

### ⚙️ Hybrid Model
- Combines both approaches using a **weighted average** of their scores for higher accuracy.  
- Balances the strengths of both methods to improve recommendation quality.

---

### 💻 Interface
- Built with **Streamlit**, providing an intuitive, web-based interface.  
- Integrates with **MySQL (via XAMPP)** to manage movie and user data efficiently.  
- Displays recommendations with **movie posters, genres, and overviews** in real-time.

---
## 🧠 How It Works

### 🎬 Content-Based Filtering (CBF)

- Uses **TF-IDF vectorization** on movie metadata (overview, genres, keywords, etc.).  
- Computes **cosine similarity** to find movies similar to the user’s choice.

---

### 🤝 Collaborative Filtering (CF)

- Uses **Singular Value Decomposition (SVD)** on user–movie ratings.  
- Predicts **unknown ratings** based on similar users.

---

### ⚙️ Hybrid Model

- Combines both models using a **weighted average** of their scores for higher accuracy.  
- Balances the strengths of both methods to improve recommendation quality.

---

### 💻 Interface and Social Features

- Built with **Streamlit**, providing an intuitive, web-based interface.  
- Integrates with **MySQL (via XAMPP)** for user, movie, and rating data.  
- Displays **movie posters, genres, and overviews** in real time.  
- Includes **social interaction tabs**:

#### 👥 Follow System
- Users can **register/login** and **follow other users**.  
- Followed users appear under the **“Following”** section in the sidebar.

#### 🔎 Friends Searched Tab
- Displays the latest movies your **friends have searched for**.  
- Encourages movie discovery based on your friends’ activity.

#### ⭐ Friends Watched Tab
- Displays movies your **friends have watched and rated**.  
- Allows you to **rate the same movies** directly from this tab.

---
# 🪪 License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.






