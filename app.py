# app.py
import streamlit as st
import pandas as pd
import pickle
import requests
from sqlalchemy import create_engine, text
from surprise import SVD

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🎬 Hybrid Movie Recommender", layout="wide")

# ---------------- SESSION STATE ----------------
if "_rerun" not in st.session_state:
    st.session_state["_rerun"] = False

def rerun_app():
    st.session_state["_rerun"] = not st.session_state["_rerun"]

# ---------------- DATABASE ----------------
DB_URL = "mysql+pymysql://root@localhost:3306/movie_recommender"
def get_db_engine():
    return create_engine(DB_URL)

# ---------------- LOAD DATA / MODELS ----------------
@st.cache_resource
def load_movies():
    engine = get_db_engine()
    return pd.read_sql(text("SELECT * FROM movies"), engine)

@st.cache_resource
def load_content_based():
    # movie_data.pkl expected to contain (movies_cb_df, cosine_sim_matrix)
    with open("movie_data.pkl", "rb") as f:
        movies_cb, cosine_sim = pickle.load(f)
    return movies_cb, cosine_sim

@st.cache_resource
def load_svd_model():
    # load pre-trained SVD (surprise) model
    with open("svd_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_clean_ratings():
    # local csv for dataset; used for building testsets
    return pd.read_csv("clean_ratings.csv")

# Global cached objects
movies = load_movies()
movies_cb, cosine_sim = load_content_based()
svd_model = load_svd_model()
ratings_df_local = load_clean_ratings()

# ---------------- POSTER FETCH ----------------
@st.cache_data
def fetch_poster(movie_id):
    api_key = "YOUR_TMDB_API_KEY"  # replace with your TMDB API key
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
    try:
        r = requests.get(url, timeout=3)
        data = r.json()
        if "poster_path" in data and data["poster_path"]:
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    except Exception:
        pass
    return "https://via.placeholder.com/150"

# ---------------- DISPLAY ----------------
def display_movies(df, show_meta=True):
    if df is None or df.empty:
        st.info("No movies to display.")
        return
    for i in range(0, len(df), 5):
        cols = st.columns(5)
        for col, j in zip(cols, range(i, i + 5)):
            if j < len(df):
                row = df.iloc[j]
                with col:
                    st.image(fetch_poster(row["movie_id"]), width=130)
                    if show_meta and "rating" in row and "username" in row:
                        st.markdown(f"{row['title']}<br>Rated {row['rating']} by *{row['username']}*",
                                    unsafe_allow_html=True)
                    elif show_meta and "searched_by" in row:
                        st.markdown(f"{row['title']}<br>Searched by *{row['searched_by']}*",
                                    unsafe_allow_html=True)
                    else:
                        st.caption(row["title"])

# ---------------- CONTENT-BASED UTILS ----------------
def get_content_recommendations(title, top_n=10):
    if title not in movies_cb['title'].values:
        return pd.DataFrame(columns=['title', 'movie_id'])
    idx = movies_cb.index[movies_cb['title'] == title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    indices = [i[0] for i in sim_scores]
    return movies_cb.iloc[indices][['title', 'movie_id']]

def get_all_other_usernames(current_username):
    engine = get_db_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT username FROM users WHERE username != :u"), {"u": current_username}).fetchall()
    return [r[0] for r in rows]

def get_following(user_id):
    engine = get_db_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT u.username
            FROM friends f
            JOIN users u ON f.friend_id = u.user_id
            WHERE f.user_id = :uid
        """), {"uid": user_id}).fetchall()
    return [r[0] for r in rows]

# ---------------- LOGIN / REGISTER ----------------
st.sidebar.title("Login / Register")

if "user" not in st.session_state:
    st.session_state.user = None

username = st.sidebar.text_input("Username", key="sid_login_username")
email = st.sidebar.text_input("Email", key="sid_login_email")

if not st.session_state.user and st.sidebar.button("Login / Register", key="sid_login_btn"):
    engine = get_db_engine()
    with engine.begin() as conn:  # ensures commit
        user_row = conn.execute(
            text("SELECT * FROM users WHERE username=:u OR email=:e"),
            {"u": username, "e": email}
        ).fetchone()

        if not user_row:
            conn.execute(
                text("INSERT INTO users (username, email) VALUES (:u, :e)"),
                {"u": username, "e": email}
            )
            user_row = conn.execute(
                text("SELECT * FROM users WHERE username=:u"),
                {"u": username}
            ).fetchone()
            st.success(f"✅ New user added: {username}")
        else:
            st.success(f"👋 Welcome back, {user_row[1]}!")

    st.session_state.user = {"id": user_row[0], "username": user_row[1], "email": user_row[2]}
    rerun_app()

elif st.session_state.user:
    st.sidebar.write(f"Logged in as: {st.session_state.user['username']}")
    if st.sidebar.button("Logout", key="sid_logout_btn"):
        st.session_state.user = None
        rerun_app()

# ---------------- FOLLOW SYSTEM ----------------
if st.session_state.user:
    engine = get_db_engine()
    all_users = get_all_other_usernames(st.session_state.user["username"])
    selected_friend = st.sidebar.selectbox("Follow a user", ["Select..."] + all_users, key="sid_follow_select")
    if st.sidebar.button("Follow", key="sid_follow_btn"):
        if selected_friend != "Select...":
            with engine.begin() as conn:
                friend_id = conn.execute(text("SELECT user_id FROM users WHERE username=:u"), {"u": selected_friend}).fetchone()[0]
                conn.execute(text("INSERT IGNORE INTO friends (user_id, friend_id) VALUES (:uid, :fid)"),
                             {"uid": st.session_state.user["id"], "fid": friend_id})
            st.sidebar.success(f"You are now following {selected_friend}")
            rerun_app()
        else:
            st.sidebar.warning("Please select a valid user to follow.")

    following = get_following(st.session_state.user["id"])
    st.sidebar.subheader("Following:")
    if following:
        st.sidebar.write(", ".join(following))
    else:
        st.sidebar.caption("You are not following anyone yet.")

# ---------------- MAIN UI ----------------
st.title("🎬 Hybrid Movie Recommender")
if not st.session_state.user:
    st.info("Please log in to access recommendations.")
    st.stop()

engine = get_db_engine()
user_id = st.session_state.user["id"]

display_movies(movies.sample(min(10, len(movies))))

tab_cb, tab_cf, tab_hybrid, tab_fsearch, tab_fwatched = st.tabs([
    "Content-Based", "Collaborative", "Hybrid", "Friends Searched", "Friends Watched"
])

# ---------------- TAB 1: CONTENT-BASED ----------------
with tab_cb:
    st.subheader("🎯 Content-Based Recommendations")
    search_term = st.text_input("Search movies by name", key="cb_search_input")
    filtered = movies_cb[movies_cb['title'].str.contains(search_term, case=False, na=False)] if search_term else movies_cb.copy()
    selected_title = st.selectbox("Select a movie", filtered['title'].tolist() if not filtered.empty else [], key="cb_selectbox") if not filtered.empty else None

    if selected_title and st.button("Get Similar Movies", key="cb_get_sim_btn"):
        similar = get_content_recommendations(selected_title, top_n=10)
        display_movies(similar)
        movie_id = int(movies_cb[movies_cb['title'] == selected_title]['movie_id'].values[0])
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT IGNORE INTO user_search_history (user_id, movie_id, search_time) VALUES (:uid, :mid, NOW())"
            ), {"uid": user_id, "mid": movie_id})
        rerun_app()

# ---------------- TAB 2: COLLABORATIVE ----------------
with tab_cf:
    st.subheader("🧠 Collaborative Recommendations (SVD)")
    ratings_df = pd.read_sql(text("SELECT user_id AS userId, movie_id AS movieId, rating FROM ratings"), engine)
    rated_by_user = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    unrated_df = movies[~movies['movie_id'].isin(rated_by_user)]

    st.markdown("### Rate More Movies")
    if not unrated_df.empty:
        sel_to_rate = st.multiselect("Pick movies to rate", unrated_df['title'].tolist(), key="cf_rate_multiselect")
        for title in sel_to_rate:
            slider_key = f"cf_slider_{title}"
            btn_key = f"cf_btn_{title}"
            rating_val = st.slider(f"Rate '{title}'", 1, 5, 3, key=slider_key)
            if st.button(f"Submit rating for '{title}'", key=btn_key):
                mid = int(unrated_df[unrated_df['title'] == title]['movie_id'].values[0])
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO ratings (user_id, movie_id, rating, rating_time)
                        VALUES (:uid, :mid, :rating, NOW())
                        ON DUPLICATE KEY UPDATE rating=:rating, rating_time=NOW()
                    """), {"uid": user_id, "mid": mid, "rating": rating_val})
                st.success(f"Rated '{title}' {rating_val} ★")
                rerun_app()

    user_ratings_db = ratings_df[ratings_df['userId'] == user_id]
    if not user_ratings_db.empty:
        unseen = movies[~movies['movie_id'].isin(user_ratings_db['movieId'])]
        if not unseen.empty:
            testset = [(user_id, mid, 0) for mid in unseen['movie_id']]
            preds = svd_model.test(testset)
            top_preds = sorted(preds, key=lambda x: x.est, reverse=True)[:10]
            top_ids = [p.iid for p in top_preds]
            recs = movies[movies['movie_id'].isin(top_ids)]
            st.markdown("### Top Collaborative Picks For You")
            display_movies(recs)
        else:
            st.info("You have rated all available movies!")
    else:
        st.info("Rate some movies (above) to enable collaborative recommendations.")

# ---------------- TAB 3: HYBRID ----------------
with tab_hybrid:
    st.subheader("🔀 Hybrid Recommendations (Content + Collaborative)")
    search_term_h = st.text_input("Pick a movie to seed hybrid recommendations", key="hybrid_search_input")
    filtered_h = movies_cb[movies_cb['title'].str.contains(search_term_h, case=False, na=False)] if search_term_h else movies_cb.copy()
    seed_title = st.selectbox("Seed movie (content-based)", filtered_h['title'].tolist() if not filtered_h.empty else [], key="hybrid_seed_select") if not filtered_h.empty else None

    if st.button("Get Hybrid Top 10", key="hybrid_get_btn"):
        ratings_df_live = pd.read_sql(text("SELECT user_id AS userId, movie_id AS movieId, rating FROM ratings"), engine)
        user_rated_list = ratings_df_live[ratings_df_live['userId'] == user_id]['movieId'].tolist()
        unseen_movies = movies[~movies['movie_id'].isin(user_rated_list)]

        if unseen_movies.empty:
            st.info("Rate more movies to enable hybrid recommendations.")
        else:
            testset = [(user_id, mid, 0) for mid in unseen_movies['movie_id']]
            preds = svd_model.test(testset)
            cf_scores = {p.iid: p.est for p in preds}

            cb_scores = {}
            if seed_title and seed_title in movies_cb['title'].values:
                seed_idx = movies_cb.index[movies_cb['title'] == seed_title][0]
                for mid in unseen_movies['movie_id']:
                    if mid in movies_cb['movie_id'].values:
                        idx_mid = movies_cb.index[movies_cb['movie_id'] == mid][0]
                        cb_scores[mid] = cosine_sim[seed_idx][idx_mid]
                    else:
                        cb_scores[mid] = 0
            else:
                cb_scores = {mid: 0 for mid in unseen_movies['movie_id']}

            hybrid_scores = {mid: 0.6*cf_scores.get(mid,0)+0.4*cb_scores.get(mid,0) for mid in unseen_movies['movie_id']}
            top_ids = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:10]
            hybrid_recs = movies[movies['movie_id'].isin(top_ids)].reset_index(drop=True)

            st.markdown("### Top 10 Hybrid Recommendations")
            display_movies(hybrid_recs)

            st.markdown("### Rate any of the above recommendations")
            for _, r in hybrid_recs.iterrows():
                mid = int(r['movie_id'])
                title = r['title']
                key_slider = f"hyb_slider_{mid}"
                key_btn = f"hyb_btn_{mid}"
                rating_val = st.slider(f"Rate '{title}'", 1, 5, 3, key=key_slider)
                if st.button(f"Submit rating for '{title}'", key=key_btn):
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO ratings (user_id, movie_id, rating, rating_time)
                            VALUES (:uid, :mid, :rating, NOW())
                            ON DUPLICATE KEY UPDATE rating=:rating, rating_time=NOW()
                        """), {"uid": user_id, "mid": mid, "rating": rating_val})
                    st.success(f"Rated '{title}' {rating_val} ★")
                    rerun_app()

            if seed_title:
                mid_seed = int(movies_cb[movies_cb['title'] == seed_title]['movie_id'].values[0])
                with engine.begin() as conn:
                    conn.execute(text("""
                        INSERT IGNORE INTO user_search_history (user_id, movie_id, search_time)
                        VALUES (:uid, :mid, NOW())
                    """), {"uid": user_id, "mid": mid_seed})
                rerun_app()

# ---------------- TAB 4 & 5 (Friends Searched & Watched) ----------------
# (code remains the same, but all INSERT/UPDATE use engine.begin() now)
# ... You can copy your original tab_fsearch and tab_fwatched blocks, 
# just make sure any conn.execute uses "with engine.begin() as conn"



# ---------------- TAB 4: FRIENDS SEARCHED ----------------
with tab_fsearch:
    st.subheader("🔎 Movies Your Friends Searched")
    # get list of friends' searches (most recent first)
    friend_searches = pd.read_sql(text("""
        SELECT s.user_id, u.username AS searched_by, s.movie_id, m.title, s.search_time
        FROM user_search_history s
        JOIN users u ON s.user_id = u.user_id
        JOIN movies m ON s.movie_id = m.movie_id
        WHERE s.user_id IN (SELECT friend_id FROM friends WHERE user_id=:uid)
        ORDER BY s.search_time DESC
        LIMIT 50
    """), engine, params={"uid": user_id})

    if friend_searches.empty:
        friends_list = get_following(user_id)
        if not friends_list:
            st.info("Follow friends to see their searches.")
        else:
            st.info("Your friends haven't searched any movies yet.")
    else:
        # show 'Already searched by you' status
        my_searches = pd.read_sql(text("SELECT movie_id FROM user_search_history WHERE user_id = :uid"),
                                   engine, params={"uid": user_id})['movie_id'].tolist()
        # adapt display DataFrame columns to match display_movies expectations
        disp_df = friend_searches[['movie_id', 'title', 'searched_by']].rename(columns={'title': 'title'})
        display_movies(disp_df, show_meta=True)

# ---------------- TAB 5: FRIENDS WATCHED ----------------
with tab_fwatched:
    st.subheader("⭐ Movies Your Friends Watched (Rated)")
    friend_ratings = pd.read_sql(text("""
        SELECT r.user_id, u.username, r.movie_id, m.title, r.rating, r.rating_time
        FROM ratings r
        JOIN users u ON r.user_id = u.user_id
        JOIN movies m ON r.movie_id = m.movie_id
        WHERE r.user_id IN (SELECT friend_id FROM friends WHERE user_id=:uid)
        ORDER BY r.rating_time DESC
        LIMIT 50
    """), engine, params={"uid": user_id})

    if friend_ratings.empty:
        friends_list = get_following(user_id)
        if not friends_list:
            st.info("Follow friends to see what they've watched/rated.")
        else:
            st.info("Your friends haven't rated any movies yet.")
    else:
        # provide display and option for current user to rate any of the same movies (if not rated yet by them)
        display_movies(friend_ratings, show_meta=True)
        st.markdown("### Rate the same movie (if you want)")
        # build a set of movie ids friend rated for quick access
        friend_movie_ids = friend_ratings['movie_id'].unique().tolist()
        # find which of these the current user hasn't rated yet
        my_ratings_db = pd.read_sql(text("SELECT user_id AS userId, movie_id AS movieId, rating FROM ratings WHERE user_id = :uid"),
                                    engine, params={"uid": user_id})
        my_rated_ids = my_ratings_db['movieId'].tolist()
        to_rate_ids = [mid for mid in friend_movie_ids if mid not in my_rated_ids]
        if to_rate_ids:
            to_rate_movies = movies[movies['movie_id'].isin(to_rate_ids)]
            sel_titles = to_rate_movies['title'].tolist()
            choice = st.selectbox("Select a movie from friends' list to rate", ["Select..."] + sel_titles, key="fw_rate_select")
            if choice != "Select..." and choice:
                mid_choice = int(to_rate_movies[to_rate_movies['title'] == choice]['movie_id'].values[0])
                val = st.slider(f"Rate '{choice}'", 1, 5, 3, key=f"fw_slider_{mid_choice}")
                if st.button(f"Submit rating for '{choice}'", key=f"fw_btn_{mid_choice}"):
                    with engine.connect() as conn:
                        conn.execute(text("""
                            INSERT INTO ratings (user_id, movie_id, rating, rating_time)
                            VALUES (:uid, :mid, :rating, NOW())
                            ON DUPLICATE KEY UPDATE rating=:rating, rating_time=NOW()
                        """), {"uid": user_id, "mid": mid_choice, "rating": val})
                    st.success(f"You rated '{choice}' {val} ★")
                    rerun_app()
        else:
            st.caption("You've already rated the movies your friends have recently rated.")

# End of file