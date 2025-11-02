-- ------------------------
-- Database: movie_recommender
-- ------------------------
CREATE DATABASE IF NOT EXISTS movie_recommender;
USE movie_recommender;

-- Movies table
CREATE TABLE movies (
    movie_id INT PRIMARY KEY,
    title VARCHAR(255) NOT NULL
);

-- USERS TABLE
CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL,
    UNIQUE(username),
    UNIQUE(email)
);

-- RATINGS TABLE
CREATE TABLE ratings (
    rating_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    movie_id INT NOT NULL,
    rating FLOAT NOT NULL DEFAULT 5.0,
    rating_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(user_id)
);

-- FRIENDS TABLE
CREATE TABLE friends (
    user_id INT NOT NULL,
    friend_id INT NOT NULL,
    PRIMARY KEY(user_id, friend_id),
    FOREIGN KEY(user_id) REFERENCES users(user_id),
    FOREIGN KEY(friend_id) REFERENCES users(user_id)
);


CREATE TABLE IF NOT EXISTS user_search_history (
    user_id INT,
    movie_id INT,
    search_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, movie_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);
