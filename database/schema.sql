PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS games (
    app_id          INTEGER PRIMARY KEY,
    title           TEXT    NOT NULL,
    date_release    TEXT,
    win             INTEGER,
    mac             INTEGER,
    linux           INTEGER,
    rating          TEXT,
    positive_ratio  INTEGER,
    user_reviews    INTEGER,
    price_final     REAL,
    price_original  REAL,
    discount        REAL,
    steam_deck      INTEGER,
    description     TEXT
);

CREATE TABLE IF NOT EXISTS users (
    user_id   INTEGER PRIMARY KEY,
    products  INTEGER,
    reviews   INTEGER
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id       INTEGER PRIMARY KEY,
    user_id         INTEGER NOT NULL,
    app_id          INTEGER NOT NULL,
    review_date     TEXT    NOT NULL,
    is_recommended  INTEGER NOT NULL CHECK (is_recommended IN (0, 1)),
    hours           REAL    CHECK (hours >= 0),
    FOREIGN KEY (user_id) REFERENCES users (user_id),
    FOREIGN KEY (app_id)  REFERENCES games (app_id)
);

CREATE TABLE IF NOT EXISTS tags (
    tag_id  INTEGER PRIMARY KEY,
    name    TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS game_tags (
    app_id  INTEGER NOT NULL,
    tag_id  INTEGER NOT NULL,
    PRIMARY KEY (app_id, tag_id),
    FOREIGN KEY (app_id) REFERENCES games (app_id),
    FOREIGN KEY (tag_id) REFERENCES tags (tag_id)
);

-- @indexes
CREATE INDEX IF NOT EXISTS idx_reviews_user ON reviews (user_id);
CREATE INDEX IF NOT EXISTS idx_reviews_game ON reviews (app_id);
CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews (review_date);
CREATE INDEX IF NOT EXISTS idx_game_tags_tag ON game_tags (tag_id);