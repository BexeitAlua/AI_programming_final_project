-- name: q01_table_sizes
SELECT 'games' AS table_name, COUNT(*) AS n_rows FROM games
UNION ALL SELECT 'users', COUNT(*) FROM users
UNION ALL SELECT 'reviews', COUNT(*) FROM reviews
UNION ALL SELECT 'tags', COUNT(*) FROM tags
UNION ALL SELECT 'game_tags', COUNT(*) FROM game_tags;

-- name: q02_top20_games_by_reviews
SELECT g.app_id, g.title, COUNT(*) AS review_count
FROM reviews AS r
JOIN games AS g ON g.app_id = r.app_id
GROUP BY g.app_id, g.title
ORDER BY review_count DESC
LIMIT 20;

-- name: q03_avg_hours_per_game
SELECT g.app_id, g.title,
       COUNT(*)               AS review_count,
       ROUND(AVG(r.hours), 1) AS avg_hours,
       ROUND(MAX(r.hours), 1) AS max_hours
FROM reviews AS r
JOIN games AS g ON g.app_id = r.app_id
GROUP BY g.app_id, g.title
HAVING COUNT(*) >= 1000
ORDER BY avg_hours DESC
LIMIT 20;

-- name: q04_popular_and_liked_games
SELECT g.app_id, g.title,
       COUNT(*)                              AS review_count,
       ROUND(AVG(r.is_recommended) * 100, 1) AS positive_pct
FROM reviews AS r
JOIN games AS g ON g.app_id = r.app_id
GROUP BY g.app_id, g.title
HAVING COUNT(*) >= 100 AND AVG(r.is_recommended) > 0.75
ORDER BY review_count DESC
LIMIT 20;

-- name: q05_activity_by_year
SELECT strftime('%Y', review_date)           AS year,
       COUNT(*)                              AS review_count,
       COUNT(DISTINCT user_id)               AS active_users,
       ROUND(AVG(is_recommended) * 100, 1)   AS positive_pct
FROM reviews
GROUP BY year
ORDER BY year;

-- name: q06_games_above_average_popularity
WITH game_counts AS (
    SELECT app_id, COUNT(*) AS review_count
    FROM reviews
    GROUP BY app_id
)
SELECT g.title, gc.review_count,
       ROUND((SELECT AVG(review_count) FROM game_counts), 1) AS avg_reviews_per_game
FROM game_counts AS gc
JOIN games AS g ON g.app_id = gc.app_id
WHERE gc.review_count > (SELECT AVG(review_count) FROM game_counts)
ORDER BY gc.review_count DESC
LIMIT 20;

-- name: q07_rank_within_price_category
WITH game_stats AS (
    SELECT g.app_id, g.title, g.price_final,
           CASE
               WHEN g.price_final = 0  THEN '1 Free'
               WHEN g.price_final < 10 THEN '2 Under 10'
               WHEN g.price_final < 30 THEN '3 10-30'
               ELSE '4 30+'
           END AS price_category,
           COUNT(*) AS review_count
    FROM reviews AS r
    JOIN games AS g ON g.app_id = r.app_id
    GROUP BY g.app_id, g.title, g.price_final
),
ranked AS (
    SELECT *, RANK() OVER (PARTITION BY price_category ORDER BY review_count DESC) AS rank_in_category
    FROM game_stats
)
SELECT price_category, rank_in_category, title, price_final, review_count
FROM ranked
WHERE rank_in_category <= 5
ORDER BY price_category, rank_in_category;

-- name: q08_top_game_each_year
WITH yearly AS (
    SELECT strftime('%Y', review_date) AS year, app_id, COUNT(*) AS review_count
    FROM reviews
    GROUP BY year, app_id
),
ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY year ORDER BY review_count DESC) AS rn
    FROM yearly
)
SELECT r.year, g.title, r.review_count
FROM ranked AS r
JOIN games AS g ON g.app_id = r.app_id
WHERE r.rn = 1
ORDER BY r.year;

-- name: q09_tags_by_positive_share
WITH game_stats AS (
    SELECT app_id, COUNT(*) AS review_count, SUM(is_recommended) AS positive_count
    FROM reviews
    GROUP BY app_id
)
SELECT t.name AS tag,
       COUNT(*)                                                    AS n_games,
       SUM(gs.review_count)                                        AS review_count,
       ROUND(100.0 * SUM(gs.positive_count) / SUM(gs.review_count), 1) AS positive_pct
FROM game_stats AS gs
JOIN game_tags AS gt ON gt.app_id = gs.app_id
JOIN tags AS t ON t.tag_id = gt.tag_id
GROUP BY t.tag_id, t.name
HAVING COUNT(*) >= 50
ORDER BY review_count DESC
LIMIT 20;

-- name: q10_user_activity_segments
WITH user_counts AS (
    SELECT user_id, COUNT(*) AS n_reviews
    FROM reviews
    GROUP BY user_id
)
SELECT CASE
           WHEN n_reviews = 1  THEN '1'
           WHEN n_reviews <= 4 THEN '2-4'
           WHEN n_reviews <= 9 THEN '5-9'
           WHEN n_reviews <= 49 THEN '10-49'
           ELSE '50+'
       END AS reviews_per_user,
       COUNT(*)       AS n_users,
       SUM(n_reviews) AS n_reviews
FROM user_counts
GROUP BY reviews_per_user
ORDER BY MIN(n_reviews);