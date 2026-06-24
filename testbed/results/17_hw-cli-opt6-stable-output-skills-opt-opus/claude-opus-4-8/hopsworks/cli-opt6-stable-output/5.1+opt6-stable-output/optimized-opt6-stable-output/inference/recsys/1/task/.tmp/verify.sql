SELECT count(*) AS total_rows,
       count(DISTINCT user_id) AS users,
       min(rank) AS min_rank,
       max(rank) AS max_rank,
       count(DISTINCT rec_id) AS distinct_rec_ids
FROM recsda1abd_1
