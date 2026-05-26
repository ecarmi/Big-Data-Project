# K-means Analysis Notes

Course instruction context: the final paper should include at least one unsupervised learning method and explain the dataset, model, and results concisely. Treat this file as factual analysis notes and rewrite the final paper section in your own words.

## K-means Model Used

- Code: `src/run_kmeans_silhouette_5_variables_k4.py`
- Plot code: `src/plot_kmeans_silhouette_5var_k4_2d.py`
- Input data: `outputs/philadelphia_no_review_category_attribute_dummy_features_with_log_review_count_price_range.csv`
- Clustered output: `outputs\kmeans_5_variable_k4\kmeans_silhouette_5_variables_k4_clustered.csv`
- Summary output: `outputs\kmeans_5_variable_k4\kmeans_silhouette_5_variables_k4_summary.txt`
- Number of clusters: `k = 4`
- Observations used: `5,854`
- Features used:
  - `log_review_count`
  - `avg_tip_compliments`
  - `checkin_count`
  - `photo_count`
  - `inside_photo_share`

This version drops `photo_file_count` because it is exactly equal to `photo_count` in this dataset. Keeping only one avoids double-weighting photo volume in the K-means distance calculation.

## Main Scores

- Silhouette score: `0.47848`
- Star-group majority accuracy: `0.46481`

## Cluster Summary

```text
 kmeans_cluster  restaurant_count  avg_stars  avg_log_review_count  avg_tip_compliments  avg_checkin_count  avg_photo_count  avg_inside_photo_share     low  middle    high
              0              4016    3.44808               3.26937              0.00462           65.90712          1.16833                 0.00956 0.37649 0.21439 0.40911
              1              1597    3.78240               4.79666              0.01033          355.28616          6.53100                 0.52691 0.16468 0.26612 0.56919
              2               178    3.98034               6.76984              0.01176         2202.65169         39.76404                 0.29069 0.03933 0.16854 0.79213
              3                63    3.50794               3.20806              0.44498           52.98413          1.68254                 0.09682 0.34921 0.20635 0.44444
```

## Interpretation

Cluster 0 is the largest and lowest-engagement group. These restaurants have low review volume, low check-ins, very few photos, and almost no inside-photo share.

Cluster 1 is a medium-to-high engagement group with many inside photos. It has more reviews, check-ins, and photos than Cluster 0, and the highest inside-photo share.

Cluster 2 is a small, very high-engagement group. It has the highest average review volume, check-ins, and photos by a large margin, as well as the highest average star rating.

Cluster 3 is a small unusual group driven mainly by tip compliments. It has very high average tip compliments but relatively low check-ins and photo volume.

## Main Takeaway

The K-means model segments Philadelphia restaurants mostly by user engagement and photo behavior. The clearest pattern is that restaurants with very high review volume, check-ins, and photos form a small cluster with the highest average rating. The clusters are not pure rating categories because K-means is unsupervised and was not trained to predict ratings directly.

## Graph Outputs

- `outputs/kmeans_5_variable_k4/kmeans_silhouette_5_variables_k4_2d_pca.png`
- `outputs/kmeans_5_variable_k4/kmeans_silhouette_5_variables_k4_2d_pca_transparent.png`
- `outputs/kmeans_5_variable_k4/kmeans_silhouette_5_variables_k4_2d_pca_no_outliers.png`
- `outputs/kmeans_5_variable_k4/kmeans_silhouette_5_variables_k4_2d_pca_facets.png`
- `outputs/kmeans_5_variable_k4/kmeans_silhouette_5_variables_k4_pca_loadings.txt`
