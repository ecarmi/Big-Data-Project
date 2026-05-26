from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs/kmeans_5_variable_k4")
SUMMARY_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_summary.txt"
NOTES_PATH = OUTPUT_DIR / "kmeans_5_variable_k4_analysis_notes.md"
CLUSTERED_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_clustered.csv"
FEATURES = [
    "log_review_count",
    "avg_tip_compliments",
    "checkin_count",
    "photo_count",
    "inside_photo_share",
]


def main() -> None:
    clustered = pd.read_csv(CLUSTERED_PATH)
    cluster_summary = (
        clustered.groupby("kmeans_cluster", as_index=False)
        .agg(
            restaurant_count=("business_id", "count"),
            avg_stars=("stars", "mean"),
            avg_log_review_count=("log_review_count", "mean"),
            avg_tip_compliments=("avg_tip_compliments", "mean"),
            avg_checkin_count=("checkin_count", "mean"),
            avg_photo_count=("photo_count", "mean"),
            avg_inside_photo_share=("inside_photo_share", "mean"),
        )
        .sort_values("kmeans_cluster")
    )
    low_share = (
        clustered.assign(
            star_group=pd.cut(
                clustered["stars"],
                bins=[0, 3.0, 3.5, 5.0],
                labels=["low", "middle", "high"],
                include_lowest=True,
            )
        )
        .groupby("kmeans_cluster")["star_group"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reset_index()
    )
    cluster_summary = cluster_summary.merge(low_share, on="kmeans_cluster", how="left")

    with NOTES_PATH.open("w", encoding="utf-8") as notes:
        notes.write("# K-means Analysis Notes\n\n")
        notes.write(
            "Course instruction context: the final paper should include at least one "
            "unsupervised learning method and explain the dataset, model, and results "
            "concisely. Treat this file as factual analysis notes and rewrite the final "
            "paper section in your own words.\n\n"
        )
        notes.write("## K-means Model Used\n\n")
        notes.write("- Code: `src/run_kmeans_silhouette_5_variables_k4.py`\n")
        notes.write("- Plot code: `src/plot_kmeans_silhouette_5var_k4_2d.py`\n")
        notes.write("- Input data: `outputs/philadelphia_no_review_category_attribute_dummy_features_with_log_review_count_price_range.csv`\n")
        notes.write(f"- Clustered output: `{CLUSTERED_PATH}`\n")
        notes.write(f"- Summary output: `{SUMMARY_PATH}`\n")
        notes.write("- Number of clusters: `k = 4`\n")
        notes.write(f"- Observations used: `{len(clustered):,}`\n")
        notes.write("- Features used:\n")
        for feature in FEATURES:
            notes.write(f"  - `{feature}`\n")
        notes.write(
            "\nThis version drops `photo_file_count` because it is exactly equal to "
            "`photo_count` in this dataset. Keeping only one avoids double-weighting "
            "photo volume in the K-means distance calculation.\n\n"
        )
        notes.write("## Main Scores\n\n")
        notes.write("- Silhouette score: `0.47848`\n")
        notes.write("- Star-group majority accuracy: `0.46481`\n\n")
        notes.write("## Cluster Summary\n\n")
        notes.write("```text\n")
        notes.write(cluster_summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
        notes.write("\n```")
        notes.write("\n\n")
        notes.write("## Interpretation\n\n")
        notes.write(
            "Cluster 0 is the largest and lowest-engagement group. These restaurants "
            "have low review volume, low check-ins, very few photos, and almost no "
            "inside-photo share.\n\n"
        )
        notes.write(
            "Cluster 1 is a medium-to-high engagement group with many inside photos. "
            "It has more reviews, check-ins, and photos than Cluster 0, and the highest "
            "inside-photo share.\n\n"
        )
        notes.write(
            "Cluster 2 is a small, very high-engagement group. It has the highest "
            "average review volume, check-ins, and photos by a large margin, as well as "
            "the highest average star rating.\n\n"
        )
        notes.write(
            "Cluster 3 is a small unusual group driven mainly by tip compliments. "
            "It has very high average tip compliments but relatively low check-ins and "
            "photo volume.\n\n"
        )
        notes.write("## Main Takeaway\n\n")
        notes.write(
            "The K-means model segments Philadelphia restaurants mostly by user "
            "engagement and photo behavior. The clearest pattern is that restaurants "
            "with very high review volume, check-ins, and photos form a small cluster "
            "with the highest average rating. The clusters are not pure rating "
            "categories because K-means is unsupervised and was not trained to predict "
            "ratings directly.\n\n"
        )
        notes.write("## Graph Outputs\n\n")
        notes.write("- `outputs/kmeans_5_variable_k4/kmeans_silhouette_5_variables_k4_2d_pca.png`\n")
        notes.write("- `outputs/kmeans_5_variable_k4/kmeans_silhouette_5_variables_k4_2d_pca_transparent.png`\n")
        notes.write("- `outputs/kmeans_5_variable_k4/kmeans_silhouette_5_variables_k4_2d_pca_no_outliers.png`\n")
        notes.write("- `outputs/kmeans_5_variable_k4/kmeans_silhouette_5_variables_k4_2d_pca_facets.png`\n")
        notes.write("- `outputs/kmeans_5_variable_k4/kmeans_silhouette_5_variables_k4_pca_loadings.txt`\n")

    print(f"Wrote K-means analysis notes to {NOTES_PATH}")


if __name__ == "__main__":
    main()
