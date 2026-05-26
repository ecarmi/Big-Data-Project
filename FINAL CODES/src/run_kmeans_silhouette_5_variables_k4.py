from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler


INPUT_PATH = Path(
    "outputs/philadelphia_no_review_category_attribute_dummy_features_with_log_review_count_price_range.csv"
)
OUTPUT_DIR = Path("outputs/kmeans_5_variable_k4")
OUTPUT_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_clustered.csv"
SUMMARY_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_summary.txt"
FEATURES = [
    "log_review_count",
    "avg_tip_compliments",
    "checkin_count",
    "photo_count",
    "inside_photo_share",
]
N_CLUSTERS = 4
RANDOM_STATE = 42


def star_group(stars: pd.Series) -> pd.Series:
    return pd.cut(
        stars,
        bins=[0, 3.0, 3.5, 5.0],
        labels=["low", "middle", "high"],
        include_lowest=True,
    )


def cluster_majority_accuracy(
    true_groups: pd.Series, cluster_labels
) -> tuple[float, dict[int, str]]:
    cluster_to_group = {}
    predicted_groups = []

    for cluster in sorted(set(cluster_labels)):
        in_cluster = cluster_labels == cluster
        majority_group = true_groups[in_cluster].mode().iloc[0]
        cluster_to_group[int(cluster)] = str(majority_group)

    for cluster in cluster_labels:
        predicted_groups.append(cluster_to_group[int(cluster)])

    return accuracy_score(true_groups.astype(str), predicted_groups), cluster_to_group


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    restaurants = pd.read_csv(INPUT_PATH)
    model_data = restaurants[["business_id", "stars"] + FEATURES].dropna().copy()
    true_groups = star_group(model_data["stars"])

    scaled_features = StandardScaler().fit_transform(model_data[FEATURES])
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=25)
    labels = kmeans.fit_predict(scaled_features)
    model_data["kmeans_cluster"] = labels

    accuracy, cluster_mapping = cluster_majority_accuracy(true_groups, labels)
    silhouette = silhouette_score(scaled_features, labels)

    clustered = restaurants.merge(
        model_data[["business_id", "kmeans_cluster"]], on="business_id", how="left"
    )
    clustered.to_csv(OUTPUT_PATH, index=False)

    cluster_summary = (
        model_data.assign(star_group=true_groups)
        .groupby("kmeans_cluster", as_index=False)
        .agg(
            restaurant_count=("business_id", "count"),
            avg_stars=("stars", "mean"),
            avg_log_review_count=("log_review_count", "mean"),
            avg_tip_compliments=("avg_tip_compliments", "mean"),
            avg_checkin_count=("checkin_count", "mean"),
            avg_photo_count=("photo_count", "mean"),
            avg_inside_photo_share=("inside_photo_share", "mean"),
            low_share=("star_group", lambda values: (values == "low").mean()),
            middle_share=("star_group", lambda values: (values == "middle").mean()),
            high_share=("star_group", lambda values: (values == "high").mean()),
        )
        .sort_values("kmeans_cluster")
    )

    with SUMMARY_PATH.open("w", encoding="utf-8") as summary_file:
        summary_file.write("K-means with 5 variables and k=4\n")
        summary_file.write(f"Input file: {INPUT_PATH}\n")
        summary_file.write(f"Output file: {OUTPUT_PATH}\n")
        summary_file.write(f"Features: {', '.join(FEATURES)}\n")
        summary_file.write("Dropped photo_file_count because it duplicates photo_count in this dataset\n")
        summary_file.write(f"Clusters: {N_CLUSTERS}\n")
        summary_file.write(f"Rows used: {len(model_data):,}\n")
        summary_file.write(f"Rows skipped for missing features: {len(restaurants) - len(model_data):,}\n")
        summary_file.write(f"Silhouette score: {silhouette:.5f}\n")
        summary_file.write(f"Star-group majority accuracy: {accuracy:.5f}\n")
        summary_file.write(f"Cluster to star-group mapping: {cluster_mapping}\n\n")
        summary_file.write("Cluster summary:\n")
        summary_file.write(
            cluster_summary.to_string(index=False, float_format=lambda value: f"{value:.5f}")
        )
        summary_file.write("\n")

    print(f"Silhouette score: {silhouette:.5f}")
    print(f"Star-group majority accuracy: {accuracy:.5f}")
    print(cluster_summary.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print(f"Wrote clustered data to {OUTPUT_PATH}")
    print(f"Wrote summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
