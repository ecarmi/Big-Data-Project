from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = Path("outputs/kmeans_5_variable_k4")
INPUT_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_clustered.csv"
OUTPUT_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_2d_pca.png"
TRANSPARENT_OUTPUT_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_2d_pca_transparent.png"
NO_OUTLIERS_OUTPUT_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_2d_pca_no_outliers.png"
NO_OUTLIERS_TRANSPARENT_OUTPUT_PATH = (
    OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_2d_pca_no_outliers_transparent.png"
)
FACET_OUTPUT_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_2d_pca_facets.png"
COORDINATES_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_2d_pca_coordinates.csv"
LOADINGS_CSV_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_pca_loadings.csv"
LOADINGS_TXT_PATH = OUTPUT_DIR / "kmeans_silhouette_5_variables_k4_pca_loadings.txt"
FEATURES = [
    "log_review_count",
    "avg_tip_compliments",
    "checkin_count",
    "photo_count",
    "inside_photo_share",
]


def main() -> None:
    clustered = pd.read_csv(INPUT_PATH)
    plot_data = clustered[["business_id", "stars", "kmeans_cluster"] + FEATURES].dropna().copy()

    scaled_features = StandardScaler().fit_transform(plot_data[FEATURES])
    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(scaled_features)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=FEATURES,
        columns=["PC1_loading", "PC2_loading"],
    ).reset_index().rename(columns={"index": "variable"})
    loadings["PC1_abs_loading"] = loadings["PC1_loading"].abs()
    loadings["PC2_abs_loading"] = loadings["PC2_loading"].abs()
    loadings.to_csv(LOADINGS_CSV_PATH, index=False)
    with LOADINGS_TXT_PATH.open("w", encoding="utf-8") as loadings_file:
        loadings_file.write("PCA loadings for 5-variable K-means 2D graph\n")
        loadings_file.write(f"Input file: {INPUT_PATH}\n")
        loadings_file.write(f"Features: {', '.join(FEATURES)}\n")
        loadings_file.write(f"PC1 explained variance: {pca.explained_variance_ratio_[0]:.5f}\n")
        loadings_file.write(f"PC2 explained variance: {pca.explained_variance_ratio_[1]:.5f}\n")
        loadings_file.write(f"Total explained variance: {pca.explained_variance_ratio_.sum():.5f}\n\n")
        loadings_file.write(loadings.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
        loadings_file.write("\n")

    plot_data["pc1"] = coordinates[:, 0]
    plot_data["pc2"] = coordinates[:, 1]
    plot_data.to_csv(COORDINATES_PATH, index=False)
    pc1_low, pc1_high = plot_data["pc1"].quantile([0.01, 0.99])
    pc2_low, pc2_high = plot_data["pc2"].quantile([0.01, 0.99])
    no_outlier_data = plot_data[
        plot_data["pc1"].between(pc1_low, pc1_high)
        & plot_data["pc2"].between(pc2_low, pc2_high)
    ].copy()

    colors = {
        0: "#0072B2",
        1: "#009E73",
        2: "#D55E00",
        3: "#CC79A7",
    }
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 6), dpi=190)
    for cluster in [0, 1, 2, 3]:
        cluster_data = plot_data[plot_data["kmeans_cluster"] == cluster]
        ax.scatter(
            cluster_data["pc1"],
            cluster_data["pc2"],
            s=9,
            alpha=0.18,
            color=colors.get(cluster, "#555555"),
            edgecolors="none",
            label=f"Cluster {cluster} (n={len(cluster_data):,})",
        )

    for cluster in sorted(plot_data["kmeans_cluster"].dropna().unique()):
        cluster = int(cluster)
        cluster_data = plot_data[plot_data["kmeans_cluster"] == cluster]
        center = cluster_data[["pc1", "pc2"]].mean()
        ax.scatter(
            center["pc1"],
            center["pc2"],
            s=120,
            marker="X",
            color=colors.get(cluster, "#555555"),
            edgecolors="black",
            linewidths=0.8,
        )
        ax.text(center["pc1"], center["pc2"], f" {cluster}", fontsize=10, fontweight="bold")

    ax.set_title("K-means Clusters Projected to Two Dimensions")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.grid(True, linewidth=0.5, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    fig.savefig(TRANSPARENT_OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)

    for output_path, point_alpha, point_size in [
        (NO_OUTLIERS_OUTPUT_PATH, 0.45, 12),
        (NO_OUTLIERS_TRANSPARENT_OUTPUT_PATH, 0.22, 9),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=190)
        for cluster in [0, 1, 2, 3]:
            cluster_data = no_outlier_data[no_outlier_data["kmeans_cluster"] == cluster]
            ax.scatter(
                cluster_data["pc1"],
                cluster_data["pc2"],
                s=point_size,
                alpha=point_alpha,
                color=colors.get(cluster, "#555555"),
                edgecolors="none",
                label=f"Cluster {cluster} (n={len(cluster_data):,})",
            )

        for cluster in sorted(no_outlier_data["kmeans_cluster"].dropna().unique()):
            cluster = int(cluster)
            cluster_data = no_outlier_data[no_outlier_data["kmeans_cluster"] == cluster]
            center = cluster_data[["pc1", "pc2"]].mean()
            ax.scatter(
                center["pc1"],
                center["pc2"],
                s=120,
                marker="X",
                color=colors.get(cluster, "#555555"),
                edgecolors="black",
                linewidths=0.8,
            )
            ax.text(center["pc1"], center["pc2"], f" {cluster}", fontsize=10, fontweight="bold")

        ax.set_title("K-means Clusters Projected to Two Dimensions")
        ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
        ax.grid(True, linewidth=0.5, alpha=0.25)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=170, sharex=True, sharey=True)
    axes = axes.ravel()
    x_limits = (plot_data["pc1"].min() - 0.5, plot_data["pc1"].max() + 0.5)
    y_limits = (plot_data["pc2"].min() - 0.5, plot_data["pc2"].max() + 0.5)
    for ax, cluster in zip(axes, sorted(plot_data["kmeans_cluster"].dropna().unique())):
        cluster = int(cluster)
        cluster_data = plot_data[plot_data["kmeans_cluster"] == cluster]
        ax.scatter(
            plot_data["pc1"],
            plot_data["pc2"],
            s=6,
            alpha=0.08,
            color="#999999",
            edgecolors="none",
        )
        ax.scatter(
            cluster_data["pc1"],
            cluster_data["pc2"],
            s=13,
            alpha=0.72,
            color=colors.get(cluster, "#555555"),
            edgecolors="none",
        )
        ax.set_title(f"Cluster {cluster} (n={len(cluster_data):,})")
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.grid(True, linewidth=0.4, alpha=0.22)
    fig.supxlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    fig.supylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    fig.suptitle("K-means Clusters, One Panel per Cluster")
    fig.tight_layout()
    fig.savefig(FACET_OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote 2D cluster plot to {OUTPUT_PATH}")
    print(f"Wrote transparent 2D cluster plot to {TRANSPARENT_OUTPUT_PATH}")
    print(f"Wrote no-outliers 2D cluster plot to {NO_OUTLIERS_OUTPUT_PATH}")
    print(
        "Wrote transparent no-outliers 2D cluster plot to "
        f"{NO_OUTLIERS_TRANSPARENT_OUTPUT_PATH}"
    )
    print(f"Wrote faceted 2D cluster plot to {FACET_OUTPUT_PATH}")
    print(f"Wrote PCA coordinates to {COORDINATES_PATH}")
    print(f"Wrote PCA loadings to {LOADINGS_CSV_PATH}")
    print(f"Wrote PCA loadings summary to {LOADINGS_TXT_PATH}")
    print(f"Rows plotted after outlier trim: {len(no_outlier_data):,} of {len(plot_data):,}")
    print(f"PC1 explained variance: {explained[0]:.5f}")
    print(f"PC2 explained variance: {explained[1]:.5f}")
    print(f"Total 2D explained variance: {explained.sum():.5f}")


if __name__ == "__main__":
    main()
