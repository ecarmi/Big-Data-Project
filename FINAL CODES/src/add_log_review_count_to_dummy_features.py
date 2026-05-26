from pathlib import Path

import pandas as pd


DUMMY_FEATURES_PATH = Path("outputs/philadelphia_no_review_category_attribute_dummy_features.csv")
CONSOLIDATED_PATH = Path("data/processed/philadelphia_restaurants_consolidated.csv")
OUTPUT_PATH = Path(
    "outputs/philadelphia_no_review_category_attribute_dummy_features_with_log_review_count.csv"
)


def main() -> None:
    dummy_features = pd.read_csv(DUMMY_FEATURES_PATH)
    log_review_counts = pd.read_csv(
        CONSOLIDATED_PATH, usecols=["business_id", "log_review_count"]
    )

    if "log_review_count" in dummy_features.columns:
        dummy_features = dummy_features.drop(columns=["log_review_count"])

    enriched = dummy_features.merge(log_review_counts, on="business_id", how="left")
    if enriched["log_review_count"].isna().any():
        missing_count = enriched["log_review_count"].isna().sum()
        raise ValueError(f"Missing log_review_count for {missing_count:,} rows")

    # Keep the clustering variable near the original business-level predictors.
    columns = enriched.columns.tolist()
    columns.remove("log_review_count")
    insert_after = columns.index("is_open") + 1
    columns.insert(insert_after, "log_review_count")
    enriched = enriched[columns]

    enriched.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(enriched):,} rows and {len(enriched.columns):,} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
