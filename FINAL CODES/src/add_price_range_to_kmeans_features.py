from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "outputs/philadelphia_no_review_category_attribute_dummy_features_with_log_review_count.csv"
)
OUTPUT_PATH = Path(
    "outputs/philadelphia_no_review_category_attribute_dummy_features_with_log_review_count_price_range.csv"
)
PRICE_DUMMY_COLUMNS = {
    "attr_restaurantspricerange2_1": 1,
    "attr_restaurantspricerange2_2": 2,
    "attr_restaurantspricerange2_3": 3,
    "attr_restaurantspricerange2_4": 4,
}


def main() -> None:
    features = pd.read_csv(INPUT_PATH)

    available_price_columns = [
        column for column in PRICE_DUMMY_COLUMNS if column in features.columns
    ]
    if not available_price_columns:
        raise KeyError("No attr_restaurantspricerange2_* dummy columns found")

    features["price_range"] = pd.NA
    for column in available_price_columns:
        features.loc[features[column] == 1, "price_range"] = PRICE_DUMMY_COLUMNS[column]

    columns = features.columns.tolist()
    columns.remove("price_range")
    insert_after = columns.index("log_review_count") + 1
    columns.insert(insert_after, "price_range")
    features = features[columns]

    features.to_csv(OUTPUT_PATH, index=False)
    non_missing = features["price_range"].notna().sum()
    print(f"Wrote {len(features):,} rows and {len(features.columns):,} columns to {OUTPUT_PATH}")
    print(f"Rows with non-missing price_range: {non_missing:,}")
    print("price_range counts:")
    print(features["price_range"].value_counts(dropna=False).sort_index().to_string())


if __name__ == "__main__":
    main()
