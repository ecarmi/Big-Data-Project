# 08_run_text_alignment_gap.py

import pandas as pd
import numpy as np


# ============================================================
# 1. File paths
# ============================================================

INPUT_FILE = "philadelphia_restaurants_consolidated.csv"

OUTPUT_RESTAURANT_ALIGNMENT = "philadelphia_restaurants_with_text_alignment.csv"
OUTPUT_ALIGNMENT_BY_STAR = "text_alignment_by_star.csv"
OUTPUT_ALIGNMENT_CATEGORY_SUMMARY = "text_alignment_category_summary.csv"
OUTPUT_ALIGNMENT_DIRECTION_SUMMARY = "text_alignment_direction_summary.csv"
OUTPUT_LOW_ALIGNMENT = "low_text_alignment_restaurants.csv"


# ============================================================
# 2. Helper functions
# ============================================================

def convert_sentiment_to_1_5_scale(sentiment_score):
    """
    The dataset's avg_sentiment / avg_review_sentiment_polarity appears to use
    a polarity-style sentiment scale, usually ranging from about -1 to 1.

    We convert it to a 1-5 implied rating scale:

        -1 maps to 1
         0 maps to 3
         1 maps to 5

    Formula:
        text_implied_rating = 3 + 2 * sentiment_score

    This allows us to compare text sentiment directly with Yelp star ratings.
    """

    return 3 + 2 * sentiment_score


def classify_alignment(abs_gap):
    """
    Classify the absolute difference between actual Yelp rating and
    text-implied rating.

    High alignment:
        actual rating and text-implied rating differ by at most 0.50 stars

    Moderate alignment:
        difference is greater than 0.50 but at most 1.00 star

    Low alignment:
        difference is greater than 1.00 star
    """

    if abs_gap <= 0.50:
        return "High alignment"
    elif abs_gap <= 1.00:
        return "Moderate alignment"
    else:
        return "Low alignment"


def classify_gap_direction(gap):
    """
    Classify the direction of the gap.

    Positive gap:
        actual Yelp rating is higher than the sentiment-implied rating

    Negative gap:
        actual Yelp rating is lower than the sentiment-implied rating
    """

    if gap > 0.25:
        return "Rating higher than text sentiment"
    elif gap < -0.25:
        return "Rating lower than text sentiment"
    else:
        return "Rating and text sentiment similar"


def make_rating_group(stars):
    if stars <= 2.5:
        return "Low-rated"
    elif stars <= 3.5:
        return "Middle-rated"
    else:
        return "High-rated"


# ============================================================
# 3. Main workflow
# ============================================================

def main():
    print("Loading consolidated restaurant dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset shape: {df.shape[0]:,} rows and {df.shape[1]:,} columns")

    required_cols = [
        "business_id",
        "name",
        "stars",
        "review_count",
        "avg_sentiment"
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"The input file is missing required columns: {missing_cols}")

    # Keep only observations with usable rating and sentiment information.
    df = df.copy()
    df = df[df["stars"].notna()].copy()
    df = df[df["avg_sentiment"].notna()].copy()

    print(f"Restaurants with usable rating and sentiment data: {len(df):,}")

    # ========================================================
    # 4. Convert sentiment into comparable 1-5 scale
    # ========================================================

    df["text_implied_rating"] = convert_sentiment_to_1_5_scale(df["avg_sentiment"])

    # Clip to valid 1-5 range in case any sentiment value falls slightly outside.
    df["text_implied_rating"] = df["text_implied_rating"].clip(lower=1, upper=5)

    # ========================================================
    # 5. Compute text-rating alignment gap
    # ========================================================

    df["text_alignment_gap"] = df["stars"] - df["text_implied_rating"]

    df["abs_text_alignment_gap"] = df["text_alignment_gap"].abs()

    df["text_alignment_level"] = df["abs_text_alignment_gap"].apply(
        classify_alignment
    )

    df["text_alignment_direction"] = df["text_alignment_gap"].apply(
        classify_gap_direction
    )

    df["rating_group_3"] = df["stars"].apply(make_rating_group)

    # ========================================================
    # 6. Summary by exact Yelp star rating
    # ========================================================

    alignment_by_star = (
        df
        .groupby("stars")
        .agg(
            restaurant_count=("business_id", "count"),
            avg_actual_rating=("stars", "mean"),
            avg_sentiment=("avg_sentiment", "mean"),
            avg_text_implied_rating=("text_implied_rating", "mean"),
            avg_text_alignment_gap=("text_alignment_gap", "mean"),
            avg_abs_text_alignment_gap=("abs_text_alignment_gap", "mean"),
            avg_review_count=("review_count", "mean"),
            avg_review_stars=("avg_review_stars", "mean") if "avg_review_stars" in df.columns else ("stars", "mean"),
            avg_positive_review_share=("positive_review_share", "mean") if "positive_review_share" in df.columns else ("stars", "mean"),
            avg_negative_review_share=("negative_review_share", "mean") if "negative_review_share" in df.columns else ("stars", "mean")
        )
        .reset_index()
        .sort_values("stars")
    )

    # Remove fallback columns if they were created only because the original variables were missing.
    if "avg_review_stars" not in df.columns:
        alignment_by_star = alignment_by_star.drop(columns=["avg_review_stars"])

    if "positive_review_share" not in df.columns:
        alignment_by_star = alignment_by_star.drop(columns=["avg_positive_review_share"])

    if "negative_review_share" not in df.columns:
        alignment_by_star = alignment_by_star.drop(columns=["avg_negative_review_share"])

    # ========================================================
    # 7. Summary by three broad rating groups
    # ========================================================

    alignment_by_rating_group = (
        df
        .groupby("rating_group_3")
        .agg(
            restaurant_count=("business_id", "count"),
            avg_actual_rating=("stars", "mean"),
            avg_sentiment=("avg_sentiment", "mean"),
            avg_text_implied_rating=("text_implied_rating", "mean"),
            avg_text_alignment_gap=("text_alignment_gap", "mean"),
            avg_abs_text_alignment_gap=("abs_text_alignment_gap", "mean"),
            avg_review_count=("review_count", "mean")
        )
        .reset_index()
    )

    rating_group_order = ["Low-rated", "Middle-rated", "High-rated"]

    alignment_by_rating_group["rating_group_3"] = pd.Categorical(
        alignment_by_rating_group["rating_group_3"],
        categories=rating_group_order,
        ordered=True
    )

    alignment_by_rating_group = alignment_by_rating_group.sort_values("rating_group_3")

    # ========================================================
    # 8. Summary by alignment level
    # ========================================================

    alignment_category_summary = (
        df
        .groupby("text_alignment_level")
        .agg(
            restaurant_count=("business_id", "count"),
            avg_actual_rating=("stars", "mean"),
            avg_sentiment=("avg_sentiment", "mean"),
            avg_text_implied_rating=("text_implied_rating", "mean"),
            avg_text_alignment_gap=("text_alignment_gap", "mean"),
            avg_abs_text_alignment_gap=("abs_text_alignment_gap", "mean"),
            avg_review_count=("review_count", "mean")
        )
        .reset_index()
    )

    alignment_category_summary["share_of_restaurants"] = (
        alignment_category_summary["restaurant_count"]
        / alignment_category_summary["restaurant_count"].sum()
    )

    alignment_level_order = [
        "High alignment",
        "Moderate alignment",
        "Low alignment"
    ]

    alignment_category_summary["text_alignment_level"] = pd.Categorical(
        alignment_category_summary["text_alignment_level"],
        categories=alignment_level_order,
        ordered=True
    )

    alignment_category_summary = alignment_category_summary.sort_values(
        "text_alignment_level"
    )

    # ========================================================
    # 9. Summary by alignment direction
    # ========================================================

    alignment_direction_summary = (
        df
        .groupby("text_alignment_direction")
        .agg(
            restaurant_count=("business_id", "count"),
            avg_actual_rating=("stars", "mean"),
            avg_sentiment=("avg_sentiment", "mean"),
            avg_text_implied_rating=("text_implied_rating", "mean"),
            avg_text_alignment_gap=("text_alignment_gap", "mean"),
            avg_abs_text_alignment_gap=("abs_text_alignment_gap", "mean"),
            avg_review_count=("review_count", "mean")
        )
        .reset_index()
    )

    alignment_direction_summary["share_of_restaurants"] = (
        alignment_direction_summary["restaurant_count"]
        / alignment_direction_summary["restaurant_count"].sum()
    )

    # ========================================================
    # 10. Low-alignment restaurants
    # ========================================================

    low_alignment_df = (
        df[df["text_alignment_level"] == "Low alignment"]
        .copy()
        .sort_values("abs_text_alignment_gap", ascending=False)
    )

    selected_low_alignment_cols = [
        "business_id",
        "name",
        "address",
        "city",
        "stars",
        "review_count",
        "avg_review_stars",
        "avg_sentiment",
        "avg_review_sentiment_subjectivity",
        "positive_review_share",
        "negative_review_share",
        "text_implied_rating",
        "text_alignment_gap",
        "abs_text_alignment_gap",
        "text_alignment_direction",
        "text_alignment_level",
        "categories"
    ]

    selected_low_alignment_cols = [
        col for col in selected_low_alignment_cols if col in low_alignment_df.columns
    ]

    low_alignment_df = low_alignment_df[selected_low_alignment_cols]

    # ========================================================
    # 11. Save outputs
    # ========================================================

    df.to_csv(OUTPUT_RESTAURANT_ALIGNMENT, index=False)
    alignment_by_star.to_csv(OUTPUT_ALIGNMENT_BY_STAR, index=False)
    alignment_category_summary.to_csv(OUTPUT_ALIGNMENT_CATEGORY_SUMMARY, index=False)
    alignment_direction_summary.to_csv(OUTPUT_ALIGNMENT_DIRECTION_SUMMARY, index=False)
    low_alignment_df.to_csv(OUTPUT_LOW_ALIGNMENT, index=False)

    # Also save rating-group summary as an extra useful table.
    alignment_by_rating_group.to_csv("text_alignment_by_rating_group.csv", index=False)

    print("\nDone. Output files created:")
    print(f"1. {OUTPUT_RESTAURANT_ALIGNMENT}")
    print(f"2. {OUTPUT_ALIGNMENT_BY_STAR}")
    print(f"3. text_alignment_by_rating_group.csv")
    print(f"4. {OUTPUT_ALIGNMENT_CATEGORY_SUMMARY}")
    print(f"5. {OUTPUT_ALIGNMENT_DIRECTION_SUMMARY}")
    print(f"6. {OUTPUT_LOW_ALIGNMENT}")

    print("\nText alignment by star rating:")
    print(alignment_by_star.to_string(index=False))

    print("\nText alignment by broad rating group:")
    print(alignment_by_rating_group.to_string(index=False))

    print("\nText alignment category summary:")
    print(alignment_category_summary.to_string(index=False))

    print("\nText alignment direction summary:")
    print(alignment_direction_summary.to_string(index=False))


if __name__ == "__main__":
    main()